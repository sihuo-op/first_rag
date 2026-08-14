"""KG 抽取管道入口：编排 rule_parser -> llm_extractor -> entity_resolver -> graph_store。"""
import logging
import time
from dataclasses import dataclass

from app.core.observability import get_tracer
from app.knowledge_graph.entity_resolver import EntityResolver, ResolvedEntity
from app.knowledge_graph.llm_extractor import extract_from_chunk, ExtractionResult
from app.knowledge_graph.rule_parser import COUNTRY_PREFIX, parse_document
from app.llm.providers import get_extraction_llm

logger = logging.getLogger(__name__)

tracer = get_tracer("kg.extract")


def _normalize_law_name(name: str) -> str:
    """法名归一化：去空白 + 去 '中华人民共和国' 前缀，用于 article 引用的跨法比对。"""
    name = name.strip()
    if name.startswith(COUNTRY_PREFIX):
        return name[len(COUNTRY_PREFIX):]
    return name


@dataclass
class ExtractionReport:
    document_id: str
    entities_count: int = 0
    relations_count: int = 0
    conflicts_count: int = 0
    duration_ms: int = 0


class KGExtractor:
    def __init__(self, store, embedding_fn, chunks_loader, document_loader, conflict_detector=None):
        """
        store: Neo4jStore
        embedding_fn: callable(str) -> list[float]
        chunks_loader: callable(document_id) -> list[dict]，每个 dict 含 id/content/char_start/char_end/document_id
        document_loader: callable(document_id) -> str，返回文档全文
        conflict_detector: ConflictDetector 或 None（None 时自动构造默认实例）
        """
        self.store = store
        self.embedding_fn = embedding_fn
        self.chunks_loader = chunks_loader
        self.document_loader = document_loader
        if conflict_detector is None:
            from app.knowledge_graph.conflict_detector import ConflictDetector
            conflict_detector = ConflictDetector(store, get_extraction_llm())
        self.conflict_detector = conflict_detector

    def run(self, document_id: str) -> ExtractionReport:
        start = time.time()
        with tracer.start_as_current_span("kg.extract.pipeline") as span:
            span.set_attribute("document.id", document_id)
            report = self._run_impl(document_id, span)
        report.duration_ms = int((time.time() - start) * 1000)
        return report

    def _run_impl(self, document_id: str, span) -> ExtractionReport:
        # Step 1: 加载 chunks + 文档全文
        chunks = self.chunks_loader(document_id)
        text = self.document_loader(document_id)
        span.set_attribute("chunks.count", len(chunks))

        # Step 2: 规则解析
        with tracer.start_as_current_span("kg.extract.rule_parse"):
            parsed = parse_document(text=text, document_id=document_id, chunks=chunks)

        # 写 Law / Document / Articles
        self.store.upsert_law(parsed.law)
        self.store.upsert_document(parsed.document)
        article_id_by_no = {}
        for art in parsed.articles:
            self.store.upsert_article(art)
            self.store.merge_relation(parsed.law.id, art.id, "CONTAINS", {})
            self.store.merge_relation(parsed.document.id, art.id, "CONTAINS", {})
            article_id_by_no[art.article_no] = art.id

        # Step 3: LLM 抽取 per chunk
        llm = get_extraction_llm()
        resolver = EntityResolver(self.store, self.embedding_fn, llm)

        all_resolved: list[ResolvedEntity] = []
        all_relations: list[tuple[str, str, str, float]] = []  # (from_id, to_id, edge_type, confidence)

        for chunk in chunks:
            with tracer.start_as_current_span("kg.extract.llm_extract") as cspan:
                cspan.set_attribute("chunk.id", chunk["id"])
                try:
                    self._process_chunk(chunk, parsed, resolver, llm, cspan,
                                        all_resolved, all_relations, article_id_by_no)
                except Exception as exc:  # noqa: BLE001 - 单 chunk 失败不应中断整个文档
                    cspan.record_exception(exc)
                    cspan.set_attribute("error.kind", "chunk_extract_failed")
                    logger.warning(
                        "kg.extract chunk failed, skipped chunk_id=%s document_id=%s: %s",
                        chunk["id"], document_id, exc,
                    )
                    continue

        # Step 6: 写关系
        for from_id, to_id, edge_type, conf in all_relations:
            with tracer.start_as_current_span("kg.extract.graph_write"):
                self.store.merge_relation(from_id, to_id, edge_type, {"confidence": conf})

        # Step 7: 冲突检测（如注入）
        conflicts_count = 0
        if self.conflict_detector is not None:
            for art in parsed.articles:
                conflicts_count += self.conflict_detector.detect_for_article(art.id)

        span.set_attribute("entities.count", len(all_resolved))
        span.set_attribute("relations.count", len(all_relations))
        span.set_attribute("conflicts.count", conflicts_count)

        return ExtractionReport(
            document_id=document_id,
            entities_count=len(all_resolved),
            relations_count=len(all_relations),
            conflicts_count=conflicts_count,
        )

    def _process_chunk(
        self, chunk: dict, parsed, resolver: EntityResolver, llm, cspan,
        all_resolved: list[ResolvedEntity],
        all_relations: list[tuple[str, str, str, float]],
        article_id_by_no: dict[int, str],
    ) -> None:
        """单个 chunk 的 LLM 抽取 + 实体合并 + 关系端点解析。"""
        article_no = self._find_article_no_for_chunk(chunk, parsed.articles)
        result: ExtractionResult = extract_from_chunk(
            chunk_text=chunk["content"], chunk_id=chunk["id"],
            article_no=article_no, llm=llm,
        )
        cspan.set_attribute("extract.entities", len(result.entities))
        cspan.set_attribute("extract.relations", len(result.relations))

        # Step 4: 实体合并
        resolved = resolver.resolve(result.entities, source_chunk_id=chunk["id"])
        all_resolved.extend(resolved)

        # 写节点 + 收集 relation 端点 ID
        name_to_id: dict[tuple[str, str], str] = {}
        for r in resolved:
            if r.existing_node_id:
                node_id = r.existing_node_id
            else:
                try:
                    node_id = self._create_new_node(r, chunk["id"])
                except ValueError as exc:
                    # 不支持的节点类型（如 Region 透传）：跳过该实体，不影响同 chunk 其余实体
                    logger.warning(
                        "kg.extract skip unsupported node type=%s name=%s chunk_id=%s: %s",
                        r.node_type, r.name, chunk["id"], exc,
                    )
                    continue
            name_to_id[(r.node_type, r.name)] = node_id

        # Step 5: 关系解析（resolve from/to 引用）
        for rel in result.relations:
            from_id = self._resolve_ref(rel.from_ref, name_to_id, article_id_by_no, parsed.law.name)
            to_id = self._resolve_ref(rel.to_ref, name_to_id, article_id_by_no, parsed.law.name)
            if from_id and to_id:
                all_relations.append((from_id, to_id, rel.type, rel.confidence))

    def _find_article_no_for_chunk(self, chunk: dict, articles: list) -> int | None:
        for art in articles:
            if chunk["id"] in art.chunk_ids:
                return art.article_no
        return None

    def _create_new_node(self, resolved: ResolvedEntity, chunk_id: str) -> str:
        if resolved.node_type == "Concept":
            embedding = self.embedding_fn(resolved.name)
            return self.store.upsert_concept(
                name=resolved.name, aliases=resolved.aliases_to_add,
                embedding=embedding, source_chunk_ids=[chunk_id],
            )
        elif resolved.node_type == "Party":
            return self.store.upsert_party(
                name=resolved.name, aliases=resolved.aliases_to_add,
                source_chunk_ids=[chunk_id],
            )
        else:
            raise ValueError(f"未支持的节点类型: {resolved.node_type}")

    def _resolve_ref(
        self, ref: str, name_to_id: dict, article_id_by_no: dict, law_name: str
    ) -> str | None:
        """解析 'article:<法名>:<条号>' / 'concept:<name>' / 'party:<name>' 引用。

        article 引用带法名：归一化后与当前文档法名不一致视为跨法引用，直接丢弃
        （返回 None），避免误链到当前法同号条文或产生自环。
        """
        parts = ref.split(":", 2)
        if len(parts) < 2:
            return None
        ref_type = parts[0]
        if ref_type == "concept":
            return name_to_id.get(("Concept", parts[1]))
        if ref_type == "party":
            return name_to_id.get(("Party", parts[1]))
        if ref_type == "article":
            # format: article:<法名>:<条号>
            if len(parts) == 3:
                if _normalize_law_name(parts[1]) != _normalize_law_name(law_name):
                    return None
                try:
                    article_no = int(parts[2])
                except ValueError:
                    return None
                return article_id_by_no.get(article_no)
        return None
