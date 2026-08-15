"""法律文档规则解析：法名/层级/生效日期/条款编号/地域。

走规则的部分：法名、条款编号、生效日期、发文机关、地域。
不走规则的部分（Concept/Party/关系）：交给 llm_extractor。
"""
import hashlib
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime

from app.knowledge_graph.schema import ArticleNode, DocumentNode, LawNode


LAW_NAME_PATTERN = re.compile(r"《?([^《》\n]{2,30}(?:法|条例|规定|办法|司法解释))》?")
EFFECTIVE_DATE_PATTERN = re.compile(r"自(\d{4})年(\d{1,2})月(\d{1,2})日起施行")
# 条款标题锚定行首：正文中的交叉引用（如「本法第三十九条规定的情形」）
# 不会被误识别为条款标题，法律文本的条款标题独占一行。
# head 组从「第」开始（不含行首空白）：char_start 定位到「第」，
# 保证 text[char_start:char_end] 即条款原文（content 字段直接喂给 LLM 冲突判定）。
ARTICLE_PATTERN = re.compile(
    r"^\s*(?P<head>第(?P<no>[一二三四五六七八九十百千零\d]+)条)", re.MULTILINE
)

# 中文数字转换（简化版）
CHINESE_NUM = {
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
    "百": 100, "千": 1000, "零": 0,
}


def chinese_to_int(s: str) -> int:
    if s.isdigit():
        return int(s)
    # 简化处理：支持 1-9999，正向扫描。
    # "十"~"十九"（10-19）没有前导数字，单位前默认为 1。
    total = 0
    current = 0  # 当前积累的个位数字
    for ch in s:
        digit = CHINESE_NUM.get(ch)
        if ch in ("十", "百", "千"):
            unit = CHINESE_NUM[ch]
            total += (current or 1) * unit
            current = 0
        elif digit is not None:
            current = digit  # 含 "零"：重置为 0
    return total + current


def detect_law_level(name: str) -> str:
    if "司法解释" in name:
        return "司法解释"
    if name.endswith("法"):
        return "法律"
    if name.endswith("条例"):
        return "法规"
    if name.endswith("规定"):
        return "规章"
    if name.endswith("办法"):
        return "规章"
    return "其他"


# 国名前缀：法律标题中的"中华人民共和国"前缀在简称中使用时通常省略。
COUNTRY_PREFIX = "中华人民共和国"


def _strip_country_prefix(name: str) -> str:
    """去掉 '中华人民共和国' 前缀，使用简称（如 '劳动合同法'）。"""
    if name.startswith(COUNTRY_PREFIX):
        return name[len(COUNTRY_PREFIX):]
    return name


@dataclass
class ParsedDocument:
    law: LawNode
    document: DocumentNode
    articles: list[ArticleNode] = field(default_factory=list)


def parse_document(text: str, document_id: str, chunks: list[dict]) -> ParsedDocument:
    # 1. 法名（取第一个匹配，通常是标题）
    law_name_match = LAW_NAME_PATTERN.search(text)
    if not law_name_match:
        raise ValueError(f"无法从文档中解析法名: {document_id}")
    law_name = _strip_country_prefix(law_name_match.group(1))
    law_level = detect_law_level(law_name)

    # 2. 生效日期
    effective_date = None
    date_match = EFFECTIVE_DATE_PATTERN.search(text)
    if date_match:
        y, m, d = date_match.groups()
        effective_date = f"{int(y):04d}-{int(m):02d}-{int(d):02d}"

    # 3. 构造 Law 节点
    law = LawNode(
        id=f"law-{uuid.uuid4().hex[:12]}",
        name=law_name,
        level=law_level,
        effective_date=effective_date,
    )

    # 4. 解析条款
    articles = []
    matches = list(ARTICLE_PATTERN.finditer(text))
    for i, m in enumerate(matches):
        article_no = chinese_to_int(m.group("no"))
        # char_start 从「第」字起（跳过行首空白），content/content_hash 均基于此切片
        char_start = m.start("head")
        char_end = matches[i + 1].start("head") if i + 1 < len(matches) else len(text)

        # 通过 char range overlap 找 chunk_ids
        chunk_ids = [
            c["id"] for c in chunks
            if c.get("char_start", 0) < char_end and c.get("char_end", 0) > char_start
        ]

        # content_hash：取 article 文本片段的 hash
        article_text = text[char_start:char_end]
        content_hash = hashlib.sha256(article_text.encode()).hexdigest()[:32]

        articles.append(ArticleNode(
            id=f"art-{uuid.uuid4().hex[:12]}",
            law_id=law.id,
            article_no=article_no,
            content_hash=content_hash,
            content=article_text,
            chunk_ids=chunk_ids,
            status="active",
            char_start=char_start,
            char_end=char_end,
        ))

    # 5. Document 节点
    document = DocumentNode(
        id=document_id,
        source_file=f"{law_name}.txt",
        uploaded_at=datetime.now().isoformat(),
        doc_type="law",
    )

    return ParsedDocument(law=law, document=document, articles=articles)
