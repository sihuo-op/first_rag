"""
文本切分器

三层分块策略：Large/Medium/Small
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import re
import tiktoken


class TextSplitter(ABC):
    """切分器基类"""

    @abstractmethod
    def split(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        pass


class ThreeLayerSplitter(TextSplitter):
    """
    三层文档切分器

    - Large: 按语义单元（段落组）切分，保留完整上下文
    - Medium: 按句子组切分，平衡上下文和精确性
    - Small: 按词组切分，提供精确匹配
    """

    def __init__(self, large_size: int = 2000, medium_size: int = 500, small_size: int = 150, overlap: int = 50):
        self.large_size = large_size
        self.medium_size = medium_size
        self.small_size = small_size
        self.overlap = overlap
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def split(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        all_chunks = []

        # 第一层：大段切分（带 offset）
        large_chunks_with_offsets = self._split_by_semantic_units_with_offset(text, self.large_size)
        for large_idx, (large_chunk, l_start, l_end) in enumerate(large_chunks_with_offsets):
            all_chunks.append({
                "content": large_chunk,
                "chunk_type": "large",
                "parent_chunk_id": None,
                "position": large_idx,
                "token_count": self._count_tokens(large_chunk),
                "char_start": l_start,
                "char_end": l_end,
                "metadata": metadata or {}
            })

            # 第二层：中段切分（offset 相对于原文）
            medium_chunks_with_offsets = self._split_by_paragraphs_with_offset(large_chunk, self.medium_size, l_start)
            for medium_idx, (medium_chunk, m_start, m_end) in enumerate(medium_chunks_with_offsets):
                all_chunks.append({
                    "content": medium_chunk,
                    "chunk_type": "medium",
                    "parent_idx": large_idx,
                    "position": medium_idx,
                    "token_count": self._count_tokens(medium_chunk),
                    "char_start": m_start,
                    "char_end": m_end,
                    "metadata": metadata or {}
                })

                # 第三层：小段切分
                small_chunks_with_offsets = self._split_by_sentences_with_offset(medium_chunk, self.small_size, m_start)
                for small_idx, (small_chunk, s_start, s_end) in enumerate(small_chunks_with_offsets):
                    all_chunks.append({
                        "content": small_chunk,
                        "chunk_type": "small",
                        "parent_idx": len(all_chunks) - len(small_chunks_with_offsets) - 1,
                        "position": small_idx,
                        "token_count": self._count_tokens(small_chunk),
                        "char_start": s_start,
                        "char_end": s_end,
                        "metadata": metadata or {}
                    })

        return all_chunks

    def _split_by_semantic_units_with_offset(self, text: str, target_size: int) -> List[tuple]:
        paragraphs = re.split(r'\n\s*\n', text.strip())
        chunks, current, current_size, current_start = [], [], 0, 0
        pos = 0
        # 重建原文以追踪 offset（re.split 丢弃分隔符，需要重新扫描）
        for para in paragraphs:
            # 找到 para 在原文中的位置
            para_start = text.find(para, pos)
            if para_start == -1:
                para_start = pos
            para_end = para_start + len(para)
            pos = para_end
            para_size = self._count_tokens(para)
            if current_size + para_size > target_size and current:
                chunk_text = "\n\n".join(current)
                chunks.append((chunk_text, current_start, current_start + len(chunk_text)))
                current, current_size = [para], para_size
                current_start = para_start
            else:
                if not current:
                    current_start = para_start
                current.append(para)
                current_size += para_size
        if current:
            chunk_text = "\n\n".join(current)
            chunks.append((chunk_text, current_start, current_start + len(chunk_text)))
        return chunks

    def _split_by_paragraphs_with_offset(self, text: str, target_size: int, base_offset: int) -> List[tuple]:
        sentences = re.split(r'(?<=[.!?。！？])\s+', text.strip())
        chunks, current, current_size, current_start = [], [], 0, 0
        pos = 0
        for sent in sentences:
            sent_start = text.find(sent, pos)
            if sent_start == -1:
                sent_start = pos
            sent_end = sent_start + len(sent)
            pos = sent_end
            sent_size = self._count_tokens(sent)
            if current_size + sent_size > target_size and current:
                chunk_text = " ".join(current)
                chunks.append((chunk_text, base_offset + current_start, base_offset + current_start + len(chunk_text)))
                current, current_size = [sent], sent_size
                current_start = sent_start
            else:
                if not current:
                    current_start = sent_start
                current.append(sent)
                current_size += sent_size
        if current:
            chunk_text = " ".join(current)
            chunks.append((chunk_text, base_offset + current_start, base_offset + current_start + len(chunk_text)))
        return chunks

    def _split_by_sentences_with_offset(self, text: str, target_size: int, base_offset: int) -> List[tuple]:
        words = re.split(r'\s+', text.strip())
        chunks, current, current_size, current_start = [], [], 0, 0
        pos = 0
        for word in words:
            word_start = text.find(word, pos)
            if word_start == -1:
                word_start = pos
            word_end = word_start + len(word)
            pos = word_end
            word_size = self._count_tokens(word)
            if current_size + word_size > target_size and current:
                chunk_text = " ".join(current)
                chunks.append((chunk_text, base_offset + current_start, base_offset + current_start + len(chunk_text)))
                current, current_size = [word], word_size
                current_start = word_start
            else:
                if not current:
                    current_start = word_start
                current.append(word)
                current_size += word_size
        if current:
            chunk_text = " ".join(current)
            chunks.append((chunk_text, base_offset + current_start, base_offset + current_start + len(chunk_text)))
        return chunks

    def _split_by_semantic_units(self, text: str, target_size: int) -> List[str]:
        paragraphs = re.split(r'\n\s*\n', text.strip())
        chunks, current_chunk, current_size = [], [], 0
        for para in paragraphs:
            para_size = self._count_tokens(para)
            if current_size + para_size > target_size and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk, current_size = [para], para_size
            else:
                current_chunk.append(para)
                current_size += para_size
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
        return chunks

    def _split_by_paragraphs(self, text: str, target_size: int) -> List[str]:
        sentences = re.split(r'(?<=[.!?。！？])\s+', text.strip())
        chunks, current_chunk, current_size = [], [], 0
        for sent in sentences:
            sent_size = self._count_tokens(sent)
            if current_size + sent_size > target_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk, current_size = [sent], sent_size
            else:
                current_chunk.append(sent)
                current_size += sent_size
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

    def _split_by_sentences(self, text: str, target_size: int) -> List[str]:
        words = re.split(r'\s+', text.strip())
        chunks, current_chunk, current_size = [], [], 0
        for word in words:
            word_size = self._count_tokens(word)
            if current_size + word_size > target_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk, current_size = [word], word_size
            else:
                current_chunk.append(word)
                current_size += word_size
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, disallowed_special=()))
