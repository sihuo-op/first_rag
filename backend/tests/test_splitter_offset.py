"""测试 ThreeLayerSplitter 的 char_start/char_end 偏移跟踪。

每个 chunk dict 应该带 `char_start` 和 `char_end` 两个 int 字段，表示在原文中的字符 range，
满足 `text[char_start:char_end] == chunk["content"]`（large 类型）。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.rag.splitter import ThreeLayerSplitter


def test_chunks_have_char_offsets():
    text = "这是第一段。\n\n这是第二段，包含一些内容。"
    splitter = ThreeLayerSplitter(large_size=2000, medium_size=500, small_size=150, overlap=0)
    chunks = splitter.split(text)
    assert len(chunks) > 0
    for chunk in chunks:
        assert "char_start" in chunk
        assert "char_end" in chunk
        assert isinstance(chunk["char_start"], int)
        assert isinstance(chunk["char_end"], int)
        assert chunk["char_start"] >= 0
        assert chunk["char_end"] > chunk["char_start"]


def test_chunk_content_matches_text_range():
    text = "这是第一段。\n\n这是第二段。"
    splitter = ThreeLayerSplitter(large_size=2000, medium_size=500, small_size=150, overlap=0)
    chunks = splitter.split(text)
    for chunk in chunks:
        if chunk["chunk_type"] == "large":
            assert text[chunk["char_start"]:chunk["char_end"]] == chunk["content"]


def test_offsets_within_document_length():
    text = "这是测试文本。\n\n第二段内容。" * 10
    splitter = ThreeLayerSplitter(large_size=2000, medium_size=500, small_size=150, overlap=0)
    chunks = splitter.split(text)
    for chunk in chunks:
        assert chunk["char_end"] <= len(text)
