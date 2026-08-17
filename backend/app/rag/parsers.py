"""
文档解析器

支持 PDF、Word、Markdown、纯文本等格式
"""

import re
from abc import ABC, abstractmethod
from typing import Dict, Any, List


class DocumentParser(ABC):
    """文档解析器基类"""

    @abstractmethod
    def parse(self, file_path: str) -> tuple[str, Dict[str, Any]]:
        pass

    @abstractmethod
    def supported_extensions(self) -> list[str]:
        pass


class PdfParser(DocumentParser):
    """PDF 解析器"""

    def parse(self, file_path: str) -> tuple[str, Dict[str, Any]]:
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        content_parts = [page.extract_text() for page in reader.pages if page.extract_text()]
        content = "\n\n".join(content_parts)
        metadata = {"file_type": "pdf", "page_count": len(reader.pages), "character_count": len(content)}
        return content, metadata

    def supported_extensions(self) -> list[str]:
        return [".pdf"]


class DocxParser(DocumentParser):
    """Word 解析器"""

    def parse(self, file_path: str) -> tuple[str, Dict[str, Any]]:
        from docx import Document
        doc = Document(file_path)
        content_parts = [p.text for p in doc.paragraphs if p.text.strip()]
        content = "\n\n".join(content_parts)
        metadata = {"file_type": "docx", "paragraph_count": len(content_parts), "character_count": len(content)}
        return content, metadata

    def supported_extensions(self) -> list[str]:
        return [".docx"]


class MarkdownParser(DocumentParser):
    """Markdown 解析器"""

    def parse(self, file_path: str) -> tuple[str, Dict[str, Any]]:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        plain_text = self._markdown_to_plain_text(content)
        metadata = {"file_type": "md", "character_count": len(plain_text), "original_character_count": len(content)}
        return plain_text, metadata

    def _markdown_to_plain_text(self, md_text: str) -> str:
        text = re.sub(r'#{1,6}\s*', '', md_text)
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = re.sub(r'\*(.*?)\*', r'\1', text)
        text = re.sub(r'`{3}.*?\n(.*?)\n`{3}', r'\1', text, flags=re.DOTALL)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def supported_extensions(self) -> list[str]:
        return [".md", ".markdown"]


class TxtParser(DocumentParser):
    """纯文本解析器"""

    def parse(self, file_path: str) -> tuple[str, Dict[str, Any]]:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        metadata = {"file_type": "txt", "character_count": len(content)}
        return content, metadata

    def supported_extensions(self) -> list[str]:
        return [".txt"]


# 解析器工厂
PARSERS = {
    ".pdf": PdfParser,
    ".docx": DocxParser,
    ".md": MarkdownParser,
    ".markdown": MarkdownParser,
    ".txt": TxtParser,
}


def get_parser(file_extension: str) -> DocumentParser:
    """根据文件扩展名获取解析器（容忍缺省前导点）。"""
    ext = file_extension.lower()
    if ext and not ext.startswith("."):
        ext = f".{ext}"
    parser_class = PARSERS.get(ext)
    if parser_class:
        return parser_class()
    raise ValueError(f"Unsupported file type: {file_extension}")
