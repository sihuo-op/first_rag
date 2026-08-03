from typing import Dict, List


class TokenBudget:
    def __init__(self, model_name: str = ""):
        self.model_name = model_name
        self._encoding = self._get_encoding(model_name)

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        if self._encoding:
            return len(self._encoding.encode(text))
        return self._fallback_count(text)

    def count_messages(self, messages: List[Dict[str, str]]) -> int:
        total = 0
        for message in messages:
            total += 4
            total += self.count_text(str(message.get("role", "")))
            total += self.count_text(str(message.get("content", "")))
        return total + 2

    def _get_encoding(self, model_name: str):
        try:
            import tiktoken
            try:
                return tiktoken.encoding_for_model(model_name)
            except Exception:
                return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None

    def _fallback_count(self, text: str) -> int:
        ascii_count = sum(1 for char in text if ord(char) < 128)
        non_ascii_count = len(text) - ascii_count
        return non_ascii_count + max(1, ascii_count // 4)
