import json
from unittest.mock import MagicMock, patch

from app.knowledge_graph.llm_extractor import extract_from_chunk, ExtractionResult


def test_extract_returns_entities_and_relations():
    fake_response = MagicMock()
    fake_response.content = json.dumps({
        "entities": [
            {"type": "Concept", "name": "试用期", "aliases": ["试用期间"]},
            {"type": "Party", "name": "用人单位"}
        ],
        "relations": [
            {"type": "EXPLAINS", "from": "article:19", "to": "concept:试用期", "confidence": 0.9},
            {"type": "APPLIES_TO", "from": "article:19", "to": "party:用人单位", "confidence": 0.95}
        ]
    }, ensure_ascii=False)

    with patch("app.knowledge_graph.llm_extractor.invoke_llm_threadsafe", return_value=fake_response):
        result = extract_from_chunk(
            chunk_text="第十九条 劳动合同期限...",
            chunk_id="chunk-1",
            article_no=19,
            llm=MagicMock(),
        )

    assert isinstance(result, ExtractionResult)
    assert len(result.entities) == 2
    assert result.entities[0].type == "Concept"
    assert result.entities[0].name == "试用期"
    assert result.entities[0].aliases == ["试用期间"]
    assert len(result.relations) == 2
    assert result.relations[0].type == "EXPLAINS"
    assert result.relations[0].from_ref == "article:19"
    assert result.relations[0].to_ref == "concept:试用期"
    assert result.relations[0].confidence == 0.9


def test_extract_handles_invalid_json():
    fake_response = MagicMock()
    fake_response.content = "not valid json"

    with patch("app.knowledge_graph.llm_extractor.invoke_llm_threadsafe", return_value=fake_response):
        result = extract_from_chunk("text", "chunk-1", article_no=1, llm=MagicMock())

    assert result.entities == []
    assert result.relations == []


def test_extract_filters_unknown_entity_types():
    fake_response = MagicMock()
    fake_response.content = json.dumps({
        "entities": [
            {"type": "Concept", "name": "试用期"},
            {"type": "UnknownType", "name": "foo"}
        ],
        "relations": []
    })

    with patch("app.knowledge_graph.llm_extractor.invoke_llm_threadsafe", return_value=fake_response):
        result = extract_from_chunk("text", "chunk-1", article_no=1, llm=MagicMock())

    assert len(result.entities) == 1
    assert result.entities[0].type == "Concept"


def test_extract_filters_unknown_relation_types():
    """Relations with unknown types or missing from/to should be dropped."""
    fake_response = MagicMock()
    fake_response.content = json.dumps({
        "entities": [],
        "relations": [
            {"type": "UNKNOWN_REL", "from": "article:1", "to": "concept:x"},
            {"type": "EXPLAINS", "from": "", "to": "concept:y"},
            {"type": "EXPLAINS", "from": "article:1", "to": "concept:z", "confidence": 0.8},
        ],
    })

    with patch("app.knowledge_graph.llm_extractor.invoke_llm_threadsafe", return_value=fake_response):
        result = extract_from_chunk("text", "chunk-1", article_no=1, llm=MagicMock())

    assert len(result.relations) == 1
    assert result.relations[0].type == "EXPLAINS"
    assert result.relations[0].to_ref == "concept:z"
    assert result.relations[0].confidence == 0.8


def test_extract_retries_on_llm_failure():
    """LLM call should retry up to 2 times on exception, then succeed."""
    good_response = MagicMock()
    good_response.content = json.dumps({
        "entities": [{"type": "Concept", "name": "经济补偿"}],
        "relations": [],
    })

    side_effects = [RuntimeError("api timeout"), good_response]
    with patch(
        "app.knowledge_graph.llm_extractor.invoke_llm_threadsafe",
        side_effect=side_effects,
    ) as mock_invoke:
        result = extract_from_chunk("text", "chunk-1", article_no=1, llm=MagicMock())

    assert mock_invoke.call_count == 2
    assert len(result.entities) == 1
    assert result.entities[0].name == "经济补偿"


def test_extract_returns_empty_after_all_retries_fail():
    """If all retries fail, return empty ExtractionResult."""
    with patch(
        "app.knowledge_graph.llm_extractor.invoke_llm_threadsafe",
        side_effect=RuntimeError("api down"),
    ) as mock_invoke:
        result = extract_from_chunk("text", "chunk-1", article_no=1, llm=MagicMock())

    # 1 initial + 2 retries = 3 total attempts
    assert mock_invoke.call_count == 3
    assert result.entities == []
    assert result.relations == []


def test_extract_handles_markdown_codeblock_wrap():
    """LLM may wrap JSON in ```json ... ``` blocks; extractor should still parse."""
    fake_response = MagicMock()
    fake_response.content = "```json\n" + json.dumps({
        "entities": [{"type": "Party", "name": "劳动者"}],
        "relations": [],
    }, ensure_ascii=False) + "\n```"

    with patch("app.knowledge_graph.llm_extractor.invoke_llm_threadsafe", return_value=fake_response):
        result = extract_from_chunk("text", "chunk-1", article_no=1, llm=MagicMock())

    assert len(result.entities) == 1
    assert result.entities[0].type == "Party"
    assert result.entities[0].name == "劳动者"


def test_extract_handles_null_aliases():
    """``aliases: null`` should not raise TypeError; entity is kept with empty aliases."""
    fake_response = MagicMock()
    fake_response.content = json.dumps({
        "entities": [
            {"type": "Concept", "name": "试用期", "aliases": None},
        ],
        "relations": [],
    })

    with patch("app.knowledge_graph.llm_extractor.invoke_llm_threadsafe", return_value=fake_response):
        result = extract_from_chunk("text", "chunk-1", article_no=1, llm=MagicMock())

    assert len(result.entities) == 1
    assert result.entities[0].name == "试用期"
    assert result.entities[0].aliases == []


def test_extract_handles_non_numeric_confidence():
    """Non-numeric confidence (e.g. "high") should fall back to 0.5, not raise ValueError."""
    fake_response = MagicMock()
    fake_response.content = json.dumps({
        "entities": [],
        "relations": [
            {"type": "EXPLAINS", "from": "article:1", "to": "concept:x", "confidence": "high"},
        ],
    })

    with patch("app.knowledge_graph.llm_extractor.invoke_llm_threadsafe", return_value=fake_response):
        result = extract_from_chunk("text", "chunk-1", article_no=1, llm=MagicMock())

    assert len(result.relations) == 1
    assert result.relations[0].confidence == 0.5
