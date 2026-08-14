import pytest

from app.knowledge_graph.rule_parser import chinese_to_int, parse_document, ParsedDocument


SAMPLE_TEXT = """中华人民共和国劳动合同法

第一章 总则

第一条 为了完善劳动合同制度，制定本法。

第二条 中华人民共和国境内的企业、个体经济组织、民办非企业单位等组织与劳动者建立劳动关系，订立、履行、变更、解除或者终止劳动合同，适用本法。

本法自2008年1月1日起施行。
"""


def test_parse_extracts_law_name():
    result = parse_document(SAMPLE_TEXT, document_id="doc-1", chunks=[])
    assert result.law.name == "劳动合同法"
    assert result.law.level == "法律"


def test_parse_extracts_articles():
    result = parse_document(SAMPLE_TEXT, document_id="doc-1", chunks=[])
    assert len(result.articles) == 2
    assert result.articles[0].article_no == 1
    assert result.articles[1].article_no == 2


def test_parse_extracts_effective_date():
    result = parse_document(SAMPLE_TEXT, document_id="doc-1", chunks=[])
    assert result.law.effective_date == "2008-01-01"


def test_parse_articles_have_char_range():
    result = parse_document(SAMPLE_TEXT, document_id="doc-1", chunks=[])
    for art in result.articles:
        assert art.char_start >= 0
        assert art.char_end > art.char_start
        # Article content should be within text range
        assert art.char_end <= len(SAMPLE_TEXT)


def test_parse_articles_chunk_ids_filled_from_overlap():
    # 造一个 chunk 覆盖第1条
    chunks = [{
        "id": "chunk-1",
        "content": "第一条 为了完善劳动合同制度，制定本法。",
        "char_start": 0,
        "char_end": 100,
    }]
    result = parse_document(SAMPLE_TEXT, document_id="doc-1", chunks=chunks)
    assert "chunk-1" in result.articles[0].chunk_ids


def test_chinese_to_int_handles_teens():
    # "十一"~"十九"（10+1..10+9），不能解析成 1..9
    assert chinese_to_int("十一") == 11
    assert chinese_to_int("十五") == 15
    assert chinese_to_int("十九") == 19


def test_chinese_to_int_handles_compound_numbers():
    assert chinese_to_int("十") == 10
    assert chinese_to_int("二十") == 20
    assert chinese_to_int("二十三") == 23
    assert chinese_to_int("九十八") == 98  # 劳动合同法最后一条
    assert chinese_to_int("一百零五") == 105
    assert chinese_to_int("123") == 123


def test_parse_article_numbers_beyond_ten():
    text = (
        "劳动合同法\n\n"
        "第十条 建立劳动关系，应当订立书面劳动合同。\n\n"
        "第十一条 未订立书面劳动合同时的处理。\n\n"
        "第十九条 试用期期限规定。\n"
    )
    result = parse_document(text, document_id="doc-3", chunks=[])
    assert [a.article_no for a in result.articles] == [10, 11, 19]


def test_parse_raises_when_no_law_name():
    # 注意：正则匹配任何以 法/条例/规定/办法/司法解释 结尾的片段，
    # 所以这里不能出现这些字样
    with pytest.raises(ValueError, match="无法从文档中解析法名"):
        parse_document("这是一段普通文本，不包含任何标题。", document_id="doc-x", chunks=[])


def test_parse_returns_empty_articles_when_none_found():
    text = "劳动合同法\n\n本文件没有分条内容。\n"
    result = parse_document(text, document_id="doc-2", chunks=[])
    assert result.articles == []
    assert result.law.name == "劳动合同法"


def test_parse_article_chunk_boundary_no_overlap_when_adjacent():
    # chunk 恰好在第一条之前结束：[0, art1_start) 与 [art1_start, ...) 相邻不重叠
    probe = parse_document(SAMPLE_TEXT, document_id="doc-1", chunks=[])
    art1_start = probe.articles[0].char_start
    chunks = [{"id": "chunk-x", "content": "标题和章名", "char_start": 0, "char_end": art1_start}]
    result = parse_document(SAMPLE_TEXT, document_id="doc-1", chunks=chunks)
    assert "chunk-x" not in result.articles[0].chunk_ids


def test_parse_does_not_split_on_in_text_cross_references():
    # 真实《劳动合同法》第二十一条正文引用了第三十九条/第四十条：
    # 「…除劳动者有本法第三十九条和第四十条第一项、第二项规定的情形外…」
    # 只有行首的「第X条」才是条款标题，正文中的交叉引用不得拆分条款。
    text = (
        "中华人民共和国劳动合同法\n\n"
        "第二十条 劳动者在试用期的工资不得低于本单位相同岗位最低档工资"
        "或者劳动合同约定工资的百分之八十。\n\n"
        "第二十一条 在试用期中，除劳动者有本法第三十九条和第四十条"
        "第一项、第二项规定的情形外，用人单位不得解除劳动合同。"
        "用人单位在试用期解除劳动合同的，应当向劳动者说明理由。\n\n"
        "第二十二条 劳动合同期满，劳动合同终止。\n"
    )
    result = parse_document(text, document_id="doc-4", chunks=[])

    # 不得产生虚假的 39/40 条节点
    assert [a.article_no for a in result.articles] == [20, 21, 22]

    # 第二十一条的范围必须覆盖正文中的交叉引用（不得在引用处截断）
    art21 = result.articles[1]
    assert art21.char_start < text.index("第三十九条")
    assert art21.char_end > text.index("第四十条")
    # 且下一条的真实起点（第二十二条）就是它的终点
    assert art21.char_end == result.articles[2].char_start
