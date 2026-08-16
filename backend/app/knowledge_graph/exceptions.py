"""KG 异常类型，用于失败回退。"""


class KGError(Exception):
    """KG 模块根异常。"""


class KGConnectionError(KGError):
    """Neo4j 连接失败。"""


class KGExtractionError(KGError):
    """抽取管道失败。"""


class KGQueryError(KGError):
    """Cypher 查询失败。"""
