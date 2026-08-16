"""知识图谱模块。

实体-关系-属性 KG，作为 HybridRetriever 的第三路检索源。
节点类型：Law / Article / Concept / Party / Region / Document
边类型：cites / is_a / conflicts_with / applies_to / contains / explains

具体设计见 docs/superpowers/specs/ 下的 KG 设计文档。
"""
