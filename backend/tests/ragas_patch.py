"""猴子补丁：绕过 ragas 对 vertexai 的硬导入"""
import sys

# 创建一个假的 vertexai 模块，让 ragas 的导入不会失败
fake_module = type(sys)('langchain_community.chat_models.vertexai')
fake_module.ChatVertexAI = type('ChatVertexAI', (), {})
sys.modules['langchain_community.chat_models.vertexai'] = fake_module