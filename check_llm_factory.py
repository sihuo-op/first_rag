import inspect
from ragas.llms import llm_factory, base, BaseRagasLLM
from ragas.embeddings import embedding_factory, BaseRagasEmbedding

print('=== BaseRagasLLM ===')
print(inspect.signature(BaseRagasLLM.__init__))

print('\n=== llm_factory ===')
print(inspect.signature(llm_factory))

print('\n=== embedding_factory ===')
print(inspect.signature(embedding_factory))

print('\n=== BaseRagasEmbedding ===')
print(inspect.signature(BaseRagasEmbedding.__init__))
