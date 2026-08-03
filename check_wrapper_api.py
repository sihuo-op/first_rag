from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
import inspect

print('=== LangchainLLMWrapper signature ===')
print(inspect.signature(LangchainLLMWrapper.__init__))

print('\n=== LangchainEmbeddingsWrapper signature ===')
print(inspect.signature(LangchainEmbeddingsWrapper.__init__))

print('\n=== LangchainLLMWrapper docstring ===')
print(LangchainLLMWrapper.__doc__)
