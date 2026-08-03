import sys
sys.path.insert(0, '/app')
import os
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-m3')
result = model.encode('测试')
print('OK, shape:', result.shape)