"""下载 bge-m3 权重文件"""
import os, urllib.request, ssl

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

SNAP_DIR = "/root/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181"
BASE_URL = "https://hf-mirror.com/BAAI/bge-m3/resolve/main"

files = [
    "pytorch_model.bin",
    "colbert_linear.pt",
    "sparse_linear.pt",
    "sentencepiece.bpe.model",
    "config_sentence_transformers.json",
]

for fname in files:
    dest = os.path.join(SNAP_DIR, fname)
    if os.path.exists(dest):
        size = os.path.getsize(dest)
        if size > 1000:
            print(f"SKIP (exists, {size} bytes): {fname}")
            continue
    url = f"{BASE_URL}/{fname}"
    print(f"Downloading {fname}...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req, context=ctx, timeout=600)
        with open(dest, 'wb') as f:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                f.write(chunk)
        size = os.path.getsize(dest)
        print(f"  Done! {size/1024/1024:.1f} MB")
    except Exception as e:
        print(f"  FAILED: {e}")

# 验证
print("\n--- 验证模型加载 ---")
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(SNAP_DIR)
    emb = model.encode(["劳动合同试用期最长是多久"])
    print(f"SUCCESS! Embedding shape: {emb.shape}")
except Exception as e:
    print(f"FAILED: {e}")
