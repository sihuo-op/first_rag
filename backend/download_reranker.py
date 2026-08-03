"""下载 BAAI/bge-reranker-v2-m3 模型"""
import os, json, urllib.request, ssl

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

MODEL_ID = "BAAI/bge-reranker-v2-m3"
CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3")
SNAP_DIR = os.path.join(CACHE_DIR, "snapshots")

# 获取模型信息
api_url = f"https://hf-mirror.com/api/models/{MODEL_ID}"
req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
resp = urllib.request.urlopen(req, context=ctx, timeout=30)
model_info = json.loads(resp.read().decode())
siblings = model_info.get("siblings", [])

# 创建目录
os.makedirs(CACHE_DIR, exist_ok=True)
refs_dir = os.path.join(CACHE_DIR, "refs")
os.makedirs(refs_dir, exist_ok=True)
main_ref = model_info.get("sha", "main")
with open(os.path.join(refs_dir, "main"), "w") as f:
    f.write(main_ref)
snapshot_dir = os.path.join(SNAP_DIR, main_ref)
os.makedirs(snapshot_dir, exist_ok=True)

# 下载文件
skip_patterns = [".onnx", ".tflite", ".h5", ".ot", ".msgpack", "flax_model"]
downloaded = 0
for sib in siblings:
    rfname = sib.get("rfilename", "")
    if any(rfname.endswith(p) for p in skip_patterns):
        continue
    file_url = f"https://hf-mirror.com/{MODEL_ID}/resolve/main/{rfname}"
    dest_dir = os.path.join(snapshot_dir, os.path.dirname(rfname))
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(snapshot_dir, rfname)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        size = os.path.getsize(dest_path)
        print(f"SKIP (exists, {size/1024/1024:.1f} MB): {rfname}")
        continue
    print(f"Downloading {rfname}...")
    try:
        req = urllib.request.Request(file_url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req, context=ctx, timeout=600)
        with open(dest_path, 'wb') as f:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                f.write(chunk)
        size = os.path.getsize(dest_path)
        print(f"  Done! {size/1024/1024:.1f} MB")
        downloaded += 1
    except Exception as e:
        print(f"  FAILED: {e}")

print(f"\nDownloaded {downloaded} files")

# 验证
try:
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(snapshot_dir)
    score = model.predict([("劳动合同", "第二十条 劳动合同的期限分为有固定期限")])
    print(f"Reranker loaded! Score: {score}")
except Exception as e:
    print(f"Reranker load test: {e}")
