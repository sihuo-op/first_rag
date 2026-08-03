"""在容器内下载 BAAI/bge-m3 模型"""
import os
import json
import urllib.request
import ssl

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 忽略 SSL 验证问题
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

MODEL_ID = "BAAI/bge-m3"
CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub/models--BAAI--bge-m3")
SNAP_DIR = os.path.join(CACHE_DIR, "snapshots")

def download_file(url, dest):
    """下载文件"""
    print(f"  Downloading: {url}")
    print(f"  To: {dest}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    resp = urllib.request.urlopen(req, context=ctx, timeout=300)
    with open(dest, 'wb') as f:
        while True:
            chunk = resp.read(8192)
            if not chunk:
                break
            f.write(chunk)
    size = os.path.getsize(dest)
    print(f"  Done! Size: {size} bytes")

# 1. 获取模型文件列表
print("Getting model info from hf-mirror.com...")
api_url = f"https://hf-mirror.com/api/models/{MODEL_ID}"
req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
resp = urllib.request.urlopen(req, context=ctx, timeout=30)
model_info = json.loads(resp.read().decode())

siblings = model_info.get("siblings", [])
print(f"Found {len(siblings)} files")

# 2. 下载必要文件
essential_patterns = ["config.json", "tokenizer.json", "tokenizer_config.json",
                      "special_tokens_map.json", "model.safetensors",
                      "sentence_bert_config.json", "modules.json",
                      "1_Pooling/", "2_Normalize/", "0_Transformer/"]

os.makedirs(CACHE_DIR, exist_ok=True)

# 创建 refs
refs_dir = os.path.join(CACHE_DIR, "refs")
os.makedirs(refs_dir, exist_ok=True)
main_ref = model_info.get("sha", "main")
refs_main_path = os.path.join(refs_dir, "main")
if not os.path.exists(refs_main_path):
    with open(refs_main_path, "w") as f:
        f.write(main_ref)

# 创建 snapshot 目录
snapshot_dir = os.path.join(SNAP_DIR, main_ref)
os.makedirs(snapshot_dir, exist_ok=True)

downloaded = 0
for sib in siblings:
    rfname = sib.get("rfilename", "")
    # 检查是否是必要文件
    should_download = False
    for pat in essential_patterns:
        if rfname.startswith(pat) or rfname == pat:
            should_download = True
            break

    # 跳过大文件中的非必要文件（如 onnx, tflite 等）
    skip_patterns = [".onnx", ".tflite", ".h5", ".ot", ".msgpack"]
    if any(rfname.endswith(p) for p in skip_patterns):
        should_download = False

    if should_download:
        file_url = f"https://hf-mirror.com/{MODEL_ID}/resolve/main/{rfname}"
        dest_dir = os.path.join(snapshot_dir, os.path.dirname(rfname))
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(snapshot_dir, rfname)
        if not os.path.exists(dest_path):
            try:
                download_file(file_url, dest_path)
                downloaded += 1
            except Exception as e:
                print(f"  FAILED: {e}")
        else:
            print(f"  SKIP (exists): {rfname}")

print(f"\nDownloaded {downloaded} files to {snapshot_dir}")

# 验证
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(snapshot_dir)
    emb = model.encode(["test"])
    print(f"Model loaded successfully! Embedding shape: {emb.shape}")
except Exception as e:
    print(f"Model load test failed: {e}")
