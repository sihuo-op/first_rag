"""
一键演示种子脚本：等待后端就绪 -> 登录 -> 上传演示语料 -> 等待处理完成 -> 打印示例问题。

用法：
    python scripts/demo_setup.py [--base-url http://localhost:8000]

前提：
    1. 后端已启动（默认 http://localhost:8000，管理员 admin/admin123 由首次启动自动创建）
    2. Milvus / Neo4j（可选）等依赖服务已就绪
"""
import argparse
import sys
import time
from pathlib import Path

import httpx

DEMO_CORPUS_DIR = Path(__file__).parent / "demo_corpus"
DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_USERNAME = "admin"
DEFAULT_PASSWORD = "admin123"

SAMPLE_QUESTIONS = [
    "劳动合同期限三个月以上不满一年的，试用期最长是多久？",
    "劳动者连续工作一年以上的，年休假有多少天？",
    "用人单位拖欠工资，劳动者可以怎么维权？",
    "女职工生育享受多少天产假？",
    "什么情况下用人单位可以解除劳动合同且不用支付经济补偿？",
]


def wait_for_backend(client: httpx.Client, timeout: float = 120.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = client.get("/health", timeout=5.0)
            if resp.status_code == 200:
                print("[demo] backend is healthy")
                return
        except httpx.HTTPError:
            pass
        time.sleep(2.0)
    print("[demo] ERROR: backend did not become healthy in time", file=sys.stderr)
    sys.exit(1)


def login(client: httpx.Client, username: str, password: str) -> None:
    resp = client.post(
        "/api/v1/auth/login",
        json={"username": username, "password": password},
        timeout=15.0,
    )
    if resp.status_code != 200:
        print(f"[demo] ERROR: login failed ({resp.status_code}): {resp.text}", file=sys.stderr)
        sys.exit(1)
    token = resp.json()["access_token"]
    client.headers["Authorization"] = f"Bearer {token}"
    print(f"[demo] logged in as {username}")


def upload_corpus(client: httpx.Client) -> list:
    doc_ids = []
    for path in sorted(DEMO_CORPUS_DIR.iterdir()):
        if path.suffix not in (".txt", ".md", ".pdf", ".docx"):
            continue
        with open(path, "rb") as f:
            resp = client.post(
                "/api/v1/documents/upload",
                files={"file": (path.name, f, "text/plain")},
                timeout=120.0,
            )
        if resp.status_code not in (200, 201):
            print(f"[demo] WARN: upload {path.name} failed ({resp.status_code}): {resp.text}")
            continue
        doc = resp.json()
        doc_ids.append(doc["id"])
        print(f"[demo] uploaded {path.name} -> document_id={doc['id']} status={doc['status']}")
    if not doc_ids:
        print("[demo] ERROR: no documents uploaded", file=sys.stderr)
        sys.exit(1)
    return doc_ids


def wait_for_processing(client: httpx.Client, doc_ids: list, timeout: float = 300.0) -> None:
    deadline = time.time() + timeout
    pending = set(doc_ids)
    while pending and time.time() < deadline:
        for doc_id in list(pending):
            resp = client.get(f"/api/v1/documents/{doc_id}/status", timeout=15.0)
            if resp.status_code != 200:
                continue
            status = resp.json().get("status")
            if status == "completed":
                pending.discard(doc_id)
                print(f"[demo] document {doc_id}: completed")
            elif status == "failed":
                print(f"[demo] ERROR: document {doc_id} processing failed", file=sys.stderr)
                sys.exit(1)
        if pending:
            time.sleep(3.0)
    if pending:
        print(f"[demo] WARN: documents still processing after {timeout}s: {pending}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed demo corpus into a running backend")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--username", default=DEFAULT_USERNAME)
    parser.add_argument("--password", default=DEFAULT_PASSWORD)
    args = parser.parse_args()

    with httpx.Client(base_url=args.base_url) as client:
        wait_for_backend(client)
        login(client, args.username, args.password)
        doc_ids = upload_corpus(client)
        wait_for_processing(client, doc_ids)

    print("\n[demo] seed complete. Try these questions in the frontend or via /api/v1/chat:")
    for q in SAMPLE_QUESTIONS:
        print(f"  - {q}")


if __name__ == "__main__":
    main()
