"""Command-line entry point for enterprise RAG evaluations."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from evals.dataset import DEFAULT_DATASET_PATH, filter_cases, load_cases, validate_dataset
from evals.metrics import check_thresholds
from evals.report import write_reports
from evals.runners import run_e2e_api_cases_sync, run_rag_tool_cases, run_retrieval_cases


DEFAULT_THRESHOLDS_PATH = Path(__file__).with_name("thresholds.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run enterprise RAG evaluations.")
    parser.add_argument("--mode", choices=["validate", "retrieval", "rag", "e2e", "all"], default="all")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--thresholds", default=str(DEFAULT_THRESHOLDS_PATH))
    parser.add_argument("--category", default=None)
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--username", default="admin")
    parser.add_argument("--password", default="admin123")
    parser.add_argument("--no-fail", action="store_true", help="Always exit 0 after writing reports.")
    return parser.parse_args()


def load_thresholds(path: str | Path) -> dict[str, Any]:
    threshold_path = Path(path)
    if not threshold_path.exists():
        return {}
    return json.loads(threshold_path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    try:
        if args.mode == "validate":
            info = validate_dataset(args.dataset)
            print(json.dumps(info, ensure_ascii=False, indent=2))
            return 0

        cases = filter_cases(load_cases(args.dataset), category=args.category, case_id=args.case_id)
        if not cases:
            print("No eval cases selected.", file=sys.stderr)
            return 2

        thresholds = load_thresholds(args.thresholds)
        results: dict[str, Any] = {}
        threshold_checks: dict[str, Any] = {}

        if args.mode in ("retrieval", "all"):
            print(f"Running retrieval eval: {len(cases)} cases")
            results["retrieval"] = run_retrieval_cases(cases, top_k=args.top_k)
            threshold_checks["retrieval"] = check_thresholds(results["retrieval"]["summary"], thresholds, "retrieval")

        if args.mode in ("rag", "all"):
            print(f"Running RAG tool eval: {len(cases)} cases")
            results["rag"] = run_rag_tool_cases(cases)
            threshold_checks["rag"] = check_thresholds(results["rag"]["summary"], thresholds, "rag")

        if args.mode in ("e2e", "all"):
            print(f"Running E2E API eval: {len(cases)} cases")
            results["e2e"] = run_e2e_api_cases_sync(cases, args.base_url, args.username, args.password)
            threshold_checks["e2e"] = check_thresholds(results["e2e"]["summary"], thresholds, "e2e")

        passed = all(check.get("passed", True) for check in threshold_checks.values())
        payload = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "dataset_path": str(Path(args.dataset)),
            "thresholds_path": str(Path(args.thresholds)),
            "mode": args.mode,
            "category": args.category,
            "case_id": args.case_id,
            "passed": passed,
            "results": results,
            "threshold_checks": threshold_checks,
        }
        report_paths = write_reports(payload)
        print(json.dumps({"passed": passed, "reports": report_paths}, ensure_ascii=False, indent=2))
        if args.no_fail:
            return 0
        return 0 if passed else 1
    except (ConnectionError, TimeoutError) as exc:
        print(f"Infrastructure error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
