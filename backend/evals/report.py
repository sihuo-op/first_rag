"""Report generation for RAG evaluation runs."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


REPORT_DIR = Path(__file__).with_name("reports")


def write_reports(payload: dict[str, Any], report_dir: str | Path = REPORT_DIR) -> dict[str, str]:
    output_dir = Path(report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"{timestamp}.json"
    latest_json_path = output_dir / "latest.json"
    latest_md_path = output_dir / "latest.md"

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    json_path.write_text(text, encoding="utf-8")
    latest_json_path.write_text(text, encoding="utf-8")
    latest_md_path.write_text(render_markdown(payload), encoding="utf-8")

    return {
        "json": str(json_path),
        "latest_json": str(latest_json_path),
        "latest_markdown": str(latest_md_path),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# RAG Evaluation Report",
        "",
        f"- Timestamp: `{payload.get('timestamp')}`",
        f"- Dataset: `{payload.get('dataset_path')}`",
        f"- Overall passed: **{payload.get('passed')}**",
        "",
        "## Summary",
        "",
    ]

    for mode, result in (payload.get("results") or {}).items():
        summary = result.get("summary") or {}
        lines.extend([
            f"### {mode}",
            "",
            "| Metric | Value |",
            "|---|---:|",
        ])
        for key, value in summary.items():
            lines.append(f"| {key} | {value} |")
        lines.append("")

        threshold = (payload.get("threshold_checks") or {}).get(mode)
        if threshold:
            lines.extend(["| Threshold | Value | Target | Passed |", "|---|---:|---:|:---:|"])
            for check in threshold.get("checks", []):
                lines.append(
                    f"| {check.get('metric')} | {check.get('value')} | {check.get('threshold')} | {check.get('passed')} |"
                )
            lines.append("")

    failures = collect_failures(payload)
    lines.extend(["## Failed / Low-score Cases", ""])
    if not failures:
        lines.append("No failed cases recorded.")
    else:
        lines.extend(["| Mode | Case | Category | Reason |", "|---|---|---|---|"])
        for failure in failures[:50]:
            lines.append(
                f"| {failure['mode']} | {failure['id']} | {failure['category']} | {failure['reason']} |"
            )
    lines.append("")

    slow_cases = collect_slow_cases(payload)
    lines.extend(["## Slowest Cases", "", "| Mode | Case | Latency seconds |", "|---|---|---:|"])
    for item in slow_cases[:20]:
        lines.append(f"| {item['mode']} | {item['id']} | {item['latency']} |")
    lines.append("")

    return "\n".join(lines)


def collect_failures(payload: dict[str, Any]) -> list[dict[str, Any]]:
    failures = []
    for mode, result in (payload.get("results") or {}).items():
        for case in result.get("cases", []):
            metrics = case.get("metrics") or {}
            if case.get("error"):
                failures.append({"mode": mode, "id": case.get("id"), "category": case.get("category"), "reason": case.get("error")})
                continue
            if metrics.get("has_error"):
                failures.append({"mode": mode, "id": case.get("id"), "category": case.get("category"), "reason": "case has runtime errors"})
                continue
            if metrics.get("answer_rule_score", 1.0) < 0.5:
                failures.append({"mode": mode, "id": case.get("id"), "category": case.get("category"), "reason": "low answer_rule_score"})
                continue
            if metrics.get("recall_at_10", 1.0) < 0.5:
                failures.append({"mode": mode, "id": case.get("id"), "category": case.get("category"), "reason": "low recall_at_10"})
    return failures


def collect_slow_cases(payload: dict[str, Any]) -> list[dict[str, Any]]:
    slow_cases = []
    for mode, result in (payload.get("results") or {}).items():
        for case in result.get("cases", []):
            latency = float((case.get("metrics") or {}).get("latency_seconds") or 0.0)
            slow_cases.append({"mode": mode, "id": case.get("id"), "latency": latency})
    return sorted(slow_cases, key=lambda item: item["latency"], reverse=True)
