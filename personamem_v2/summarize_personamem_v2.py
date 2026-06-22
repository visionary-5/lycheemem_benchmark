#!/usr/bin/env python3
"""Summarize PersonaMem-v2 LycheeMem JSONL predictions."""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_rows(patterns: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    obj["_file"] = path
                    rows.append(obj)
    return rows


def acc(rows: list[dict[str, Any]]) -> tuple[int, int, float]:
    total = len(rows)
    correct = sum(1 for r in rows if r.get("is_correct") is True)
    return correct, total, correct / total if total else 0.0


def by_field(rows: list[dict[str, Any]], field: str) -> list[tuple[str, int, int, float]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(field) or "")].append(row)
    out = []
    for key, group in groups.items():
        correct, total, score = acc(group)
        out.append((key, correct, total, score))
    out.sort(key=lambda x: (-x[3], x[0]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, nargs="+")
    ap.add_argument("--json_out", default="")
    args = ap.parse_args()

    rows = load_rows(args.glob)
    correct, total, score = acc(rows)
    avg_retrieved = sum(float(r.get("retrieved_total") or 0) for r in rows) / total if total else 0.0
    avg_ctx = sum(float(r.get("retrieved_context_chars") or 0) for r in rows) / total if total else 0.0
    print(f"Overall MCQ Accuracy: {score:.3f} ({correct}/{total})")
    print(f"Avg retrieved_total: {avg_retrieved:.2f}")
    print(f"Avg retrieved_context_chars: {avg_ctx:.0f}")

    summary: dict[str, Any] = {
        "overall": {
            "correct": correct,
            "total": total,
            "accuracy": score,
            "avg_retrieved_total": avg_retrieved,
            "avg_retrieved_context_chars": avg_ctx,
        },
        "breakdowns": {},
    }
    for field in [
        "pref_type",
        "updated",
        "who",
        "sensitive_info",
        "conversation_scenario",
        "topic_query",
        "topic_preference",
    ]:
        vals = by_field(rows, field)
        if not vals:
            continue
        print(f"\nBy {field}:")
        summary["breakdowns"][field] = []
        for key, c, t, s in vals[:30]:
            print(f"  {key or '<blank>'}: {s:.3f} ({c}/{t})")
            summary["breakdowns"][field].append(
                {"value": key, "correct": c, "total": t, "accuracy": s}
            )

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
