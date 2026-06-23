#!/usr/bin/env python3
"""Compare PersonaMem-v2 prediction JSONL files by row_index."""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_run(spec: str) -> tuple[str, dict[str, dict[str, Any]]]:
    if "=" not in spec:
        raise ValueError(f"Run spec must be LABEL=GLOB, got: {spec}")
    label, pattern = spec.split("=", 1)
    rows: dict[str, dict[str, Any]] = {}
    for path in sorted(glob.glob(pattern)):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                row_index = obj.get("row_index")
                if row_index is None:
                    continue
                obj["_file"] = path
                rows[str(row_index)] = obj
    return label, rows


def score(rows: dict[str, dict[str, Any]]) -> tuple[int, int, float]:
    total = len(rows)
    correct = sum(1 for row in rows.values() if row.get("is_correct") is True)
    return correct, total, correct / total if total else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run to compare, formatted as LABEL=GLOB. Quote globs in the shell.",
    )
    ap.add_argument("--json_out", default="")
    args = ap.parse_args()

    loaded = [load_run(spec) for spec in args.run]
    if not loaded:
        raise RuntimeError("No runs loaded")

    summary: dict[str, Any] = {"runs": {}, "comparisons": []}
    for label, rows in loaded:
        correct, total, acc = score(rows)
        summary["runs"][label] = {"correct": correct, "total": total, "accuracy": acc}
        print(f"{label}: {acc:.3f} ({correct}/{total})")

    base_label, base_rows = loaded[0]
    base_keys = set(base_rows)
    for label, rows in loaded[1:]:
        keys = sorted(base_keys & set(rows), key=lambda x: int(x) if x.isdigit() else x)
        wins = losses = ties = 0
        by_history: dict[str, list[int]] = defaultdict(list)
        for key in keys:
            base_ok = base_rows[key].get("is_correct") is True
            run_ok = rows[key].get("is_correct") is True
            delta = int(run_ok) - int(base_ok)
            if delta > 0:
                wins += 1
            elif delta < 0:
                losses += 1
            else:
                ties += 1
            history = str(rows[key].get("history_path") or "")
            by_history[history].append(delta)

        net = wins - losses
        print(f"{label} vs {base_label}: overlap={len(keys)} net={net:+d} wins={wins} losses={losses} ties={ties}")
        worst = sorted(
            ((sum(vals), history, len(vals)) for history, vals in by_history.items()),
            key=lambda item: (item[0], item[1]),
        )[:10]
        best = sorted(
            ((sum(vals), history, len(vals)) for history, vals in by_history.items()),
            key=lambda item: (-item[0], item[1]),
        )[:10]
        summary["comparisons"].append(
            {
                "label": label,
                "baseline": base_label,
                "overlap": len(keys),
                "net": net,
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "worst_histories": [
                    {"history": history, "net": net_delta, "total": total}
                    for net_delta, history, total in worst
                ],
                "best_histories": [
                    {"history": history, "net": net_delta, "total": total}
                    for net_delta, history, total in best
                ],
            }
        )

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
