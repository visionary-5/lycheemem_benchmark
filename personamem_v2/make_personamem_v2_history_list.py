#!/usr/bin/env python3
"""Create a reproducible PersonaMem-v2 history-index list."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


def load_history_keys(benchmark_csv: Path, size: str) -> list[str]:
    column = f"chat_history_{size}_link"
    keys: list[str] = []
    seen: set[str] = set()
    with benchmark_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            key = str(row.get(column) or row.get("chat_history_link") or "").strip()
            if key and key not in seen:
                seen.add(key)
                keys.append(key)
    return keys


def parse_excludes(values: list[str]) -> set[int]:
    out: set[int] = set()
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                left, right = part.split("-", 1)
                start, end = int(left), int(right)
                out.update(range(start, end + 1))
            else:
                out.add(int(part))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark_csv", required=True)
    ap.add_argument("--size", default="32k")
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--seed", type=int, default=20260624)
    ap.add_argument("--exclude", action="append", default=[])
    ap.add_argument("--output", required=True)
    ap.add_argument("--metadata_output", default="")
    args = ap.parse_args()

    benchmark_csv = Path(args.benchmark_csv).resolve()
    keys = load_history_keys(benchmark_csv, args.size)
    excludes = parse_excludes(args.exclude)
    candidates = [i for i in range(len(keys)) if i not in excludes]
    if args.count > len(candidates):
        raise ValueError(f"Requested {args.count} histories, only {len(candidates)} candidates")

    rng = random.Random(args.seed)
    selected = sorted(rng.sample(candidates, args.count))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(str(i) for i in selected) + "\n", encoding="utf-8")

    metadata = {
        "benchmark_csv": str(benchmark_csv),
        "size": args.size,
        "seed": args.seed,
        "count": args.count,
        "exclude": args.exclude,
        "total_histories": len(keys),
        "selected_indices": selected,
        "selected_history_keys": [keys[i] for i in selected],
    }
    metadata_path = Path(args.metadata_output) if args.metadata_output else output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"wrote {len(selected)} indices to {output}")
    print(f"metadata: {metadata_path}")


if __name__ == "__main__":
    main()
