#!/usr/bin/env python3
"""Download the PersonaMem-v2 text benchmark subset needed for MCQ evaluation.

The HF dataset is file-based. This script first downloads benchmark.csv, then
downloads only the chat histories referenced by the selected rows/histories.
It also supports falling back to a broad snapshot of benchmark/text when the
CSV path cannot be resolved directly.
"""

from __future__ import annotations

import argparse
import ast
import csv
import os
from pathlib import Path
from typing import Iterable

from huggingface_hub import hf_hub_download, snapshot_download


REPO_ID = "bowen-upenn/PersonaMem-v2"
DEFAULT_CSV_CANDIDATES = (
    "benchmark/text/benchmark.csv",
    "data/benchmark/text/benchmark.csv",
    "benchmark.csv",
)


def _try_download(filename: str, data_root: Path) -> Path | None:
    candidates = [filename]
    if filename.startswith("data/"):
        candidates.append(filename[len("data/") :])
    else:
        candidates.append(f"data/{filename}")
    if filename.startswith("/"):
        candidates.append(filename.lstrip("/"))

    seen: set[str] = set()
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            path = hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=candidate,
                local_dir=str(data_root),
                local_dir_use_symlinks=False,
            )
            return Path(path)
        except Exception:
            continue
    return None


def download_benchmark_csv(data_root: Path) -> Path:
    for candidate in DEFAULT_CSV_CANDIDATES:
        path = _try_download(candidate, data_root)
        if path and path.exists():
            print(f"[download] benchmark csv: {path}", flush=True)
            return path

    print("[download] direct CSV lookup failed; snapshotting benchmark/text/**", flush=True)
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(data_root),
        local_dir_use_symlinks=False,
        allow_patterns=["benchmark/text/**", "data/benchmark/text/**"],
    )
    for candidate in DEFAULT_CSV_CANDIDATES:
        local = data_root / candidate
        if local.exists():
            print(f"[download] benchmark csv: {local}", flush=True)
            return local
    raise FileNotFoundError("Could not locate PersonaMem-v2 benchmark.csv after download")


def parse_user_query(value: str) -> str:
    try:
        obj = ast.literal_eval(value)
        if isinstance(obj, dict):
            return str(obj.get("content") or "")
    except Exception:
        pass
    return str(value or "")


def selected_history_paths(
    benchmark_csv: Path,
    *,
    size: str,
    max_histories: int,
    max_rows: int,
) -> list[str]:
    column = f"chat_history_{size}_link"
    paths: list[str] = []
    seen: set[str] = set()
    with benchmark_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_rows and i >= max_rows:
                break
            path = str(row.get(column) or row.get("chat_history_link") or "").strip()
            if not path or path in seen:
                continue
            seen.add(path)
            paths.append(path)
            if max_histories and len(paths) >= max_histories:
                break
    if not paths:
        raise RuntimeError(f"No history paths found in {benchmark_csv} column {column}")
    return paths


def download_histories(paths: Iterable[str], data_root: Path) -> None:
    ok = 0
    failed: list[str] = []
    for idx, history_path in enumerate(paths, 1):
        local = data_root / history_path
        if local.exists():
            ok += 1
            print(f"[history {idx}] exists: {history_path}", flush=True)
            continue
        got = _try_download(history_path, data_root)
        if got and got.exists():
            ok += 1
            print(f"[history {idx}] downloaded: {history_path}", flush=True)
        else:
            failed.append(history_path)
            print(f"[history {idx}] FAILED: {history_path}", flush=True)
    print(f"[download] histories ok={ok} failed={len(failed)}", flush=True)
    if failed:
        print("[download] failed history paths:", flush=True)
        for path in failed[:20]:
            print(f"  {path}", flush=True)
        raise SystemExit(2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--size", choices=["32k", "128k"], default="32k")
    ap.add_argument("--max_histories", type=int, default=0)
    ap.add_argument("--max_rows", type=int, default=0)
    ap.add_argument(
        "--snapshot_text",
        action="store_true",
        help="Download the whole benchmark/text subtree instead of referenced histories only.",
    )
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    benchmark_csv = download_benchmark_csv(data_root)

    if args.snapshot_text:
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            local_dir=str(data_root),
            local_dir_use_symlinks=False,
            allow_patterns=["benchmark/text/**", "data/benchmark/text/**"],
        )
        return

    paths = selected_history_paths(
        benchmark_csv,
        size=args.size,
        max_histories=args.max_histories,
        max_rows=args.max_rows,
    )
    print(f"[download] selected {len(paths)} unique {args.size} histories", flush=True)
    download_histories(paths, data_root)


if __name__ == "__main__":
    main()

