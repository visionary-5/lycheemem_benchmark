#!/usr/bin/env python3
"""Run PersonaMem-v2 MCQ reader-only baselines.

This uses the same option construction and answer parsing as the LycheeMem
adapter, but does not call LycheeMem.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from run_personamem_v2_lycheemem import (
    build_mcq_instruction,
    build_reader_messages,
    call_reader,
    extract_answer_letter,
    load_history,
    load_rows,
    make_options,
    official_question,
    resolve_history_path,
    select_rows,
    stable_id,
    write_summary,
)


def build_context(mode: str, messages: list[dict[str, str]], max_chars: int) -> str:
    if mode == "none":
        return ""
    if mode == "system":
        if messages and messages[0].get("role") == "system":
            return messages[0].get("content", "")[:max_chars]
        return ""
    if mode == "full_history":
        parts = [f"{m['role']}: {m['content']}" for m in messages]
        return "\n\n".join(parts)[-max_chars:]
    raise ValueError(f"unknown context mode: {mode}")


def build_baseline_messages(
    mode: str,
    history_messages: list[dict[str, str]],
    memory_context: str,
    question: str,
    option_text: str,
) -> list[dict[str, str]]:
    if mode == "full_history":
        messages = [dict(m) for m in history_messages]
        instruction = build_mcq_instruction(option_text)
        messages.append({"role": "user", "content": f"{question}\n\n{instruction}"})
        return messages
    return build_reader_messages(memory_context, question, option_text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--benchmark_file", default="")
    ap.add_argument("--size", choices=["32k", "128k"], default="32k")
    ap.add_argument("--history_start", type=int, default=0)
    ap.add_argument("--max_histories", type=int, default=4)
    ap.add_argument("--max_questions_per_history", type=int, default=5)
    ap.add_argument("--max_rows", type=int, default=0)
    ap.add_argument("--context_mode", choices=["none", "system", "full_history"], default="none")
    ap.add_argument("--max_context_chars", type=int, default=120000)
    ap.add_argument("--reader_url", default="http://10.251.171.6:28043/v1")
    ap.add_argument("--reader_model", default="my-llm-qwen")
    ap.add_argument("--reader_api_key", default="dummy")
    ap.add_argument("--output_dir", default="./outputs")
    ap.add_argument("--run_id", default="")
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    benchmark_file = Path(args.benchmark_file) if args.benchmark_file else data_root / "benchmark/text/benchmark.csv"
    if not benchmark_file.exists():
        alt = data_root / "data/benchmark/text/benchmark.csv"
        benchmark_file = alt if alt.exists() else benchmark_file
    if not benchmark_file.exists():
        raise FileNotFoundError(f"benchmark CSV not found: {benchmark_file}")

    rows = select_rows(
        load_rows(benchmark_file),
        size=args.size,
        history_start=args.history_start,
        max_histories=args.max_histories,
        max_questions_per_history=args.max_questions_per_history,
        max_rows=args.max_rows,
    )
    if not rows:
        raise RuntimeError("No rows selected")

    run_id = args.run_id or f"reader_{args.context_mode}_{args.size}_{stable_id(str(time.time()), 6)}"
    out_dir = Path(args.output_dir).resolve() / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    summary_path = out_dir / "summary.json"

    column = f"chat_history_{args.size}_link"
    resolver_cache: dict[str, Path | None] = {}
    context_cache: dict[str, str] = {}
    history_cache: dict[str, list[dict[str, str]]] = {}
    client = OpenAI(base_url=args.reader_url, api_key=args.reader_api_key)

    meta = {
        "args": vars(args),
        "benchmark_file": str(benchmark_file),
        "rows": len(rows),
        "prompt_mode": "official_mcq_text_qwen_user_final",
        "option_seed_mode": "official_python_hash",
        "python_hash_seed": os.environ.get("PYTHONHASHSEED", ""),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    with pred_path.open("w", encoding="utf-8") as f:
        for row in rows:
            row_index = int(row["_row_index"])
            history_key = str(row.get(column) or row.get("chat_history_link") or "")
            if history_key not in history_cache:
                history_path = resolve_history_path(data_root, history_key, resolver_cache)
                history_cache[history_key] = load_history(history_path)
                context_cache[history_key] = build_context(
                    args.context_mode,
                    history_cache[history_key],
                    args.max_context_chars,
                )

            t0 = time.time()
            try:
                question = official_question(row)
                option_text, option_mapping, correct_letter = make_options(row, question)
                messages = build_baseline_messages(
                    args.context_mode,
                    history_cache[history_key],
                    context_cache[history_key],
                    question,
                    option_text,
                )
                response = call_reader(client, args.reader_model, messages)
                pred_letter = extract_answer_letter(response, option_mapping)
                result: dict[str, Any] = {
                    "row_index": row_index,
                    "persona_id": row.get("persona_id"),
                    "history_path": history_key,
                    "user_query": question,
                    "topic_query": row.get("topic_query"),
                    "topic_preference": row.get("topic_preference"),
                    "conversation_scenario": row.get("conversation_scenario"),
                    "pref_type": row.get("pref_type"),
                    "who": row.get("who"),
                    "updated": row.get("updated"),
                    "sensitive_info": row.get("sensitive_info"),
                    "correct_answer": row.get("correct_answer"),
                    "incorrect_answers": row.get("incorrect_answers"),
                    "option_mapping": option_mapping,
                    "correct_letter": correct_letter,
                    "predicted_letter": pred_letter,
                    "is_correct": pred_letter == correct_letter,
                    "model_response": response,
                    "context_mode": args.context_mode,
                    "context_chars": len(context_cache[history_key]),
                    "prompt_mode": "official_mcq_text_qwen_user_final",
                    "option_seed_mode": "official_python_hash",
                    "error": "",
                }
            except Exception as exc:
                result = {"row_index": row_index, "is_correct": False, "error": str(exc)}
            result["elapsed_sec"] = time.time() - t0
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            marker = "OK" if result.get("is_correct") else "NO"
            print(
                f"[q row={row_index}] {marker} pred={result.get('predicted_letter')} "
                f"gold={result.get('correct_letter')} {result['elapsed_sec']:.1f}s",
                flush=True,
            )
            write_summary(pred_path, summary_path)

    write_summary(pred_path, summary_path)
    print(f"[done] predictions -> {pred_path}", flush=True)
    print(f"[done] summary -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
