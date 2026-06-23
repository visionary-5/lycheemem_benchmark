#!/usr/bin/env python3
"""Run PersonaMem-v2 MCQ evaluation through LycheeMem.

This is an external adapter. It does not import or modify LycheeMem internals.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import requests
from openai import OpenAI


LETTERS = "ABCD"
OFFICIAL_RECALL_SUFFIX = (
    " Please recall my related preferences from our conversation history to give personalized responses."
)


def stable_id(text: str, n: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:n]


def post_json(url: str, path: str, payload: dict[str, Any], timeout: int, tries: int = 4) -> dict[str, Any]:
    last = ""
    for k in range(tries):
        try:
            r = requests.post(url + path, json=payload, timeout=timeout)
            if r.ok:
                return r.json()
            last = f"{r.status_code}: {r.text[:300]}"
        except Exception as exc:
            last = str(exc)
        time.sleep(1.5 * (k + 1))
    raise RuntimeError(f"POST {path} failed after {tries}: {last}")


def clear_memory(url: str) -> None:
    r = requests.delete(url + "/memory/clear", timeout=60)
    r.raise_for_status()


def append_turn(url: str, session_id: str, role: str, content: str, max_chars: int = 95000) -> None:
    role = role if role in {"system", "user", "assistant"} else "user"
    text = str(content or "").strip()
    if not text:
        return
    for start in range(0, len(text), max_chars):
        chunk = text[start : start + max_chars]
        post_json(
            url,
            "/memory/append-turn",
            {"session_id": session_id, "role": role, "content": chunk},
            timeout=120,
        )


def consolidate(url: str, session_id: str, session_date: str | None) -> dict[str, Any]:
    return post_json(
        url,
        "/memory/consolidate",
        {
            "session_id": session_id,
            "background": False,
            "flush_session": True,
            "force_ingest": True,
            "session_date": session_date,
        },
        timeout=3600,
        tries=2,
    )


def search_memory(url: str, query: str, top_k: int, reference_time: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"query": query[:1000], "top_k": min(top_k, 50), "include_semantic": True}
    if reference_time:
        payload["reference_time"] = reference_time
    return post_json(url, "/memory/search", payload, timeout=120)


def parse_literal_or_json(value: str) -> Any:
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        return ast.literal_eval(text)
    except Exception:
        return text


def parse_user_query(value: str) -> dict[str, str]:
    obj = parse_literal_or_json(value)
    if isinstance(obj, dict):
        role = str(obj.get("role") or "user")
        content = str(obj.get("content") or "")
        return {"role": role, "content": content}
    return {"role": "user", "content": str(value or "").strip("\"'")}


def parse_incorrect_answers(value: str) -> list[str]:
    obj = parse_literal_or_json(value)
    if obj is None:
        return []
    if isinstance(obj, list):
        return [str(x) for x in obj]
    return [str(obj)]


def normalize_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text") or ""))
                elif "text" in item:
                    parts.append(str(item.get("text") or ""))
                elif item.get("type"):
                    parts.append(f"[{item.get('type')} omitted]")
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p.strip())
    return str(content or "")


def load_history(path: Path) -> list[dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "chat_history" in data:
        raw = data["chat_history"]
    elif isinstance(data, dict) and "conversations" in data:
        raw = data["conversations"]
    elif isinstance(data, list):
        raw = data
    elif isinstance(data, dict):
        raw = []
        for value in data.values():
            if isinstance(value, dict) and "conversations" in value:
                raw = value["conversations"]
                break
            if isinstance(value, list):
                raw = value
                break
    else:
        raw = []

    messages: list[dict[str, str]] = []
    for msg in raw:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "user").lower()
        if role not in {"system", "user", "assistant"}:
            role = "user"
        content = normalize_content(msg.get("content"))
        if content.strip():
            messages.append({"role": role, "content": content.strip()})
    return messages


def resolve_history_path(data_root: Path, raw_path: str, cache: dict[str, Path | None]) -> Path:
    raw_path = str(raw_path or "").strip()
    if raw_path in cache:
        cached = cache[raw_path]
        if cached is None:
            raise FileNotFoundError(raw_path)
        return cached

    candidates = [
        Path(raw_path),
        data_root / raw_path,
        data_root / raw_path.lstrip("/"),
    ]
    if raw_path.startswith("data/"):
        candidates.append(data_root / raw_path[len("data/") :])
    else:
        candidates.append(data_root / "data" / raw_path)

    for candidate in candidates:
        if candidate.exists():
            cache[raw_path] = candidate
            return candidate

    basename = Path(raw_path).name
    matches = list(data_root.rglob(basename)) if basename else []
    if matches:
        cache[raw_path] = matches[0]
        return matches[0]
    cache[raw_path] = None
    raise FileNotFoundError(f"Cannot resolve history path: {raw_path}")


def official_question(row: dict[str, str]) -> str:
    user_query = parse_user_query(row.get("user_query", ""))
    question = user_query["content"]
    if question:
        question += OFFICIAL_RECALL_SUFFIX
    return question


def raw_question(row: dict[str, str]) -> str:
    return parse_user_query(row.get("user_query", ""))["content"]


def make_options(row: dict[str, str], question: str) -> tuple[str, dict[str, str], str]:
    correct = str(row.get("correct_answer") or "")
    incorrect = parse_incorrect_answers(row.get("incorrect_answers", "[]"))
    options = [correct] + incorrect
    options = [x for x in options if str(x).strip()]
    if len(options) < 4:
        raise ValueError(f"Need 4 MCQ options, got {len(options)}")
    options = options[:4]
    row_seed = hash(f"{row.get('persona_id','')}_{question}") % 2**32
    rng = random.Random(row_seed)
    rng.shuffle(options)
    mapping = {LETTERS[i]: options[i] for i in range(4)}
    correct_letter = next(letter for letter, answer in mapping.items() if answer == correct)
    option_text = "\n".join(f"{letter}. {answer}" for letter, answer in mapping.items())
    return option_text, mapping, correct_letter


def extract_answer_letter(text: str, option_mapping: dict[str, str]) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    patterns = [
        r"\$\\boxed\{([A-Z])\}\$",
        r"\\boxed\{([A-Z])\}",
        r"Final Answer:\s*([A-Z])",
        r"final answer:\s*([A-Z])",
        r"Answer:\s*([A-Z])",
        r"answer:\s*([A-Z])",
        r"final answer is\s*\$?\\boxed\{([A-Z])\}\$?",
        r"final answer is\s*([A-Z])",
        r"the answer is\s*\$?\\boxed\{([A-Z])\}\$?",
        r"the answer is\s*([A-Z])",
        r"\b([A-Z])\.\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, raw, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            return str(match.group(1)).upper()
    return ""


def normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(text).lower())).strip()


def build_mcq_instruction(option_text: str) -> str:
    return (
        "Please choose the best answer from the following options:\n\n"
        f"{option_text}"
        "\n\nThink step by step about which answer best fits the user's query and conversation context. "
        "Provide your reasoning first, then give your final answer as 'Final Answer: [Letter]'"
    )


def build_reader_messages(
    context: str,
    user_query: str,
    option_text: str,
    *,
    prompt_mode: str,
) -> list[dict[str, str]]:
    if not context.strip():
        context = "(No relevant memory found)"
    instruction = build_mcq_instruction(option_text)
    memory_message = {
        "role": "system",
        "content": (
            "Relevant conversation history has been compressed into the memory context below. "
            "Use it as the available conversation context for personalization.\n\n"
            f"{context}"
        ),
    }
    if prompt_mode == "qwen_user_final":
        return [
            memory_message,
            {"role": "user", "content": f"{user_query}\n\n{instruction}"},
        ]
    if prompt_mode == "official_system_final":
        return [
            memory_message,
            {"role": "user", "content": user_query},
            {"role": "system", "content": instruction},
        ]
    raise ValueError(f"Unknown prompt_mode: {prompt_mode}")


def build_search_query(
    row: dict[str, str],
    *,
    question: str,
    raw_query: str,
    option_text: str,
    search_mode: str,
) -> str:
    if search_mode == "query":
        return question
    if search_mode == "query_raw":
        return raw_query
    if search_mode == "query_options":
        return f"{question}\n\nCandidate responses:\n{option_text}"
    if search_mode == "query_raw_options":
        return f"{raw_query}\n\nCandidate responses:\n{option_text}"

    metadata_parts = []
    for key, label in [
        ("topic_query", "Topic"),
        ("conversation_scenario", "Scenario"),
    ]:
        value = str(row.get(key) or "").strip()
        if value:
            metadata_parts.append(f"{label}: {value}")
    metadata = "\n".join(metadata_parts)

    if search_mode == "query_metadata":
        return "\n\n".join(part for part in [raw_query, metadata] if part)
    if search_mode == "query_metadata_options":
        return "\n\n".join(
            part
            for part in [raw_query, metadata, f"Candidate responses:\n{option_text}"]
            if part
        )
    raise ValueError(f"Unknown search_mode: {search_mode}")


def call_reader(client: OpenAI, model: str, messages: list[dict[str, str]], max_tokens: int = 4096) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return (resp.choices[0].message.content or "").strip()


def group_messages(
    messages: list[dict[str, str]],
    turns_per_session: int,
    *,
    separate_system_session: bool,
) -> list[list[dict[str, str]]]:
    prefix: list[list[dict[str, str]]] = []
    if separate_system_session and messages and messages[0].get("role") == "system":
        prefix = [[messages[0]]]
        messages = messages[1:]

    groups: list[list[dict[str, str]]] = []
    current: list[dict[str, str]] = []
    for msg in messages:
        current.append(msg)
        if len(current) >= turns_per_session:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return prefix + groups


def ingest_history(
    url: str,
    history_key: str,
    messages: list[dict[str, str]],
    *,
    turns_per_session: int,
    separate_system_session: bool,
    ingest_workers: int,
    session_date_start: date,
) -> dict[str, Any]:
    groups = group_messages(
        messages,
        turns_per_session,
        separate_system_session=separate_system_session,
    )
    base = f"pmv2_{stable_id(history_key, 10)}"
    print(f"  [ingest] {len(messages)} messages -> {len(groups)} sessions", flush=True)

    def ingest_group(item: tuple[int, list[dict[str, str]]]) -> dict[str, Any]:
        idx, group = item
        sid = f"{base}_s{idx:03d}"
        for msg in group:
            append_turn(url, sid, msg["role"], msg["content"])
        sdate = (session_date_start + timedelta(days=idx)).isoformat()
        result = consolidate(url, sid, sdate)
        return {"session_id": sid, "date": sdate, **result}

    t0 = time.time()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, ingest_workers)) as ex:
        futs = [ex.submit(ingest_group, (i, group)) for i, group in enumerate(groups)]
        for j, fut in enumerate(as_completed(futs), 1):
            result = fut.result()
            results.append(result)
            if j == 1 or j % 5 == 0 or j == len(futs):
                print(f"    ... consolidated {j}/{len(futs)} ({time.time() - t0:.0f}s)", flush=True)
    return {
        "message_count": len(messages),
        "session_count": len(groups),
        "separate_system_session": separate_system_session,
        "elapsed_sec": time.time() - t0,
        "consolidations": results,
    }


def load_rows(benchmark_csv: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with benchmark_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            row = dict(row)
            row["_row_index"] = str(i)
            rows.append(row)
    return rows


def select_rows(
    rows: list[dict[str, str]],
    *,
    size: str,
    history_start: int,
    max_histories: int,
    max_questions_per_history: int,
    max_rows: int,
) -> list[dict[str, str]]:
    column = f"chat_history_{size}_link"
    grouped: dict[str, list[dict[str, str]]] = {}
    ordered_keys: list[str] = []
    for row in rows:
        path = str(row.get(column) or row.get("chat_history_link") or "").strip()
        if not path:
            continue
        if path not in grouped:
            ordered_keys.append(path)
        grouped.setdefault(path, []).append(row)

    selected_history_keys = ordered_keys[history_start:]
    if max_histories:
        selected_history_keys = selected_history_keys[:max_histories]

    out: list[dict[str, str]] = []
    for key in selected_history_keys:
        group = grouped[key]
        if max_questions_per_history:
            group = group[:max_questions_per_history]
        out.extend(group)
        if max_rows and len(out) >= max_rows:
            out = out[:max_rows]
            break
    return out


def load_done_rows(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("row_index") is not None:
                done.add(str(obj["row_index"]))
    return done


def answer_row(
    row: dict[str, str],
    row_index: int,
    *,
    url: str,
    client: OpenAI,
    model: str,
    top_k: int,
    search_mode: str,
    max_context_chars: int,
    prompt_mode: str,
) -> dict[str, Any]:
    question = official_question(row)
    raw_query = raw_question(row)
    option_text, option_mapping, correct_letter = make_options(row, question)
    search_query = build_search_query(
        row,
        question=question,
        raw_query=raw_query,
        option_text=option_text,
        search_mode=search_mode,
    )
    search_resp = search_memory(url, search_query, top_k=top_k)
    context = search_resp.get("raw_retrieved_context") or ""
    if max_context_chars and len(context) > max_context_chars:
        context = context[:max_context_chars]
    messages = build_reader_messages(context, question, option_text, prompt_mode=prompt_mode)
    response = call_reader(client, model, messages)
    pred_letter = extract_answer_letter(response, option_mapping)
    is_correct = pred_letter == correct_letter
    return {
        "row_index": row_index,
        "persona_id": row.get("persona_id"),
        "history_path": row.get("chat_history_32k_link") or row.get("chat_history_link"),
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
        "is_correct": is_correct,
        "model_response": response,
        "retrieved_total": search_resp.get("total", 0),
        "retrieved_context_chars": len(context),
        "search_mode": search_mode,
        "search_query_chars": len(search_query),
        "top_k": top_k,
        "max_context_chars": max_context_chars,
        "prompt_mode": prompt_mode,
        "option_seed_mode": "official_python_hash",
    }


def write_summary(prediction_path: Path, summary_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    with prediction_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    total = len(rows)
    correct = sum(1 for r in rows if r.get("is_correct") is True)
    summary = {
        "total": total,
        "correct": correct,
        "accuracy": (correct / total if total else 0.0),
        "avg_retrieved_total": (
            sum(float(r.get("retrieved_total") or 0) for r in rows) / total if total else 0.0
        ),
        "avg_retrieved_context_chars": (
            sum(float(r.get("retrieved_context_chars") or 0) for r in rows) / total if total else 0.0
        ),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--benchmark_file", default="")
    ap.add_argument("--size", choices=["32k", "128k"], default="32k")
    ap.add_argument("--history_start", type=int, default=0)
    ap.add_argument("--max_histories", type=int, default=1)
    ap.add_argument("--max_questions_per_history", type=int, default=20)
    ap.add_argument("--max_rows", type=int, default=0)
    ap.add_argument("--turns_per_session", type=int, default=12)
    ap.add_argument("--separate_system_session", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--ingest_workers", type=int, default=4)
    ap.add_argument("--lycheemem_url", default="http://localhost:8000")
    ap.add_argument("--reader_url", default="http://10.251.171.6:28043/v1")
    ap.add_argument("--reader_model", default="my-llm-qwen")
    ap.add_argument("--reader_api_key", default="dummy")
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--max_context_chars", type=int, default=0)
    ap.add_argument(
        "--prompt_mode",
        choices=["qwen_user_final", "official_system_final"],
        default="qwen_user_final",
    )
    ap.add_argument(
        "--search_mode",
        choices=[
            "query",
            "query_raw",
            "query_options",
            "query_raw_options",
            "query_metadata",
            "query_metadata_options",
        ],
        default="query",
    )
    ap.add_argument("--clear_before_ingest", action="store_true")
    ap.add_argument("--skip_ingest", action="store_true")
    ap.add_argument("--output_dir", default="./outputs")
    ap.add_argument("--run_id", default="")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    benchmark_file = Path(args.benchmark_file) if args.benchmark_file else data_root / "benchmark/text/benchmark.csv"
    if not benchmark_file.exists():
        alt = data_root / "data/benchmark/text/benchmark.csv"
        benchmark_file = alt if alt.exists() else benchmark_file
    if not benchmark_file.exists():
        raise FileNotFoundError(f"benchmark CSV not found: {benchmark_file}")

    run_id = args.run_id or f"pmv2_{args.size}_h{args.history_start}_{stable_id(str(time.time()), 6)}"
    out_dir = Path(args.output_dir).resolve() / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    summary_path = out_dir / "summary.json"
    meta_path = out_dir / "run_meta.json"

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

    column = f"chat_history_{args.size}_link"
    history_keys: list[str] = []
    for row in rows:
        key = str(row.get(column) or row.get("chat_history_link") or "")
        if key and key not in history_keys:
            history_keys.append(key)
    print(f"[run] selected rows={len(rows)} histories={len(history_keys)} size={args.size}", flush=True)

    meta_path.write_text(
        json.dumps(
            {
                "args": vars(args),
                "benchmark_file": str(benchmark_file),
                "rows": len(rows),
                "histories": history_keys,
                "prompt_mode": args.prompt_mode,
                "search_mode": args.search_mode,
                "top_k": args.top_k,
                "max_context_chars": args.max_context_chars,
                "option_seed_mode": "official_python_hash",
                "python_hash_seed": os.environ.get("PYTHONHASHSEED", ""),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    client = OpenAI(base_url=args.reader_url, api_key=args.reader_api_key)
    resolver_cache: dict[str, Path | None] = {}
    done = load_done_rows(pred_path) if args.resume else set()

    # Current implementation is intended for clean one-history runs. Multiple histories
    # are allowed for smoke testing, but clean headline runs should use isolated shell loop.
    for hi, history_key in enumerate(history_keys):
        history_rows = [r for r in rows if str(r.get(column) or r.get("chat_history_link") or "") == history_key]
        print(f"\n=== history {args.history_start + hi}: rows={len(history_rows)} ===", flush=True)
        if args.clear_before_ingest:
            print("  [clear] API clear", flush=True)
            clear_memory(args.lycheemem_url)
            time.sleep(1)
        if not args.skip_ingest:
            history_path = resolve_history_path(data_root, history_key, resolver_cache)
            messages = load_history(history_path)
            ingest_stats = ingest_history(
                args.lycheemem_url,
                history_key,
                messages,
                turns_per_session=args.turns_per_session,
                separate_system_session=args.separate_system_session,
                ingest_workers=args.ingest_workers,
                session_date_start=date(2024, 1, 1),
            )
            (out_dir / f"ingest_{stable_id(history_key)}.json").write_text(
                json.dumps(ingest_stats, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        with pred_path.open("a", encoding="utf-8") as f:
            for row in history_rows:
                row_index = int(row["_row_index"])
                if str(row_index) in done:
                    continue
                t0 = time.time()
                try:
                    result = answer_row(
                        row,
                        row_index,
                        url=args.lycheemem_url,
                        client=client,
                        model=args.reader_model,
                        top_k=args.top_k,
                        search_mode=args.search_mode,
                        max_context_chars=args.max_context_chars,
                        prompt_mode=args.prompt_mode,
                    )
                    result["error"] = ""
                except Exception as exc:
                    result = {"row_index": row_index, "is_correct": False, "error": str(exc)}
                result["elapsed_sec"] = time.time() - t0
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                done.add(str(row_index))
                marker = "OK" if result.get("is_correct") else "NO"
                print(
                    f"  [q row={row_index}] {marker} pred={result.get('predicted_letter')} "
                    f"gold={result.get('correct_letter')} retrieved={result.get('retrieved_total')} "
                    f"{result['elapsed_sec']:.1f}s",
                    flush=True,
                )
                write_summary(pred_path, summary_path)

    write_summary(pred_path, summary_path)
    print(f"\n[done] predictions -> {pred_path}", flush=True)
    print(f"[done] summary -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
