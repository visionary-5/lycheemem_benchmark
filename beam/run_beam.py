"""
BEAM × LycheeMem — 批量评测主脚本
将 BEAM chat.json 写入 LycheeMem 记忆，然后基于检索回答 probing questions。

用法:
  python run_beam.py \
    --input_directory chats/100K \
    --chat_size 100K \
    --start_index 0 --end_index 7 \
    --lycheemem_url http://localhost:8000 \
    --reader_url http://your-llm-server/v1 \
    --reader_model your-model-name \
    --top_k 15
"""

import argparse
import json
import os
import time
import traceback
import requests
from openai import OpenAI


def clear_memory(base_url: str):
    resp = requests.delete(f"{base_url}/memory/clear", timeout=30)
    resp.raise_for_status()
    return resp.json()


def append_turn(base_url: str, session_id: str, role: str, content: str):
    resp = requests.post(
        f"{base_url}/memory/append-turn",
        json={"session_id": session_id, "role": role, "content": content},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def consolidate_sync(base_url: str, session_id: str):
    resp = requests.post(
        f"{base_url}/memory/consolidate",
        json={
            "session_id": session_id,
            "flush_session": True,
            "background": False,
            "force_ingest": True,
        },
        timeout=5400,
    )
    resp.raise_for_status()
    return resp.json()


def search_memory(base_url: str, query: str, top_k: int = 15):
    resp = requests.post(
        f"{base_url}/memory/search",
        json={"query": query, "top_k": top_k, "include_semantic": True},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def wait_search_ready(base_url: str, max_retries: int = 30):
    for attempt in range(max_retries):
        try:
            r = requests.post(
                f"{base_url}/memory/search",
                json={"query": "test", "top_k": 1, "include_semantic": True},
                timeout=30,
            )
            if r.status_code == 200:
                data = r.json()
                print(f"  [ready] search OK (records={data.get('total', 0)})")
                return True
            print(f"  [wait] search returned {r.status_code}, retrying...")
        except Exception as e:
            print(f"  [wait] search error: {e}, retrying...")
        time.sleep(20)
    print("  [warn] search not ready after retries, proceeding anyway")
    return False


def ingest_chat(base_url: str, chat_data: list, session_id: str):
    total_msgs = 0
    for batch_idx, batch in enumerate(chat_data):
        turns = batch["turns"]
        for turn in turns:
            for msg in turn:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if not content.strip():
                    continue
                for attempt in range(3):
                    try:
                        append_turn(base_url, session_id, role, content)
                        total_msgs += 1
                        break
                    except Exception as e:
                        if attempt == 2:
                            print(f"  [warn] msg {total_msgs} failed: {e}")
                        time.sleep(5)
        print(f"  [ingest] batch {batch_idx+1}/{len(chat_data)} ({total_msgs} msgs)")

    print(f"  [ingest] {total_msgs} msgs appended. Consolidating...")
    result = consolidate_sync(base_url, session_id)
    turns_consumed = result.get("turns_consumed", "?")
    facts_added = result.get("facts_added", "?")
    print(f"  [ingest] done: turns_consumed={turns_consumed}, facts={facts_added}")
    return result


ANSWER_PROMPT = (
    "You are a memory assistant. Based ONLY on the memory context below, "
    "answer the user's question.\n\n"
    "## Memory Context\n{context}\n\n"
    "## Question\n{query}\n\n"
    "## Rules\n"
    "1. UPDATES: If a value changed over time, always report the MOST RECENT value only.\n"
    "2. CONTRADICTIONS: If the user said opposite things at different times, "
    "state both and note the contradiction.\n"
    "3. TEMPORAL: If asked about duration/gaps between dates, calculate the answer.\n"
    "4. ORDERING: Memory entries are numbered [1],[2],[3]... in chronological order. "
    "Lower numbers = earlier. Use this to answer sequence questions.\n"
    "5. SPECIFICS: Include exact numbers, dates, names, versions from context.\n"
    "6. NO INFO: Only say you don't know if context truly contains NOTHING relevant. "
    "If there's partial information, use it.\n"
    "7. FORMAT: Answer directly and concisely. No preamble.\n"
)


def answer_question(base_url, reader_client, reader_model, query, top_k=15):
    search_resp = search_memory(base_url, query, top_k=top_k)
    context = search_resp.get("raw_retrieved_context", "")
    if not context.strip():
        context = "(No relevant memory found)"

    prompt = ANSWER_PROMPT.format(context=context, query=query)

    for attempt in range(3):
        resp = reader_client.chat.completions.create(
            model=reader_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=2048,
        )
        content = resp.choices[0].message.content
        if content and content.strip():
            return content.strip()
        time.sleep(2)
    return "I don't know."


def process_single_chat(chat_dir, chat_size, base_url, reader_client,
                        reader_model, top_k, result_save_dir):
    chat_file = os.path.join(chat_dir, "chat.json")
    pq_file = os.path.join(chat_dir, "probing_questions", "probing_questions.json")

    if not os.path.exists(pq_file):
        print(f"  [skip] no probing_questions.json")
        return

    with open(chat_file) as f:
        chat_data = json.load(f)
    with open(pq_file) as f:
        probing_questions = json.load(f)

    chat_name = os.path.basename(chat_dir)
    session_id = f"beam_{chat_size}_{chat_name}"

    print("  [clear] clearing memory...")
    result = clear_memory(base_url)
    print(f"  [clear] {result}")

    ingest_chat(base_url, chat_data, session_id)
    wait_search_ready(base_url)

    results = {}
    total_q = sum(len(v) for v in probing_questions.values())
    q_idx = 0
    start = time.time()

    for q_type, questions in probing_questions.items():
        type_results = []
        for question in questions:
            q_idx += 1
            q_text = question["question"]
            print(f"  [{q_idx}/{total_q}] {q_type}: {q_text[:60]}...")
            try:
                answer = answer_question(
                    base_url, reader_client, reader_model, q_text, top_k
                )
                print(f"    -> {answer[:80]}")
            except Exception as e:
                answer = f"Error: {e}"
                print(f"    [error] {e}")
            obj = dict(question)
            obj["llm_response"] = answer
            type_results.append(obj)
        results[q_type] = type_results

    os.makedirs(result_save_dir, exist_ok=True)
    output_file = os.path.join(result_save_dir, "lycheemem-results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    elapsed = int(time.time() - start)
    print(f"  [done] {total_q} questions in {elapsed}s -> {output_file}")


def main():
    parser = argparse.ArgumentParser(description="BEAM x LycheeMem evaluation")
    parser.add_argument("--input_directory", required=True)
    parser.add_argument("--chat_size", required=True)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, required=True)
    parser.add_argument("--lycheemem_url", default="http://localhost:8000")
    parser.add_argument("--reader_url", required=True)
    parser.add_argument("--reader_model", required=True)
    parser.add_argument("--reader_api_key", default="dummy")
    parser.add_argument("--top_k", type=int, default=15)
    args = parser.parse_args()

    reader_client = OpenAI(api_key=args.reader_api_key, base_url=args.reader_url)

    input_dir = args.input_directory
    dirs = sorted(
        [d for d in os.listdir(input_dir)
         if os.path.isdir(os.path.join(input_dir, d)) and d.isdigit()],
        key=lambda x: int(x),
    )

    total_start = time.time()
    for idx in range(args.start_index, min(args.end_index, len(dirs))):
        dirname = dirs[idx]
        chat_dir = os.path.join(input_dir, dirname)
        result_save_dir = os.path.join("results", args.chat_size, dirname)

        print(f"\n{'='*60}")
        print(f"Chat {dirname} ({idx+1-args.start_index}/{args.end_index-args.start_index})")
        print(f"{'='*60}")

        try:
            process_single_chat(
                chat_dir, args.chat_size, args.lycheemem_url,
                reader_client, args.reader_model, args.top_k, result_save_dir,
            )
        except Exception as e:
            print(f"  [FATAL] {dirname}: {e}")
            traceback.print_exc()

    elapsed = int(time.time() - total_start)
    print(f"\nTotal: {elapsed//3600:02}:{(elapsed%3600)//60:02}:{elapsed%60:02}")


if __name__ == "__main__":
    main()
