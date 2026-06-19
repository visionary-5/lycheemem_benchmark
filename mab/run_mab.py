"""
MemoryAgentBench runner for LycheeMem.

Flow per context:
  1. Clear memory
  2. Ingest all chunks via append-turn
  3. Consolidate (sync)
  4. Wait for search readiness
  5. Answer each query using search + reader LLM
  6. Score with substring_exact_match / exact_match

Usage:
  python run_mab.py \
    --mab_repo ~/MemoryAgentBench \
    --dataset eventqa_full \
    --lycheemem_url http://localhost:8000 \
    --reader_url http://10.251.171.6:28043/v1 \
    --reader_model my-llm-qwen \
    --top_k 15
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# LycheeMem API helpers
# ---------------------------------------------------------------------------

def clear_memory(base_url: str):
    for attempt in range(3):
        try:
            r = requests.delete(f"{base_url}/memory/clear", timeout=60)
            if r.status_code in (200, 204):
                return True
        except Exception as e:
            print(f"  clear_memory attempt {attempt+1} failed: {e}")
            time.sleep(5)
    return False


def append_turn(base_url: str, session_id: str, role: str, content: str):
    for attempt in range(3):
        try:
            r = requests.post(f"{base_url}/memory/append-turn",
                              json={"session_id": session_id, "role": role, "content": content},
                              timeout=60)
            r.raise_for_status()
            return True
        except Exception as e:
            if attempt == 2:
                print(f"  append_turn failed: {e}")
            time.sleep(3)
    return False


def consolidate_sync(base_url: str, session_id: str, timeout: int = 5400):
    try:
        r = requests.post(f"{base_url}/memory/consolidate",
                          json={"session_id": session_id, "flush_session": True,
                                "background": False, "force_ingest": True},
                          timeout=timeout)
        if r.status_code == 200:
            data = r.json()
            print(f"  Consolidate OK: turns={data.get('turns_consumed','?')}, facts={data.get('facts_added','?')}")
            return True
        else:
            print(f"  Consolidate error {r.status_code}: {r.text[:200]}")
            return False
    except requests.exceptions.ReadTimeout:
        print("  Consolidate timed out, polling...")
        for i in range(60):
            time.sleep(60)
            try:
                r2 = requests.post(f"{base_url}/memory/search",
                                   json={"query": "test", "top_k": 1, "include_semantic": True},
                                   timeout=30)
                if r2.status_code == 200 and r2.json().get("total", 0) > 0:
                    print(f"  Consolidate complete (poll {i+1})")
                    return True
            except:
                pass
        return False


def wait_search_ready(base_url: str, max_retries: int = 30, interval: int = 20):
    for attempt in range(max_retries):
        try:
            r = requests.post(f"{base_url}/memory/search",
                              json={"query": "general information", "top_k": 1, "include_semantic": True},
                              timeout=30)
            if r.status_code == 200:
                total = r.json().get("total", 0)
                print(f"  Search ready (records={total})")
                return True
            else:
                print(f"  Search returned {r.status_code}, retry {attempt+1}/{max_retries}...")
        except Exception as e:
            print(f"  Search error: {e}, retry {attempt+1}/{max_retries}...")
        time.sleep(interval)
    print("  WARNING: search not ready after retries")
    return False


def search_memory(base_url: str, query: str, top_k: int = 15):
    try:
        r = requests.post(f"{base_url}/memory/search",
                          json={"query": query, "top_k": top_k, "include_semantic": True},
                          timeout=60)
        if r.status_code == 200:
            data = r.json()
            results = data.get("results", [])
            return results
        else:
            print(f"  search error {r.status_code}")
            return []
    except Exception as e:
        print(f"  search exception: {e}")
        return []


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

ANSWER_PROMPT = """You are a helpful assistant with access to a memory system.
Based on the retrieved memory context below, answer the user's question.

Rules:
1. If memory contains the answer, state it directly and concisely.
2. If there are UPDATES or CONTRADICTIONS, use the LATEST information.
3. For temporal questions, pay attention to dates and ordering in context.
4. For specific values (numbers, names, dates), quote them exactly.
5. Keep answers SHORT — just the essential information.
6. If memory does not contain relevant information, say "I don't know."
7. Do NOT add speculation or information not in the context.

Retrieved Memory Context:
{context}

Question: {question}
Answer:"""


def answer_question(reader_client: OpenAI, reader_model: str,
                    base_url: str, question: str, top_k: int = 15) -> dict:
    start_time = time.time()
    results = search_memory(base_url, question, top_k)
    memory_time = time.time() - start_time

    if results:
        context_parts = []
        for i, r in enumerate(results, 1):
            text = r.get("content", r.get("text", str(r)))
            context_parts.append(f"[{i}] {text}")
        context = "\n".join(context_parts)
    else:
        context = "(No relevant memories found)"

    prompt = ANSWER_PROMPT.format(context=context, question=question)

    try:
        response = reader_client.chat.completions.create(
            model=reader_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200,
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        answer = f"Error: {e}"

    query_time = time.time() - start_time - memory_time

    return {
        "output": answer,
        "input_len": len(prompt),
        "output_len": len(answer),
        "memory_construction_time": memory_time,
        "query_time_len": query_time,
    }


# ---------------------------------------------------------------------------
# Dataset loading (standalone, doesn't import MABench code)
# ---------------------------------------------------------------------------

def load_mab_dataset(mab_repo: str, sub_dataset: str, max_samples: int = 0):
    """Load dataset from HuggingFace cache or local MABench repo."""
    try:
        from datasets import load_dataset
        ds = load_dataset("ai-hyz/MemoryAgentBench", split="data")
        items = [item for item in ds if item.get("sub_dataset") == sub_dataset
                 or sub_dataset in str(item.get("source", ""))]
        if not items:
            items = list(ds)
        print(f"  Loaded {len(items)} items from HuggingFace")
    except Exception as e:
        print(f"  HuggingFace load failed ({e}), trying local...")
        data_path = os.path.join(mab_repo, "data", f"{sub_dataset}.json")
        if os.path.exists(data_path):
            with open(data_path) as f:
                items = json.load(f)
            if isinstance(items, dict) and "data" in items:
                items = items["data"]
            print(f"  Loaded {len(items)} items from local file")
        else:
            print(f"  ERROR: cannot find dataset at {data_path}")
            sys.exit(1)

    if max_samples > 0:
        items = items[:max_samples]
    return items


def chunk_text(text: str, chunk_size: int = 4096) -> list:
    """Split text into chunks by sentences, respecting chunk_size (chars)."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) + 1 > chunk_size and current:
            chunks.append(current.strip())
            current = sent
        else:
            current = current + " " + sent if current else sent
    if current.strip():
        chunks.append(current.strip())
    return chunks


# ---------------------------------------------------------------------------
# Evaluation metrics (matching MABench's approach)
# ---------------------------------------------------------------------------

def substring_exact_match(prediction: str, ground_truth) -> float:
    """Check if ground truth appears as substring in prediction (case-insensitive)."""
    if isinstance(ground_truth, list):
        return max(substring_exact_match(prediction, gt) for gt in ground_truth)
    pred_lower = prediction.lower().strip()
    gt_lower = str(ground_truth).lower().strip()
    return 1.0 if gt_lower in pred_lower else 0.0


def exact_match(prediction: str, ground_truth) -> float:
    """Strict exact match after normalization."""
    if isinstance(ground_truth, list):
        return max(exact_match(prediction, gt) for gt in ground_truth)
    pred_clean = prediction.lower().strip().rstrip(".")
    gt_clean = str(ground_truth).lower().strip().rstrip(".")
    return 1.0 if pred_clean == gt_clean else 0.0


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

def process_single_context(context_text: str, queries: list, answers: list,
                           context_idx: int, args, reader_client: OpenAI) -> list:
    """Process one context: ingest chunks → consolidate → answer queries."""
    session_id = f"mab_{args.dataset}_{context_idx}"

    print(f"\n{'='*60}")
    print(f"Context {context_idx}: {len(context_text)} chars, {len(queries)} queries")
    print(f"{'='*60}")

    # Step 1: Clear
    print("  [1/4] Clearing memory...")
    clear_memory(args.lycheemem_url)
    time.sleep(2)

    # Step 2: Ingest chunks as conversation turns
    chunks = chunk_text(context_text, args.chunk_size)
    print(f"  [2/4] Ingesting {len(chunks)} chunks...")
    for i, chunk in enumerate(chunks):
        role = "user" if i % 2 == 0 else "assistant"
        append_turn(args.lycheemem_url, session_id, role, chunk)
        if (i + 1) % 50 == 0:
            print(f"    ... {i+1}/{len(chunks)} chunks appended")

    # Step 3: Consolidate
    print("  [3/4] Consolidating...")
    t0 = time.time()
    consolidate_sync(args.lycheemem_url, session_id)
    print(f"    Consolidate took {time.time()-t0:.0f}s")

    # Step 4: Wait for search
    wait_search_ready(args.lycheemem_url)

    # Step 5: Answer queries
    print(f"  [4/4] Answering {len(queries)} queries...")
    results = []
    for qi, (query, answer) in enumerate(zip(queries, answers)):
        output = answer_question(reader_client, args.reader_model,
                                 args.lycheemem_url, query, args.top_k)
        prediction = output["output"]

        score = substring_exact_match(prediction, answer)
        results.append({
            "context_idx": context_idx,
            "query_idx": qi,
            "query": query,
            "answer": answer,
            "prediction": prediction,
            "score": score,
            "memory_time": output["memory_construction_time"],
            "query_time": output["query_time_len"],
        })
        status = "OK" if score > 0 else "MISS"
        print(f"    Q{qi+1}: {status} | pred={prediction[:80]}")

    return results


def main():
    parser = argparse.ArgumentParser(description="MemoryAgentBench runner for LycheeMem")
    parser.add_argument("--mab_repo", type=str, default="~/MemoryAgentBench",
                        help="Path to cloned MemoryAgentBench repo (for local data fallback)")
    parser.add_argument("--dataset", type=str, default="eventqa_full",
                        help="Sub-dataset name (eventqa_full, ruler_qa1, fact_mh, etc.)")
    parser.add_argument("--max_samples", type=int, default=5,
                        help="Max contexts to evaluate (0=all)")
    parser.add_argument("--chunk_size", type=int, default=4096,
                        help="Chunk size in characters for splitting context")
    parser.add_argument("--lycheemem_url", type=str, default="http://localhost:8000")
    parser.add_argument("--reader_url", type=str, default="http://localhost:8080/v1")
    parser.add_argument("--reader_model", type=str, default="my-llm-qwen")
    parser.add_argument("--top_k", type=int, default=15)
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    args.mab_repo = os.path.expanduser(args.mab_repo)

    # Init reader LLM
    reader_client = OpenAI(base_url=args.reader_url, api_key="dummy")

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    items = load_mab_dataset(args.mab_repo, args.dataset, args.max_samples)

    # Process each context
    all_results = []
    for idx, item in enumerate(items):
        context = item.get("context", "")
        if not context or len(context) < 100:
            print(f"  Skipping context {idx}: too short ({len(context)} chars)")
            continue

        questions = item.get("questions", [])
        answers_raw = item.get("answers", [])
        if isinstance(questions, str):
            questions = [questions]
        if isinstance(answers_raw, str):
            answers_raw = [answers_raw]

        results = process_single_context(
            context, questions, answers_raw, idx, args, reader_client
        )
        all_results.extend(results)

        # Save incrementally
        os.makedirs(os.path.join(args.output_dir, args.dataset), exist_ok=True)
        out_path = os.path.join(args.output_dir, args.dataset, "lycheemem-results.json")
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Summary
    if all_results:
        avg_score = sum(r["score"] for r in all_results) / len(all_results)
        print(f"\n{'='*60}")
        print(f"FINAL: {args.dataset} — {len(all_results)} queries, accuracy={avg_score:.1%}")
        print(f"{'='*60}")
    else:
        print("No results generated.")


if __name__ == "__main__":
    main()
