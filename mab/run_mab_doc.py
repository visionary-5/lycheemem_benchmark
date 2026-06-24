"""Parallel document ingestion for MABench (throughput fix).

run_mab_v2 ingests a whole context as ONE session then calls consolidate once.
For large document contexts (ruler/eventqa/infbench/recsys/icl/detective/FC) the
single consolidate fires thousands of serial extraction calls -> 10-30+ min/ctx,
infeasible for the full Overall.

This adapter keeps the EXACT same chunking (chunk_text_into_sentences, 4096) and the
EXACT same query/metric path (reuses run_mab_v2._answer_queries), but distributes the
chunks round-robin across N sessions and consolidates them CONCURRENTLY (the remote
vLLM batches concurrent requests). Memory content + retrieval口径 unchanged; only the
ingestion is parallelized. Same trick that made LME/BEAM structured runs feasible.

Clean memory is provided by the orchestrator (restart LycheeMem before each context).
"""
import argparse, time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import run_mab_v2 as M


def reason_answer_queries(url, context_idx, questions, answers, sub_dataset, mc_time,
                          timeout=300, tries=3):
    """Answer via LycheeMem's native /memory/reason endpoint (does its own retrieval
    + reasoning over memory) instead of external search+reader. This is the system's
    designed QA口径; on ruler it ~doubled substring vs search+reader.

    Retries transient failures (ReadTimeout / non-200) up to `tries` times with backoff:
    a single ReadTimeout used to score 0, which silently cost points (e.g. FC) in the
    scan. Mirrors run_format_ab.py:reason_with_retry."""
    results = []
    for qi, (question, answer) in enumerate(zip(questions, answers)):
        t0 = time.time()
        prediction = "(reason no-response)"
        for k in range(tries):
            try:
                r = requests.post(url + "/memory/reason",
                                  json={"user_query": question, "session_id": f"q_{context_idx}_{qi}",
                                        "append_to_session": False}, timeout=timeout)
                if r.status_code == 200:
                    prediction = (r.json().get("response") or "").strip()
                    break
                prediction = f"(reason err {r.status_code})"
            except Exception as e:
                prediction = f"(reason exception {type(e).__name__})"
            if k < tries - 1:
                time.sleep(3 * (k + 1))  # 3s, 6s backoff before retry
        metrics, info = M.post_process(prediction, answer, sub_dataset)
        results.append({"output": prediction, "answer": answer, "query": question,
                        "query_id": qi, "context_id": context_idx,
                        "query_time_len": time.time() - t0,
                        "memory_construction_time": mc_time, **metrics, **info})
        status = "HIT" if metrics.get("substring_exact_match", 0) else "MISS"
        print(f"    Q{qi+1}: {status} | sem={metrics.get('substring_exact_match',0):.0f} "
              f"em={metrics.get('exact_match',0):.0f} | pred={prediction[:80]}", flush=True)
    return results


def parallel_ingest(url, base_sid, chunks, chunks_per_session, workers, raw_ingest, memorize_tpl):
    # contiguous chunks -> small sessions (cap intra-session graph linking cost).
    # Small sessions keep the per-consolidate evidence-graph linking ~O(k^2) with
    # small k, which is far cheaper/safer than one giant session over all chunks.
    n_sessions = max(1, (len(chunks) + chunks_per_session - 1) // chunks_per_session)
    buckets = [[] for _ in range(n_sessions)]
    for i, ch in enumerate(chunks):
        buckets[i // chunks_per_session].append((i, ch))

    def ingest_session(si, items):
        sid = f"{base_sid}_s{si:02d}"
        for i, ch in items:
            content = ch if raw_ingest else memorize_tpl.format(
                context=ch, time_stamp=time.strftime("%Y-%m-%d %H:%M:%S"))
            role = "user" if i % 2 == 0 else "assistant"
            M.lycheemem_append_turn(url, sid, role, content)
        M.lycheemem_consolidate(url, sid)
        return sid, len(items)

    done = 0
    with ThreadPoolExecutor(max_workers=min(workers, len(buckets))) as ex:
        futs = [ex.submit(ingest_session, si, items) for si, items in enumerate(buckets)]
        for f in as_completed(futs):
            sid, n = f.result()
            done += 1
            print(f"    [consolidated] {sid}: {n} chunks  ({done}/{len(buckets)} sessions)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--sub_dataset", required=True)
    ap.add_argument("--context_idx", type=int, required=True)
    ap.add_argument("--max_questions", type=int, default=0)
    ap.add_argument("--chunk_size", type=int, default=4096)
    ap.add_argument("--retrieve_num", type=int, default=50)
    ap.add_argument("--chunks_per_session", type=int, default=4)
    ap.add_argument("--ingest_workers", type=int, default=6)
    ap.add_argument("--raw_ingest", action="store_true")
    ap.add_argument("--lycheemem_url", default="http://localhost:8000")
    ap.add_argument("--llm_url", default="http://10.251.171.6:28043/v1")
    ap.add_argument("--llm_model", default="my-llm-qwen")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--generation_max_length", type=int, default=60)
    ap.add_argument("--strong_reader", action="store_true")
    ap.add_argument("--query_mode", choices=["search", "reason"], default="search",
                    help="search = external reader over /memory/search; reason = native /memory/reason")
    ap.add_argument("--output_dir", default="./outputs_doc")
    args = ap.parse_args()

    llm_client = OpenAI(base_url=args.llm_url, api_key="dummy")

    all_items = M.load_data(args.data_dir, args.dataset, args.sub_dataset, args.context_idx + 1)
    item = all_items[args.context_idx]
    context_text = item["context"]
    questions = item.get("questions", [])
    answers = item.get("answers", [])
    if isinstance(questions, str):
        questions = [questions]
    if isinstance(answers, str):
        answers = [answers]
    if args.max_questions > 0:
        questions = questions[: args.max_questions]
        answers = answers[: args.max_questions]

    query_tpl = M.get_query_template(args.sub_dataset)
    memorize_tpl = M.get_memorize_template(args.sub_dataset)
    base_sid = f"mabdoc_{args.sub_dataset}_{args.context_idx}"

    print(f"\n{'='*70}\nContext {args.context_idx}: {len(context_text)} chars, "
          f"{len(questions)} queries [PARALLEL DOC]\n{'='*70}")

    chunks = M.chunk_text_into_sentences(context_text, chunk_size=args.chunk_size)
    nses = max(1, (len(chunks) + args.chunks_per_session - 1) // args.chunks_per_session)
    print(f"  [ingest] {len(chunks)} chunks -> {nses} sessions "
          f"({args.chunks_per_session}/session), {args.ingest_workers} workers")
    t0 = time.time()
    parallel_ingest(args.lycheemem_url, base_sid, chunks, args.chunks_per_session,
                    args.ingest_workers, args.raw_ingest, memorize_tpl)
    mc_time = time.time() - t0
    print(f"  [ingest] done in {mc_time:.0f}s")

    if args.query_mode == "reason":
        print(f"  [answer] via /memory/reason (native QA口径)")
        results = reason_answer_queries(args.lycheemem_url, args.context_idx, questions,
                                        answers, args.sub_dataset, mc_time)
    else:
        results = M._answer_queries(args.context_idx, questions, answers, query_tpl, args,
                                    llm_client, mc_time)

    import os, json
    out_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.sub_dataset}_ctx{args.context_idx}_doc_results.json")
    json.dump(results, open(out_path, "w"), indent=2, ensure_ascii=False)

    # quick aggregate
    keys = ["substring_exact_match", "exact_match", "f1", "eventqa_recall"]
    agg = {k: round(sum(r.get(k, 0) for r in results) / max(1, len(results)), 4)
           for k in keys if any(k in r for r in results)}
    print(f"  [agg ctx{args.context_idx}] n={len(results)} {agg}  -> {out_path}")


if __name__ == "__main__":
    main()
