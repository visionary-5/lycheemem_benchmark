"""Structure-aware BEAM ingestion.

BEAM chat.json is a list of batches; every message carries a real `time_anchor`
date (e.g. 'March-15-2024'). The default ingest_chat throws all of it away: it
appends every message into ONE session "beam" with no session_date and consolidates
once. That discards the temporal/session structure -> event_ordering / temporal /
multi_session questions score ~0.

This adapter ingests per-batch as a dated session (session_date = batch time_anchor),
the way LycheeMem is designed for. Answering + result format identical to
run_beam_lycheemem; reader thinking disabled + configurable top_k.
"""
import argparse, ast, json, os, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from openai import OpenAI
import run_beam_lycheemem as B

READER_PROMPT = (
    "You are a helpful assistant answering questions based on retrieved memory context.\n\n"
    "### Retrieved Context ###\n{context}\n\n"
    "### Instructions ###\n"
    "Only provide the answer without any explanations.\n"
    "If the information is not available in the context, say 'I don't know'.\n\n"
    "### Question ###\n{query}"
)


def batch_turns(batch):
    turns = batch["turns"]
    return ast.literal_eval(turns) if isinstance(turns, str) else turns


def batch_date(batch):
    for t in batch_turns(batch):
        for m in (t if isinstance(t, list) else [t]):
            ta = m.get("time_anchor")
            if ta and ta != "None":
                return ta
    ta = batch.get("time_anchor")
    return ta if ta and ta != "None" else None


def consolidate_dated(url, sid, session_date):
    for k in range(4):
        try:
            r = requests.post(url + "/memory/consolidate",
                              json={"session_id": sid, "background": False, "flush_session": True,
                                    "force_ingest": True, "session_date": session_date}, timeout=3600)
            if r.ok:
                return
        except Exception:
            pass
        time.sleep(2 * (k + 1))
    raise RuntimeError(f"consolidate failed for {sid}")


def _ingest_batch(url, bi, batch):
    sid = f"beam_batch{bi:02d}"
    sdate = batch_date(batch)
    n = 0
    for t in batch_turns(batch):
        for m in (t if isinstance(t, list) else [t]):
            c = (m.get("content") or "").strip()
            if c:
                B.append_turn(url, sid, m.get("role", "user"), c)
                n += 1
    consolidate_dated(url, sid, sdate)
    print(f"  [ingest] batch{bi} date={sdate} msgs={n} consolidated", flush=True)


def structured_ingest(url, chat_data, workers=4):
    with ThreadPoolExecutor(max_workers=min(workers, len(chat_data) or 1)) as ex:
        futs = [ex.submit(_ingest_batch, url, bi, b) for bi, b in enumerate(chat_data)]
        for f in as_completed(futs):
            f.result()


def answer(url, client, model, q, top_k):
    ctx = B.search_memory(url, q, top_k=top_k).get("raw_retrieved_context", "") or "(No relevant memory found)"
    prompt = READER_PROMPT.format(context=ctx, query=q)
    resp = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        temperature=0, max_tokens=512,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return (resp.choices[0].message.content or "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_directory", required=True)   # e.g. chats
    ap.add_argument("--chat_size", default="100K")
    ap.add_argument("--start_index", type=int, default=1)
    ap.add_argument("--end_index", type=int, default=1)
    ap.add_argument("--lycheemem_url", default="http://localhost:8000")
    ap.add_argument("--reader_url", default="http://10.251.171.6:28043/v1")
    ap.add_argument("--reader_model", default="my-llm-qwen")
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--result_dir", default="./results_structured")
    args = ap.parse_args()

    client = OpenAI(base_url=args.reader_url, api_key="dummy")

    for idx in range(args.start_index, args.end_index + 1):
        chat_dir = os.path.join(args.input_directory, args.chat_size, str(idx))
        chat_file = os.path.join(chat_dir, "chat.json")
        pq_file = os.path.join(chat_dir, "probing_questions", "probing_questions.json")
        if not os.path.exists(pq_file):
            print(f"[skip] no probing_questions in {chat_dir}"); continue

        chat_data = json.load(open(chat_file))
        probing = json.load(open(pq_file))
        print(f"\n==== chat {args.chat_size}/{idx}: {len(chat_data)} batches, "
              f"{sum(len(v) for v in probing.values())} questions [STRUCTURED] ====")

        B.clear_memory(args.lycheemem_url); time.sleep(2)
        structured_ingest(args.lycheemem_url, chat_data)

        results = {}
        for qtype, qs in probing.items():
            rows = []
            for q in qs:
                try:
                    a = answer(args.lycheemem_url, client, args.reader_model, q["question"], args.top_k)
                except Exception as e:
                    a = f"Error: {e}"
                obj = dict(q); obj["llm_response"] = a
                rows.append(obj)
            results[qtype] = rows
            print(f"  [answer] {qtype}: {len(rows)} done", flush=True)

        save_dir = os.path.join(args.result_dir, args.chat_size, str(idx))
        os.makedirs(save_dir, exist_ok=True)
        json.dump(results, open(os.path.join(save_dir, "lycheemem-results.json"), "w"),
                  indent=4, ensure_ascii=False)
        print(f"  saved -> {save_dir}/lycheemem-results.json")


if __name__ == "__main__":
    main()
