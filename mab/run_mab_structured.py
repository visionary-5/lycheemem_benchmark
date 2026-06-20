"""Structure-aware MABench ingestion for longmemeval_s*.

MABench's item["context"] is actually a Python-literal string:
  ['Chat Time: 2022/11/17 (Thu) 12:04', [{'role':'user','content':..}, ...], 'Chat Time:..', [...], ...]
i.e. it ALREADY contains the per-session real timestamps + real user/assistant turns.
The default run_mab_v2 --raw_ingest throws that structure away (chunks the whole blob,
wall-clock timestamps). This adapter parses the structure back and ingests it the way
LycheeMem is designed for: per-session, turn-by-turn, consolidate with the real session_date.

Query + scoring reuse run_mab_v2 unchanged, so the ONLY difference vs the raw run is ingestion.
"""
import argparse, ast, json, os, re, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from openai import OpenAI

import run_mab_v2 as M


def parse_sessions(context_str):
    """Return list of (session_date_str, [ {role,content}, ... ]) in order."""
    parsed = ast.literal_eval(context_str)
    sessions, i = [], 0
    while i < len(parsed):
        el = parsed[i]
        if isinstance(el, str) and i + 1 < len(parsed) and isinstance(parsed[i + 1], list):
            date_raw = re.sub(r"^\s*Chat Time:\s*", "", el).strip()
            date_clean = re.sub(r"\s*\([A-Za-z]{3}\)", "", date_raw).strip()  # drop "(Thu)"
            sessions.append((date_clean, parsed[i + 1]))
            i += 2
        else:
            i += 1
    return sessions


def _post(url, path, payload, timeout, tries=4):
    last = ""
    for k in range(tries):
        try:
            r = requests.post(url + path, json=payload, timeout=timeout)
            if r.ok:
                return r
            last = f"{r.status_code}: {r.text[:160]}"
        except Exception as e:
            last = str(e)
        time.sleep(1.5 * (k + 1))
    raise RuntimeError(f"POST {path} failed after {tries}: {last}")


def append_turn(url, sid, role, content):
    role = role if role in ("user", "assistant", "system") else "user"
    content = (content or "").strip()
    if not content:
        return
    _post(url, "/memory/append-turn", {"session_id": sid, "role": role, "content": content[:99000]}, 120)


def consolidate(url, sid, session_date):
    _post(url, "/memory/consolidate",
          {"session_id": sid, "background": False, "flush_session": True,
           "force_ingest": True, "session_date": session_date}, 3600)


def structured_ingest(url, context_idx, sessions, workers=6):
    total_turns = sum(len(t) for _, t in sessions)
    print(f"  [ingest] {len(sessions)} sessions, {total_turns} turns, {workers} parallel workers (timestamped)")
    t0 = time.time()
    done = [0]

    def do_session(si_sess):
        si, (sdate, turns) = si_sess
        sid = f"mabstr_{context_idx}_sess{si:03d}"
        for t in turns:
            append_turn(url, sid, t.get("role", "user"), t.get("content", ""))
        consolidate(url, sid, session_date=sdate)
        return si

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(do_session, (si, s)) for si, s in enumerate(sessions)]
        for f in as_completed(futs):
            f.result()
            done[0] += 1
            if done[0] % 20 == 0:
                print(f"    ... {done[0]}/{len(sessions)} sessions consolidated ({time.time()-t0:.0f}s)", flush=True)
    dt = time.time() - t0
    print(f"  [ingest] done in {dt:.0f}s")
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--dataset", default="Accurate_Retrieval")
    ap.add_argument("--sub_dataset", default="longmemeval_s*")
    ap.add_argument("--context_idx", type=int, default=0)
    ap.add_argument("--lycheemem_url", default="http://localhost:8000")
    ap.add_argument("--llm_url", default="http://10.251.171.6:28043/v1")
    ap.add_argument("--llm_model", default="my-llm-qwen")
    ap.add_argument("--retrieve_num", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--generation_max_length", type=int, default=100)
    ap.add_argument("--strong_reader", action="store_true")
    ap.add_argument("--ingest_workers", type=int, default=6)
    ap.add_argument("--output_dir", default="./outputs_structured")
    args = ap.parse_args()

    llm_client = OpenAI(base_url=args.llm_url, api_key="dummy")
    items = M.load_data(args.data_dir, args.dataset, args.sub_dataset, args.context_idx + 1)
    item = items[args.context_idx]
    questions = item.get("questions", [])
    answers = item.get("answers", [])
    if isinstance(questions, str): questions = [questions]
    if isinstance(answers, str): answers = [answers]

    sessions = parse_sessions(item["context"])
    query_tpl = M.get_query_template(args.sub_dataset)

    print(f"\n{'='*70}\nContext {args.context_idx}: {len(sessions)} sessions, {len(questions)} queries  [STRUCTURED]\n{'='*70}")
    print("  [1/3] Clearing memory..."); M.lycheemem_clear(args.lycheemem_url); time.sleep(2)
    mct = structured_ingest(args.lycheemem_url, args.context_idx, sessions, workers=args.ingest_workers)

    results = M._answer_queries(args.context_idx, questions, answers, query_tpl, args, llm_client, mct)

    out_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.sub_dataset}_ctx{args.context_idx}_lycheemem_results.json")
    json.dump({"data": results}, open(out_path, "w"), ensure_ascii=False, indent=1)

    def avg(k):
        vs = [r[k] for r in results if isinstance(r.get(k), (int, float, bool))]
        return 100 * sum(vs) / len(vs) if vs else 0.0
    print(f"\n{'='*60}\nSTRUCTURED RESULTS ctx{args.context_idx} (n={len(results)})")
    print(f"  substring_exact_match: {avg('substring_exact_match'):.1f}%")
    print(f"  exact_match:           {avg('exact_match'):.1f}%")
    print(f"  f1:                    {avg('f1'):.1f}%")
    print(f"  saved -> {out_path}\n{'='*60}")


if __name__ == "__main__":
    main()
