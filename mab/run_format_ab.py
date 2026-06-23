"""口径 A/B for instruction-style columns (ICL / FactConsolidation).

These columns carry task-specific RULES that the paper feeds the model in a prompt
template (ICL: 'use the mapping to assign a label'; FC: 'newer fact has larger serial
number, answer from the knowledge pool not the real world'). The /memory/reason
endpoint ignores those rules (it just answers the user's question from memory), so it
scores ~0. The correct口径 for these columns is search + external reader WITH the
task template — exactly what the paper intends, just fed to the reader not swallowed
by reason.

Ingest each column ONCE, then score the SAME questions under:
  REASON  : /memory/reason (with retry; current scan config)
  SEARCH  : /memory/search + reader + dataset-specific query template + post_process
Reports the OFFICIAL metric (ICL=exact_match, FC=substring_exact_match).
"""
import argparse, json, time, requests
from openai import OpenAI
import run_mab_v2 as M
import run_mab_doc as D

OFFICIAL = {"in_context_learning": "exact_match", "factconsolidation": "substring_exact_match"}


def reason_with_retry(url, q, tries=3, timeout=300):
    for k in range(tries):
        try:
            r = requests.post(url + "/memory/reason",
                              json={"user_query": q, "session_id": "ab_probe", "append_to_session": False},
                              timeout=timeout)
            if r.status_code == 200:
                return (r.json().get("response") or "").strip()
        except Exception:
            pass
        time.sleep(3 * (k + 1))
    return "(reason failed)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--sub_dataset", required=True)
    ap.add_argument("--context_idx", type=int, default=0)
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--lycheemem_url", default="http://localhost:8000")
    ap.add_argument("--llm_url", default="http://10.251.171.6:28043/v1")
    ap.add_argument("--llm_model", default="my-llm-qwen")
    ap.add_argument("--data_dir", default="./data")
    args = ap.parse_args()

    key = M.get_dataset_key(args.sub_dataset)
    metric = OFFICIAL.get(key, "substring_exact_match")
    client = OpenAI(base_url=args.llm_url, api_key="dummy")

    items = M.load_data(args.data_dir, args.dataset, args.sub_dataset, args.context_idx + 1)
    it = items[args.context_idx]
    qs = it["questions"][: args.n]
    ans = it["answers"][: args.n]
    qtpl = M.get_query_template(args.sub_dataset)

    chunks = M.chunk_text_into_sentences(it["context"], 4096)
    print(f"[{args.sub_dataset}] ingest {len(chunks)} chunks (small sessions)...", flush=True)
    t0 = time.time()
    D.parallel_ingest(args.lycheemem_url, f"abf_{args.sub_dataset}", chunks, 4, 6, True,
                      M.get_memorize_template(args.sub_dataset))
    print(f"ingest done {time.time()-t0:.0f}s; official metric = {metric}\n", flush=True)

    s_search = s_reason = 0
    for i, (q, a) in enumerate(zip(qs, ans)):
        fq = qtpl.format(question=q)  # task template with the rules
        raw, _ = M.lycheemem_search(args.lycheemem_url, q, args.top_k)
        pred_s = M.generate_answer(client, args.llm_model, fq, raw, temperature=0.1, max_tokens=64)
        pred_r = reason_with_retry(args.lycheemem_url, q)
        ms, _ = M.post_process(pred_s, a, args.sub_dataset)
        mr, _ = M.post_process(pred_r, a, args.sub_dataset)
        hs, hr = int(ms.get(metric, 0)), int(mr.get(metric, 0))
        s_search += hs; s_reason += hr
        print(f"Q{i+1} gold={str(a)[:24]:24s} | SEARCH={'H' if hs else '.'} REASON={'H' if hr else '.'} "
              f"| S:{pred_s[:42]!r} R:{pred_r[:42]!r}", flush=True)

    n = len(qs)
    print(f"\n==== {args.sub_dataset}  official={metric}  n={n} ====")
    print(f"  SEARCH+template : {s_search}/{n} = {s_search/n:.1%}")
    print(f"  REASON          : {s_reason}/{n} = {s_reason/n:.1%}")


if __name__ == "__main__":
    main()
