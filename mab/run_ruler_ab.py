"""Root-cause A/B for weak doc-task scores (per user: weak number -> check
format/params/prompt/口径 before blaming the system).

Ingest ruler_qa1 ctx0 ONCE (small-session consolidate), then answer the same N
questions under several retrieval/query口径 and compare substring_exact_match:
  A) search include_semantic=True  + external reader   (current scan config)
  B) search include_semantic=False + external reader   (raw episodic turns)
  C) /memory/reason endpoint (LycheeMem native QA over memory, no external reader)

This isolates whether ruler 10% is the system's true ability or a config artifact.
"""
import argparse, json, time, requests
from openai import OpenAI
import run_mab_v2 as M
import run_mab_doc as D

READER_SYS = ("You are a helpful AI. Answer the question based ONLY on the following "
              "retrieved context. Do NOT use your own world knowledge.\n\n[Retrieved Context]\n{ctx}\n")


def substr(pred, ans):
    al = ans if isinstance(ans, list) else [ans]
    al = [a for sub in al for a in (sub if isinstance(sub, list) else [sub])]
    return int(any(M.normalize_answer(str(a)) in M.normalize_answer(pred or "") for a in al))


def search_ctx(url, q, top_k, include_semantic):
    r = requests.post(url + "/memory/search",
                      json={"query": q, "top_k": top_k, "include_semantic": include_semantic},
                      timeout=60)
    d = r.json() if r.status_code == 200 else {}
    return d.get("raw_retrieved_context", "") or ""


def reason(url, q):
    r = requests.post(url + "/memory/reason",
                      json={"user_query": q, "append_to_session": False}, timeout=120)
    if r.status_code != 200:
        return f"(reason error {r.status_code})"
    return (r.json().get("response") or "").strip()


def reader(client, model, q, ctx, max_tokens):
    msg = [{"role": "system", "content": READER_SYS.format(ctx=ctx)},
           {"role": "user", "content": q}]
    resp = client.chat.completions.create(model=model, messages=msg, temperature=0.1,
                                          max_tokens=max_tokens,
                                          extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    return (resp.choices[0].message.content or "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Accurate_Retrieval")
    ap.add_argument("--sub_dataset", default="ruler_qa1_197K")
    ap.add_argument("--context_idx", type=int, default=0)
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--lycheemem_url", default="http://localhost:8000")
    ap.add_argument("--llm_url", default="http://10.251.171.6:28043/v1")
    ap.add_argument("--llm_model", default="my-llm-qwen")
    ap.add_argument("--data_dir", default="./data")
    args = ap.parse_args()

    client = OpenAI(base_url=args.llm_url, api_key="dummy")
    items = M.load_data(args.data_dir, args.dataset, args.sub_dataset, args.context_idx + 1)
    it = items[args.context_idx]
    qs = it["questions"][: args.n]
    ans = it["answers"][: args.n]
    qtpl = M.get_query_template(args.sub_dataset)

    # ingest once (small sessions)
    chunks = M.chunk_text_into_sentences(it["context"], 4096)
    print(f"ingest {len(chunks)} chunks (small sessions)...", flush=True)
    t0 = time.time()
    D.parallel_ingest(args.lycheemem_url, f"ab_{args.sub_dataset}", chunks, 4, 6, True,
                      M.get_memorize_template(args.sub_dataset))
    print(f"ingest done {time.time()-t0:.0f}s\n", flush=True)

    modes = {"A_sem_true": "search_sem", "B_sem_false": "search_raw", "C_reason": "reason"}
    score = {k: 0 for k in modes}
    rows = []
    for i, (q, a) in enumerate(zip(qs, ans)):
        fq = qtpl.format(question=q)
        outs = {}
        ctxT = search_ctx(args.lycheemem_url, q, args.top_k, True)
        ctxF = search_ctx(args.lycheemem_url, q, args.top_k, False)
        outs["A_sem_true"] = reader(client, args.llm_model, fq, ctxT, 60)
        outs["B_sem_false"] = reader(client, args.llm_model, fq, ctxF, 60)
        outs["C_reason"] = reason(args.lycheemem_url, q)
        line = {"q": q, "answer": a}
        for k in modes:
            h = substr(outs[k], a)
            score[k] += h
            line[k] = {"hit": h, "pred": outs[k][:80]}
        rows.append(line)
        print(f"Q{i+1} gold={str(a)[:30]} | "
              + " ".join(f"{k}={'H' if substr(outs[k],a) else '.'}" for k in modes)
              + f" | ctxT={len(ctxT)}c ctxF={len(ctxF)}c", flush=True)

    print("\n==== SUBSTRING by口径 (n=%d) ====" % len(qs))
    for k in modes:
        print(f"  {k}: {score[k]}/{len(qs)} = {score[k]/len(qs):.1%}")
    json.dump(rows, open(f"/tmp/ab_{args.sub_dataset}.json", "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
