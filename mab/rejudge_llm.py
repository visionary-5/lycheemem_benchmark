"""Re-score existing MABench longmemeval_s predictions with the official
LongMemEval-style LLM judge (Qwen3.6-35B-A3B / my-llm-qwen), instead of substring.

Reads outputs_final/Accurate_Retrieval/longmemeval_s*_ctx*_results.json (300 rows),
extracts (question, gold, prediction), asks the judge yes/no, aggregates accuracy.
"""
import argparse, glob, json, os, re, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# Official LongMemEval default judge prompt (verbatim).
JUDGE_PROMPT = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
    "If the response is equivalent to the correct answer or contains all the intermediate "
    "steps to get the correct answer, you should also answer yes. If the response only "
    "contains a subset of the information required by the answer, answer no. \n\n"
    "Question: {q}\n\nCorrect Answer: {a}\n\nModel Response: {r}\n\n"
    "Is the model response correct? Answer yes or no only."
)

Q_RE = re.compile(r"Now Answer the Question:\s*(.*?)\s*\n\s*Answer:", re.DOTALL)


def extract_question(query: str) -> str:
    m = Q_RE.search(query or "")
    if m:
        return m.group(1).strip()
    return (query or "").strip()


def gold_str(answer) -> str:
    if isinstance(answer, list):
        return " / ".join(str(a) for a in answer)
    return str(answer)


def judge_one(client, model, question, gold, pred):
    pred = (pred or "").strip()
    if not pred:
        return False, "empty_pred"
    prompt = JUDGE_PROMPT.format(q=question, a=gold, r=pred)
    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=8,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            content = (resp.choices[0].message.content or "").strip().lower()
            if not content:
                time.sleep(1.5)
                continue
            verdict = content.split()[0].strip(".,:!")
            return verdict.startswith("yes"), content
        except Exception as exc:
            time.sleep(2 * (attempt + 1))
            last = str(exc)
    return False, f"err:{last}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="outputs_final/Accurate_Retrieval/longmemeval_s*_ctx*_results.json")
    ap.add_argument("--judge_url", default="http://10.251.171.6:28043/v1")
    ap.add_argument("--judge_model", default="my-llm-qwen")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="outputs_final/rejudge_llm_results.json")
    args = ap.parse_args()

    client = OpenAI(base_url=args.judge_url, api_key="dummy")
    files = sorted(glob.glob(args.glob))
    print(f"Found {len(files)} ctx files")

    tasks = []  # (ctx_name, idx, question, gold, pred, substring_bool)
    for f in files:
        ctx = os.path.basename(f)
        d = json.load(open(f))
        rows = d["data"] if isinstance(d, dict) and "data" in d else d
        for i, r in enumerate(rows):
            tasks.append((
                ctx, i,
                extract_question(r.get("query", "")),
                gold_str(r.get("answer", "")),
                str(r.get("parsed_output") if r.get("parsed_output") is not None else r.get("output", "")),
                bool(r.get("substring_exact_match")),
            ))
    print(f"Total rows: {len(tasks)}")

    results = []

    def run(t):
        ctx, i, q, g, p, sub = t
        ok, raw = judge_one(client, args.judge_model, q, g, p)
        return {"ctx": ctx, "idx": i, "question": q, "gold": g, "pred": p,
                "substring": sub, "judge": ok, "judge_raw": raw}

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(run, t) for t in tasks]
        for fut in as_completed(futs):
            results.append(fut.result())
            done += 1
            if done % 25 == 0:
                print(f"  judged {done}/{len(tasks)}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(results, open(args.out, "w"), ensure_ascii=False, indent=1)

    # Aggregate
    per_ctx = {}
    for r in results:
        c = per_ctx.setdefault(r["ctx"], {"n": 0, "judge": 0, "sub": 0})
        c["n"] += 1
        c["judge"] += int(r["judge"])
        c["sub"] += int(r["substring"])

    print("\n" + "=" * 72)
    print(f"{'ctx':<46} {'n':>4} {'substr':>7} {'JUDGE':>7}")
    for c in sorted(per_ctx):
        v = per_ctx[c]
        print(f"{c:<46} {v['n']:>4} {100*v['sub']/v['n']:>6.1f}% {100*v['judge']/v['n']:>6.1f}%")
    N = len(results)
    sj = sum(int(r["judge"]) for r in results)
    ss = sum(int(r["substring"]) for r in results)
    print("-" * 72)
    print(f"{'OVERALL':<46} {N:>4} {100*ss/N:>6.1f}% {100*sj/N:>6.1f}%")
    print("=" * 72)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
