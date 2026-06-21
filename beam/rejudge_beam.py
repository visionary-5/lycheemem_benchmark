"""Re-judge existing BEAM lycheemem-results.json with a FIXED rubric LLM judge.

The original eval_llm_judge.py did not disable Qwen3.6 thinking mode, so the judge
returned content=None and everything scored 0. This re-judges the already-generated
llm_response values with enable_thinking disabled.
"""
import argparse, glob, json, os, re, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

JUDGE_PROMPT = """You are an evaluation judge. Given a question, a rubric (list of criteria), and an LLM response, score whether the response satisfies each rubric criterion.

### Question
{question}

### Rubric Criteria
{rubric}

### LLM Response
{response}

### Instructions
For each rubric criterion, output YES if the response satisfies it, NO if not.
Then give an overall score as a fraction (e.g., 3/4 means 3 out of 4 criteria met).

Format your response EXACTLY as:
CRITERION_1: YES/NO
CRITERION_2: YES/NO
...
SCORE: X/Y"""

SCORE_RE = re.compile(r"SCORE:\s*(\d+)\s*/\s*(\d+)", re.IGNORECASE)


def judge_one(client, model, question, rubric_list, response):
    response = (response or "").strip()
    if not response or "Error" in response:
        return 0.0, "empty"
    rubric_text = "\n".join(f"{i+1}. {r}" for i, r in enumerate(rubric_list))
    prompt = JUDGE_PROMPT.format(question=question, rubric=rubric_text, response=response)
    last = ""
    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=400,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            content = (resp.choices[0].message.content or "").strip()
            if not content:
                time.sleep(1.5)
                continue
            m = SCORE_RE.search(content)
            if m:
                num, den = int(m.group(1)), int(m.group(2))
                return (num / den if den else 0.0), content[:80]
            yes = len(re.findall(r":\s*YES", content, re.IGNORECASE))
            no = len(re.findall(r":\s*NO", content, re.IGNORECASE))
            if yes + no:
                return yes / (yes + no), content[:80]
            return 0.0, content[:80]
        except Exception as exc:
            last = str(exc)
            time.sleep(2 * (attempt + 1))
    return 0.0, f"err:{last}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="results/100K/[0-9]/lycheemem-results.json")
    ap.add_argument("--judge_url", default="http://10.251.171.6:28043/v1")
    ap.add_argument("--judge_model", default="my-llm-qwen")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="results/100K/rejudge_beam_results.json")
    args = ap.parse_args()

    client = OpenAI(base_url=args.judge_url, api_key="dummy")
    files = sorted(glob.glob(args.glob))
    print(f"Found {len(files)} conversation result files: {[f.split('/')[-2] for f in files]}")

    tasks = []  # (conv, qtype, idx, question, rubric, response)
    for f in files:
        conv = f.split("/")[-2]
        d = json.load(open(f))
        for qtype, rows in d.items():
            if not isinstance(rows, list):
                continue
            for idx, r in enumerate(rows):
                tasks.append((conv, qtype, idx, r.get("question", ""),
                              r.get("rubric", []) or [], r.get("llm_response", "")))
    print(f"Total rows: {len(tasks)}")

    def run(t):
        conv, qtype, idx, q, rub, resp = t
        score, raw = judge_one(client, args.judge_model, q, rub, resp)
        return {"conv": conv, "qtype": qtype, "idx": idx, "score": score, "raw": raw}

    results = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(run, t) for t in tasks]
        for fut in as_completed(futs):
            results.append(fut.result())
            done += 1
            if done % 20 == 0:
                print(f"  judged {done}/{len(tasks)}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(results, open(args.out, "w"), ensure_ascii=False, indent=1)

    by_type = defaultdict(list)
    for r in results:
        by_type[r["qtype"]].append(r["score"])
    print("\n" + "=" * 56)
    print(f"{'qtype':<28} {'n':>4} {'score':>8}")
    for t in sorted(by_type):
        v = by_type[t]
        print(f"{t:<28} {len(v):>4} {100*sum(v)/len(v):>7.1f}%")
    alls = [r["score"] for r in results]
    print("-" * 56)
    print(f"{'OVERALL':<28} {len(alls):>4} {100*sum(alls)/len(alls):>7.1f}%")
    print("=" * 56)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
