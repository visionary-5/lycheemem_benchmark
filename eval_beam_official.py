"""
BEAM LLM-Judge Evaluation (aligned with official BEAM evaluation methodology)
Uses the same unified_llm_judge_base_prompt from BEAM paper.

Usage:
  python eval_beam_official.py \
    --results_file results/100K/1_v3b/lycheemem-results.json \
    --pq_file chats/100K/1/probing_questions/probing_questions.json \
    --judge_url http://10.251.171.6:28043/v1 \
    --judge_model my-llm-qwen
"""
import argparse, json, os, time, re
from openai import OpenAI

UNIFIED_JUDGE_PROMPT = """
You are an expert evaluator tasked with judging whether the LLM's response demonstrates compliance with the specified RUBRIC CRITERION.

## EVALUATION INPUTS
- RUBRIC CRITERION (what to check): {rubric_item}
- RESPONSE TO EVALUATE: {llm_response}

## EVALUATION RUBRIC:
The rubric defines a specific requirement, constraint, or expected behavior that the LLM response should demonstrate.

**IMPORTANT**: Pay careful attention to whether the rubric specifies:
- **Positive requirements** (things the response SHOULD include/do)
- **Negative constraints** (things the response SHOULD NOT include/do)

## SEMANTIC TOLERANCE RULES:
Judge by meaning, not exact wording.
- Accept paraphrases and synonyms that preserve intent.
- Case/punctuation/whitespace differences must be ignored.
- Numbers/currencies/dates may appear in equivalent forms. Treat them as equal when numerically equivalent.

## SCORING SCALE:
- **1.0 (Complete Compliance)**: Fully complies with the rubric criterion.
- **0.5 (Partial Compliance)**: Partially complies (element present but minor inaccuracies).
- **0.0 (No Compliance)**: Fails to comply (required element missing or incorrect, or response is non-responsive).

## OUTPUT FORMAT:
Return your evaluation in JSON format:
{{"score": [1.0, 0.5, or 0.0], "reason": "[brief explanation]"}}

NOTE: ONLY output the json object."""


def parse_judge_response(content):
    """Parse judge response to extract score."""
    if not content:
        return 0.0
    content = content.strip()
    if content.startswith("```"):
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        if match:
            content = match.group(1)
    try:
        d = json.loads(content)
        return float(d.get("score", 0.0))
    except (json.JSONDecodeError, ValueError):
        match = re.search(r'"score"\s*:\s*([0-9.]+)', content)
        if match:
            return float(match.group(1))
    return 0.0


def judge_single_criterion(client, model, rubric_item, llm_response):
    """Judge a single rubric item against the response. Returns 0.0-1.0."""
    if not llm_response or "Error" in str(llm_response):
        return 0.0

    prompt = UNIFIED_JUDGE_PROMPT.format(
        rubric_item=rubric_item,
        llm_response=llm_response,
    )

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=300,
            )
            content = resp.choices[0].message.content
            if content:
                return parse_judge_response(content)
            time.sleep(2)
        except Exception as e:
            print(f"    judge error (attempt {attempt+1}): {e}")
            time.sleep(3)
    return 0.0


def evaluate(results_file, pq_file, judge_url, judge_model, judge_api_key="dummy"):
    with open(results_file) as f:
        results = json.load(f)
    with open(pq_file) as f:
        pq_data = json.load(f)

    client = OpenAI(api_key=judge_api_key, base_url=judge_url)

    scores_by_type = {}
    total_score = 0.0
    total_count = 0
    detailed_results = {}

    for q_type, questions in results.items():
        type_scores = []
        type_details = []
        for idx, q in enumerate(questions):
            response = q.get("llm_response", "")
            rubric = []
            if q_type in pq_data and idx < len(pq_data[q_type]):
                rubric = pq_data[q_type][idx].get("rubric", [])
            if not rubric:
                continue

            item_scores = []
            for r_item in rubric:
                score = judge_single_criterion(client, judge_model, r_item, response)
                item_scores.append(score)

            avg_score = sum(item_scores) / len(item_scores) if item_scores else 0.0
            type_scores.append(avg_score)
            type_details.append({
                "question": q.get("question", ""),
                "score": avg_score,
                "rubric_scores": item_scores,
            })
            print(f"  [{q_type}][{idx+1}] score: {avg_score:.0%} ({item_scores})")

        if type_scores:
            avg = sum(type_scores) / len(type_scores)
            scores_by_type[q_type] = avg
            total_score += sum(type_scores)
            total_count += len(type_scores)
            detailed_results[q_type] = type_details

    overall = total_score / total_count if total_count else 0.0

    print("\n" + "=" * 60)
    print("BEAM x LycheeMem - Official-Style Evaluation")
    print("=" * 60)
    print(f"\n{'Category':<30} {'Score'}")
    print("-" * 45)
    for q_type, score in scores_by_type.items():
        bar = "#" * int(score * 10)
        print(f"{q_type:<30} {score:.0%}  [{bar:<10}]")
    print("-" * 45)
    print(f"{'OVERALL':<30} {overall:.0%}")
    print("=" * 60)
    print(f"\nRef: Mem0 on 100K = 64.1%")

    out_file = results_file.replace("lycheemem-results", "eval-official")
    output = {
        "overall": round(overall, 4),
        "by_type": {k: round(v, 4) for k, v in scores_by_type.items()},
        "details": detailed_results,
    }
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", required=True)
    parser.add_argument("--pq_file", required=True)
    parser.add_argument("--judge_url", required=True)
    parser.add_argument("--judge_model", required=True)
    parser.add_argument("--judge_api_key", default="dummy")
    args = parser.parse_args()
    evaluate(args.results_file, args.pq_file, args.judge_url, args.judge_model, args.judge_api_key)
