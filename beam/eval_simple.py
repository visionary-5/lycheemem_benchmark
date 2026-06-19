"""
BEAM 简化评测 — 检查 rubric 覆盖率
rubric 是 list[str]，每条描述一个要求。检查 LLM 回答是否满足。

用法:
  python eval_simple.py --results_file results/100K/1/lycheemem-results.json \
                        --pq_file chats/100K/1/probing_questions/probing_questions.json
"""
import argparse, json, re


def check_rubric_item(response: str, rubric_item: str) -> float:
    """检查单条 rubric 是否被回答覆盖。返回 0.0 或 1.0"""
    response_lower = response.lower().strip()
    rubric_lower = rubric_item.lower().strip()

    # 1. abstention 类：rubric 说"no information"，回答应该拒答
    if "no information" in rubric_lower or "no relevant" in rubric_lower:
        abstained = any(x in response_lower for x in [
            "i don't know", "i do not know", "not available",
            "no information", "cannot find", "not mentioned",
            "no relevant", "don't have"
        ])
        return 1.0 if abstained else 0.0

    # 2. "should state" / "should mention" 类：提取关键内容检查
    match = re.search(r"should (?:state|mention|include|contain)[:\s]*(.+)", rubric_lower)
    if match:
        key_content = match.group(1).strip().strip("\"'")
        # 检查关键词是否在回答中
        keywords = [w for w in key_content.split() if len(w) > 2]
        if keywords:
            hits = sum(1 for w in keywords if w in response_lower)
            return 1.0 if hits / len(keywords) >= 0.5 else 0.0
        return 1.0 if key_content in response_lower else 0.0

    # 3. 短 rubric（关键词/短语）：直接在回答中查找
    if len(rubric_lower.split()) <= 5:
        # 短的关键词直接匹配
        return 1.0 if rubric_lower in response_lower else 0.0

    # 4. 长描述类 rubric：提取关键词做模糊匹配
    keywords = [w for w in rubric_lower.split() if len(w) > 3]
    if keywords:
        hits = sum(1 for w in keywords if w in response_lower)
        ratio = hits / len(keywords)
        return 1.0 if ratio >= 0.4 else 0.0

    return 0.0


def evaluate(results_file: str, pq_file: str):
    with open(results_file) as f:
        results = json.load(f)
    with open(pq_file) as f:
        pq_data = json.load(f)

    scores_by_type = {}
    total_score = 0.0
    total_count = 0

    for q_type, questions in results.items():
        type_scores = []
        for idx, q in enumerate(questions):
            response = q.get("llm_response", "")
            if not response or "Error" in str(response):
                type_scores.append(0.0)
                continue

            # 获取 rubric
            rubric = []
            if q_type in pq_data and idx < len(pq_data[q_type]):
                rubric = pq_data[q_type][idx].get("rubric", [])

            if not rubric:
                continue

            # 对每条 rubric 评分，取平均
            item_scores = [check_rubric_item(response, r) for r in rubric]
            score = sum(item_scores) / len(item_scores) if item_scores else 0.0
            type_scores.append(score)

        if type_scores:
            avg = sum(type_scores) / len(type_scores)
            scores_by_type[q_type] = {
                "avg_score": avg,
                "count": len(type_scores),
                "scores": type_scores,
            }
            total_score += sum(type_scores)
            total_count += len(type_scores)

    overall = total_score / total_count if total_count else 0.0

    print("\n" + "=" * 60)
    print("BEAM x LycheeMem - Evaluation Results (100K, chat 1)")
    print("=" * 60)
    print(f"\n{'Category':<30} {'Score':<10} {'N'}")
    print("-" * 50)
    for q_type, data in scores_by_type.items():
        bar = "#" * int(data["avg_score"] * 10)
        print(f"{q_type:<30} {data['avg_score']:.0%}  [{bar:<10}] {data['count']}")
    print("-" * 50)
    print(f"{'OVERALL':<30} {overall:.0%}")
    print("=" * 60)
    print(f"\nRef: Mem0 baseline on 100K = 64.1%")

    # 保存
    output = {
        "overall_accuracy": round(overall, 4),
        "by_type": {k: round(v["avg_score"], 4) for k, v in scores_by_type.items()},
    }
    out_file = results_file.replace("lycheemem-results", "evaluation-summary")
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", required=True)
    parser.add_argument("--pq_file", required=True)
    args = parser.parse_args()
    evaluate(args.results_file, args.pq_file)
