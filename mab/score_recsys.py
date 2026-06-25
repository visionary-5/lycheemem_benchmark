"""Official MABench recsys_redial Recall@k scorer.

Ported verbatim from MemoryAgentBench utils/eval_other_utils.py
(_process_recsys_dataset + clean_*/extract_*/find_nearest_movie helpers) so the
number is directly comparable to the paper. Recall@5 is the official metric.

Re-scores run_mab_doc recsys predictions (predictions already stored — no re-run).
Needs ./processed_data/Recsys_Redial/entity2id.json (from the HF dataset) and the
`editdistance` package.

Usage:
  python score_recsys.py                # score outputs_mid recsys results
  python score_recsys.py --selftest     # verify the matching logic
  python score_recsys.py "<glob>"       # custom results glob
"""
import os, re, json, sys, glob
from editdistance import eval as edit_distance

ENTITY2ID = "./processed_data/Recsys_Redial/entity2id.json"


def clean_parentheses(text):
    return re.sub(r"\([^()]*\)", "", text)


def normalize_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()


def clean_text_elements(text, remove_parentheses=True, normalize_ws=True, remove_nums=True):
    if remove_parentheses:
        text = re.sub(r"\([^()]*\)", "", text)
    if remove_nums:
        text = re.sub(r"^(?:\d+[\.\)、]?\s*[\-\—\–]?\s*)?", "", text)
    if normalize_ws:
        text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_movie_name(text):
    filename = text.split("/")[-1]
    cleaned = filename.replace("_", " ").replace("-", " ").replace(">", " ")
    return normalize_whitespace(clean_parentheses(cleaned))


def find_nearest_movie(target_name, unique_candidates):
    # edit-distance nearest neighbour (official). unique_candidates precomputed once.
    distances = [edit_distance(target_name.lower(), c.lower()) for c in unique_candidates]
    return unique_candidates[min(range(len(unique_candidates)), key=distances.__getitem__)]


def extract_recommendation_list(text, unique_candidates):
    # Official: split on the first "1.", clean numbering, match each item to nearest candidate.
    try:
        _, recommendation_text = text.split("1.", maxsplit=1)
    except Exception:
        recommendation_text = text.replace(",", "\n")
    raw = [clean_text_elements(item.strip()) for item in recommendation_text.split("\n")]
    return [find_nearest_movie(item, unique_candidates) for item in raw]


def load_candidates():
    name_to_id = json.load(open(ENTITY2ID))
    id_to_name = {eid: extract_movie_name(name) for name, eid in name_to_id.items()}
    unique_candidates = list(set(id_to_name.values()))
    return id_to_name, unique_candidates


def score(results_glob):
    id_to_name, cands = load_candidates()
    print(f"candidates: {len(cands)} unique movie names")
    files = sorted(glob.glob(results_glob))
    if not files:
        print("no result files:", results_glob); return
    n = 0; r1 = r5 = r10 = 0.0
    for f in files:
        data = json.load(open(f))
        for d in data:
            predicted = extract_recommendation_list(d["output"], cands)
            gt_ids = [int(x.strip()) for x in d["answer"]]
            gt_movies = [id_to_name[g] for g in gt_ids if g in id_to_name]
            if not gt_movies:
                continue
            r1 += sum(m in predicted[:1] for m in gt_movies) / len(gt_movies)
            r5 += sum(m in predicted[:5] for m in gt_movies) / len(gt_movies)
            r10 += sum(m in predicted[:10] for m in gt_movies) / len(gt_movies)
            n += 1
    if n == 0:
        print("no scorable questions"); return
    print(f"n={n}  Recall@1={r1/n:.4f}  Recall@5={r5/n:.4f}  Recall@10={r10/n:.4f}")


def selftest():
    id_to_name, cands = load_candidates()
    # 7008 -> "This Is Spinal Tap", 4611 -> "Panic Room"
    print("id 7008 ->", id_to_name.get(7008), "| id 4611 ->", id_to_name.get(4611))
    pred = ("The recommendations are: \n1.This Is Spinal Tap\n2.Some Other Film\n"
            "3.Panic Room\n4.Another Movie\n5.Yet Another\n")
    predicted = extract_recommendation_list(pred, cands)
    print("parsed top5:", predicted[:5])
    for gid in (7008, 4611):
        gm = id_to_name[gid]
        print(f"gold {gid} '{gm}' in top5? {gm in predicted[:5]}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        selftest()
    else:
        g = sys.argv[1] if len(sys.argv) > 1 else \
            "outputs_mid/Test_Time_Learning/recsys_redial_full_ctx*_doc_results.json"
        score(g)
