"""Re-score existing detective_qa results with the fixed post_process (unwrap the
MCQ JSON answer field) — no re-run needed, predictions are already stored.

Reports per-ctx and题-weighted overall: exact_match (official), substring, and
mcq_letter_match (A/B/C/D choice match, the most robust MCQ口径).
"""
import json, glob, sys
import run_mab_v2 as M

pat = sys.argv[1] if len(sys.argv) > 1 else "outputs_mid/Long_Range_Understanding/detective_qa_ctx*_doc_results.json"
files = sorted(glob.glob(pat))
tot_n = tot_em = tot_sem = tot_letter = 0
print(f"{len(files)} ctx files")
print("ctx    n     em    sem  letter")
for f in files:
    r = json.load(open(f))
    n = len(r); em = sem = letter = 0
    for d in r:
        metrics, _ = M.post_process(d["output"], d["answer"], "detective_qa")
        em += metrics["exact_match"]
        sem += metrics["substring_exact_match"]
        letter += metrics.get("mcq_letter_match", 0)
    tot_n += n; tot_em += em; tot_sem += sem; tot_letter += letter
    ctx = f.split("_ctx")[1].split("_")[0]
    print(f"ctx{ctx:>2} {n:>3}  {em/n:5.2f}  {sem/n:5.2f}  {letter/n:5.2f}")
print("-" * 34)
print(f"ALL  n={tot_n}  em={tot_em/tot_n:.3f}  sem={tot_sem/tot_n:.3f}  letter={tot_letter/tot_n:.3f}")
print(f"  hits: em={tot_em} sem={tot_sem} letter={tot_letter} / {tot_n}")
