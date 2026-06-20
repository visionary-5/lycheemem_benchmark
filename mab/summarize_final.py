"""Aggregate per-context longmemeval results into one overall score (300 queries)."""
import json, glob, os

out_dir = "outputs_final/Accurate_Retrieval"
files = sorted(glob.glob(os.path.join(out_dir, "longmemeval_s*_ctx*_results.json")))
print(f"Found {len(files)} per-context result files\n")

all_rows = []
per_ctx = []
for f in files:
    d = json.load(open(f))
    rows = d["data"]
    all_rows.extend(rows)
    def m(key):
        vals = [r[key] for r in rows if key in r]
        return 100 * sum(vals) / len(vals) if vals else 0.0
    per_ctx.append((os.path.basename(f), len(rows), m("substring_exact_match"), m("exact_match"), m("f1")))

print(f"{'file':<42} {'n':>4} {'substr':>7} {'EM':>6} {'F1':>6}")
for name, n, s, e, fl in per_ctx:
    print(f"{name:<42} {n:>4} {s:>6.1f}% {e:>5.1f}% {fl:>5.1f}%")

def overall(key):
    vals = [r[key] for r in all_rows if key in r]
    return 100 * sum(vals) / len(vals) if vals else 0.0

print(f"\n{'='*70}")
print(f"OVERALL longmemeval_s*  ({len(all_rows)} queries)")
print(f"  substring_exact_match: {overall('substring_exact_match'):.1f}%")
print(f"  exact_match:           {overall('exact_match'):.1f}%")
print(f"  f1:                    {overall('f1'):.1f}%")
print(f"  rougeL_f1:             {overall('rougeL_f1'):.1f}%")
print(f"{'='*70}")
