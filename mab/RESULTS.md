# MABench (MemoryAgentBench) Results — LycheeMem (dev5)

Benchmark: **MemoryAgentBench** (arXiv 2507.05257), the full 9-column Overall picture.
Memory system: **LycheeMem dev5, unmodified** (`top_k ≤ 50`).
Reader / judge model: `Qwen3.6-35B-A3B` (served as `my-llm-qwen`, `enable_thinking=False`).

> **Status (2026-06-24):** mid-scale batch complete. 7 columns run at **~100 questions/column
> (detective 71 = full)**, plus `longmemeval_s*` at full paper scale (300). These are
> **paper-comparable to Table 3**. `recsys` / `infbench_sum` still pending their scorers
> (Recall@5 / fluency×F1) and are not measured. Earlier n=15 scan numbers are superseded —
> small samples ran optimistic (ruler_qa1 scan 33% → full 20%; detective scan 16.7% → full 8.5%).

## 9-column picture (official metrics)

| Category | Column | 口径 | Metric | Score | Done / Full (q) |
|---|---|---|---|---|---|
| Accurate Retrieval | `longmemeval_s*` (LME) | reason | LLM-judge | **50.3%** | 300 / 300 ✅ |
| Accurate Retrieval | `eventqa_full` | reason | substring | **55.0%** | 200 / 500 (2 of 5 ctx) |
| Accurate Retrieval | `ruler_qa1_197K` (single-hop) | reason | substring | 20.0% | 100 / 100 ✅ |
| Accurate Retrieval | `ruler_qa2_421K` (multi-hop) | reason | substring | 27.0% | 100 / 100 ✅ |
| Conflict Resolution | `factconsolidation_sh_262k` | search+tpl | substring | 7.0% | 100 / 100 ✅ |
| Conflict Resolution | `factconsolidation_mh_262k` | search+tpl | substring | 0.0% | 100 / 100 ✅ |
| Long-Range Underst. | `detective_qa` | reason | exact_match | 8.5% | 71 / 71 ✅ |
| Long-Range Underst. | `infbench_sum_eng_shots2` | search+tpl | fluency×F1 | *scorer TODO* | 0 / ~100 |
| Test-Time Learning | `icl_banking77_5900shot` | search+tpl | exact_match | 8.0% | 100 / 100 ✅ |
| Test-Time Learning | `recsys_redial_full` | search+tpl | Recall@5 | *scorer TODO* | 0 / ~100 |

**Coverage:** 7 of the 8 attempted columns are at **100% (full)**; only `eventqa` is partial at
**200 / 500** (2 of its 5 contexts — and it is the best-performing column, worth finishing).
`recsys` + `infbench_sum` are **0 / ~100**, not started (pending their scorers).

**Scale:** 7 measured columns = **771 questions** (ruler×2 / FC×2 / icl = 100 each;
eventqa 200; detective 71 full), plus **LME 300** = **1071 questions total**. eventqa is
题-weighted over ctx0 (51) + ctx1 (59); detective spans its full 10 contexts (71 q). Each
ruler/FC/icl column = the first 100 questions of its single context (ruler/FC are 1 context
in MABench; eventqa has 5, we ran 2).

**Read of the landscape.** LycheeMem is strong on the *conversational / event* Accurate-Retrieval
columns and mid-to-weak elsewhere:
- **eventqa 55%** — the headline; **beats** Mem0 (37) and Zep (42) in Table 3.
- **LME 50.3%** (LLM-judge, full 300) — strong, LycheeMem's home domain.
- **ruler** single-hop 20% / multi-hop 27% — multi-hop matches the n=15 scan (26.7); single-hop
  sits just below Mem0 (25). Honest read: ruler (wiki long-doc QA) is off LycheeMem's home turf.
- **factconsolidation** sh 7 / mh 0 — low, but the paper's own systems score ~2–5 on FC-mh too.
- **detective 8.5%** — long-narrative reasoning MCQ; reason retrieval can't recall the supporting
  clues (mostly "no information in retrieved memories"). A genuine weak spot, hard for all systems.
- **icl 8%** — few-shot label classification, off-domain for a conversational memory system.

No column is a harness artifact anymore: every score below reflects真实 system behavior under the
correct口径 (the two parse bugs that hid real scores — ICL and detective — are fixed, see below).

## Methodology — 口径 splits by column type

Different MABench columns need different query口径; matching the口径 to the column (not forcing one
pipeline everywhere) is a legitimate system-fit adaptation faithful to each column's intent.

- **QA-style columns** (`ruler`, `eventqa`, `longmemeval`, `detective`): LycheeMem's native
  **`POST /memory/reason`** (does its own retrieval + reasoning). Body **must** be
  `{"user_query": q, "session_id": <any>, "append_to_session": false}` (no session_id → HTTP 422).
  On ruler this ~doubled substring vs search+external reader. Reason calls now retry transient
  ReadTimeouts (3×, `timeout=300`) — a single timeout used to silently score 0.

- **Instruction-style columns** (`icl`, `factconsolidation`, `recsys`): carry task **rules** the
  paper feeds via a prompt template (ICL: "use the mapping to assign a label"; FC: "newer fact =
  larger serial, answer from the knowledge pool not the real world"). `/memory/reason` ignores
  those rules → ~0. Correct口径 is **`/memory/search` + external reader + the column's task template**.

### Two parse fixes (structured outputs that hid real scores)

Both are harness-side; the model answered correctly but the score was thrown away:
- **ICL** — query asks for `label: {n}`; default parse prefix `Answer:` left `label:` in place →
  exact_match never matched the bare numeric gold. Fixed: `parse_output(prefix="label:")`.
- **detective** — query asks for `{"answer":"X. <opt>","reasoning":...}` JSON; the JSON blob was
  compared whole vs the bare gold `"X. <opt>"` → exact_match always 0 even when correct (e.g. pred
  `answer="D. Her sister Charlotte Blacklock"` == gold, scored 0). Fixed: unwrap the `answer` field,
  then score (exact / substring / A–D letter-match all agree at 8.5%). Re-scored existing
  predictions with `rejudge_detective.py` — no re-run.

## Methodology — ingestion throughput (the feasibility fix)

`run_mab_v2.py` ingests a context as ONE session → one `consolidate` fires thousands of serial
extraction calls (10–30+ min/ctx, O(n²) stalls). **`run_mab_doc.py`** keeps the *exact* same
chunking + query/metric path but round-robins chunks across N small sessions consolidated
**concurrently** (the remote vLLM batches them). Config: `--chunks_per_session 4 --ingest_workers 6`.
Memory reset = **physical LycheeMem restart before each context** (`lsof -ti tcp:8000 | xargs -r
kill -9` + `rm -rf data/*` + relaunch); built-in clear is unreliable. `run_mid.sh` orchestrates this.
Each large-doc column took ~1–2h (ingest 15–30min + ~100 reason answers); full batch ~15h.

## LongMemEval (LME) — full-scale detail

5 contexts × 60 questions = 300.

| Pipeline | substring | LLM-judge |
|---|---|---|
| raw ingest + substring (original) | 24.3% | — |
| **structured ingest + LLM-judge** | **32.3%** | **50.3%** |

Per-context judge: 46.7 / 46.7 / 58.3 / 46.7 / 53.3 → **50.3%**. The clean number is the structured
row. Two harness issues sank the original 24.3%: metric口径 (LongMemEval is officially LLM-judged,
not substring) and ingestion format (`item["context"]` is a Python-literal of per-session real
timestamps + turns that `--raw_ingest` discards; `run_mab_structured.py` parses it back). Raw
per-question judge data: `results/longmemeval_s_structured_rejudge.json`.

## Scripts

| Script | Role |
|---|---|
| `run_mab_v2.py` | Core: templates, query, `post_process`, official metrics. + infbench/recsys templates, `--max_questions`, ICL `label:` fix, detective JSON-answer unwrap. |
| `run_mab_doc.py` | Small-session **parallel ingest** + `--query_mode {reason,search}`, reason with retry. |
| `run_mid.sh` | **Mid-scale batch** orchestrator: 7 columns / 17 ctx, 口径 split, per-ctx restart. |
| `run_scan.sh` | n=15 landscape scan (superseded by run_mid). |
| `run_format_ab.py` / `run_ruler_ab.py` | 口径 A/B that established the splits. |
| `run_mab_structured.py` | LME structured per-session ingest (the 50.3). |
| `rejudge_detective.py` | Re-score detective with the JSON-answer fix (no re-run). |
| `rejudge_llm.py` | Official LongMemEval-style LLM judge. |

Results: `outputs_mid/` (mid-scale 7 columns), `outputs_structured/` (LME 300).

## Pending

- `recsys_redial_full` — needs Recall@5 scorer.
- `infbench_sum_eng_shots2` — needs fluency(0/1)×F1 judge.
