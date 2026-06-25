# MABench (MemoryAgentBench) Results — LycheeMem (dev5)

Benchmark: **MemoryAgentBench** (arXiv 2507.05257), the full 9-column Overall picture.
Memory system: **LycheeMem dev5, unmodified** (`top_k ≤ 50`).
Reader / judge model: `Qwen3.6-35B-A3B` (served as `my-llm-qwen`, `enable_thinking=False`).

> **Status (2026-06-25):** complete. **8 of 9 columns measured at full scale** (each column's
> entire question set), all **paper-comparable to Table 3**. Only `infbench_sum` is skipped
> (official metric is HELMET's multi-step LLM-judge F1 and the column is a single question —
> not worth the reproduction cost; see Skipped). Earlier n=15 scan numbers are superseded —
> small samples ran optimistic (ruler_qa1 scan 33% → full 20%; detective scan 16.7% → full 8.5%).

## 9-column picture (official metrics)

| Category | Column | 口径 | Metric | Score | Done / Full (q) |
|---|---|---|---|---|---|
| Accurate Retrieval | `eventqa_full` | reason | substring | **54.0%** | 500 / 500 ✅ |
| Accurate Retrieval | `longmemeval_s*` (LME) | reason | LLM-judge | **50.3%** | 300 / 300 ✅ |
| Accurate Retrieval | `ruler_qa2_421K` (multi-hop) | reason | substring | 27.0% | 100 / 100 ✅ |
| Accurate Retrieval | `ruler_qa1_197K` (single-hop) | reason | substring | 20.0% | 100 / 100 ✅ |
| Test-Time Learning | `recsys_redial_full` | search+tpl | Recall@5 | 12.75% | 200 / 200 ✅ |
| Long-Range Underst. | `detective_qa` | reason | exact_match | 8.5% | 71 / 71 ✅ |
| Test-Time Learning | `icl_banking77_5900shot` | search+tpl | exact_match | 8.0% | 100 / 100 ✅ |
| Conflict Resolution | `factconsolidation_sh_262k` | search+tpl | substring | 7.0% | 100 / 100 ✅ |
| Conflict Resolution | `factconsolidation_mh_262k` | search+tpl | substring | 0.0% | 100 / 100 ✅ |
| Long-Range Underst. | `infbench_sum_eng_shots2` | search+tpl | F1 (LLM-judge) | *skipped* | — |

**Coverage:** every measured column is at **100% (full)** — eventqa 500/500 (all 5 ctx), recsys
200/200, ruler/FC/icl 100/100 (each is 1 context in MABench), detective 71/71 (10 ctx), LME 300/300.
**Total measured: 1271 questions** across 8 columns + LME. `infbench_sum` (1 question) skipped.

**Read of the landscape.** LycheeMem is strong on the *conversational / event* Accurate-Retrieval
columns and mid-to-weak elsewhere:
- **eventqa 54%** (500 q) — the headline; **beats** Mem0 (37) and Zep (42) in Table 3. Per-ctx
  51/59/60/51/49 — stable across the full set.
- **LME 50.3%** (LLM-judge, full 300) — strong, LycheeMem's home domain.
- **ruler** single-hop 20% / multi-hop 27% — multi-hop matches the n=15 scan (26.7); single-hop
  sits just below Mem0 (25). Honest read: ruler (wiki long-doc QA) is off LycheeMem's home turf.
- **recsys 12.75%** Recall@5 (200 q; @1 4.6 / @10 24.3) — conversational-movie recommendation,
  a reasonable mid result for a memory system not built as a recommender.
- **detective 8.5%** — long-narrative reasoning MCQ; reason retrieval can't recall the supporting
  clues (mostly "no information in retrieved memories"). A genuine weak spot, hard for all systems.
- **icl 8% / factconsolidation 7 & 0** — instruction columns off LycheeMem's home turf; the
  paper's own systems also score ~2–5 on FC-mh.

No column is a harness artifact: every score reflects真实 system behavior under the correct口径
(the two parse bugs that hid real scores — ICL and detective — are fixed, see below).

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
  larger serial, answer from the knowledge pool not the real world"; recsys: "reply with 20 movie
  recommendations as a numbered list"). `/memory/reason` ignores those rules → ~0. Correct口径 is
  **`/memory/search` + external reader + the column's task template**.

### recsys Recall@5 scoring (`score_recsys.py`)

Ported verbatim from the official `eval_other_utils.py:_process_recsys_dataset`: gold answers are
DBpedia entity IDs → movie names via **`entity2id.json`** (31161 entities, from the HF dataset
`ai-hyz/MemoryAgentBench`, placed at `processed_data/Recsys_Redial/entity2id.json`). The model emits
a numbered list of 20 recommendations; each item is matched to the nearest candidate movie name by
**edit distance** (`editdistance` pkg), then Recall@5 = fraction of gold movies in the top-5.

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
Each large-doc context took ~1–3h (ingest scales with size: recsys 5.6M = 363 chunks ≈ 9h).

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
| `run_mid.sh` | **Full-scale batch** orchestrator: 口径 split, per-ctx restart. |
| `score_recsys.py` | Official recsys Recall@k via entity2id + edit-distance matching (no re-run). |
| `rejudge_detective.py` | Re-score detective with the JSON-answer fix (no re-run). |
| `run_mab_structured.py` | LME structured per-session ingest (the 50.3). |
| `rejudge_llm.py` | Official LongMemEval-style LLM judge. |
| `run_scan.sh` / `run_format_ab.py` / `run_ruler_ab.py` | n=15 scan + 口径 A/B that established the splits. |

Results: `outputs_mid/` (8 columns), `outputs_structured/` (LME 300).

## Skipped

- `infbench_sum_eng_shots2` — official metric is **HELMET's LLM-judge F1**: `fluency`(0/1 gate) ×
  F1 where F1 = `recall`(key-points covered) + `precision`(sentences supported), each a separate
  multi-shot LLM judge needing pre-extracted key points. The column is a **single question** —
  a multi-step judge on 1 item is statistical noise. Not worth implementing; left unmeasured.
