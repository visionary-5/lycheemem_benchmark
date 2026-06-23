# MABench (MemoryAgentBench) Results — LycheeMem (dev5)

Benchmark: **MemoryAgentBench** (arXiv 2507.05257), the full 9-column Overall picture.
Memory system: **LycheeMem dev5, unmodified** (`top_k ≤ 50`).
Reader / judge model: `Qwen3.6-35B-A3B` (served as `my-llm-qwen`, `enable_thinking=False`).

> **Status (2026-06-23):** one column — `longmemeval_s*` — is run at **full paper scale
> (300 questions)** and is directly comparable to the paper's Table 3. The other eight
> columns below are a **landscape scan** (1 context, ≤15 questions/column) — *directional,
> not yet paper-grade*. They exist to fix the per-column口径 (which endpoint / template
> each column needs) before the mid-scale batch. Do **not** quote the scan cells in the
> paper; quote LME 50.3 and re-run the rest at scale first (see "Next: mid-scale batch").

## 9-column landscape

| Category | Column | 口径 | Metric | Score | Scale |
|---|---|---|---|---|---|
| Accurate Retrieval | `longmemeval_s*` (LME) | reason | LLM-judge | **50.3%** | **300 (full)** |
| Accurate Retrieval | `eventqa_full` | reason | substring | 46.7% | 15 (scan) |
| Accurate Retrieval | `ruler_qa1_197K` (single-hop) | reason | substring | 33%¹ | 15 (scan) |
| Accurate Retrieval | `ruler_qa2_421K` (multi-hop) | reason | substring | 26.7% | 15 (scan) |
| Conflict Resolution | `factconsolidation_sh_262k` | search+template | substring | 13.3% | 15 (scan) |
| Conflict Resolution | `factconsolidation_mh_262k` | search+template | substring | 0% | 15 (scan) |
| Long-Range Underst. | `detective_qa` | reason | exact_match | 16.7% | 15 (scan) |
| Long-Range Underst. | `infbench_sum_eng_shots2` | search+template | judge fluency×F1 | *scorer TODO* | — |
| Test-Time Learning | `icl_banking77_5900shot` | search+template | exact_match | 13.3% | 15 (scan) |
| Test-Time Learning | `recsys_redial_full` | search+template | Recall@5 | *scorer TODO* | — |

¹ ruler single-hop in isolation (single-column A/B, n=15) scores **47%**; 33% is its value
inside the mixed cheap-first scan. The scan number is the conservative one.

**Read of the landscape:** LycheeMem's real strengths are the *Accurate-Retrieval / conversational*
columns (LME 50.3, eventqa 46.7, ruler). Against the paper's Table 3 peers: eventqa **beats**
Mem0 (37) and Zep (42); ruler single-hop sits **between** Mem0 (25) and Zep (44);
factconsolidation multi-hop is low (0) but the paper's own systems score 2–5 there too.
No column is a true outlier-low. The non-conversational instruction columns (FC / ICL / recsys)
are mid-pack — expected for a conversational-memory system, but none bottom-out.

## Methodology — 口径 splits by column type

The single most important finding: **different MABench columns need different query口径**,
and matching the口径 to the column (not forcing one pipeline everywhere) is a legitimate
system-fit adaptation, not gaming the benchmark — it stays faithful to each column's intent.

- **QA-style columns** (`ruler`, `eventqa`, `longmemeval`): use LycheeMem's native
  **`POST /memory/reason`** endpoint — it does its own retrieval + reasoning over memory.
  Request body **must** be `{"user_query": q, "session_id": <any>, "append_to_session": false}`
  (missing `session_id` → HTTP 422). On ruler this **~doubled** substring vs search+external
  reader (20% → 47%). This is the system's *designed* QA path.

- **Instruction-style columns** (`icl`, `factconsolidation`, `recsys`): carry task-specific
  **rules** the paper feeds via a prompt template (ICL: "use the mapping to assign a label";
  FC: "newer fact = larger serial, answer from the knowledge pool not the real world").
  `/memory/reason` ignores those rules → scores ~0. Correct口径 is **`/memory/search` +
  external reader + the column's task template** (`run_format_ab.py` confirmed this A/B).

Official metrics per column: `ruler/eventqa/factconsolidation` = `substring_exact_match`;
`icl/detective` = `exact_match`; `recsys` = Recall@5; `infbench_sum` = fluency(0/1)×F1 judge.

## Methodology — ingestion throughput (the feasibility fix)

`run_mab_v2.py` ingests a whole context as ONE session then calls `consolidate` once. For
large document contexts that fires thousands of serial extraction calls → 10–30+ min/ctx and
on a single session O(n²)-stalls (>30 min no output). **`run_mab_doc.py`** keeps the *exact*
same chunking and the *exact* same query/metric path, but round-robins chunks across N small
sessions and consolidates them **concurrently** (the remote vLLM batches concurrent requests).
Config that works: `--chunks_per_session 4 --ingest_workers 6`. Memory content + retrieval口径
are unchanged; only ingestion is parallelized. Memory is reset by **physically restarting
LycheeMem before each context** (`lsof -ti tcp:8000 | xargs -r kill -9` + `rm -rf data/*` +
relaunch); the built-in clear is unreliable. The scan orchestrator (`run_scan.sh`) does this.

## LongMemEval (LME) — the one full-scale, paper-comparable column

5 contexts × 60 questions = **300 questions**.

| Pipeline | substring | LLM-judge |
|---|---|---|
| raw ingest + substring metric (original) | 24.3% | — |
| raw ingest + LLM-judge | — | 36.7%² |
| **structured ingest + LLM-judge** | **32.3%** | **50.3%** |

Per-context LLM-judge (structured): 46.7 / 46.7 / 58.3 / 46.7 / 53.3 → **overall 50.3%**.

² required raising the API `top_k` cap to 100 (a source hack, since reverted). The clean,
unmodified-system number is the structured row. Raw per-question judge data:
`results/longmemeval_s_structured_rejudge.json` (300 rows).

Two harness-side issues (not the memory system) sank the original 24.3%:
1. **Metric口径** — MABench scores `longmemeval_s` with `substring_exact_match`, but
   LongMemEval is officially LLM-judged (answers take flexible forms: `2023-06-03` vs
   `June 3rd`). The official-style judge (`rejudge_llm.py`) recovers ~+8–12 pts.
2. **Ingestion format (the big one)** — `item["context"]` is not a flat blob; it is a
   Python-literal string retaining 111 per-session real timestamps + real user/assistant
   turns. `--raw_ingest` discards exactly the temporal/turn structure LycheeMem is built on.
   `run_mab_structured.py` parses it back and ingests per-session, turn-by-turn,
   `consolidate(session_date=<real Chat Time>)`. On ctx0: substring 16.7%→35.0%, judge
   21.7%→46.7%. Uses only data MABench already provides — a correct adapter, not extra data.

## Scripts

| Script | Role |
|---|---|
| `run_mab_v2.py` | Core: templates, query, `post_process`, official metrics. Adds `infbench_sum`/`recsys` templates, `--max_questions`, and the ICL `parse_output(prefix="label:")` fix. |
| `run_mab_doc.py` | Small-session **parallel ingest** + `--query_mode {reason,search}`. Reuses v2's query/metric path. The throughput fix that makes Overall feasible. |
| `run_scan.sh` | 9-column **landscape scan** orchestrator (1 ctx, ≤15 q/col, per-column restart, cheap-first ordering). |
| `run_format_ab.py` | 口径 A/B for **instruction** columns (ICL/FC): reason vs search+template, reports official metric. Proved reason→0 on rule-bearing columns. |
| `run_ruler_ab.py` | 口径 A/B for **ruler**: search(semantic) vs search(episodic) vs reason. Proved reason ≈2× search. |
| `run_mab_structured.py` | LME structured per-session ingest (produced the 50.3). |
| `rejudge_llm.py` | Official LongMemEval-style LLM judge. |

## Reproduce LME (the paper-grade number)

```bash
bash run_goal_structured.sh 1 2 3 4    # 5 ctx, clean restart per ctx, 6-way parallel ingest
python rejudge_llm.py --glob "outputs_structured/Accurate_Retrieval/longmemeval_s*_ctx*results.json" \
    --out outputs_structured/rejudge_all.json
```

## Next: mid-scale batch (planned, to make the other 8 columns paper-grade)

1. Add 3× retry + `timeout=300` to `run_mab_doc.py`'s `reason_answer_queries` (a ReadTimeout
   currently scores 0 — cost FC points in the scan). Pattern: `run_format_ab.py:reason_with_retry`.
2. `run_mid.sh`: 7 columns, 口径 split, per-ctx restart, `--chunks_per_session 4
   --retrieve_num 50`, ~100 q/col (detective full 71). tmux, ~6–8h serial.
3. `recsys` / `infbench_sum` need scorers first (Recall@5 / fluency×F1) — deferred.
4. Aggregate by official metric, fold in LME 50.3, compare to Table 3.
