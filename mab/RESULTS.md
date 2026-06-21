# MABench (MemoryAgentBench) Results — LycheeMem

Benchmark: **MemoryAgentBench / Accurate_Retrieval / `longmemeval_s*`** (multi-session
conversational memory QA). 5 contexts × 60 questions = **300 questions**.
Reader & judge model: `Qwen3.6-35B-A3B` (served as `my-llm-qwen`). System: LycheeMem, **unmodified** (`top_k ≤ 50`).

## Headline

| Pipeline | substring | LLM-judge |
|---|---|---|
| raw ingest + substring metric (original) | 24.3% | — |
| raw ingest + LLM-judge | — | 36.7%¹ |
| **structured ingest + LLM-judge** | **32.3%** | **50.3%** |

Per-context LLM-judge (structured): 46.7 / 46.7 / 58.3 / 46.7 / 53.3 → **overall 50.3%**.

¹ required raising the API `top_k` cap to 100 (a source hack, since reverted). The
clean, unmodified-system number is the structured row.

## Why the original number was misleadingly low

Two independent issues, both in the **evaluation harness**, not the memory system:

1. **Metric口径** — MABench scores `longmemeval_s` with `substring_exact_match`, but
   LongMemEval is officially graded by an LLM judge (answers take flexible forms, e.g.
   `2023-06-03` vs `June 3rd`, `a pilsner or lager` vs `Pilsner or Lager`). Switching to
   the official LongMemEval-style judge (`rejudge_llm.py`) recovers ~+8–12 points.

2. **Ingestion format (the big one)** — `item["context"]` is **not** a flat blob. It is a
   Python-literal string `['Chat Time: 2022/11/17 ...', [{'role':'user','content':..}, ..], ...]`
   that retains **111 per-session real timestamps + real user/assistant turns**. The default
   `--raw_ingest` throws that away: it splits the whole blob into token chunks with
   wall-clock timestamps. That discards exactly the temporal/turn structure LycheeMem is
   built on, so temporal/ordering/aggregation questions fail.

   `run_mab_structured.py` parses the structure back and ingests it the way LycheeMem is
   designed for: per-session, turn-by-turn, `consolidate(session_date=<real Chat Time>)`.
   Query + scoring are reused from `run_mab_v2.py` unchanged, so the **only** difference vs
   the raw run is ingestion. On ctx0: substring 16.7%→35.0%, judge **21.7%→46.7%**.

This uses only information MABench already provides in the context — it is a correct
ingestion adapter, not extra data.

## Reproduce

```bash
# full structured run (per-context clean restart, 6-way parallel timestamped ingest)
bash run_goal_structured.sh 1 2 3 4        # ctx0 via: run_mab_structured.py --context_idx 0
# LLM-judge all contexts
python rejudge_llm.py --glob "outputs_structured/Accurate_Retrieval/longmemeval_s*_ctx*results.json" \
    --out outputs_structured/rejudge_all.json
```

Raw per-question judge data: `results/longmemeval_s_structured_rejudge.json` (300 rows).

## Scope note

Only the `longmemeval_s` sub-dataset (the conversational-memory one, LycheeMem's target
domain) was run. MABench's other sub-datasets — `ruler` (wiki long-doc QA),
`factconsolidation` (world-knowledge conflict), `infbench_sum` (summarization),
`icl_*` (few-shot classification) — are non-conversational task types off-domain for a
conversational memory system, and are not measured here.
