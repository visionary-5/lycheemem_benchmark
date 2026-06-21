# BEAM Results — LycheeMem

Benchmark: **BEAM** probing questions, size **100K**, 20 conversations × 20 questions =
**400 questions**, 10 question types. Reader & judge: `Qwen3.6-35B-A3B` (`my-llm-qwen`).
Scoring: rubric LLM-judge (`rejudge_beam.py`). System: LycheeMem, `top_k=50`.

## Headline

| Pipeline | overall |
|---|---|
| raw ingest + LLM-judge | ~20%¹ |
| **structured ingest + LLM-judge** | **28.5%** |

Per-type (structured, 40 q each):

| type | raw | structured |
|---|---|---|
| abstention | 50%² | 85.0% |
| preference_following | 0% | 62.9% |
| information_extraction | 50% | 40.4% |
| multi_session_reasoning | **0%** | **23.7%** |
| instruction_following | 0% | 22.5% |
| event_ordering | **0%** | **12.7%** |
| knowledge_update | 50% | 12.5% |
| temporal_reasoning | **0%** | **12.5%** |
| summarization | 10% | 9.4% |
| contradiction_resolution | 25% | 3.1% |

¹ original full-set rubric judge over conversations 1–7 ≈ 20.3% (and the heuristic
`eval_simple` ≈ 21.7%). ² raw per-type figures are from conversation 1 (2 q/type, noisy).

## Two issues, same as MABench

1. **Metric口径 (already fixed)** — `eval_llm_judge.py` did not disable Qwen3.6 thinking
   mode, so the judge returned `content=None` and scored almost everything 0 (reported
   ~5% overall). With thinking disabled, the real raw number is ~20%, matching the
   `eval_simple` heuristic.

2. **Ingestion format** — `chat.json` is a list of batches and **every message carries a
   real `time_anchor` date** (e.g. `March-15-2024`). The default `ingest_chat` drops all of
   it: every message is appended into one session `"beam"` with no `session_date`, then
   consolidated once. That discards temporal/session structure, so the timestamp-dependent
   categories score 0.

   `run_beam_structured.py` ingests **per batch as a dated session**
   (`session_date = batch time_anchor`). The three 0% categories recover:
   event_ordering 0→12.7%, temporal_reasoning 0→12.5%, multi_session_reasoning 0→23.7%,
   and preference_following 0→62.9%. Single-conversation A/B (conv 1): 18.5% → 29.2%.

The lift is smaller than MABench's (24.3→50.3) because BEAM already preserved real turns —
only the timestamps were being thrown away, so there was less to recover.

## Reproduce

```bash
# full structured run (per-conversation clean restart, parallel timestamped batch ingest)
bash run_beam_goal_structured.sh 100K 1 20
# rubric LLM-judge (thinking disabled) over all conversations
python rejudge_beam.py --glob "results_structured/100K/*/lycheemem-results.json" \
    --out results_structured/rejudge_all.json
```

Raw per-question judge data: `results/beam_100k_structured_rejudge.json` (400 rows).

> Note: `run_beam_structured.py` imports helpers from `run_beam_lycheemem.py`; both live
> here. The upstream BEAM benchmark repo is github.com/mohammadtavakoli78/BEAM.
