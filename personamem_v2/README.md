# PersonaMem-v2 x LycheeMem

External adapter for running LycheeMem dev5 on PersonaMem-v2 MCQ evaluation.

Scope:

- System under test stays unchanged: no edits to `lycheemem_code`.
- Main metric is official-style MCQ micro accuracy: `correct / total`.
- Default retrieval query is the official question text, including the official
  recall suffix. `--search_mode query_raw` uses only the raw user query.
  Option-aware modes are available for visible-input retrieval ablations, but
  should be reported separately from the conservative question-only result.
- For clean runs, use an isolated LycheeMem process with its own DB/vector paths.
- MCQ prompt text, recall suffix, option construction, and answer extraction are
  aligned with the official `inference.py`. The local Qwen endpoint rejects a
  final `system` message, so `--prompt_mode qwen_user_final` appends the
  official MCQ instruction to the final `user` message. Use
  `--prompt_mode official_system_final` when running against an API/model that
  accepts the exact official message order.
- Wrapper scripts set `PYTHONHASHSEED=0` because official option shuffling uses
  Python `hash()`.
- Wrapper scripts do not pin a GPU by default. Pass the optional trailing
  `GPU_ID` argument to set `CUDA_VISIBLE_DEVICES` for the LycheeMem server
  process. The reader model is called through the configured OpenAI-compatible
  endpoint and does not use this server's A100s.

## Files

| File | Purpose |
| --- | --- |
| `download_personamem_v2.py` | Download benchmark CSV and the needed chat histories from HuggingFace. |
| `run_personamem_v2_lycheemem.py` | Ingest one or more histories, answer MCQ rows, save JSONL results. |
| `run_personamem_v2_reader_baseline.py` | Reader-only baselines: no memory, system persona only, or full history. |
| `summarize_personamem_v2.py` | Summarize prediction JSONL files into accuracy tables. |
| `compare_personamem_v2_runs.py` | Compare multiple prediction sets on overlapping `row_index` values. |
| `run_personamem_v2_isolated.sh` | Start an isolated LycheeMem server on a chosen port and run a small slice. |
| `run_personamem_v2_replay.sh` | Reuse an existing isolated DB/vector and rerun answering with new reader/retrieval parameters. |
| `run_personamem_v2_batch.sh` | Run isolated per-history slices in parallel workers. |
| `run_personamem_v2_replay_batch.sh` | Replay already-ingested histories in parallel for parameter sweeps. |

## Server Quick Start

```bash
cd /home/ldf/benchmark_lycheemem/PersonaMemV2
source /home/ldf/anaconda3/etc/profile.d/conda.sh
conda activate lycheemem

python download_personamem_v2.py \
  --data_root ./data \
  --size 32k \
  --max_histories 3

bash run_personamem_v2_isolated.sh small_32k_h0 0 1 20 32k 8010 20
python summarize_personamem_v2.py --glob "outputs/small_32k_h0/*.jsonl"
```

`run_personamem_v2_isolated.sh` does not delete the main LycheeMem `data/`
directory. It starts `lycheemem_code/main.py` with per-run SQLite/vector paths
under `./lychee_runs/<run_id>/`.

Replay an already-ingested history with a different `top_k`:

```bash
bash run_personamem_v2_replay.sh \
  pmv2_h0_q5_official_userfinal_k20 \
  pmv2_h0_q5_replay_k10 \
  0 5 32k 8010 query 1 10
```

The replay wrapper accepts optional trailing parameters:

```bash
bash run_personamem_v2_replay.sh \
  DB_RUN_ID OUT_RUN_ID HISTORY_START MAX_Q SIZE PORT SEARCH_MODE MAX_HISTORIES \
  TOP_K MAX_CONTEXT_CHARS PROMPT_MODE GPU_ID
```

For example, pin a replay worker to GPU 2:

```bash
bash run_personamem_v2_replay.sh \
  pmv2_h0_q5_official_userfinal_k20 \
  pmv2_h0_q5_query_raw_k20_gpu2 \
  0 5 32k 8010 query_raw 1 20 0 qwen_user_final 2
```

Search modes:

- `query`: official question plus recall suffix.
- `query_raw`: raw user query only.
- `query_options`: official question plus visible MCQ options.
- `query_raw_options`: raw user query plus visible MCQ options.
- `query_metadata` / `query_metadata_options`: diagnostic only; these use CSV
  metadata fields and should not be headline results unless the paper protocol
  explicitly includes them.

Replay a 40-history / 200-question parameter check:

```bash
bash run_personamem_v2_replay_batch.sh \
  0 39 5 32k 20 query_raw_options 8040 3 \
  pmv2_replay_h0_h39_q5_query_raw_options_k20 0 qwen_user_final 2
```

The last argument pins each worker's LycheeMem server to GPU 2. Omit it to use
the default visible device set.

Compare two parameter runs on the same rows:

```bash
python compare_personamem_v2_runs.py \
  --run 'query=outputs/pmv2_h*_q5_official_userfinal_k20/predictions.jsonl' \
  --run 'raw_options=outputs/pmv2_h*_q5_official_userfinal_query_raw_options_k20/predictions.jsonl'
```

Reader baselines:

```bash
PYTHONHASHSEED=0 python -u run_personamem_v2_reader_baseline.py \
  --data_root ./data \
  --size 32k \
  --history_start 0 \
  --max_histories 4 \
  --max_questions_per_history 5 \
  --context_mode full_history \
  --run_id baseline_full_h0_h3_q5
```

## Current Smoke Results

Server path: `/home/ldf/benchmark_lycheemem/PersonaMemV2`.

- `outputs/official_userfinal_k20_h0_h9_q5_summary.json`: LycheeMem, 32k,
  first 10 histories x 5 questions, `top_k=20`: `28/50 = 0.560`.
- First 40 histories x 5 questions, 32k text MCQ, same rows, Qwen reader:
  - `query`, `top_k=20`: `93/200 = 0.465`.
  - `query`, `top_k=50`: `98/200 = 0.490`; row-level comparison vs
    `query/top_k=20`: net `+5` questions (`25` wins, `20` losses, `155` ties).
  - `query_raw`, `top_k=20`: `88/200 = 0.440`.
  - `query_raw_options`, `top_k=20`: `86/200 = 0.430`. This looked strong on
    the first 50 questions (`30/50 = 0.600`) but regressed badly at 200, so do
    not use it as a headline config.
- Current best clean screened config is `query/top_k=50`, but the gain is small
  and still below the PersonaMem-v2 agentic-memory paper anchor, so it needs
  another 200-question ablation before scaling to 500+ questions.
- `outputs/baseline_none_h0_h3_q5_official_userfinal/summary.json`: no-memory
  baseline on first 4 histories x 5 questions: `4/20 = 0.200`.
- `outputs/baseline_system_h0_h3_q5_official_userfinal/summary.json`: system
  persona only: `8/20 = 0.400`.
- `outputs/baseline_full_h0_h3_q5_official_userfinal/summary.json`: full
  history baseline: `9/20 = 0.450`.
