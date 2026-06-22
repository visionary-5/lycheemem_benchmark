# PersonaMem-v2 x LycheeMem

External adapter for running LycheeMem dev5 on PersonaMem-v2 MCQ evaluation.

Scope:

- System under test stays unchanged: no edits to `lycheemem_code`.
- Main metric is official-style MCQ micro accuracy: `correct / total`.
- Default retrieval query is the user query only. `--search_mode query_options`
  is available as an ablation, not the clean headline number.
- For clean runs, use an isolated LycheeMem process with its own DB/vector paths.
- MCQ prompt text, recall suffix, option construction, and answer extraction are
  aligned with the official `inference.py`. The local Qwen endpoint rejects a
  final `system` message, so the current compatibility mode appends the official
  MCQ instruction to the final `user` message. Use the exact official message
  order again when running against an API/model that accepts it.
- Wrapper scripts set `PYTHONHASHSEED=0` because official option shuffling uses
  Python `hash()`.

## Files

| File | Purpose |
| --- | --- |
| `download_personamem_v2.py` | Download benchmark CSV and the needed chat histories from HuggingFace. |
| `run_personamem_v2_lycheemem.py` | Ingest one or more histories, answer MCQ rows, save JSONL results. |
| `run_personamem_v2_reader_baseline.py` | Reader-only baselines: no memory, system persona only, or full history. |
| `summarize_personamem_v2.py` | Summarize prediction JSONL files into accuracy tables. |
| `run_personamem_v2_isolated.sh` | Start an isolated LycheeMem server on a chosen port and run a small slice. |
| `run_personamem_v2_replay.sh` | Reuse an existing isolated DB/vector and rerun answering with new reader/retrieval parameters. |

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
- `outputs/baseline_none_h0_h3_q5_official_userfinal/summary.json`: no-memory
  baseline on first 4 histories x 5 questions: `4/20 = 0.200`.
- `outputs/baseline_system_h0_h3_q5_official_userfinal/summary.json`: system
  persona only: `8/20 = 0.400`.
- `outputs/baseline_full_h0_h3_q5_official_userfinal/summary.json`: full
  history baseline: `9/20 = 0.450`.
