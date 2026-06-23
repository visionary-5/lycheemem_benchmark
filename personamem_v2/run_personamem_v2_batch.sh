#!/bin/bash
# Run PersonaMem-v2 isolated slices in parallel workers.
#
# Usage:
#   bash run_personamem_v2_batch.sh START_HISTORY END_HISTORY MAX_Q SIZE TOP_K PORT_BASE WORKERS [BATCH_ID] [MAX_CONTEXT_CHARS] [SEARCH_MODE] [PROMPT_MODE] [GPU_ID] [MEMORY_POLICY] [INGEST_MODE]
#
# END_HISTORY is inclusive. Existing outputs/<run_id>/summary.json files are
# skipped so the batch can be resumed safely.
set -euo pipefail

START_HISTORY=${1:?START_HISTORY is required}
END_HISTORY=${2:?END_HISTORY is required}
MAX_Q=${3:-5}
SIZE=${4:-32k}
TOP_K=${5:-20}
PORT_BASE=${6:-8030}
WORKERS=${7:-3}
BATCH_ID=${8:-pmv2_h${START_HISTORY}_h${END_HISTORY}_q${MAX_Q}_${SIZE}_k${TOP_K}}
MAX_CONTEXT_CHARS=${9:-0}
SEARCH_MODE=${10:-query}
PROMPT_MODE=${11:-qwen_user_final}
GPU_ID=${12:-}
MEMORY_POLICY=${13:-standard}
INGEST_MODE=${14:-turns}

ROOT=/home/ldf/benchmark_lycheemem/PersonaMemV2
LOG_DIR="$ROOT/batch_logs/$BATCH_ID"
mkdir -p "$LOG_DIR"

source /home/ldf/anaconda3/etc/profile.d/conda.sh
conda activate lycheemem

echo "[batch] id=$BATCH_ID start=$START_HISTORY end=$END_HISTORY max_q=$MAX_Q size=$SIZE top_k=$TOP_K search_mode=$SEARCH_MODE prompt_mode=$PROMPT_MODE memory_policy=$MEMORY_POLICY ingest_mode=$INGEST_MODE workers=$WORKERS max_context_chars=$MAX_CONTEXT_CHARS gpu_id=${GPU_ID:-<default>}"
echo "[batch] logs=$LOG_DIR"

run_worker() {
  local worker_id=$1
  local port=$((PORT_BASE + worker_id))
  local h
  for ((h = START_HISTORY + worker_id; h <= END_HISTORY; h += WORKERS)); do
    local run_id="pmv2_h${h}_q${MAX_Q}_official_userfinal_k${TOP_K}"
    if [ "$SEARCH_MODE" != "query" ]; then
      run_id="${run_id}_${SEARCH_MODE}"
    fi
    if [ "$MAX_CONTEXT_CHARS" != "0" ]; then
      run_id="${run_id}_c${MAX_CONTEXT_CHARS}"
    fi
    if [ "$PROMPT_MODE" != "qwen_user_final" ]; then
      run_id="${run_id}_${PROMPT_MODE}"
    fi
    if [ "$MEMORY_POLICY" != "standard" ]; then
      run_id="${run_id}_mp${MEMORY_POLICY}"
    fi
    if [ "$INGEST_MODE" != "turns" ]; then
      run_id="${run_id}_ing${INGEST_MODE}"
    fi
    local out_dir="$ROOT/outputs/$run_id"
    local log="$LOG_DIR/worker${worker_id}_h${h}.log"

    if [ -f "$out_dir/summary.json" ]; then
      echo "[worker $worker_id] skip h=$h existing $out_dir/summary.json" | tee -a "$LOG_DIR/worker${worker_id}.log"
      continue
    fi

    echo "[worker $worker_id] start h=$h run_id=$run_id port=$port" | tee -a "$LOG_DIR/worker${worker_id}.log"
    if bash "$ROOT/run_personamem_v2_isolated.sh" "$run_id" "$h" 1 "$MAX_Q" "$SIZE" "$port" "$TOP_K" "$MAX_CONTEXT_CHARS" "$SEARCH_MODE" "$PROMPT_MODE" "$GPU_ID" "$MEMORY_POLICY" "$INGEST_MODE" > "$log" 2>&1; then
      echo "[worker $worker_id] done h=$h" | tee -a "$LOG_DIR/worker${worker_id}.log"
    else
      code=$?
      echo "[worker $worker_id] FAILED h=$h code=$code log=$log" | tee -a "$LOG_DIR/worker${worker_id}.log"
    fi
  done
}

pids=()
for ((w = 0; w < WORKERS; w++)); do
  run_worker "$w" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

python - <<PY
import json
import subprocess
import sys
from pathlib import Path

root = Path("$ROOT")
start = int("$START_HISTORY")
end = int("$END_HISTORY")
max_q = int("$MAX_Q")
top_k = int("$TOP_K")
batch_id = "$BATCH_ID"
search_mode = "$SEARCH_MODE"
max_context_chars = "$MAX_CONTEXT_CHARS"
prompt_mode = "$PROMPT_MODE"
memory_policy = "$MEMORY_POLICY"
ingest_mode = "$INGEST_MODE"
files = [
    str(root / "outputs" / (
        f"pmv2_h{h}_q{max_q}_official_userfinal_k{top_k}"
        + (f"_{search_mode}" if search_mode != "query" else "")
        + (f"_c{max_context_chars}" if max_context_chars != "0" else "")
        + (f"_{prompt_mode}" if prompt_mode != "qwen_user_final" else "")
        + (f"_mp{memory_policy}" if memory_policy != "standard" else "")
        + (f"_ing{ingest_mode}" if ingest_mode != "turns" else "")
    ) / "predictions.jsonl")
    for h in range(start, end + 1)
    if (root / "outputs" / (
        f"pmv2_h{h}_q{max_q}_official_userfinal_k{top_k}"
        + (f"_{search_mode}" if search_mode != "query" else "")
        + (f"_c{max_context_chars}" if max_context_chars != "0" else "")
        + (f"_{prompt_mode}" if prompt_mode != "qwen_user_final" else "")
        + (f"_mp{memory_policy}" if memory_policy != "standard" else "")
        + (f"_ing{ingest_mode}" if ingest_mode != "turns" else "")
    ) / "predictions.jsonl").exists()
]
summary = root / "outputs" / f"{batch_id}_summary.json"
if files:
    subprocess.run(
        [sys.executable, str(root / "summarize_personamem_v2.py"), "--glob", *files, "--json_out", str(summary)],
        check=False,
    )
    print(f"[batch] summary={summary}")
    print(f"[batch] files={len(files)}")
else:
    print("[batch] no prediction files found")
PY

exit "$status"
