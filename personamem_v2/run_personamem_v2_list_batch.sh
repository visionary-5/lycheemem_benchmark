#!/bin/bash
# Run PersonaMem-v2 isolated slices for an explicit list of history indices.
#
# Usage:
#   bash run_personamem_v2_list_batch.sh HISTORY_LIST MAX_Q SIZE TOP_K PORT_BASE WORKERS [BATCH_ID] [MAX_CONTEXT_CHARS] [SEARCH_MODE] [PROMPT_MODE] [GPU_ID] [MEMORY_POLICY] [INGEST_MODE]
#
# HISTORY_LIST is a newline-delimited file of history indices. Existing
# outputs/<run_id>/summary.json files are skipped, so the batch is resumable.
set -euo pipefail

HISTORY_LIST=${1:?HISTORY_LIST is required}
MAX_Q=${2:-5}
SIZE=${3:-32k}
TOP_K=${4:-50}
PORT_BASE=${5:-8110}
WORKERS=${6:-4}
BATCH_ID=${7:-pmv2_list_q${MAX_Q}_${SIZE}_k${TOP_K}}
MAX_CONTEXT_CHARS=${8:-0}
SEARCH_MODE=${9:-query}
PROMPT_MODE=${10:-qwen_user_final}
GPU_ID=${11:-}
MEMORY_POLICY=${12:-standard}
INGEST_MODE=${13:-turns}

ROOT=${PERSONAMEM_V2_ROOT:-/home/ldf/benchmark_lycheemem/PersonaMemV2}
ISOLATED_SCRIPT=${ISOLATED_SCRIPT:-run_personamem_v2_isolated.sh}
LOG_DIR="$ROOT/batch_logs/$BATCH_ID"
mkdir -p "$LOG_DIR"

source /home/ldf/anaconda3/etc/profile.d/conda.sh
conda activate lycheemem

mapfile -t HISTORIES < <(grep -v '^[[:space:]]*$' "$HISTORY_LIST")

echo "[list-batch] id=$BATCH_ID histories=${#HISTORIES[@]} max_q=$MAX_Q size=$SIZE top_k=$TOP_K search_mode=$SEARCH_MODE prompt_mode=$PROMPT_MODE memory_policy=$MEMORY_POLICY ingest_mode=$INGEST_MODE workers=$WORKERS max_context_chars=$MAX_CONTEXT_CHARS gpu_id=${GPU_ID:-<default>}"
echo "[list-batch] list=$HISTORY_LIST"
echo "[list-batch] logs=$LOG_DIR"
echo "[list-batch] isolated_script=$ISOLATED_SCRIPT"

run_id_for_history() {
  local h=$1
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
  echo "$run_id"
}

run_worker() {
  local worker_id=$1
  local port=$((PORT_BASE + worker_id))
  local pos
  local worker_status=0
  for ((pos = worker_id; pos < ${#HISTORIES[@]}; pos += WORKERS)); do
    local h="${HISTORIES[$pos]}"
    local run_id
    run_id=$(run_id_for_history "$h")
    local out_dir="$ROOT/outputs/$run_id"
    local log="$LOG_DIR/worker${worker_id}_h${h}.log"

    if [ -f "$out_dir/summary.json" ]; then
      echo "[worker $worker_id] skip h=$h existing $out_dir/summary.json" | tee -a "$LOG_DIR/worker${worker_id}.log"
      continue
    fi

    echo "[worker $worker_id] start h=$h run_id=$run_id port=$port" | tee -a "$LOG_DIR/worker${worker_id}.log"
    if bash "$ROOT/$ISOLATED_SCRIPT" "$run_id" "$h" 1 "$MAX_Q" "$SIZE" "$port" "$TOP_K" "$MAX_CONTEXT_CHARS" "$SEARCH_MODE" "$PROMPT_MODE" "$GPU_ID" "$MEMORY_POLICY" "$INGEST_MODE" > "$log" 2>&1; then
      echo "[worker $worker_id] done h=$h" | tee -a "$LOG_DIR/worker${worker_id}.log"
    else
      code=$?
      echo "[worker $worker_id] FAILED h=$h code=$code log=$log" | tee -a "$LOG_DIR/worker${worker_id}.log"
      worker_status=1
    fi
  done
  return "$worker_status"
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
import subprocess
import sys
from pathlib import Path

root = Path("$ROOT")
max_q = int("$MAX_Q")
top_k = int("$TOP_K")
batch_id = "$BATCH_ID"
search_mode = "$SEARCH_MODE"
max_context_chars = "$MAX_CONTEXT_CHARS"
prompt_mode = "$PROMPT_MODE"
memory_policy = "$MEMORY_POLICY"
ingest_mode = "$INGEST_MODE"
histories = [line.strip() for line in Path("$HISTORY_LIST").read_text(encoding="utf-8").splitlines() if line.strip()]

def run_id_for_history(h: str) -> str:
    return (
        f"pmv2_h{h}_q{max_q}_official_userfinal_k{top_k}"
        + (f"_{search_mode}" if search_mode != "query" else "")
        + (f"_c{max_context_chars}" if max_context_chars != "0" else "")
        + (f"_{prompt_mode}" if prompt_mode != "qwen_user_final" else "")
        + (f"_mp{memory_policy}" if memory_policy != "standard" else "")
        + (f"_ing{ingest_mode}" if ingest_mode != "turns" else "")
    )

files = [
    str(root / "outputs" / run_id_for_history(h) / "predictions.jsonl")
    for h in histories
    if (root / "outputs" / run_id_for_history(h) / "predictions.jsonl").exists()
]
summary = root / "outputs" / f"{batch_id}_summary.json"
if files:
    subprocess.run(
        [sys.executable, str(root / "summarize_personamem_v2.py"), "--glob", *files, "--json_out", str(summary)],
        check=False,
    )
    print(f"[list-batch] summary={summary}")
    print(f"[list-batch] files={len(files)} expected={len(histories)}")
else:
    print("[list-batch] no prediction files found")
PY

exit "$status"
