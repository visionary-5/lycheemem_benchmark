#!/bin/bash
# Replay PersonaMem-v2 answer-time experiments over already-ingested runs.
#
# Usage:
#   bash run_personamem_v2_replay_batch.sh START_HISTORY END_HISTORY MAX_Q SIZE TOP_K SEARCH_MODE PORT_BASE WORKERS [BATCH_ID] [MAX_CONTEXT_CHARS] [PROMPT_MODE] [GPU_ID] [MEMORY_POLICY]
#
# END_HISTORY is inclusive. This does not re-ingest histories. It reuses
# lychee_runs/pmv2_h{h}_q{MAX_Q}_official_userfinal_k20 when available, and
# falls back to lychee_runs/pmv2_h{h}_q{MAX_Q}_syssep for early smoke runs.
set -euo pipefail

START_HISTORY=${1:?START_HISTORY is required}
END_HISTORY=${2:?END_HISTORY is required}
MAX_Q=${3:-5}
SIZE=${4:-32k}
TOP_K=${5:-20}
SEARCH_MODE=${6:-query}
PORT_BASE=${7:-8040}
WORKERS=${8:-3}
BATCH_ID=${9:-pmv2_replay_h${START_HISTORY}_h${END_HISTORY}_q${MAX_Q}_${SIZE}_${SEARCH_MODE}_k${TOP_K}}
MAX_CONTEXT_CHARS=${10:-0}
PROMPT_MODE=${11:-qwen_user_final}
GPU_ID=${12:-}
MEMORY_POLICY=${13:-standard}

ROOT=/home/ldf/benchmark_lycheemem/PersonaMemV2
LOG_DIR="$ROOT/batch_logs/$BATCH_ID"
mkdir -p "$LOG_DIR"

source /home/ldf/anaconda3/etc/profile.d/conda.sh
conda activate lycheemem

echo "[replay-batch] id=$BATCH_ID start=$START_HISTORY end=$END_HISTORY max_q=$MAX_Q size=$SIZE top_k=$TOP_K search_mode=$SEARCH_MODE prompt_mode=$PROMPT_MODE memory_policy=$MEMORY_POLICY workers=$WORKERS max_context_chars=$MAX_CONTEXT_CHARS gpu_id=${GPU_ID:-<default>}"
echo "[replay-batch] logs=$LOG_DIR"

db_run_id_for_history() {
  local h=$1
  local primary="pmv2_h${h}_q${MAX_Q}_official_userfinal_k20"
  local fallback="pmv2_h${h}_q${MAX_Q}_syssep"
  if [ -d "$ROOT/lychee_runs/$primary" ]; then
    echo "$primary"
  elif [ -d "$ROOT/lychee_runs/$fallback" ]; then
    echo "$fallback"
  else
    echo ""
  fi
}

run_worker() {
  local worker_id=$1
  local port=$((PORT_BASE + worker_id))
  local h
  for ((h = START_HISTORY + worker_id; h <= END_HISTORY; h += WORKERS)); do
    local db_run_id
    db_run_id=$(db_run_id_for_history "$h")
    if [ -z "$db_run_id" ]; then
      echo "[worker $worker_id] missing DB for h=$h" | tee -a "$LOG_DIR/worker${worker_id}.log"
      continue
    fi

    local run_id="pmv2_h${h}_q${MAX_Q}_official_userfinal_${SEARCH_MODE}_k${TOP_K}"
    if [ "$MAX_CONTEXT_CHARS" != "0" ]; then
      run_id="${run_id}_c${MAX_CONTEXT_CHARS}"
    fi
    if [ "$PROMPT_MODE" != "qwen_user_final" ]; then
      run_id="${run_id}_${PROMPT_MODE}"
    fi
    if [ "$MEMORY_POLICY" != "standard" ]; then
      run_id="${run_id}_mp${MEMORY_POLICY}"
    fi
    local out_dir="$ROOT/outputs/$run_id"
    local log="$LOG_DIR/worker${worker_id}_h${h}.log"

    if [ -f "$out_dir/summary.json" ]; then
      echo "[worker $worker_id] skip h=$h existing $out_dir/summary.json" | tee -a "$LOG_DIR/worker${worker_id}.log"
      continue
    fi

    echo "[worker $worker_id] start h=$h db=$db_run_id run_id=$run_id port=$port" | tee -a "$LOG_DIR/worker${worker_id}.log"
    if bash "$ROOT/run_personamem_v2_replay.sh" "$db_run_id" "$run_id" "$h" "$MAX_Q" "$SIZE" "$port" "$SEARCH_MODE" 1 "$TOP_K" "$MAX_CONTEXT_CHARS" "$PROMPT_MODE" "$GPU_ID" "$MEMORY_POLICY" > "$log" 2>&1; then
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
import subprocess
import sys
from pathlib import Path

root = Path("$ROOT")
start = int("$START_HISTORY")
end = int("$END_HISTORY")
max_q = int("$MAX_Q")
top_k = int("$TOP_K")
search_mode = "$SEARCH_MODE"
batch_id = "$BATCH_ID"
max_context_chars = "$MAX_CONTEXT_CHARS"
prompt_mode = "$PROMPT_MODE"
memory_policy = "$MEMORY_POLICY"
files = []
for h in range(start, end + 1):
    run_id = f"pmv2_h{h}_q{max_q}_official_userfinal_{search_mode}_k{top_k}"
    if max_context_chars != "0":
        run_id = f"{run_id}_c{max_context_chars}"
    if prompt_mode != "qwen_user_final":
        run_id = f"{run_id}_{prompt_mode}"
    if memory_policy != "standard":
        run_id = f"{run_id}_mp{memory_policy}"
    path = root / "outputs" / run_id / "predictions.jsonl"
    if path.exists():
        files.append(str(path))
summary = root / "outputs" / f"{batch_id}_summary.json"
if files:
    subprocess.run(
        [sys.executable, str(root / "summarize_personamem_v2.py"), "--glob", *files, "--json_out", str(summary)],
        check=False,
    )
    print(f"[replay-batch] summary={summary}")
    print(f"[replay-batch] files={len(files)}")
else:
    print("[replay-batch] no prediction files found")
PY

exit "$status"
