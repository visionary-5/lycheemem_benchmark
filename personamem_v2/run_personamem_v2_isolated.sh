#!/bin/bash
# Start an isolated LycheeMem process and run a PersonaMem-v2 slice.
#
# Usage:
#   bash run_personamem_v2_isolated.sh RUN_ID HISTORY_START MAX_HISTORIES MAX_Q_PER_HISTORY SIZE [PORT] [TOP_K]
#
# This script does not remove the main lycheemem_code/data directory. It writes
# per-run DB/vector/log files under ./lychee_runs/RUN_ID and stops only the
# LycheeMem process it started.
set -euo pipefail

RUN_ID=${1:-small_32k_h0}
HISTORY_START=${2:-0}
MAX_HISTORIES=${3:-1}
MAX_Q=${4:-20}
SIZE=${5:-32k}
PORT=${6:-8010}
TOP_K=${7:-50}

ROOT=/home/ldf/benchmark_lycheemem/PersonaMemV2
LM_DIR=/home/ldf/benchmark_lycheemem/BEAM/lycheemem_code
RUN_DATA="$ROOT/lychee_runs/$RUN_ID"
LOG="$RUN_DATA/server.log"

source /home/ldf/anaconda3/etc/profile.d/conda.sh
conda activate lycheemem

mkdir -p "$RUN_DATA"

if lsof -ti tcp:"$PORT" >/dev/null 2>&1; then
  echo "ERROR: port $PORT is already in use. Choose another port or stop the old process." >&2
  exit 2
fi

cd "$LM_DIR"
API_PORT="$PORT" \
SQLITE_DB_PATH="$RUN_DATA/sessions.db" \
COMPACT_MEMORY_DB_PATH="$RUN_DATA/compact.db" \
COMPACT_VECTOR_DB_PATH="$RUN_DATA/compact_vector" \
EVOLVE_DB_PATH="$RUN_DATA/evolve.db" \
EVOLVE_ENABLED=false \
LYCHEE_STATS_DIR="$RUN_DATA" \
TOKEN_STATS_PATH="$RUN_DATA/token_stats.json" \
EMBEDDING_STATS_PATH="$RUN_DATA/embedding_stats.json" \
LYCHEE_TRACE_PATH="$RUN_DATA/trace.jsonl" \
LYCHEE_TRACE_RUN_ID="$RUN_ID" \
nohup python main.py --port "$PORT" > "$LOG" 2>&1 &
SERVER_PID=$!

cleanup() {
  if kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

for t in $(seq 1 60); do
  if curl -s "http://localhost:$PORT/health" 2>/dev/null | grep -q ok; then
    echo "LycheeMem isolated server ready on port $PORT after $t checks"
    break
  fi
  if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    echo "ERROR: LycheeMem exited during startup" >&2
    tail -80 "$LOG" >&2 || true
    exit 3
  fi
  sleep 3
  if [ "$t" = "60" ]; then
    echo "ERROR: LycheeMem did not become healthy" >&2
    tail -80 "$LOG" >&2 || true
    exit 4
  fi
done

cd "$ROOT"
PYTHONHASHSEED=0 python -u run_personamem_v2_lycheemem.py \
  --data_root "$ROOT/data" \
  --size "$SIZE" \
  --history_start "$HISTORY_START" \
  --max_histories "$MAX_HISTORIES" \
  --max_questions_per_history "$MAX_Q" \
  --lycheemem_url "http://localhost:$PORT" \
  --reader_url "http://10.251.171.6:28043/v1" \
  --reader_model "my-llm-qwen" \
  --top_k "$TOP_K" \
  --turns_per_session 12 \
  --ingest_workers 4 \
  --search_mode query \
  --output_dir "$ROOT/outputs" \
  --run_id "$RUN_ID"
