#!/bin/bash
# Start LycheeMem against an existing isolated run DB/vector and rerun MCQ
# answering without re-ingesting history.
#
# Usage:
#   bash run_personamem_v2_replay.sh DB_RUN_ID OUT_RUN_ID HISTORY_START MAX_Q [SIZE] [PORT] [SEARCH_MODE] [MAX_HISTORIES] [TOP_K] [MAX_CONTEXT_CHARS] [PROMPT_MODE] [GPU_ID] [MEMORY_POLICY]
set -euo pipefail

DB_RUN_ID=${1:?DB_RUN_ID is required}
OUT_RUN_ID=${2:?OUT_RUN_ID is required}
HISTORY_START=${3:?HISTORY_START is required}
MAX_Q=${4:?MAX_Q is required}
SIZE=${5:-32k}
PORT=${6:-8010}
SEARCH_MODE=${7:-query}
MAX_HISTORIES=${8:-1}
TOP_K=${9:-50}
MAX_CONTEXT_CHARS=${10:-0}
PROMPT_MODE=${11:-qwen_user_final}
GPU_ID=${12:-}
MEMORY_POLICY=${13:-standard}

ROOT=/home/ldf/benchmark_lycheemem/PersonaMemV2
LM_DIR=/home/ldf/benchmark_lycheemem/BEAM/lycheemem_code
RUN_DATA="$ROOT/lychee_runs/$DB_RUN_ID"
LOG="$RUN_DATA/server_replay_${OUT_RUN_ID}.log"

source /home/ldf/anaconda3/etc/profile.d/conda.sh
conda activate lycheemem

if [ ! -d "$RUN_DATA" ]; then
  echo "ERROR: run data not found: $RUN_DATA" >&2
  exit 2
fi

if lsof -ti tcp:"$PORT" >/dev/null 2>&1; then
  echo "ERROR: port $PORT is already in use. Choose another port or stop the old process." >&2
  exit 3
fi

cd "$LM_DIR"
if [ -n "$GPU_ID" ]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi
API_PORT="$PORT" \
SQLITE_DB_PATH="$RUN_DATA/sessions.db" \
COMPACT_MEMORY_DB_PATH="$RUN_DATA/compact.db" \
COMPACT_VECTOR_DB_PATH="$RUN_DATA/compact_vector" \
EVOLVE_DB_PATH="$RUN_DATA/evolve.db" \
EVOLVE_ENABLED=false \
LYCHEE_STATS_DIR="$RUN_DATA" \
TOKEN_STATS_PATH="$RUN_DATA/token_stats_${OUT_RUN_ID}.json" \
EMBEDDING_STATS_PATH="$RUN_DATA/embedding_stats_${OUT_RUN_ID}.json" \
COMPACT_STATS_PATH="$RUN_DATA/compact_stats_${OUT_RUN_ID}.json" \
LYCHEE_TRACE_PATH="$RUN_DATA/trace_${OUT_RUN_ID}.jsonl" \
LYCHEE_TRACE_RUN_ID="$OUT_RUN_ID" \
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
    echo "LycheeMem replay server ready on port $PORT after $t checks"
    break
  fi
  if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    echo "ERROR: LycheeMem exited during startup" >&2
    tail -80 "$LOG" >&2 || true
    exit 4
  fi
  sleep 3
  if [ "$t" = "60" ]; then
    echo "ERROR: LycheeMem did not become healthy" >&2
    tail -80 "$LOG" >&2 || true
    exit 5
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
  --max_context_chars "$MAX_CONTEXT_CHARS" \
  --turns_per_session 12 \
  --ingest_workers 4 \
  --search_mode "$SEARCH_MODE" \
  --prompt_mode "$PROMPT_MODE" \
  --memory_policy "$MEMORY_POLICY" \
  --output_dir "$ROOT/outputs" \
  --run_id "$OUT_RUN_ID" \
  --skip_ingest
