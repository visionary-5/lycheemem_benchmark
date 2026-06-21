#!/bin/bash
# Structure-aware BEAM full runner: restart LycheeMem clean before each conversation
# (avoid stale-ANN cross-conv contamination), parse batches + timestamped ingest.
# Usage: bash run_beam_goal_structured.sh 100K 1 20
set -u
source /home/ldf/anaconda3/etc/profile.d/conda.sh
conda activate lycheemem

LM_DIR=/home/ldf/benchmark_lycheemem/BEAM/lycheemem_code
BEAM_DIR=/home/ldf/benchmark_lycheemem/BEAM/beam_repo
SIZE=${1:-100K}
START=${2:-1}
END=${3:-20}

restart_lycheemem() {
  lsof -ti tcp:8000 2>/dev/null | xargs -r kill -9
  sleep 2
  rm -rf "$LM_DIR"/data/*
  cd "$LM_DIR"
  nohup python main.py > /tmp/lycheemem.log 2>&1 &
  for t in $(seq 1 40); do
    if curl -s http://localhost:8000/health 2>/dev/null | grep -q ok; then
      echo "  LycheeMem ready (after ${t} checks)"; return 0
    fi
    sleep 3
  done
  echo "  ERROR: LycheeMem did not become healthy"; tail -20 /tmp/lycheemem.log; return 1
}

for i in $(seq "$START" "$END"); do
  if [ ! -f "$BEAM_DIR/chats/$SIZE/$i/probing_questions/probing_questions.json" ]; then
    echo "================ CONV $i : no probing_questions, skip ================"; continue
  fi
  echo "================ CONV $SIZE/$i : restarting LycheeMem clean ================"
  restart_lycheemem || exit 1
  cd "$BEAM_DIR"
  python -u run_beam_structured.py \
    --input_directory chats \
    --chat_size "$SIZE" \
    --start_index "$i" \
    --end_index "$i" \
    --top_k 50 \
    --result_dir ./results_structured
  echo "================ CONV $i DONE ================"
done
echo "ALL BEAM STRUCTURED CONVS DONE: $SIZE [$START-$END]"
