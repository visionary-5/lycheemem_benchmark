#!/bin/bash
# Correct-method MABench longmemeval_s runner: restart LycheeMem clean before
# each context (avoid stale-ANN), best config, output to outputs_goal.
# Usage: bash run_goal.sh 0          (one context)
#        bash run_goal.sh 1 2 3 4    (rest)
set -u
source /home/ldf/anaconda3/etc/profile.d/conda.sh
conda activate lycheemem

LM_DIR=/home/ldf/benchmark_lycheemem/BEAM/lycheemem_code
MAB_DIR=/home/ldf/benchmark_lycheemem/MABench
SUB="longmemeval_s*"
OUT=./outputs_goal

restart_lycheemem() {
  pkill -9 -f "python main.py" 2>/dev/null
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

for i in "$@"; do
  echo "================ CONTEXT $i : restarting LycheeMem clean ================"
  restart_lycheemem || exit 1
  cd "$MAB_DIR"
  python -u run_mab_v2.py \
    --dataset Accurate_Retrieval \
    --sub_dataset "$SUB" \
    --context_idx "$i" \
    --retrieve_num 50 \
    --temperature 0.1 \
    --output_dir "$OUT" \
    --generation_max_length 100 \
    --raw_ingest
  echo "================ CONTEXT $i DONE ================"
done
echo "ALL REQUESTED CONTEXTS DONE: $@"
