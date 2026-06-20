#!/bin/bash
# Run longmemeval_s* one context at a time, restarting LycheeMem clean before each
# to avoid stale-ANN cross-context contamination.
set -u
source /home/ldf/anaconda3/etc/profile.d/conda.sh
conda activate lycheemem

LM_DIR=/home/ldf/benchmark_lycheemem/BEAM/lycheemem_code
MAB_DIR=/home/ldf/benchmark_lycheemem/MABench
SUB="longmemeval_s*"

restart_lycheemem() {
  pkill -9 -f "python main.py" 2>/dev/null
  sleep 2
  rm -rf "$LM_DIR"/data/*
  cd "$LM_DIR"
  nohup python main.py > /tmp/lycheemem.log 2>&1 &
  for t in $(seq 1 30); do
    if curl -s http://localhost:8000/health 2>/dev/null | grep -q ok; then
      echo "  LycheeMem ready (after ${t} checks)"; return 0
    fi
    sleep 3
  done
  echo "  ERROR: LycheeMem did not become healthy"; return 1
}

for i in 0 1 2 3 4; do
  echo "================ CONTEXT $i : restarting LycheeMem clean ================"
  restart_lycheemem || exit 1
  cd "$MAB_DIR"
  python -u run_mab_v2.py \
    --dataset Accurate_Retrieval \
    --sub_dataset "$SUB" \
    --context_idx "$i" \
    --retrieve_num 100 \
    --temperature 0.1 \
    --output_dir ./outputs_final \
    --generation_max_length 100 \
    --raw_ingest
  echo "================ CONTEXT $i DONE ================"
done

echo "ALL CONTEXTS DONE"
