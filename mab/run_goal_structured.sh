#!/bin/bash
# Structure-aware MABench longmemeval_s full runner: restart LycheeMem clean before
# each context (avoid stale-ANN), parse sessions + timestamped ingest, output to outputs_structured.
# Usage: bash run_goal_structured.sh 1 2 3 4
set -u
source /home/ldf/anaconda3/etc/profile.d/conda.sh
conda activate lycheemem

LM_DIR=/home/ldf/benchmark_lycheemem/BEAM/lycheemem_code
MAB_DIR=/home/ldf/benchmark_lycheemem/MABench
OUT=./outputs_structured

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

for i in "$@"; do
  echo "================ CONTEXT $i : restarting LycheeMem clean ================"
  restart_lycheemem || exit 1
  cd "$MAB_DIR"
  python -u run_mab_structured.py \
    --context_idx "$i" \
    --ingest_workers 6 \
    --retrieve_num 50 \
    --temperature 0.1 \
    --output_dir "$OUT"
  echo "================ CONTEXT $i DONE ================"
done
echo "ALL STRUCTURED CONTEXTS DONE: $@"
