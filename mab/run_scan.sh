#!/bin/bash
# MABench LANDSCAPE SCAN: 1 context + <=15 questions per column, standard consolidate
# mode, small-session doc ingest. Cheap-first ordering so numbers stream in.
# Goal: see per-column real landscape before committing to full Overall.
set -u
source /home/ldf/anaconda3/etc/profile.d/conda.sh
conda activate lycheemem

LM_DIR=/home/ldf/benchmark_lycheemem/BEAM/lycheemem_code
MAB_DIR=/home/ldf/benchmark_lycheemem/MABench
OUT=./outputs_scan

# job: DATASET|SUB|CTX_IDX|GEN_MAX_LEN  (cheap-first by chunk count)
JOBS=(
  "Test_Time_Learning|icl_banking77_5900shot_balance|0|20"
  "Long_Range_Understanding|detective_qa|0|256"
  "Long_Range_Understanding|infbench_sum_eng_shots2|0|2048"
  "Accurate_Retrieval|ruler_qa1_197K|0|60"
  "Conflict_Resolution|factconsolidation_sh_262k|0|60"
  "Conflict_Resolution|factconsolidation_mh_262k|0|60"
  "Accurate_Retrieval|ruler_qa2_421K|0|60"
  "Accurate_Retrieval|eventqa_full|0|80"
  "Test_Time_Learning|recsys_redial_full|0|128"
)

restart_lycheemem() {
  lsof -ti tcp:8000 2>/dev/null | xargs -r kill -9 2>/dev/null
  sleep 2
  rm -rf "$LM_DIR"/data/*
  cd "$LM_DIR"
  nohup python main.py > /tmp/lycheemem.log 2>&1 &
  for t in $(seq 1 50); do
    curl -s http://localhost:8000/health 2>/dev/null | grep -q ok && return 0
    sleep 3
  done
  echo "  ERROR: LycheeMem not healthy"; tail -20 /tmp/lycheemem.log; return 1
}

echo "########## MABench SCAN start: $(date) ##########"
for job in "${JOBS[@]}"; do
  IFS='|' read -r DS SUB IDX GML <<< "$job"
  echo ""
  echo ">>> SCAN $DS / $SUB ctx=$IDX  $(date)"
  restart_lycheemem || { echo "!!! restart failed $SUB"; continue; }
  cd "$MAB_DIR"
  python -u run_mab_doc.py \
    --dataset "$DS" --sub_dataset "$SUB" --context_idx "$IDX" \
    --max_questions 15 --chunks_per_session 4 --ingest_workers 6 \
    --retrieve_num 50 --temperature 0.1 --generation_max_length "$GML" \
    --query_mode reason \
    --raw_ingest --output_dir "$OUT" 2>&1 || echo "!!! FAILED $SUB"
  echo "<<< SCAN DONE $SUB  $(date)"
done
echo "########## MABench SCAN end: $(date) ##########"
