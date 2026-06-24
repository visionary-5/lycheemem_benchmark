#!/bin/bash
# MABench MID-SCALE batch — paper-grade question counts, 口径 split per column.
#
# 7 columns / 17 contexts. For each context: physical LycheeMem restart -> parallel
# small-session ingest (run_mab_doc.py) -> answer under the column's correct口径
# (reason for QA columns, search+template for instruction columns). Serial, ~6-8h.
# Run inside tmux. Reason path now retries transient ReadTimeouts (3x, timeout 300).
#
# Columns & question counts (verified against ./data on 2026-06-23):
#   ruler_qa1_197K   reason  1 ctx  = 100 q   (single-hop)
#   ruler_qa2_421K   reason  1 ctx  = 100 q   (multi-hop)
#   eventqa_full     reason  2 ctx  = 200 q   (ctx 0,1 of 5)
#   factconsol_sh    search  1 ctx  = 100 q
#   factconsol_mh    search  1 ctx  = 100 q
#   icl_banking77    search  1 ctx  = 100 q
#   detective_qa     reason 10 ctx  =  71 q   (full)
# recsys / infbench_sum excluded — need their scorers (Recall@5 / fluency*F1) first.
set -u
source /home/ldf/anaconda3/etc/profile.d/conda.sh
conda activate lycheemem

LM_DIR=/home/ldf/benchmark_lycheemem/BEAM/lycheemem_code
MAB_DIR=/home/ldf/benchmark_lycheemem/MABench
OUT=./outputs_mid

# job: DATASET|SUB|CTX_IDX|GEN_MAX_LEN|QUERY_MODE
JOBS=(
  "Accurate_Retrieval|ruler_qa1_197K|0|60|reason"
  "Accurate_Retrieval|ruler_qa2_421K|0|60|reason"
  "Accurate_Retrieval|eventqa_full|0|80|reason"
  "Accurate_Retrieval|eventqa_full|1|80|reason"
  "Conflict_Resolution|factconsolidation_sh_262k|0|60|search"
  "Conflict_Resolution|factconsolidation_mh_262k|0|60|search"
  "Test_Time_Learning|icl_banking77_5900shot_balance|0|20|search"
)
# detective_qa: full 71 questions spread across its 10 contexts (reason)
for c in $(seq 0 9); do
  JOBS+=("Long_Range_Understanding|detective_qa|$c|256|reason")
done

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

echo "########## MABench MID start: $(date) ##########"
for job in "${JOBS[@]}"; do
  IFS='|' read -r DS SUB IDX GML MODE <<< "$job"
  echo ""
  echo ">>> MID $DS / $SUB ctx=$IDX mode=$MODE  $(date)"
  restart_lycheemem || { echo "!!! restart failed $SUB ctx$IDX"; continue; }
  cd "$MAB_DIR"
  python -u run_mab_doc.py \
    --dataset "$DS" --sub_dataset "$SUB" --context_idx "$IDX" \
    --chunks_per_session 4 --ingest_workers 6 \
    --retrieve_num 50 --temperature 0.1 --generation_max_length "$GML" \
    --query_mode "$MODE" \
    --raw_ingest --output_dir "$OUT" 2>&1 || echo "!!! FAILED $SUB ctx$IDX"
  echo "<<< MID DONE $SUB ctx$IDX  $(date)"
done
echo "########## MABench MID end: $(date) ##########"
