# MemoryAgentBench 评测

[MemoryAgentBench](https://github.com/HUST-AI-HYZ/MemoryAgentBench) (ICLR 2026) 上评测 LycheeMem 的独立脚本。

## 评测内容

MABench 评测 memory agent 的四项核心能力：

| 能力 | 数据集 | 指标 |
|------|--------|------|
| Accurate Retrieval (AR) | eventqa_full, ruler_qa1, ruler_qa2 | substring_exact_match |
| Conflict Resolution (CR) | fact_mh, fact_sh | substring_exact_match |
| Long-Range Understanding (LRU) | detectiveQA | exact_match |
| Test-Time Learning (TTL) | ICL_banking, ICL_clinic, ICL_nlu, ICL_trec_coarse, ICL_trec_fine | exact_match |

设计理念是「inject once, query multiple times」—— 一段长文本注入记忆后，用多个问题测试召回和理解。

## 与 BEAM 的区别

| | BEAM | MABench |
|--|------|---------|
| 输入 | 多轮对话 (100K-10M tokens) | 长文档切 chunk 后注入 |
| 评测重点 | 对话记忆保持 | 精准检索 + 冲突消解 + 远距理解 |
| 评分 | LLM judge (0/0.5/1) | exact_match / substring_match |
| 规模 | 20 questions/chat | 多 question/context, 多 context/dataset |

## 用法

```bash
pip install -r requirements.txt

python run_mab.py \
    --dataset eventqa_full \
    --max_samples 5 \
    --lycheemem_url http://localhost:8000 \
    --reader_url http://your-llm/v1 \
    --reader_model your-model \
    --top_k 15 \
    --chunk_size 4096
```

数据集自动从 HuggingFace (`ai-hyz/MemoryAgentBench`) 下载。

### 可选数据集

```bash
# Accurate Retrieval
--dataset eventqa_full
--dataset ruler_qa1
--dataset ruler_qa2

# Conflict Resolution
--dataset fact_mh
--dataset fact_sh

# Long-Range Understanding
--dataset detectiveQA

# Test-Time Learning
--dataset ICL_banking
--dataset ICL_clinic
```

### 批量跑所有数据集

```bash
for ds in eventqa_full ruler_qa1 ruler_qa2 fact_mh fact_sh; do
    python run_mab.py --dataset $ds --max_samples 5 \
        --lycheemem_url http://localhost:8000 \
        --reader_url http://your-llm/v1 \
        --reader_model your-model
done
```

## 流程

每个 context 的处理：

1. `/memory/clear` 清空上一个 context 的记忆
2. 将 context 按 `chunk_size` 切分，逐 chunk 调 `/memory/append-turn`
3. `/memory/consolidate` 同步等待编码完成
4. 轮询 `/memory/search` 确认索引就绪
5. 对每个 query，search → reader LLM 生成答案
6. substring_exact_match 评分

## 环境

- LycheeMem: dev5
- Reader: Qwen3.6-35B-A3B (vLLM) 或其他 OpenAI-compatible
- Embedding: bge-m3
