# BEAM Benchmark 评测

BEAM (Benchmark for Evaluating AI Memory) 上评测 LycheeMem dev5 的完整流程。

## 文件

| 文件 | 用途 |
|------|------|
| `run_beam.py` | 主脚本：清空记忆 → ingest chat → consolidate → 验证搜索就绪 → 答题 → 保存 |
| `eval_simple.py` | keyword matching 评估（开发迭代用，快但有误差） |
| `eval_beam_official.py` | LLM-as-judge 评估（BEAM 论文官方方法，需 GPT-4 级别模型） |

## 前置条件

1. LycheeMem 服务运行中（默认 `http://localhost:8000`）
2. Reader LLM 可用（OpenAI 兼容接口，用于答题）
3. BEAM 数据集结构：
```
chats/100K/{1,2,...}/chat.json
chats/100K/{1,2,...}/probing_questions/probing_questions.json
```

## 用法

### 跑评测

```bash
pip install -r requirements.txt

python run_beam.py \
    --input_directory chats/100K \
    --chat_size 100K \
    --start_index 0 --end_index 7 \
    --lycheemem_url http://localhost:8000 \
    --reader_url http://your-llm/v1 \
    --reader_model your-model \
    --top_k 15
```

单个 100K chat 耗时约 70-90 分钟（consolidate 阶段占大头）。

### 评估

```bash
# 开发迭代（快，keyword matching）
python eval_simple.py \
    --results_file results/100K/1/lycheemem-results.json \
    --pq_file chats/100K/1/probing_questions/probing_questions.json

# 论文报告（LLM judge，需 GPT-4）
python eval_beam_official.py \
    --results_file results/100K/1/lycheemem-results.json \
    --pq_file chats/100K/1/probing_questions/probing_questions.json \
    --judge_url http://your-judge/v1 \
    --judge_model gpt-4
```

## 当前结果

100K 档，7 个 chat，eval_simple 评估：

| Chat | Score | Chat | Score |
|------|-------|------|-------|
| 1 | 24% | 5 | 25% |
| 2 | 11% | 6 | 30% |
| 3 | 20% | 7 | 19% |
| 4 | 23% | **Avg** | **21.7%** |

Mem0 baseline (论文): **64.1%**

### 失败模式分析（140 题）

- **50% RETRIEVAL_FAIL**：搜索返回空/无关内容（根因：compact encoding 过度压缩）
- **26% ANSWERED_WRONG**：有 context 但答错（knowledge_update 选了旧值、summarization 信息不全）
- **20% ANSWERED_RIGHT**
- **4% INFRA_ERROR**：已通过 search-readiness check 修复

### Per-Category

| Category | Avg | 瓶颈 |
|----------|-----|------|
| temporal_reasoning | 46% | planning fallback 有效 |
| knowledge_update | 39% | 新旧值区分 |
| contradiction_resolution | 32% | 需要两侧都被召回 |
| multi_session_reasoning | 29% | - |
| information_extraction | 27% | 编码质量决定 |
| event_ordering | 0% | 时序信息完全丢失 |
| summarization | 0% | context 量不够 |

## 环境

- LycheeMem: dev5
- Reader: Qwen3.6-35B-A3B (vLLM)
- Embedding: bge-m3 (local)
- Hardware: A100-80G
