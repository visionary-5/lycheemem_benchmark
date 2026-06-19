# LycheeMem BEAM Benchmark 评测脚本

BEAM (Benchmark for Evaluating AI Memory) 上评测 LycheeMem dev5 的完整流程脚本。

## 目录结构

```
├── run_beam.py            # 主脚本：ingest + answer（批量跑多个 chat）
├── eval_simple.py         # 评估：keyword matching（开发迭代用）
├── eval_beam_official.py  # 评估：LLM-as-judge（论文报告用，需 GPT-4）
├── requirements.txt
└── README.md
```

## 前置条件

1. **LycheeMem 服务**已启动（默认 `http://localhost:8000`）
2. **Reader LLM** 服务可用（OpenAI 兼容接口）
3. **BEAM 数据集**已下载，结构如下：
```
chats/
├── 100K/
│   ├── 1/
│   │   ├── chat.json
│   │   └── probing_questions/
│   │       └── probing_questions.json
│   ├── 2/
│   ...
```

## 用法

### 1. 跑评测（ingest + 答题）

```bash
python run_beam.py \
    --input_directory chats/100K \
    --chat_size 100K \
    --start_index 0 --end_index 7 \
    --lycheemem_url http://localhost:8000 \
    --reader_url http://your-llm/v1 \
    --reader_model your-model \
    --top_k 15
```

每个 chat 流程：清空记忆 → 写入对话 → consolidate → 验证搜索就绪 → 答 20 题 → 保存结果。

单个 100K chat 耗时约 70-90 分钟（主要是 consolidate 阶段）。

### 2. 评估（keyword matching）

```bash
python eval_simple.py \
    --results_file results/100K/1/lycheemem-results.json \
    --pq_file chats/100K/1/probing_questions/probing_questions.json
```

### 3. 评估（LLM Judge，论文用）

```bash
python eval_beam_official.py \
    --results_file results/100K/1/lycheemem-results.json \
    --pq_file chats/100K/1/probing_questions/probing_questions.json \
    --judge_url http://your-judge-llm/v1 \
    --judge_model gpt-4
```

## 当前结果（100K, 7 chats, eval_simple）

| Chat | Score |
|------|-------|
| 1 | 24% |
| 2 | 11% |
| 3 | 20% |
| 4 | 23% |
| 5 | 25% |
| 6 | 30% |
| 7 | 19% |
| **平均** | **21.7%** |

对比 Mem0 baseline: 64.1%

### Per-Category 平均

| Category | Avg | 说明 |
|----------|-----|------|
| temporal_reasoning | 46% | 最强，planning fallback 兜底有效 |
| knowledge_update | 39% | 能找到更新后的值 |
| contradiction_resolution | 32% | V4 prompt 的矛盾规则有效 |
| multi_session_reasoning | 29% | 跨 session 推理 |
| information_extraction | 27% | 高方差，取决于编码质量 |
| abstention | 21% | 应拒答时能拒答 |
| instruction_following | 11% | 用户格式偏好几乎未编码 |
| preference_following | 6% | 同上 |
| event_ordering | 0% | 时间顺序信息完全丢失 |
| summarization | 0% | broad query context 不够 |

### 失败模式分布（140 题）

| 模式 | 数量 | 占比 | 原因 |
|------|------|------|------|
| RETRIEVAL_FAIL | 70 | 50% | 搜索返回空/无关内容 |
| ANSWERED_WRONG | 37 | 26% | 有 context 但答错 |
| ANSWERED_RIGHT | 28 | 20% | 正确 |
| INFRA_ERROR | 5 | 4% | 服务端 500 |

## 已知问题

1. **记忆过度压缩**：200+ 条消息 consolidate 后可能只剩十几条 records，丢失大量细节
2. **vector index 残留**：`/memory/clear` 只清 sqlite，vector index 可能保留旧 embeddings（LycheeMem bug）
3. **eval_simple 假阳性**：模型拒答时复述关键词会被错判为正确（已知，用 LLM judge 做最终评测）

## 环境

- LycheeMem: dev5 分支
- Reader LLM: Qwen3.6-35B-A3B (vLLM serving)
- Embedding: bge-m3 (local)
- 硬件: A100-80G
