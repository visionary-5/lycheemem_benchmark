# LycheeMem Benchmark

LycheeMem 在各类记忆能力 benchmark 上的评测脚本和结果。

## 目录

| Benchmark | 路径 | 说明 |
|-----------|------|------|
| BEAM | [`beam/`](beam/) | Benchmark for Evaluating AI Memory，100K/500K/1M/10M 对话记忆评测 |
| MABench | [`mab/`](mab/) | MemoryAgentBench (ICLR 2026)，精准检索/冲突消解/远距理解/测试时学习 |
| PersonaMem-v2 | [`personamem_v2/`](personamem_v2/) | 隐式用户偏好/个性化记忆评测，MCQ response selection |

## 当前结果速览

| Benchmark | 当前状态 | 关键结果 | 详细报告 |
|-----------|----------|----------|----------|
| BEAM | 已有评测脚本和结果 | 见子目录报告 | [`beam/RESULTS.md`](beam/RESULTS.md) |
| MABench | mid-scale batch 已完成，部分列仍待 scorer/full run | 见子目录报告 | [`mab/RESULTS.md`](mab/RESULTS.md) |
| PersonaMem-v2 | 32k text MCQ 固定 500 题子集完成 | LycheeMem dev5: `240/500 = 48.0%` | [`personamem_v2/RESULTS.md`](personamem_v2/RESULTS.md) |

PersonaMem-v2 的当前数字是 `100 histories x 5 questions` 的中规模固定子集
（seed `20260624`），用于 go/no-go 和配置筛选；它还不是官方 full overall。
当前最稳配置是 `query/top_k=50`、turn ingestion、official-style MCQ micro accuracy。
横向粗看，这个结果高于 DCPM 论文里公开的 Mem0 / Zep-Graphiti
PersonaMem-v2 memory-system baseline 区间，但低于 PersonaMem-v2 原论文的
Qwen3-GRPO / trained agentic-memory anchor。完整解释、口径 caveat 和 breakdown 见
[`personamem_v2/RESULTS.md`](personamem_v2/RESULTS.md)。

## 仓库结构

```
├── beam/                  # BEAM benchmark 评测
│   ├── run_beam.py        # 主脚本：ingest + answer
│   ├── eval_simple.py     # keyword matching 评估
│   ├── eval_beam_official.py  # LLM judge 评估
│   └── requirements.txt
├── mab/                   # MemoryAgentBench 评测
│   ├── run_mab.py         # 主脚本：ingest + answer + eval
│   └── requirements.txt
├── personamem_v2/         # PersonaMem-v2 评测适配和结果
│   ├── run_personamem_v2_lycheemem.py
│   ├── run_personamem_v2_list_batch.sh
│   ├── RESULTS.md         # 当前 500 题结果和横向解读
│   └── results/           # 固定子集产物、summary、predictions
└── README.md              # 本文件
```

后续新增 benchmark 在顶层加文件夹即可。
