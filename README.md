# LycheeMem Benchmark

LycheeMem 在各类记忆能力 benchmark 上的评测脚本和结果。

## 目录

| Benchmark | 路径 | 说明 |
|-----------|------|------|
| BEAM | [`beam/`](beam/) | Benchmark for Evaluating AI Memory，100K/500K/1M/10M 对话记忆评测 |
| MABench | [`mab/`](mab/) | MemoryAgentBench (ICLR 2026)，精准检索/冲突消解/远距理解/测试时学习 |

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
└── README.md              # 本文件
```

后续新增 benchmark 在顶层加文件夹即可。
