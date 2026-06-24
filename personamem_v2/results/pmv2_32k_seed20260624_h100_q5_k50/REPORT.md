# PersonaMem-v2 500-Question Screened Run

Date: 2026-06-24

## Bottom Line

LycheeMem dev5 scored **240/500 = 48.0% MCQ accuracy** on a fixed-seed PersonaMem-v2 32k text subset. This is a clean medium-scale signal: the metric and MCQ answer extraction match the official-style multiple-choice protocol, and the memory system was used through the external API without modifying LycheeMem source.

This is **not** the PersonaMem-v2 official full overall. It is a reproducible 100-history x 5-question subset for go/no-go and configuration screening before a full run.

## Configuration

| Field | Value |
| --- | --- |
| Benchmark | PersonaMem-v2 text MCQ |
| Context size | 32k |
| Subset | 100 random histories x 5 questions = 500 questions |
| History-list seed | 20260624 |
| Total available histories in CSV | 200 |
| Selected histories | `100` indices in `history_list.json` |
| Memory system | LycheeMem dev5, external API only |
| Ingestion | role-preserving turns, one isolated LycheeMem DB/vector store per history |
| Consolidation | `flush_session=True`, `force_ingest=True`, per-session dates preserved by adapter |
| Retrieval | `query`, `top_k=50` |
| Prompt mode | `qwen_user_final` |
| Reader | `my-llm-qwen` via OpenAI-compatible endpoint |
| Reader caveat | `enable_thinking=False` passed in `extra_body` |
| Option shuffling | `PYTHONHASHSEED=0`, official Python-hash style |

## Overall Result

| Metric | Value |
| --- | --- |
| Correct / total | 240/500 |
| Accuracy | 48.0% |
| Avg retrieved items | 50.00 |
| Avg retrieved context chars | 9408 |
| Mean answer latency per question | 59.9s |
| Median answer latency per question | 55.3s |
| P95 answer latency per question | 109.8s |
| Max answer latency per question | 208.6s |

Completeness check: `100` histories, `500` prediction rows, no missing outputs, no partial histories.

## Breakdown By Preference Type

| Type | Correct / total | Accuracy |
| --- | ---: | ---: |
| stereotypical_pref | 41/64 | 64.1% |
| ask_to_forget | 68/135 | 50.4% |
| neutral_preferences | 33/68 | 48.5% |
| therapy_background | 33/69 | 47.8% |
| anti_stereotypical_pref | 34/79 | 43.0% |
| health_and_medical_conditions | 15/39 | 38.5% |
| sensitive_info | 16/46 | 34.8% |

## Breakdown By Dynamic Preference

| Updated | Correct / total | Accuracy |
| --- | ---: | ---: |
| True | 68/135 | 50.4% |
| False | 172/365 | 47.1% |

## Breakdown By Preference Owner

| Owner | Correct / total | Accuracy |
| --- | ---: | ---: |
| self | 224/446 | 50.2% |
| others | 16/54 | 29.6% |

## Breakdown By Sensitive Info

| Sensitive | Correct / total | Accuracy |
| --- | ---: | ---: |
| False | 224/454 | 49.3% |
| True | 16/46 | 34.8% |

## Breakdown By Scenario

| Scenario | Correct / total | Accuracy |
| --- | ---: | ---: |
| professional_writing | 9/14 | 64.3% |
| creative_writing | 40/75 | 53.3% |
| professional_email | 78/157 | 49.7% |
| personal_email | 113/254 | 44.5% |

## Horizontal Position

Use these as orientation, not a same-backbone claim.

| Reference | PersonaMem-v2 MCQ result | Notes |
| --- | ---: | --- |
| LycheeMem dev5, this run | 48.0% | 500-question fixed subset, Qwen reader |
| PersonaMem-v2 paper GPT-5-Chat | 45.6% | Official paper MCQ anchor |
| PersonaMem-v2 paper Qwen3-4B-GRPO | 53.8% | Trained long-context reasoning model |
| PersonaMem-v2 paper agentic memory | 55.2% | Trained 2k-token memory model |
| DCPM paper Mem0 | 41.25-42.53% | Reported under kimi/deepseek no-think backbones |
| DCPM paper Mem0 + graph | 40.16-43.28% | Reported under kimi/deepseek no-think backbones |
| DCPM paper Zep/Graphiti | 38.27-39.54% | Reported under kimi/deepseek no-think backbones |
| DCPM paper DCPM lite | 46.85-54.10% | System-1 memory variant |
| DCPM paper DCPM full | 49.36-59.30% | System-1 + System-2 memory variant |

Interpretation: the current LycheeMem signal is clearly above the public Mem0/Zep/Graphiti memory-system band reported by DCPM, roughly around DCPM-lite/deepseek and slightly below DCPM-full/deepseek. It remains below the PersonaMem-v2 trained agentic-memory anchor. Because reader/backbone and subset differ, this is a go signal for a full, same-model run rather than a final paper number.

## What Worked

- The official-style MCQ metric is clean: micro accuracy over prediction rows.
- The adapter uses the official visible inputs: query text plus MCQ choices in the prompt, and question-only retrieval for the headline config.
- No LycheeMem source changes were made.
- `query/top_k=50` is the best screened clean config so far: first 40 histories were 98/200 = 49.0%, and this broader 500-question subset is 48.0%.
- Retrieval consistently returned 50 items, with mean context around 9.4k chars, well inside the reader context budget.

## Known Weak Spots

- `who=others` is weak: 16/54 = 29.6%. The reader often treats another person's preference as the user's own.
- Sensitive information is weak: 16/46 = 34.8%.
- Health/medical conditions are weak: 15/39 = 38.5%.
- Anti-stereotypical preferences remain below average: 34/79 = 43.0%.

These are consistent with PersonaMem-v2's design: the hard cases require separating ownership, privacy/sensitive cues, implicit medical constraints, and preferences that conflict with population priors.

## Reproduction

Generate the fixed list:

```bash
python make_personamem_v2_history_list.py   --benchmark_csv ./data/benchmark/text/benchmark.csv   --size 32k   --count 100   --seed 20260624   --output ./eval_lists/pmv2_32k_seed20260624_h100.txt   --metadata_output ./eval_lists/pmv2_32k_seed20260624_h100.json
```

Run the batch:

```bash
PERSONAMEM_V2_ROOT=/home/ldf/benchmark_lycheemem/PersonaMemV2 ISOLATED_SCRIPT=run_personamem_v2_isolated_ingest.sh bash /home/ldf/benchmark_lycheemem/PersonaMemV2/run_personamem_v2_list_batch.sh   /home/ldf/benchmark_lycheemem/PersonaMemV2/eval_lists/pmv2_32k_seed20260624_h100.txt   5 32k 50 8110 4 pmv2_32k_seed20260624_h100_q5_k50   0 query qwen_user_final 1 standard turns
```

Artifacts in this directory:

- `history_list.json`: fixed subset metadata.
- `history_urls.tsv`: HuggingFace URLs for selected histories; raw histories are ignored by git.
- `manifest.json`: exact per-history run outputs used in the strict summary.
- `predictions.jsonl`: all 500 prediction rows.
- `summary.json`: accuracy and breakdowns.
- `extra_stats.json`: latency and retrieval-size stats.

## Recommendation

Go for a larger/full run, but keep this exact headline protocol: `query/top_k=50`, turn ingestion, `qwen_user_final`, official-style MCQ scoring. The current score is strong enough relative to Mem0/Zep/Graphiti to justify the cost. Before paper reporting, rerun with the paper target reader/backbone and full split so the final table is not a subset/backbone caveat.
