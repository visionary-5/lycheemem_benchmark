# PersonaMem-v2 Results - LycheeMem (dev5)

Benchmark: **PersonaMem-v2** (arXiv 2512.06688), text MCQ, 32k histories.
Memory system: **LycheeMem dev5, unmodified** (`top_k <= 50`).
Reader model: `Qwen3.6-35B-A3B` served as `my-llm-qwen`, with `enable_thinking=False`.

> **Status (2026-06-24):** medium-scale fixed subset complete: **100 histories x 5 questions = 500 MCQs**.
> Score: **240/500 = 48.0%**. This is **not official full overall**; it is a reproducible
> go/no-go and configuration-screening run. The public 32k CSV has 200 histories, so this covers
> half of the histories under the current `5 questions/history` screening cap.

## Headline

| Run | Split / scale | Metric | Score | Done / Full |
|---|---|---|---:|---:|
| `pmv2_32k_seed20260624_h100_q5_k50` | 32k text MCQ, seed 20260624 | MCQ micro accuracy | **48.0%** | 500 / 500 complete |

**Clean read:** this is a positive go signal. It is above the public Mem0 / Zep-Graphiti
PersonaMem-v2 memory-system baselines reported in DCPM, and roughly around the lower DCPM-lite band.
It is still below the PersonaMem-v2 paper's trained Qwen3-GRPO and trained agentic-memory anchors.
Because this run uses our Qwen reader and a 500-question subset, treat it as a configuration and
benchmark-value result, not a final paper table number.

## Main result and runtime

| Metric | Value |
|---|---:|
| Correct / total | **240/500** |
| MCQ accuracy | **48.0%** |
| Avg retrieved memories | 50.00 |
| Avg retrieved context chars | 9408 |
| Mean answer latency / question | 59.9s |
| Median answer latency / question | 55.3s |
| P95 answer latency / question | 109.8s |

Completeness: strict manifest has **100 histories**, **500 prediction rows**, no missing outputs,
no partial histories. Raw history JSON files are deliberately ignored by git; committed artifacts
include the history list, URL list, manifest, predictions, summary, and extra stats.

## Horizontal picture

These are the closest public anchors for understanding the score. They are not all same-backbone
or same-subset comparisons.

| System / paper anchor | PersonaMem-v2 MCQ | Comparison note |
|---|---:|---|
| **LycheeMem dev5, this 500-question run** | **48.0%** | Qwen reader, fixed 100-history subset |
| PersonaMem-v2 paper: GPT-5-Chat | 45.6% | Official MCQ anchor from original paper |
| PersonaMem-v2 paper: Qwen3-4B-GRPO | 53.8% | Trained long-context reasoning model |
| PersonaMem-v2 paper: agentic memory | 55.2% | Trained 2k-token memory model |
| DCPM: Mem0 | 41.25-42.53% | Reported under kimi/deepseek no-think backbones |
| DCPM: Mem0 + graph | 40.16-43.28% | Reported under kimi/deepseek no-think backbones |
| DCPM: Zep / Graphiti | 38.27-39.54% | Reported under kimi/deepseek no-think backbones |
| DCPM: DCPM-lite | 46.85-54.10% | System-1 memory only |
| DCPM: DCPM-full | 49.36-59.30% | System-1 + System-2 memory |

**Read of the landscape.** PersonaMem-v2 is not a factual QA benchmark; it stresses implicit
personalization under noisy multi-session histories. A 48.0% clean MCQ score is meaningful because
random is 25%, frontier long-context models in the original paper are only around the 37-48% band,
and production memory baselines in DCPM sit around 38-43%. The gap to 53-55% is also real: the
strongest references are trained specifically for PersonaMem-v2-style personalization and/or
agentic memory compression.

## Methodology - what is comparable

- **Metric:** official-style MCQ micro accuracy, `correct / total`. Each row has four plausible
  choices and exactly one personalized answer.
- **Prompt / answer extraction:** aligned to the official MCQ format. The local Qwen endpoint cannot
  handle the exact final system-message form, so `qwen_user_final` appends the official final MCQ
  instruction to the final user message. This is a transport compatibility change, not a scoring
  change.
- **Option order:** `PYTHONHASHSEED=0` is pinned because the official option shuffle depends on
  Python `hash()`.
- **Memory insertion:** only external LycheeMem APIs are used. No LycheeMem source was changed.
- **Retrieval:** headline run uses question-only retrieval (`search_mode=query`) with `top_k=50`.
  Option-aware or metadata-aware retrieval modes exist only as diagnostics and are not mixed into
  this result.
- **Ingestion:** role-preserving turn ingestion into isolated per-history LycheeMem runs. This keeps
  each user/history independent and prevents cross-user leakage.

This means the **metric口径 is cleanly comparable**. The caveat is scale/backbone: this is a 500-row
screening subset with Qwen reader, not the full official overall with the paper's models.

## Configuration screening

| Config | Scale | Score | Note |
|---|---:|---:|---|
| `query`, `top_k=20` | first 40 histories x 5 = 200 | 93/200 = 46.5% | clean question-only retrieval |
| `query`, `top_k=50` | first 40 histories x 5 = 200 | 98/200 = 49.0% | best screened config |
| `query_raw`, `top_k=20` | first 40 histories x 5 = 200 | 88/200 = 44.0% | raw user query only |
| `query_raw_options`, `top_k=20` | first 40 histories x 5 = 200 | 86/200 = 43.0% | option-aware diagnostic; regressed by 200 |
| **`query`, `top_k=50`** | **seeded 100 histories x 5 = 500** | **240/500 = 48.0%** | current headline |

The key decision from screening is stable: **use `query/top_k=50`** for larger or full runs.
The 500-question result essentially confirms the 200-question signal rather than overfitting the
first 40 histories.

## Breakdowns

### Preference type

| Type | Correct / total | Accuracy |
|---|---:|---:|
| `stereotypical_pref` | 41/64 | **64.1%** |
| `ask_to_forget` | 68/135 | **50.4%** |
| `neutral_preferences` | 33/68 | **48.5%** |
| `therapy_background` | 33/69 | **47.8%** |
| `anti_stereotypical_pref` | 34/79 | **43.0%** |
| `health_and_medical_conditions` | 15/39 | **38.5%** |
| `sensitive_info` | 16/46 | **34.8%** |

### Dynamic preferences

| Updated | Correct / total | Accuracy |
|---|---:|---:|
| `True` | 68/135 | **50.4%** |
| `False` | 172/365 | **47.1%** |

### Preference owner

| Owner | Correct / total | Accuracy |
|---|---:|---:|
| `self` | 224/446 | **50.2%** |
| `others` | 16/54 | **29.6%** |

### Sensitive information

| Sensitive info | Correct / total | Accuracy |
|---|---:|---:|
| `False` | 224/454 | **49.3%** |
| `True` | 16/46 | **34.8%** |

### Scenario

| Scenario | Correct / total | Accuracy |
|---|---:|---:|
| `professional_writing` | 9/14 | **64.3%** |
| `creative_writing` | 40/75 | **53.3%** |
| `professional_email` | 78/157 | **49.7%** |
| `personal_email` | 113/254 | **44.5%** |

## Weak spots

- **Preference ownership:** `others` is only **16/54 = 29.6%**. The system/reader often fails to keep
  another person's preference separate from the user's own preference.
- **Sensitive information:** `sensitive_info=True` is **16/46 = 34.8%**. These cases need stricter
  handling of sensitive cues and likely more precise memory selection.
- **Health/medical constraints:** `health_and_medical_conditions` is **15/39 = 38.5%**.
- **Anti-stereotypical preferences:** **34/79 = 43.0%**, below the overall average. This matches the
  benchmark's intended difficulty: avoid falling back to population priors.

## Artifacts

| Path | Role |
|---|---|
| `results/pmv2_32k_seed20260624_h100_q5_k50/REPORT.md` | Detailed per-run report generated from the final summary. |
| `results/pmv2_32k_seed20260624_h100_q5_k50/summary.json` | Overall and breakdown metrics. |
| `results/pmv2_32k_seed20260624_h100_q5_k50/predictions.jsonl` | 500 row-level predictions. |
| `results/pmv2_32k_seed20260624_h100_q5_k50/manifest.json` | Exact per-history output files used for strict aggregation. |
| `results/pmv2_32k_seed20260624_h100_q5_k50/history_list.json` | Fixed-seed selected histories and metadata. |
| `results/pmv2_32k_seed20260624_h100_q5_k50/history_urls.tsv` | HF source URLs for the selected histories. |
| `make_personamem_v2_history_list.py` | Reproducible history-list generator. |
| `run_personamem_v2_list_batch.sh` | Explicit-history-list batch runner. |

## Recommendation

**Go for the larger/full run.** Keep the headline protocol unchanged: `query/top_k=50`, turn
ingestion, isolated per-history LycheeMem state, official-style MCQ scoring, and thinking disabled
for the Qwen reader. For paper reporting, rerun with the target paper reader/backbone and full split
so the final number has no subset/backbone caveat.
