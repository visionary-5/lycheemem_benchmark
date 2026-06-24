"""
MemoryAgentBench (ICLR 2026) — LycheeMem Evaluation Runner v2

Strictly follows MABench methodology:
- Same templates (memorize + query prompts per dataset)
- Same chunking (nltk sentence tokenize + tiktoken token counting)
- Same evaluation metrics (exact_match, substring_exact_match, f1, eventqa_recall)
- Same data format and output structure

Usage:
  python run_mab_v2.py \
    --data_dir ./data \
    --sub_dataset eventqa_full \
    --dataset Accurate_Retrieval \
    --max_test_samples 5 \
    --lycheemem_url http://localhost:8000 \
    --llm_url http://10.251.171.6:28043/v1 \
    --llm_model my-llm-qwen \
    --chunk_size 4096 \
    --retrieve_num 100
"""

import argparse
import json
import os
import re
import string
import sys
import time
from collections import Counter, defaultdict

import nltk
import numpy as np
import requests
import tiktoken
from openai import OpenAI
from rouge_score import rouge_scorer as rs

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

# ===========================================================================
# MABench Templates (exact copy from utils/templates.py)
# ===========================================================================

SYSTEM_MESSAGE = "You are a helpful assistant that can read the context and memorize it for future retrieval."

BASE_TEMPLATES = {
    "ruler_qa": {
        "system": SYSTEM_MESSAGE,
        "memorize": 'Dialogue between User and Assistant {time_stamp}\\n<User> The following context is the documents I have read: \n{context}\n <Assistant> I have learned the documents and I will answer the question you ask.',
        "query": "Answer the question based on the memorized documents. Only give me the answer and do not output any other words. \n\n Now Answer the Question: {question}",
    },
    "eventqa": {
        "system": SYSTEM_MESSAGE,
        "memorize": 'Dialogue between User and Assistant {time_stamp}\\n<User> The following context is the book excerpt: \n{context}\n <Assistant> I have read the book excerpt and I will answer the question you ask.',
        "query": "Based on the context you memorized, complete the task below:\n\n{question}\n\n The event that happens next is:",
    },
    "longmemeval": {
        "system": SYSTEM_MESSAGE,
        "memorize": 'Dialogue between User and Assistant \\n<User> The following context is the conversation between the user and the assistant: \n{context}\n <Assistant> I have memorized the conversation and I will answer the question you ask.',
        "query": "The history chats are between you and a user. Based on the relevant chat history, answer the question as concisely as you can, using a single phrase if possible.\n\n {question} \n\n Answer:",
    },
    "in_context_learning": {
        "system": SYSTEM_MESSAGE,
        "memorize": 'Dialogue between User and Assistant {time_stamp} \\n<User> The following context is the examples I have learned: \n{context}\n <Assistant> I have learned the examples and I will answer the question you ask.',
        "query": 'Use the provided mapping from the context to numerical label to assign a numerical label to the context. Only output "label: {{label}}" and nothing else. \n\nQuestion:{question} \n\n label:',
    },
    "factconsolidation": {
        "system": SYSTEM_MESSAGE,
        "memorize": 'Dialogue between User and Assistant {time_stamp} \\n<User> The following context is the facts I have learned: \n{context}\n <Assistant> I have learned the facts and I will answer the question you ask.',
        "query": "Pretend you are a knowledge management system. Each fact in the knowledge pool is provided with a serial number at the beginning, and the newer fact has larger serial number. \n You need to solve the conflicts of facts in the knowledge pool by finding the newest fact with larger serial number. You need to answer a question based on this rule. You should give a very concise answer without saying other words for the question **only** from the knowledge pool you have memorized rather than the real facts in real world. \n\nFor example:\n\n [Knowledge Pool] \n\n Question: Based on the provided Knowledge Pool, what is the name of the current president of Russia? \nAnswer: Donald Trump \n\n Now Answer the Question: Based on the provided Knowledge Pool, {question} \nAnswer:",
    },
    "detective_qa": {
        "system": SYSTEM_MESSAGE,
        "memorize": 'Dialogue between User and Assistant {time_stamp} \\n<User> The following context is the book I have read: \n{context}\n <Assistant> I have read the book and I will answer the question you ask.',
        "query": "Based on the context you memorized, answer the question below. You are required to answer the question based on the strict output format.\n\n {question} \n\n",
    },
    "infbench_sum": {
        "system": SYSTEM_MESSAGE,
        "memorize": 'Dialogue between User and Assistant {time_stamp} \\n<User> The following context is the book/story I have read: \n{context}\n <Assistant> I have read it and I will answer the question you ask.',
        "query": "Based on the context you memorized, complete the task below:\n\n{question}",
    },
    "recsys": {
        "system": SYSTEM_MESSAGE,
        "memorize": 'Dialogue between User and Assistant {time_stamp} \\n<User> The following context is the movie recommendation conversation and catalog: \n{context}\n <Assistant> I have memorized it and I will answer the question you ask.',
        "query": "{question}",
    },
}

DATASET_MAPPING = {
    ("ruler_",): "ruler_qa",
    ("icl_",): "in_context_learning",
    ("eventqa_",): "eventqa",
    ("longmemeval_",): "longmemeval",
    ("factconsolidation_",): "factconsolidation",
    ("detective",): "detective_qa",
    ("infbench_sum",): "infbench_sum",
    ("recsys",): "recsys",
}


def get_dataset_key(sub_dataset: str) -> str:
    for patterns, key in DATASET_MAPPING.items():
        if all(p in sub_dataset for p in patterns):
            return key
    raise ValueError(f"Unknown sub_dataset: {sub_dataset}")


def get_memorize_template(sub_dataset: str) -> str:
    return BASE_TEMPLATES[get_dataset_key(sub_dataset)]["memorize"]


def get_query_template(sub_dataset: str) -> str:
    return BASE_TEMPLATES[get_dataset_key(sub_dataset)]["query"]


def get_system_message(sub_dataset: str) -> str:
    return BASE_TEMPLATES[get_dataset_key(sub_dataset)]["system"]


# ===========================================================================
# MABench Chunking (exact replica of chunk_text_into_sentences)
# ===========================================================================

def chunk_text_into_sentences(text: str, chunk_size: int = 4096) -> list:
    """Split text into chunks by sentence boundaries, respecting token limit."""
    try:
        encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    except KeyError:
        encoding = tiktoken.encoding_for_model("gpt-4o-mini")

    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_sentences = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = len(encoding.encode(sentence, allowed_special={"<|endoftext|>"}))
        if current_tokens + sentence_tokens > chunk_size and current_sentences:
            chunks.append(" ".join(current_sentences))
            current_sentences = [sentence]
            current_tokens = sentence_tokens
        else:
            current_sentences.append(sentence)
            current_tokens += sentence_tokens

    if current_sentences:
        chunks.append(" ".join(current_sentences))
    return chunks


# ===========================================================================
# MABench Evaluation Metrics (exact replica)
# ===========================================================================

rouge_scorer_instance = rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def f1_score(prediction: str, ground_truth: str):
    pred_norm = normalize_answer(prediction)
    gt_norm = normalize_answer(ground_truth)
    special = {"yes", "no", "noanswer"}
    if (pred_norm in special or gt_norm in special) and pred_norm != gt_norm:
        return 0, 0, 0
    pred_tokens = pred_norm.split()
    gt_tokens = gt_norm.split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0, 0, 0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gt_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def substring_exact_match_score(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(ground_truth) in normalize_answer(prediction)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    if isinstance(ground_truths, str):
        gt_list = [ground_truths]
    elif ground_truths and isinstance(ground_truths[0], list):
        gt_list = [g for sub in ground_truths for g in sub]
    else:
        gt_list = ground_truths
    return max(metric_fn(prediction, gt) for gt in gt_list)


def parse_output(text: str, prefix="Answer:"):
    patterns = [
        re.compile(f"(?:{prefix})(.*)(?:\n|$)", flags=re.IGNORECASE),
        re.compile(r"(?:^)(.*)(?:\n|$)"),
    ]
    for p in patterns:
        m = p.search(text)
        if m:
            clean = re.sub(f"^{re.escape(prefix)}", "", m[1].strip(), flags=re.IGNORECASE).strip()
            return clean
    return None


def calculate_metrics(prediction, ground_truths):
    if prediction is None:
        prediction = ""
    metrics = {
        "exact_match": metric_max_over_ground_truths(exact_match_score, prediction, ground_truths),
        "f1": metric_max_over_ground_truths(lambda p, g: f1_score(p, g)[0], prediction, ground_truths),
        "substring_exact_match": metric_max_over_ground_truths(substring_exact_match_score, prediction, ground_truths),
    }
    if isinstance(ground_truths, str):
        answer_list = [ground_truths]
    elif ground_truths and isinstance(ground_truths[0], list):
        answer_list = [g for sub in ground_truths for g in sub]
    else:
        answer_list = ground_truths
    rouge_scores = [rouge_scorer_instance.score(target=a, prediction=prediction) for a in answer_list]
    for rt in rouge_scorer_instance.rouge_types:
        metrics[rt + "_f1"] = max(s[rt].fmeasure for s in rouge_scores)
        metrics[rt + "_recall"] = max(s[rt].recall for s in rouge_scores)
    return metrics


def post_process(output_text: str, answer, sub_dataset: str):
    """Dataset-specific post-processing → (metrics_dict, info_dict)."""
    if "icl" in sub_dataset:
        # ICL query asks the model to emit 'label: {label}'. The default parse prefix
        # is 'Answer:', which leaves the 'label:' prefix in place -> exact_match never
        # matches the bare numeric gold (e.g. 'label: 50' vs '50'). Parse on 'label:'
        # to recover the official intent (compare the numeric label only).
        parsed = parse_output(output_text, prefix="label:")
        return calculate_metrics(parsed, answer), {"parsed_output": parsed}
    elif "eventqa" in sub_dataset:
        recall = sum(a.lower() in output_text.lower() for a in answer) / len(answer)
        binary_recall = int(recall == 1)
        parsed = parse_output(output_text)
        m = calculate_metrics(parsed, answer)
        m["eventqa_recall"] = binary_recall
        return m, {"parsed_output": parsed}
    elif "detective" in sub_dataset:
        # detective is multiple-choice; the query prompt asks the model for a JSON object
        # {"answer":"X. <option text>","reasoning":...}. The model complies, so the raw
        # output is a JSON blob and exact_match vs the bare gold ("X. <text>") is always 0
        # even when the chosen option is correct. Unwrap the answer field first (same class
        # of bug as the ICL 'label:' parse). Also expose a letter-only match (A/B/C/D),
        # the most robust MCQ口径 since option-text wording can drift while the choice is right.
        pred_ans = output_text
        try:
            pred_ans = json.loads(output_text).get("answer", output_text)
        except Exception:
            mobj = re.search(r'"answer"\s*:\s*"([^"]*)"', output_text)
            if mobj:
                pred_ans = mobj.group(1)
        m = calculate_metrics(pred_ans, answer)

        def _letter(s):
            mm = re.match(r"\s*([A-D])\b", str(s).strip())
            return mm.group(1) if mm else None
        gold_list = answer if isinstance(answer, list) else [answer]
        gold_list = [g for sub in gold_list for g in (sub if isinstance(sub, list) else [sub])]
        pl = _letter(pred_ans)
        m["mcq_letter_match"] = int(pl is not None and any(pl == _letter(g) for g in gold_list))
        return m, {"parsed_output": pred_ans}
    else:
        m = calculate_metrics(output_text, answer)
        parsed = parse_output(output_text)
        if parsed is not None:
            pm = calculate_metrics(parsed, answer)
            m = {k: max(m[k], pm[k]) for k in m}
        return m, {"parsed_output": parsed}


# ===========================================================================
# Data Loading (from local JSON — server has no internet)
# ===========================================================================

def load_data(data_dir: str, dataset: str, sub_dataset: str, max_samples: int = 5):
    """Load from pre-exported JSON files on server."""
    json_path = os.path.join(data_dir, f"{dataset}.json")
    if not os.path.exists(json_path):
        print(f"ERROR: {json_path} not found")
        sys.exit(1)

    with open(json_path) as f:
        all_items = json.load(f)

    items = [it for it in all_items if it.get("source", "") == sub_dataset]
    if not items:
        sources = sorted(set(it.get("source", "") for it in all_items))
        print(f"ERROR: no items with source='{sub_dataset}'. Available: {sources}")
        sys.exit(1)

    if max_samples > 0:
        items = items[:max_samples]
    print(f"Loaded {len(items)} contexts for '{sub_dataset}' from {json_path}")
    return items


# ===========================================================================
# LycheeMem HTTP API
# ===========================================================================

def lycheemem_clear(url: str):
    for attempt in range(3):
        try:
            r = requests.delete(f"{url}/memory/clear", timeout=60)
            if r.status_code in (200, 204):
                return True
        except Exception as e:
            print(f"  clear attempt {attempt+1} failed: {e}")
            time.sleep(3)
    return False


def lycheemem_append_turn(url: str, session_id: str, role: str, content: str):
    for attempt in range(3):
        try:
            r = requests.post(
                f"{url}/memory/append-turn",
                json={"session_id": session_id, "role": role, "content": content},
                timeout=120,
            )
            r.raise_for_status()
            return True
        except Exception as e:
            if attempt == 2:
                print(f"  append_turn failed: {e}")
            time.sleep(3)
    return False


def lycheemem_consolidate(url: str, session_id: str, timeout: int = 5400):
    try:
        r = requests.post(
            f"{url}/memory/consolidate",
            json={"session_id": session_id, "flush_session": True, "background": False, "force_ingest": True},
            timeout=timeout,
        )
        if r.status_code == 200:
            data = r.json()
            print(f"  Consolidate OK: turns={data.get('turns_consumed','?')}, facts={data.get('facts_added','?')}")
            return True
        else:
            print(f"  Consolidate error {r.status_code}: {r.text[:200]}")
            return False
    except requests.exceptions.ReadTimeout:
        print("  Consolidate timed out, polling search...")
        for i in range(120):
            time.sleep(30)
            try:
                r2 = requests.post(
                    f"{url}/memory/search",
                    json={"query": "test", "top_k": 1, "include_semantic": True},
                    timeout=30,
                )
                if r2.status_code == 200 and r2.json().get("total", 0) > 0:
                    print(f"  Consolidate complete (poll {i+1})")
                    return True
            except Exception:
                pass
        return False


def lycheemem_search(url: str, query: str, top_k: int = 100):
    """Search LycheeMem and return (raw_context_str, result_count)."""
    try:
        r = requests.post(
            f"{url}/memory/search",
            json={"query": query, "top_k": top_k, "include_semantic": True},
            timeout=60,
        )
        if r.status_code == 200:
            data = r.json()
            raw_ctx = data.get("raw_retrieved_context", "")
            total = data.get("total", 0)
            return raw_ctx, total
        else:
            print(f"  search error {r.status_code}")
            return "", 0
    except Exception as e:
        print(f"  search exception: {e}")
        return "", 0


# ===========================================================================
# LLM Answer Generation (matching MABench mem0 handler logic)
# ===========================================================================

def generate_answer(llm_client: OpenAI, model: str, query: str, memories_str: str,
                    temperature: float = 0.7, max_tokens: int = 40, strong: bool = False) -> str:
    """Generate answer using retrieved memories — same pattern as mem0 handler."""
    # Clean up LycheeMem metadata prefixes from the context
    import re
    clean_memories = re.sub(r'\[\d+\]\n\[(user|assistant)\]:\s*', '', memories_str)

    if strong:
        system_prompt = (
            "You are a precise assistant. Answer the question using ONLY the retrieved context below; "
            "do NOT use outside world knowledge. "
            "If the question asks for a count, total, or number of days/weeks between events, find the "
            "relevant values in the context and compute the result step by step, then state the final number. "
            "Give the shortest possible answer (a single word, number, or phrase). "
            "For dates, prefer the natural-language form that appears in the context (e.g. 'June 3rd').\n\n"
            f"[Retrieved Context]\n{clean_memories}\n"
        )
    else:
        system_prompt = f"You are a helpful AI. Answer the question based ONLY on the following retrieved context. Do NOT use your own world knowledge.\n\n[Retrieved Context]\n{clean_memories}\n"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]
    try:
        response = llm_client.chat.completions.create(
            model=model, messages=messages,
            temperature=temperature, max_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        content = response.choices[0].message.content
        if content is None:
            content = getattr(response.choices[0].message, "reasoning", "") or ""
        return content.strip()
    except Exception as e:
        print(f"  LLM error: {e}")
        return f"Error: {e}"


# ===========================================================================
# Main Evaluation Loop
# ===========================================================================

def _answer_queries(context_idx, questions, answers, query_tpl, args,
                    llm_client, memory_construction_time):
    """Search + generate + evaluate for each query. Shared by full and query_only runs."""
    print(f"  [4/5] Answering {len(questions)} queries (retrieve_num={args.retrieve_num})...")
    results = []

    for qi, (question, answer) in enumerate(zip(questions, answers)):
        formatted_query = query_tpl.format(question=question)

        # Search — use the question (not full template query) for better vector match
        search_start = time.time()
        raw_ctx, result_count = lycheemem_search(args.lycheemem_url, question, args.retrieve_num)
        search_time = time.time() - search_start

        # Strip template boilerplate from retrieved context if present
        memories_str = raw_ctx.strip() if raw_ctx else "(No relevant memories found)"

        # Generate answer
        gen_start = time.time()
        prediction = generate_answer(
            llm_client, args.llm_model, formatted_query, memories_str,
            temperature=args.temperature, max_tokens=args.generation_max_length,
            strong=args.strong_reader,
        )
        gen_time = time.time() - gen_start

        # Evaluate
        metrics, info = post_process(prediction, answer, args.sub_dataset)
        query_time = search_time + gen_time

        result = {
            "output": prediction,
            "input_len": len(formatted_query) + len(memories_str),
            "output_len": len(prediction),
            "memory_construction_time": memory_construction_time,
            "query_time_len": query_time,
            "answer": answer,
            "query": formatted_query,
            "query_id": qi,
            "context_id": context_idx,
            "search_results_count": result_count,
            **metrics,
            **info,
        }
        results.append(result)

        status = "HIT" if metrics.get("substring_exact_match", 0) else "MISS"
        print(f"    Q{qi+1}: {status} | sem={metrics.get('substring_exact_match',0):.0f} "
              f"em={metrics.get('exact_match',0):.0f} f1={metrics.get('f1',0):.2f} "
              f"| pred={prediction[:80]}")

    return results


def process_context(context_idx: int, item: dict, args, llm_client: OpenAI):
    """Process one context: clear → ingest → consolidate → answer queries."""
    context_text = item["context"]
    questions = item.get("questions", [])
    answers = item.get("answers", [])
    if isinstance(questions, str):
        questions = [questions]
    if isinstance(answers, str):
        answers = [answers]
    if getattr(args, "max_questions", 0) and args.max_questions > 0:
        questions = questions[: args.max_questions]
        answers = answers[: args.max_questions]

    session_id = f"mab_{args.sub_dataset}_{context_idx}"
    memorize_tpl = get_memorize_template(args.sub_dataset)
    query_tpl = get_query_template(args.sub_dataset)

    print(f"\n{'='*70}")
    print(f"Context {context_idx}: {len(context_text)} chars, {len(questions)} queries")
    print(f"{'='*70}")

    if args.query_only:
        # Reuse already-consolidated memory; skip clear/ingest/consolidate
        print("  [query_only] Reusing existing consolidated memory, skipping to queries...")
        ingest_start = time.time()
        memory_construction_time = 0.0
        return _answer_queries(context_idx, questions, answers, query_tpl, args,
                               llm_client, memory_construction_time)

    # Step 1: Clear memory
    print("  [1/5] Clearing memory...")
    lycheemem_clear(args.lycheemem_url)
    time.sleep(2)

    # Step 2: Chunk text (MABench method: nltk + tiktoken)
    chunks = chunk_text_into_sentences(context_text, chunk_size=args.chunk_size)
    print(f"  [2/5] Ingesting {len(chunks)} chunks (chunk_size={args.chunk_size} tokens)...")

    # Step 3: Ingest each chunk
    ingest_start = time.time()
    for i, chunk in enumerate(chunks):
        if args.raw_ingest:
            content = chunk
        else:
            content = memorize_tpl.format(
                context=chunk,
                time_stamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            )
        role = "user" if i % 2 == 0 else "assistant"
        lycheemem_append_turn(args.lycheemem_url, session_id, role, content)
        if (i + 1) % 50 == 0:
            print(f"    ... {i+1}/{len(chunks)} chunks appended")
    ingest_time = time.time() - ingest_start
    print(f"    Ingestion took {ingest_time:.0f}s")

    # Step 4: Consolidate
    if args.skip_consolidate:
        print("  [3/5] Skipping consolidate (raw turn search mode)...")
    else:
        print("  [3/5] Consolidating...")
        t0 = time.time()
        lycheemem_consolidate(args.lycheemem_url, session_id)
        print(f"    Consolidate took {time.time()-t0:.0f}s")

    # Step 5: Answer queries
    memory_construction_time = time.time() - ingest_start
    return _answer_queries(context_idx, questions, answers, query_tpl, args,
                           llm_client, memory_construction_time)


def main():
    parser = argparse.ArgumentParser(description="MABench LycheeMem Evaluation (v2 - proper methodology)")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--dataset", type=str, default="Accurate_Retrieval",
                        help="Main dataset split (Accurate_Retrieval, Conflict_Resolution, etc.)")
    parser.add_argument("--sub_dataset", type=str, default="eventqa_full",
                        help="Sub-dataset source filter")
    parser.add_argument("--max_test_samples", type=int, default=5)
    parser.add_argument("--max_questions", type=int, default=0,
                        help="If >0, only answer the first N questions per context (口径 validation)")
    parser.add_argument("--chunk_size", type=int, default=4096, help="Tokens per chunk (MABench default)")
    parser.add_argument("--retrieve_num", type=int, default=100, help="Top-K for search (MABench mem0 default)")
    parser.add_argument("--lycheemem_url", type=str, default="http://localhost:8000")
    parser.add_argument("--llm_url", type=str, default="http://10.251.171.6:28043/v1")
    parser.add_argument("--llm_model", type=str, default="my-llm-qwen")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--generation_max_length", type=int, default=40)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--skip_consolidate", action="store_true")
    parser.add_argument("--raw_ingest", action="store_true",
                        help="Ingest raw text without memorize template wrapping (better vectors)")
    parser.add_argument("--query_only", action="store_true",
                        help="Reuse existing consolidated memory, skip clear/ingest/consolidate")
    parser.add_argument("--strong_reader", action="store_true",
                        help="Use enhanced reader prompt (step-by-step counting/date reasoning, concise)")
    parser.add_argument("--context_idx", type=int, default=-1,
                        help="Run only this single context index (for clean per-context restart loop)")
    args = parser.parse_args()

    llm_client = OpenAI(base_url=args.llm_url, api_key="dummy")

    # Load data; --context_idx N runs only that single context (clean per-context runs)
    if args.context_idx >= 0:
        all_items = load_data(args.data_dir, args.dataset, args.sub_dataset, args.context_idx + 1)
        targets = [(args.context_idx, all_items[args.context_idx])]
    else:
        all_items = load_data(args.data_dir, args.dataset, args.sub_dataset, args.max_test_samples)
        targets = list(enumerate(all_items))

    # Process each context
    all_results = []
    metrics_agg = defaultdict(list)

    for idx, item in targets:
        results = process_context(idx, item, args, llm_client)
        all_results.extend(results)

        for r in results:
            for key in ["exact_match", "f1", "substring_exact_match", "eventqa_recall",
                        "rouge1_f1", "rougeL_f1", "input_len", "output_len",
                        "memory_construction_time", "query_time_len"]:
                if key in r:
                    metrics_agg[key].append(r[key])

        # Save incrementally
        out_dir = os.path.join(args.output_dir, args.dataset)
        os.makedirs(out_dir, exist_ok=True)
        suffix = f"_ctx{args.context_idx}" if args.context_idx >= 0 else ""
        out_path = os.path.join(out_dir, f"{args.sub_dataset}{suffix}_lycheemem_results.json")

        averaged = {
            k: np.mean(v) * (1 if ("_len" in k or "_time" in k) else 100)
            for k, v in metrics_agg.items()
        }

        output_data = {
            "agent_config": {
                "agent_name": "Structure_rag_lycheemem",
                "model": args.llm_model,
                "temperature": args.temperature,
                "retrieve_num": args.retrieve_num,
                "agent_chunk_size": args.chunk_size,
            },
            "dataset_config": {
                "dataset": args.dataset,
                "sub_dataset": args.sub_dataset,
                "chunk_size": args.chunk_size,
                "max_test_samples": args.max_test_samples,
                "generation_max_length": args.generation_max_length,
            },
            "data": all_results,
            "metrics": {k: list(v) for k, v in metrics_agg.items()},
            "averaged_metrics": averaged,
        }
        with open(out_path, "w") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Final summary
    if all_results:
        print(f"\n{'='*70}")
        print(f"FINAL RESULTS: {args.sub_dataset}")
        print(f"{'='*70}")
        for k, v in sorted(metrics_agg.items()):
            if "_len" not in k and "_time" not in k:
                print(f"  {k}: {np.mean(v)*100:.1f}%")
            else:
                print(f"  {k}: {np.mean(v):.1f}")
        print(f"  Total queries: {len(all_results)}")
        print(f"  Results saved: {out_path}")


if __name__ == "__main__":
    main()

