"""
BEAM × LycheeMem Adapter
把 BEAM 对话写入 LycheeMem 记忆，然后基于检索回答 probing questions。

用法:
  python run_beam_lycheemem.py \
    --input_directory chats/100K \
    --chat_size 100K \
    --start_index 0 --end_index 5 \
    --lycheemem_url http://localhost:8000 \
    --reader_url http://10.251.171.6:28043/v1 \
    --reader_model my-llm-qwen \
    --reader_api_key dummy
"""

import argparse
import json
import os
import time
import traceback
import requests
from openai import OpenAI


def clear_memory(base_url: str):
    resp = requests.delete(f"{base_url}/memory/clear", timeout=30)
    resp.raise_for_status()
    return resp.json()


def append_turn(base_url: str, session_id: str, role: str, content: str):
    resp = requests.post(
        f"{base_url}/memory/append-turn",
        json={"session_id": session_id, "role": role, "content": content},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def consolidate(base_url: str, session_id: str, flush_session: bool = True):
    """异步触发 consolidate，通过 token_stats 轮询等待完成。"""
    import time as _time

    # 异步启动
    resp = requests.post(
        f"{base_url}/memory/consolidate",
        json={
            "session_id": session_id,
            "flush_session": flush_session,
            "background": True,
            "force_ingest": True,
        },
        timeout=60,
    )
    resp.raise_for_status()
    print(f"    consolidate triggered (background)")

    # 轮询 token_stats 等待完成（calls 数停止增长）
    stats_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "lycheemem_code", "data", "token_stats.json"
    )
    # 用相对于 lycheemem_code 的路径
    poll_interval = 20
    max_wait = 7200
    waited = 0
    last_calls = -1
    stable_count = 0

    while waited < max_wait:
        _time.sleep(poll_interval)
        waited += poll_interval

        # 尝试用 search 请求探测——如果 consolidate 完成了，search 能快速返回
        try:
            test = requests.post(
                f"{base_url}/memory/search",
                json={"query": "test", "top_k": 1, "include_semantic": True},
                timeout=5,
            )
            if test.status_code == 200:
                data = test.json()
                total = data.get("total", 0)
                if waited % 60 == 0:
                    print(f"    ... waiting ({waited}s, memory records: {total})")
                # 不能靠 search 判断完成，继续等
        except Exception:
            pass

        # 真正的判断：通过检查连接数看 LycheeMem 是否还在调 vLLM
        # 简化做法：等足够长时间后直接返回
        if waited >= 60 and waited % 60 == 0:
            print(f"    ... consolidate running ({waited}s)")

        # 简单策略：等到没有活跃的外部连接（意味着 LLM 调用结束）
        # 但我们没法从客户端检测这个，所以改用固定等待 + 测试
        # 用一个更好的办法：尝试同步 consolidate（如果已经完成会立刻返回 skipped）
        if waited >= 120 and waited % 60 == 0:
            try:
                check = requests.post(
                    f"{base_url}/memory/consolidate",
                    json={
                        "session_id": session_id,
                        "flush_session": False,
                        "background": False,
                        "force_ingest": False,
                    },
                    timeout=10,
                )
                if check.status_code == 200:
                    result = check.json()
                    if result.get("status") == "skipped" or result.get("skipped_reason"):
                        print(f"    consolidate done! ({waited}s)")
                        return result
            except requests.exceptions.ReadTimeout:
                # 还在忙
                pass
            except Exception:
                pass

    print(f"    consolidate wait exceeded {max_wait}s, proceeding")
    return {"status": "timeout"}


def search_memory(base_url: str, query: str, top_k: int = 10):
    resp = requests.post(
        f"{base_url}/memory/search",
        json={"query": query, "top_k": top_k, "include_semantic": True},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def ingest_chat(base_url: str, chat_data: list, chat_size: str, session_id: str = "beam"):
    """将 BEAM chat.json 的对话写入 LycheeMem，整体 append 后统一 consolidate。"""
    total_msgs = 0

    for batch_idx, batch in enumerate(chat_data):
        turns = batch["turns"]
        for turn in turns:
            for msg in turn:
                append_turn(base_url, session_id, msg["role"], msg["content"])
                total_msgs += 1
        print(f"  [ingest] batch {batch_idx+1}/{len(chat_data)} appended ({total_msgs} msgs so far)")

    # 统一一次 consolidate
    print(f"  [ingest] all {total_msgs} msgs appended. Starting consolidate (this takes a while)...")
    result = consolidate(base_url, session_id, flush_session=True)
    print(f"  [ingest] consolidate done: {result.get('status')} | turns_consumed={result.get('turns_consumed', '?')} | facts={result.get('facts_added', '?')}")


def answer_with_lycheemem(
    base_url: str,
    reader_client: OpenAI,
    reader_model: str,
    query: str,
    top_k: int = 10,
) -> str:
    """检索 LycheeMem 然后用 reader 基于上下文答题。"""
    search_resp = search_memory(base_url, query, top_k=top_k)
    context = search_resp.get("raw_retrieved_context", "")

    if not context.strip():
        context = "(No relevant memory found)"

    prompt = (
        "You are a helpful assistant answering questions based on retrieved memory context.\n\n"
        f"### Retrieved Context ###\n{context}\n\n"
        "### Instructions ###\n"
        "Only provide the answer without any explanations.\n"
        "If the information is not available in the context, say 'I don't know'.\n\n"
        f"### Question ###\n{query}"
    )

    resp = reader_client.chat.completions.create(
        model=reader_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=2048,
    )
    return resp.choices[0].message.content.strip()


def process_single_chat(
    chat_dir: str,
    chat_size: str,
    base_url: str,
    reader_client: OpenAI,
    reader_model: str,
    result_save_dir: str,
):
    """处理单个 chat: 清库 → 写入 → 答题 → 保存结果。"""
    chat_file = os.path.join(chat_dir, "chat.json")
    pq_file = os.path.join(chat_dir, "probing_questions", "probing_questions.json")

    if not os.path.exists(pq_file):
        print(f"  [skip] no probing_questions.json in {chat_dir}")
        return

    with open(chat_file, "r") as f:
        chat_data = json.load(f)
    with open(pq_file, "r") as f:
        probing_questions = json.load(f)

    # Step 1: 清除旧记忆
    print("  [clear] clearing memory...")
    clear_memory(base_url)

    # Step 2: 写入对话
    print("  [ingest] ingesting chat...")
    ingest_chat(base_url, chat_data, chat_size)

    # Step 3: 答题
    results = {}
    total_q = sum(len(v) for v in probing_questions.values())
    q_idx = 0
    for q_type, questions in probing_questions.items():
        type_results = []
        for question in questions:
            q_idx += 1
            q_text = question["question"]
            print(f"  [answer] ({q_idx}/{total_q}) {q_type}: {q_text[:60]}...")
            try:
                answer = answer_with_lycheemem(
                    base_url, reader_client, reader_model, q_text
                )
            except Exception as e:
                answer = f"Error: {e}"
                print(f"    [error] {e}")
            obj = dict(question)
            obj["llm_response"] = answer
            type_results.append(obj)
        results[q_type] = type_results

    # Step 4: 保存
    os.makedirs(result_save_dir, exist_ok=True)
    output_file = os.path.join(result_save_dir, "lycheemem-results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"  [saved] {output_file}")


def batch_run(args):
    """按 chat index 范围批量跑。"""
    input_dir = args.input_directory
    entries = os.listdir(input_dir)
    dirs = sorted(
        [name for name in entries if os.path.isdir(os.path.join(input_dir, name)) and name.isdigit()],
        key=lambda x: int(x),
    )

    reader_client = OpenAI(
        api_key=args.reader_api_key,
        base_url=args.reader_url,
    )

    start = time.time()

    for idx in range(args.start_index, min(args.end_index, len(dirs))):
        dirname = dirs[idx]
        chat_dir = os.path.join(input_dir, dirname)
        result_save_dir = os.path.join("results", args.chat_size, dirname)

        print(f"\n{'='*60}")
        print(f"Processing chat {dirname} ({idx+1-args.start_index}/{args.end_index-args.start_index})")
        print(f"{'='*60}")

        try:
            process_single_chat(
                chat_dir=chat_dir,
                chat_size=args.chat_size,
                base_url=args.lycheemem_url,
                reader_client=reader_client,
                reader_model=args.reader_model,
                result_save_dir=result_save_dir,
            )
        except Exception as e:
            print(f"  [FATAL] {dirname}: {e}")
            traceback.print_exc()

    elapsed = int(time.time() - start)
    h, m, s = elapsed // 3600, (elapsed % 3600) // 60, elapsed % 60
    print(f"\nTotal elapsed: {h:02}:{m:02}:{s:02}")


def parse_args():
    parser = argparse.ArgumentParser(description="BEAM × LycheeMem evaluation")
    parser.add_argument("--input_directory", type=str, required=True,
                        help="chats directory, e.g. chats/100K")
    parser.add_argument("--chat_size", type=str, required=True,
                        help="100K, 500K, 1M, or 10M")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, required=True)
    parser.add_argument("--lycheemem_url", type=str, default="http://localhost:8000",
                        help="LycheeMem API base URL")
    parser.add_argument("--reader_url", type=str, required=True,
                        help="Reader LLM OpenAI-compatible base URL")
    parser.add_argument("--reader_model", type=str, required=True,
                        help="Reader model name")
    parser.add_argument("--reader_api_key", type=str, default="dummy",
                        help="Reader API key")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Number of memory results to retrieve")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    batch_run(args)
