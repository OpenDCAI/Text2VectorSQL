import argparse
import json
import re
import os
from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI
# 移除了 lru_cache，引入了 Lock 用于线程安全的文件写入
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

# Load environment variables from a .env file
load_dotenv()

# --- 新增：用于线程安全地写入缓存文件的锁 ---
cache_lock = Lock()

# --- 新增：持久化磁盘缓存函数 ---
def load_cache(cache_file: str) -> dict:
    """
    从 .jsonl 文件加载缓存。
    每一行都是一个独立的 JSON 对象 {"key": prompt, "value": response}。
    这种方式可以抵抗因程序中断导致的文件损坏。
    """
    if not os.path.exists(cache_file):
        return {}
    
    cache = {}
    with open(cache_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if "key" in record and "value" in record:
                    cache[record["key"]] = record["value"]
            except json.JSONDecodeError:
                print(f"Skipping corrupted line in cache file: {line}")
    return cache

def save_to_cache(cache_file: str, key: str, value: str):
    """
    将单个键值对安全地以 .jsonl 格式追加到缓存文件中。
    使用追加模式 ('a')，既高效又安全。
    """
    with cache_lock:
        record = {"key": key, "value": value}
        with open(cache_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

# --- 修改：移除了 @lru_cache 装饰器 ---
def make_llm_call(model: str, prompt: str, api_url: str, api_key: str) -> str:
    """
    实际执行 LLM API 调用的函数。
    """
    client = OpenAI(
        api_key=api_key,
        base_url=api_url if api_url else None
    )
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling LLM API: {str(e)}")
        return ""

def parse_response(response):
    pattern = r"```sql\s*(.*?)\s*```"
    
    sql_blocks = re.findall(pattern, response, re.DOTALL)

    if sql_blocks:
        last_sql = sql_blocks[-1].strip()
        return last_sql
    else:
        print("No SQL blocks found.")
        return ""

# --- 修改：整合了新的缓存逻辑 ---
def llm_inference(
    model: str, 
    items: list, 
    api_url: str, 
    api_key: str, 
    cache_file_path: str, # 新增缓存文件路径参数
    parallel: bool = True
) -> list:
    """
    使用持久化磁盘缓存生成 LLM 响应。
    """
    
    # 1. 加载现有缓存
    cache = load_cache(cache_file_path)
    print(f"Loaded {len(cache)} items from cache file: {cache_file_path}")

    # 2. 筛选出需要新处理的任务
    items_to_process = []

    for item in items:
        # 假设每个 item 都包含 'sql_synthesis_prompt' 键
        if item.get("prompt") not in cache:
            items_to_process.append(item)
    
    print(f"Total items: {len(items)}, To process: {len(items_to_process)}")

    # 3. 定义单个任务的处理函数
    def process_item(item):
        prompt = item["prompt"]
        # 调用 API
        response = make_llm_call(model, prompt, api_url, api_key)
        # 成功后立刻写入缓存
        if response:
            save_to_cache(cache_file_path, prompt, response)
        return prompt, response

    # 4. 执行需要处理的任务（并行或顺序）
    if items_to_process:
        if parallel:
            with ThreadPoolExecutor(max_workers=32) as executor:
                # 使用 list 包装 tqdm 以立即显示进度条
                results_iterator = list(tqdm(
                    executor.map(process_item, items_to_process),
                    total=len(items_to_process),
                    desc="Generating responses"
                ))
                # 将新结果更新到内存缓存中
                for prompt, response in results_iterator:
                    cache[prompt] = response
        else:
            for prompt in tqdm(items_to_process, desc="Generating responses"):
                _, response = process_item(prompt)
                cache[prompt] = response
    
    # 5. 组装最终结果
    final_results = []
    for item in items:
        prompt = item["prompt"]
        final_results.append({
            "prompt": prompt,
            "db_id": item["db_id"],
            "response": cache.get(prompt, "") # 从更新后的缓存中获取结果
        })

    return final_results

def run_sql_synthesis(
    input_file: str,
    output_file: str,
    model_name: str,
    api_key: str,
    api_url: str,
    cache_file_path: str, # 新增缓存文件路径参数
    parallel: bool
):
    """
    主逻辑函数，现在包含缓存路径。
    """
    if not api_key or not model_name:
        raise ValueError("Error: api_key and model_name must be provided.")

    print("--- Running Synthesis with Configuration ---")
    print(f"Model: {model_name}")
    print(f"API URL: {api_url}")
    print(f"Cache File: {cache_file_path}") # 打印缓存文件路径
    print(f"Parallel Execution Enabled: {parallel}")
    print(f"Input File: {input_file}")
    print(f"Output File: {output_file}")
    print("------------------------------------------")
    
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    # 确保缓存目录也存在
    Path(cache_file_path).parent.mkdir(parents=True, exist_ok=True)
    
    input_dataset = json.load(open(input_file, encoding="utf-8"))
    
    results = llm_inference(
        model=model_name,
        items=input_dataset,
        api_url=api_url,
        api_key=api_key,
        cache_file_path=cache_file_path, # 传递缓存路径
        parallel=parallel
    )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nSynthesis complete. Results saved to {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="./prompts/sql_synthesis_prompts.json", 
                        help="Input JSON file with prompts")
    parser.add_argument("--output_file", type=str, default="./results/sql_synthesis.json", 
                        help="Output JSON file for results")
    # 新增缓存文件路径的命令行参数
    parser.add_argument("--cache_file", type=str, default="./cache/synthesis_cache.jsonl",
                        help="Path to the persistent cache file")
    opt = parser.parse_args()

    # 从环境变量加载配置
    api_key_env = os.getenv("API_KEY")
    api_url_env = os.getenv("BASE_URL")
    model_name_env = os.getenv("LLM_MODEL_NAME")
    
    no_parallel_str = os.getenv("NO_PARALLEL", "false").lower()
    parallel_execution = not (no_parallel_str == 'true')

    run_sql_synthesis(
        input_file=opt.input_file,
        output_file=opt.output_file,
        model_name=model_name_env,
        api_key=api_key_env,
        api_url=api_url_env,
        cache_file_path=opt.cache_file, # 传递缓存路径
        parallel=parallel_execution
    )
