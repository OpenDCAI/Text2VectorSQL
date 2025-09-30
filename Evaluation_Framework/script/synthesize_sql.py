import argparse
import json
import re
import os
from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

# Load environment variables from a .env file
load_dotenv()

# --- 用于线程安全地写入缓存文件的锁 ---
cache_lock = Lock()

# --- 持久化磁盘缓存函数 ---
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

def make_llm_call_openai(model: str, prompt: str, api_url: str, api_key: str) -> str:
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

def make_llm_call_vllm(model: str, prompt: str, api_url: str, api_key: str) -> str:
    """
    实际执行 LLM API 调用的函数。
    此版本经过修改，专门用于连接本地 vLLM OpenAI 兼容服务器。
    """
    # 修改点 1: 初始化 OpenAI 客户端，强制指向本地 vLLM 地址
    # api_key 对于本地服务是必需的，但内容无所谓，随便给一个非空字符串
    # base_url 硬编码或通过环境变量传入你的 vLLM 服务器地址
    client = OpenAI(
        api_key=api_key, # vLLM 不需要 key，但库要求有，所以可以传 "none" 或任何字符串
        base_url=api_url # 这个 URL 必须指向你的 vLLM 服务，例如 "http://localhost:8000/v1"
    )
    
    try:
        # 修改点 2: 'model' 参数现在必须与 vLLM 启动时加载的模型路径完全一致
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8
        )
        return response.choices[0].message.content
    except Exception as e:
        # 保持错误处理不变
        print(f"Error calling local vLLM API: {str(e)}")
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

# --- MODIFIED: 函数签名被修改，以接收一个对象列表 ---
def llm_inference(
    model: str,
    items: list,  # 修改点：不再接收 prompts, query_ids, db_ids，而是接收一个 items 列表
    api_url: str,
    api_key: str,
    cache_file_path: str,
    database_type: str,
    parallel: bool = True,
    use_vllm: bool = False
) -> list:
    """
    使用持久化磁盘缓存生成 LLM 响应。
    现在接收一个对象列表，而不是多个平行列表，以保证数据完整性。
    """
    
    # 1. 加载现有缓存
    cache = load_cache(cache_file_path)
    print(f"Loaded {len(cache)} items from cache file: {cache_file_path}")

    # 2. 筛选出需要新处理的任务
    items_to_process = []
    # 修改点：遍历对象列表，并使用对象内的 prompt 字段进行缓存检查
    for item in items:
        # 假设每个 item 都包含 'sql_synthesis_prompt' 键
        if item.get("sql_synthesis_prompt") not in cache:
            items_to_process.append(item)
    
    print(f"Total items: {len(items)}, To process: {len(items_to_process)}")

    # 3. 定义单个任务的处理函数
    # 修改点：处理函数现在接收整个 item 对象
    def process_item(item):
        prompt = item["sql_synthesis_prompt"]
        if use_vllm:
            response = make_llm_call_vllm(model, prompt, api_url, api_key)
        else:
            response = make_llm_call_openai(model, prompt, api_url, api_key)

        if response:
            save_to_cache(cache_file_path, prompt, response)
        return prompt, response

    # 4. 执行需要处理的任务（并行或顺序）
    if items_to_process:
        if parallel:
            with ThreadPoolExecutor(max_workers=32) as executor:
                # 修改点：executor.map 现在处理的是 item 对象的列表
                results_iterator = list(tqdm(
                    executor.map(process_item, items_to_process),
                    total=len(items_to_process),
                    desc="Generating responses"
                ))
                for prompt, response in results_iterator:
                    cache[prompt] = response
        else:
            for item in tqdm(items_to_process, desc="Generating responses"):
                prompt = item["sql_synthesis_prompt"]
                _, response = process_item(item)
                cache[prompt] = response
    
    # 5. 组装最终结果
    final_results = []
    # 修改点：遍历原始的 items 列表来保证顺序和完整性
    for item in items:
        prompt = item["sql_synthesis_prompt"]

        # 从缓存中获取原始的、可能包含```sql的响应
        raw_response = cache.get(prompt, "")
        
        # 如果响应非空，则调用 parse_response 函数进行清理
        if raw_response:
            parsed_response = parse_response(raw_response)
        else:
            parsed_response = ""

        final_results.append({
            "query_id": item["query_id"],
            "db_id": item["db_id"],
            "db_type": database_type,
            "response": parsed_response
        })

    return final_results

def run_sql_synthesis(
    input_file: str,
    output_file: str,
    model_name: str,
    api_key: str,
    api_url: str,
    cache_file_path: str,
    database_type: str,
    parallel: bool,
    use_vllm: bool
):
    """
    主逻辑函数，现在包含缓存路径。
    """
    if not api_key or not model_name:
        raise ValueError("Error: api_key and model_name must be provided.")

    print("--- Running Synthesis with Configuration ---")
    print(f"Model: {model_name}")
    print(f"API URL: {api_url}")
    print(f"Cache File: {cache_file_path}")
    print(f"Parallel Execution Enabled: {parallel}")
    print(f"Input File: {input_file}")
    print(f"Output File: {output_file}")
    print("------------------------------------------")
    
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    Path(cache_file_path).parent.mkdir(parents=True, exist_ok=True)
    
    input_dataset = json.load(open(input_file, encoding="utf-8"))
    
    # MODIFIED: 不再拆分列表，直接传递整个 input_dataset
    results = llm_inference(
        model=model_name,
        items=input_dataset, # 修改点：传递完整的对象列表
        api_url=api_url,
        api_key=api_key,
        cache_file_path=cache_file_path,
        database_type = database_type,
        parallel=parallel,
        use_vllm=use_vllm
    )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nSynthesis complete. Results saved to {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="../sqlite/prompts/sql_generate_prompts.json", 
                        help="Input JSON file with prompts")
    parser.add_argument("--output_file", type=str, default="../sqlite/results/toy_spider/sql_synthesis.json", 
                        help="Output JSON file for results")
    parser.add_argument("--cache_file", type=str, default="../cache/sqlite/synthesis_sql_cache.jsonl",
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
        cache_file_path=opt.cache_file,
        parallel=parallel_execution,
        use_vllm=True
    )
