import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
# 移除了 lru_cache, 引入了 Lock 用于线程安全的文件写入
from typing import List, Dict
from threading import Lock

import openai
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables from a .env file
load_dotenv()

# --- 新增：用于线程安全地写入缓存文件的锁 ---
cache_lock = Lock()

# --- 新增：持久化磁盘缓存函数 ---
def load_cache(cache_file: str) -> dict:
    """
    从 .jsonl 文件加载缓存。
    每一行都是一个独立的 JSON 对象 {"key": prompt, "value": response}。
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
    """
    with cache_lock:
        record = {"key": key, "value": value}
        with open(cache_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

# --- 修改：移除了 @lru_cache 装饰器，并重命名函数 ---
def make_llm_call(model: str, prompt: str, api_url: str, api_key: str) -> str:
    """
    实际执行 LLM API 调用的函数，并增加了对返回结果的健壮性检查。
    """
    client = openai.OpenAI(
        api_key=api_key,
        base_url=api_url if api_url else None
    )
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8
        )
        # --- 新增的健壮性检查 ---
        # 1. 检查 response 对象是否存在
        # 2. 检查 response.choices 列表是否存在且不为空
        if response and response.choices and len(response.choices) > 0:
            # 只有检查通过后，才安全地访问内容
            return response.choices[0].message.content
        else:
            # 如果响应格式不正确，打印警告信息并返回空字符串
            print(f"Warning: Received an invalid or empty response from API. Response: {response}")
            return ""
            
    except Exception as e:
        print(f"Error calling LLM API: {e}")
        return ""

# --- 修改：整合了新的持久化缓存逻辑 ---
def llm_inference(
    model: str, 
    dataset: List[Dict], 
    api_key: str, 
    cache_file_path: str, # 新增缓存文件路径参数
    api_url: str = "", 
    parallel_workers: int = 4
) -> List[Dict]:
    """
    使用持久化磁盘缓存执行 LLM 推理。
    """
    # 1. 加载现有缓存
    cache = load_cache(cache_file_path)
    print(f"Loaded {len(cache)} items from cache file: {cache_file_path}")

    # 2. 筛选出需要新处理的任务和已处理的任务
    items_to_process = []
    final_results = []
    for data in dataset:
        prompt = data["prompt"]
        if prompt in cache:
            # 如果在缓存中，直接添加到最终结果
            final_results.append({**data, "responses": [cache[prompt]]})
        else:
            # 否则，添加到待处理列表
            items_to_process.append(data)
    
    print(f"Total items: {len(dataset)}, To process: {len(items_to_process)}")

    # 3. 定义单个任务的处理函数
    def process_item(data: Dict) -> Dict:
        prompt = data["prompt"]
        response = make_llm_call(model, prompt, api_url, api_key)
        # 成功后立刻写入缓存
        if response:
            save_to_cache(cache_file_path, prompt, response)
        # 返回包含响应的完整数据
        return {**data, "responses": [response]}
    
    # 4. 执行需要处理的任务
    if items_to_process:
        newly_processed_results = []
        if parallel_workers > 1:
            with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
                newly_processed_results = list(tqdm(
                    executor.map(process_item, items_to_process),
                    total=len(items_to_process),
                    desc="Generating responses"
                ))
        else:
            newly_processed_results = [process_item(data) for data in tqdm(items_to_process, desc="Generating responses")]
        
        # 5. 将新处理的结果与已在缓存中的结果合并
        final_results.extend(newly_processed_results)
    
    return final_results

def synthesize_questions(
    input_file: str,
    output_file: str,
    model_name: str,
    api_key: str,
    api_url: str,
    max_workers: int,
    cache_file_path: str # 新增缓存文件路径参数
):
    """
    主逻辑函数，现在包含缓存路径。
    """
    if not api_key or not model_name:
        raise ValueError("Error: api_key and model_name must be provided.")
    
    print("--- Running Synthesis with Configuration ---")
    print(f"Model: {model_name}")
    print(f"API URL: {api_url}")
    print(f"Max Workers: {max_workers}")
    print(f"Cache File: {cache_file_path}")
    print(f"Input File: {input_file}")
    print(f"Output File: {output_file}")
    print("------------------------------------------")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    os.makedirs(os.path.dirname(cache_file_path), exist_ok=True)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        input_dataset = json.load(f)
    
    results = llm_inference(
        model=model_name,
        dataset=input_dataset,
        api_key=api_key,
        api_url=api_url,
        cache_file_path=cache_file_path, # 传递缓存路径
        parallel_workers=max_workers
    )
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nSynthesis complete. Results saved to {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run LLM inference for question synthesis.")
    parser.add_argument("--input_file", type=str, default="./prompts/question_synthesis_prompts.json")
    parser.add_argument("--output_file", type=str, default="./results/question_synthesis.json")
    # 新增缓存文件路径的命令行参数
    parser.add_argument("--cache_file", type=str, default="./cache/question_synthesis_cache.jsonl",
                        help="Path to the persistent cache file")
    
    opt = parser.parse_args()

    # 从环境变量加载配置
    api_key_env = os.getenv("API_KEY")
    api_url_env = os.getenv("BASE_URL")
    model_name_env = os.getenv("LLM_MODEL_NAME")
    max_workers_env = int(os.getenv("MAX_WORKERS", 32)) # 增加了默认的并行数

    synthesize_questions(
        input_file=opt.input_file,
        output_file=opt.output_file,
        model_name=model_name_env,
        api_key=api_key_env,
        api_url=api_url_env,
        max_workers=max_workers_env,
        cache_file_path=opt.cache_file # 传递缓存路径
    )
