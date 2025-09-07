# filename: llm_inference_env.py
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

# -------------------------- 环境变量读取 --------------------------
# 自动加载同目录或上级目录中的 .env 文件
load_dotenv()

API_KEY        = os.getenv("API_KEY")
BASE_URL       = os.getenv("BASE_URL")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o")
MAX_WORKERS    = int(os.getenv("MAX_WORKERS", 32))
CACHE_FILE_PATH = os.getenv("CACHE_FILE_PATH", "./results/inference_cache.json")

# 固定超参数
NUM_RESPONSES = 5
TEMPERATURE   = 0.8
# -----------------------------------------------------------------

# 创建一个线程锁，用于在多线程环境下安全地写入缓存文件
cache_lock = Lock()

def load_cache(cache_file):
    """加载缓存文件。如果文件不存在或为空，则返回一个空字典。"""
    if not os.path.exists(cache_file):
        return {}
    with open(cache_file, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}

def save_to_cache(cache_file, key, value):
    """将单个键值对安全地保存到缓存文件中。"""
    with cache_lock:
        cache = load_cache(cache_file)
        cache[key] = value
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)

def llm_inference(dataset):
    """
    使用大语言模型为数据集中的每个提示（prompt）生成回应。

    Args:
        dataset: 一个包含词典的列表，每个词典都需要一个 "cot_synthesis_prompt" 键。

    Returns:
        一个包含词典的列表，每个词典都增加了 "responses" 键，其值为生成的文本列表。
    """
    # 1. 初始化 OpenAI 客户端
    try:
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        return []

    # 2. 加载缓存
    cache = load_cache(CACHE_FILE_PATH)
    print(f"Loaded {len(cache)} items from cache file: {CACHE_FILE_PATH}")

    # 3. 筛选出需要新处理的 prompts
    prompts_to_process = []
    for data in dataset:
        prompt = data["cot_synthesis_prompt"]
        if prompt not in cache:
            prompts_to_process.append(prompt)

    print(f"Total prompts: {len(dataset)}, To process: {len(prompts_to_process)}")

    if prompts_to_process:
        # 4. 定义单个 prompt 的处理函数
        def process_prompt(prompt):
            """为单个 prompt 调用 LLM API 并返回结果"""
            system_prompt = (
                "You are an expert SQLite data analyst working with a database that has a vector search extension. "
                "Your task is to generate a step-by-step thinking process (Chain-of-Thought) to arrive at a SQL query for a given question and database schema. "
                "For vector similarity searches, you MUST use the exact syntax: "
                "`vector_column MATCH lembed('model_name', 'search_text') AND k = N`, "
                "where `k` is the number of nearest neighbors to find. Do not use other operators like `<->`. "
                "**Crucially, you MUST conclude your response with the final, complete SQL query enclosed in a markdown "
                "code block like this: ```sql\\n[YOUR SQL QUERY HERE]\\n```**. "
                "Do not include any explanation after the final SQL code block."
            )

            try:
                chat_completion = client.chat.completions.create(
                    model=LLM_MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    n=NUM_RESPONSES,
                    temperature=TEMPERATURE
                )
                responses = [choice.message.content for choice in chat_completion.choices]
                # 成功后写缓存
                save_to_cache(CACHE_FILE_PATH, prompt, responses)
                return prompt, responses
            except Exception as e:
                error_message = f"Error processing prompt: {e}"
                print(error_message)
                return prompt, [error_message] * NUM_RESPONSES

        # 5. 并行处理
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_prompt = {
                executor.submit(process_prompt, prompt): prompt
                for prompt in prompts_to_process
            }

            for future in tqdm(
                as_completed(future_to_prompt),
                total=len(prompts_to_process),
                desc="LLM Inference"
            ):
                prompt, responses = future.result()
                cache[prompt] = responses

    # 6. 组装结果
    results = []
    for data in dataset:
        prompt = data["cot_synthesis_prompt"]
        data["responses"] = cache.get(
            prompt, ["Error: Response not found in cache after processing."]
        )
        results.append(data)

    return results

# -----------------------------------------------------------------
if __name__ == "__main__":
    # 基本路径配置
    input_file  = "./prompts/cot_synthesis_prompts.json"
    output_file = "./results/cot_synthesis.json"

    # 打印当前配置
    print("Running with configuration:")
    print(f"  Model:      {LLM_MODEL_NAME}")
    print(f"  API Key:    {'SET' if API_KEY else 'NOT SET'}")
    print(f"  Base URL:   {BASE_URL}")
    print(f"  Max Workers:{MAX_WORKERS}")
    print(f"  Cache File: {CACHE_FILE_PATH}")
    print(f"  #Responses: {NUM_RESPONSES}")
    print(f"  Temperature:{TEMPERATURE}")

    # 确保目录存在
    os.makedirs(os.path.dirname(output_file),    exist_ok=True)
    os.makedirs(os.path.dirname(CACHE_FILE_PATH), exist_ok=True)

    # 读取输入数据
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            input_dataset = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_file}")
        exit(1)

    # 调用推理
    results = llm_inference(input_dataset)

    # 保存输出
    if results:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nInference complete. Results saved to {output_file}")
