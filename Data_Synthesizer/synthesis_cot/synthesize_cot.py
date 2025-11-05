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
# 这样，如果没有通过参数传递 API_KEY 和 BASE_URL，代码仍然可以尝试从 .env 文件中获取它们
load_dotenv()
# -----------------------------------------------------------------

# 创建一个线程锁，用于在多线程环境下安全地写入缓存文件
cache_lock = Lock()

def load_cache(cache_file):
    """
    加载 .jsonl 格式的缓存文件。
    每一行都是一个独立的 JSON 对象 {"key": "...", "value": ...}。
    """
    if not os.path.exists(cache_file):
        return {}
    
    cache = {}
    with open(cache_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                # 移除行尾的换行符并解析
                record = json.loads(line.strip())
                # 假设每行的格式是 {"key": prompt, "value": responses}
                if "key" in record and "value" in record:
                    cache[record["key"]] = record["value"]
            except json.JSONDecodeError:
                # 忽略损坏的行（通常是最后一行）
                print(f"Skipping corrupted line in cache file: {line.strip()}")
    return cache

def save_to_cache(cache_file, key, value):
    """
    将单个键值对安全地以 .jsonl 格式追加到缓存文件中。
    """
    with cache_lock:
        # 创建一个包含键和值的记录
        record = {"key": key, "value": value}
        # 使用追加模式 'a' 写入
        with open(cache_file, 'a', encoding='utf-8') as f:
            # 将记录转换为 JSON 字符串并写入，然后添加换行符
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

def llm_inference(dataset, api_key, base_url, llm_model_name, max_workers, cache_file_path, num_responses, temperature, db_type="sqlite"):
    """
    使用大语言模型为数据集中的每个提示（prompt）生成回应。

    Args:
        dataset: 一个包含词典的列表，每个词典都需要一个 "cot_synthesis_prompt" 键。
        api_key (str): OpenAI API 密钥。
        base_url (str): OpenAI API 的基础 URL。
        llm_model_name (str): 要使用的模型名称。
        max_workers (int): 并行处理的最大线程数。
        cache_file_path (str): 缓存文件的路径。
        num_responses (int): 每个提示要生成的回应数量。
        temperature (float): 生成文本时的温度参数。

    Returns:
        一个包含词典的列表，每个词典都增加了 "responses" 键，其值为生成的文本列表。
    """
    # 1. 初始化 OpenAI 客户端
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        return []

    # 2. 加载缓存
    cache = load_cache(cache_file_path)
    print(f"Loaded {len(cache)} items from cache file: {cache_file_path}")

    # 3. 筛选出需要新处理的 prompts
    prompts_to_process = []
    for data in dataset:
        if "cot_synthesis_prompt" not in data:
            print("Warning: 'cot_synthesis_prompt' key not found in dataset item. Skipping.")
            continue
        prompt = data["cot_synthesis_prompt"]
        if prompt not in cache:
            prompts_to_process.append(prompt)

    print(f"Total prompts: {len(dataset)}, To process: {len(prompts_to_process)}")

    if prompts_to_process:
        # 4. 定义单个 prompt 的处理函数
        def process_prompt(prompt):
            """为单个 prompt 调用 LLM API 并返回结果"""
            if db_type == "sqlite":
                system_prompt = (
                    "You are an expert SQLite data analyst working with a database that has a vector search extension. "
                    "Your task is to generate a step-by-step thinking process (Chain-of-Thought) to arrive at a SQL query for a given question and database schema. "
                    "For vector similarity searches, you MUST use the exact syntax: "
                    "`vector_column MATCH lembed('embed_model_name', 'search_text') AND k = N`, "
                    "where `k` is the number of nearest neighbors to find. Do not use other operators like `<->`. "
                    "**Crucially, you MUST conclude your response with the final, complete SQL query enclosed in a markdown "
                    "code block like this: ```sql\\n[YOUR SQL QUERY HERE]\\n```**. "
                    "Do not include any explanation after the final SQL code block."
                )
            elif db_type == "postgresql":
                system_prompt = (
                    "You are an expert PostgreSQL data analyst working with a database that has the `pgvector` extension. "
                    "Your task is to generate a step-by-step thinking process (Chain-of-Thought) to arrive at a SQL query for a given question and database schema. "
                    "For vector similarity searches, you MUST use the L2 distance operator `<->`. The query structure must calculate the distance in the SELECT clause and use it for ordering: "
                    "`SELECT ..., vector_column <-> lembed('embed_model_name', 'search_text') AS distance FROM table_name ORDER BY distance LIMIT N`. "
                    "**Crucially, you MUST conclude your response with the final, complete SQL query enclosed in a markdown "
                    "code block like this: ```sql\n[YOUR SQL QUERY HERE]\n```**. "
                    "Do not include any explanation after the final SQL code block."
                )
            elif db_type == "clickhouse":
                system_prompt = (
                    "You are an expert ClickHouse data analyst working with a database that has built-in vector search capabilities. "
                    "Your task is to generate a step-by-step thinking process (Chain-of-Thought) to arrive at a SQL query for a given question and database schema. "
                    "For vector similarity searches, you MUST use a `WITH` clause for the reference vector and a distance function (e.g., `L2Distance`). The query syntax is: "
                    "`WITH lembed('embed_model_name', 'search_text') AS ref_vec SELECT ..., L2Distance(vector_column, ref_vec) AS distance FROM table_name ORDER BY distance LIMIT N`. "
                    "**Crucially, you MUST conclude your response with the final, complete SQL query enclosed in a markdown "
                    "code block like this: ```sql\n[YOUR SQL QUERY HERE]\n```**. "
                    "Do not include any explanation after the final SQL code block."
                )
            elif db_type == "myscale":
                system_prompt = (
                    "You are an expert MyScale data analyst working with a database that has built-in vector search capabilities. "
                    "Your task is to generate a step-by-step thinking process (Chain-of-Thought) to arrive at a SQL query for a given question and database schema. "
                    "For vector similarity searches, you MUST use a `WITH` clause for the reference vector and the `distance` function (e.g., `distance(vector_column, ref_vec)`). The query syntax is: "
                    "`WITH lembed('embed_model_name', 'search_text') AS ref_vec SELECT ..., distance(vector_column, ref_vec) AS distance FROM table_name ORDER BY distance LIMIT N`. "
                    "**Crucially, you MUST conclude your response with the final, complete SQL query enclosed in a markdown "
                    "code block like this: ```sql\n[YOUR SQL QUERY HERE]\n```**. "
                    "Do not include any explanation after the final SQL code block."
                )

            try:
                chat_completion = client.chat.completions.create(
                    model=llm_model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    n=num_responses,
                    temperature=temperature
                )
                responses = [choice.message.content for choice in chat_completion.choices]
                # 成功后写缓存
                save_to_cache(cache_file_path, prompt, responses)
                return prompt, responses
            except Exception as e:
                error_message = f"Error processing prompt: {e}"
                print(error_message)
                return prompt, [error_message] * num_responses

        # 5. 并行处理
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
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
        if "cot_synthesis_prompt" not in data:
            continue
        prompt = data["cot_synthesis_prompt"]
        data["responses"] = cache.get(
            prompt, ["Error: Response not found in cache after processing."]
        )
        results.append(data)

    return results

# -----------------------------------------------------------------
def synthesize_cot(
    input_file: str,
    output_file: str,
    llm_model_name: str,
    api_key: str,
    base_url: str,
    max_workers: int,
    cache_file_path: str,
    num_responses: int,
    temperature: float,
    db_type: str = "sqlite"
):
    """
    运行完整的 CoT 合成流程，从读取输入文件到调用 LLM 推理，最后保存结果。

    Args:
        input_file (str): 包含 prompts 的输入 JSON 文件路径。
        output_file (str): 保存结果的输出 JSON 文件路径。
        llm_model_name (str): 要使用的模型名称。
        api_key (str): OpenAI API 密钥。
        base_url (str): OpenAI API 的基础 URL。
        max_workers (int): 并行处理的最大线程数。
        cache_file_path (str): 缓存文件的路径。
        num_responses (int): 每个提示要生成的回应数量。
        temperature (float): 生成文本时的温度参数。
    """
    # 打印当前配置
    print("Running with configuration:")
    print(f"  Model:      {llm_model_name}")
    print(f"  API Key:    {'SET' if api_key else 'NOT SET'}")
    print(f"  Base URL:   {base_url}")
    print(f"  Max Workers:{max_workers}")
    print(f"  Cache File: {cache_file_path}")
    print(f"  #Responses: {num_responses}")
    print(f"  Temperature:{temperature}")
    print(f"  db_type:    {db_type}")

    # 确保目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    os.makedirs(os.path.dirname(cache_file_path), exist_ok=True)

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
    results = llm_inference(
        dataset=input_dataset,
        api_key=api_key,
        base_url=base_url,
        llm_model_name=llm_model_name,
        max_workers=max_workers,
        cache_file_path=cache_file_path,
        num_responses=num_responses,
        temperature=temperature,
        db_type=db_type
    )

    # 保存输出
    if results:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nInference complete. Results saved to {output_file}")


# 示例：如何调用这个修改后的函数
if __name__ == '__main__':
    # 您现在可以从任何地方（例如，配置文件、命令行参数、环境变量）获取这些值
    # 并将它们作为参数传递给 synthesize_cot 函数。
    # 这里我们仍然从环境变量中读取，以保持原始脚本的执行行为。
    
    # 基本路径配置
    INPUT_FILE  = "./prompts/cot_synthesis_prompts.json"
    OUTPUT_FILE = "./results/cot_synthesis.json"

    # 从环境变量加载配置，并提供默认值
    API_KEY        = os.getenv("API_KEY")
    BASE_URL       = os.getenv("BASE_URL")
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o")
    MAX_WORKERS    = int(os.getenv("MAX_WORKERS", 32))
    CACHE_FILE_PATH = os.getenv("CACHE_FILE_PATH", "./results/inference_cache.json")
    
    # 固定超参数
    NUM_RESPONSES = 5
    TEMPERATURE   = 0.8
    
    # 检查必要的环境变量是否已设置
    if not API_KEY or not BASE_URL:
        print("Error: API_KEY and BASE_URL must be set as environment variables or passed as arguments.")
    else:
        synthesize_cot(
            input_file=INPUT_FILE,
            output_file=OUTPUT_FILE,
            llm_model_name=LLM_MODEL_NAME,
            api_key=API_KEY,
            base_url=BASE_URL,
            max_workers=MAX_WORKERS,
            cache_file_path=CACHE_FILE_PATH,
            num_responses=NUM_RESPONSES,
            temperature=TEMPERATURE
        )
