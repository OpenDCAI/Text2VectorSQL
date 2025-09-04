import argparse
import json
import re
import os # <-- Import the os module
from dotenv import load_dotenv # <-- Import the load_dotenv function
from tqdm import tqdm
from openai import OpenAI
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Load environment variables from a .env file
load_dotenv()

# 缓存装饰器，最多缓存10000个结果
@lru_cache(maxsize=10000)
def cached_llm_call(model: str, prompt: str, api_url: str, api_key: str) -> str:
    """
    Cached version of the LLM call to avoid redundant requests for same prompts.
    使用OpenAI Python SDK 1.0.0+版本的新API调用方式
    """
    # 创建客户端实例
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
        # Extract the last SQL query in the response text and remove extra whitespace characters
        last_sql = sql_blocks[-1].strip()
        return last_sql
    else:
        print("No SQL blocks found.")
        return ""

def llm_inference(model: str, prompts: list, db_ids: list, api_url: str, api_key: str, parallel: bool = True) -> list:
    """
    Generates responses using an LLM for given prompts with caching and parallel execution.

    Args:
        model: The LLM to use for generating responses.
        prompts (list of str): A list of prompts for the model.
        db_ids (list of str): A list of database IDs corresponding to each prompt.
        api_url (str): OpenAI API URL
        api_key (str): OpenAI API key
        parallel (bool): Whether to use parallel execution

    Returns:
        list of dict: A list of dictionaries containing the prompt, db_id, and generated response.
    """
    
    def process_item(prompt, db_id):
        response = cached_llm_call(model, prompt, api_url, api_key)
        return {
            "prompt": prompt,
            "db_id": db_id,
            "response": response
        }
    
    if parallel:
        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=32) as executor:
            results = list(tqdm(
                executor.map(process_item, prompts, db_ids),
                total=len(prompts),
                desc="Generating responses"
            ))
    else:
        # 顺序处理
        results = []
        for prompt, db_id in tqdm(zip(prompts, db_ids), total=len(prompts), desc="Generating responses"):
            results.append(process_item(prompt, db_id))
    
    return results

if __name__ == '__main__':
    # --- MODIFICATION START ---
    # The argparse section is modified to remove arguments that are now in the .env file.
    
    parser = argparse.ArgumentParser()
    # These two arguments are kept as they are file paths, which are better suited for command-line arguments.
    parser.add_argument("--input_file", type=str, default="./prompts/sql_synthesis_prompts.json", 
                        help="Input JSON file with prompts")
    parser.add_argument("--output_file", type=str, default="./results/sql_synthesis.json", 
                        help="Output JSON file for results")
    
    # The following arguments are removed because they will be loaded from the .env file:
    # parser.add_argument("--model", type=str, required=True, help="OpenAI model name")
    # parser.add_argument("--api_url", type=str, default="", help="OpenAI API URL")
    # parser.add_argument("--api_key", type=str, required=True, help="OpenAI API key")
    # parser.add_argument("--no_parallel", action="store_true", help="Disable parallel execution")
    
    opt = parser.parse_args()

    # Load configuration from environment variables
    api_key = os.getenv("API_KEY")
    api_url = os.getenv("BASE_URL")
    model_name = os.getenv("LLM_MODEL_NAME")
    
    # Handle the boolean value for NO_PARALLEL
    # os.getenv returns a string, so we compare it to 'true'
    no_parallel_str = os.getenv("NO_PARALLEL", "false").lower()
    no_parallel = no_parallel_str == 'true'

    # Add a check to ensure required variables are set
    if not api_key or not model_name:
        raise ValueError("Error: API_KEY and LLM_MODEL_NAME must be set in your .env file.")

    print("--- Configuration Loaded from .env ---")
    print(f"Model: {model_name}")
    print(f"API URL: {api_url}")
    print(f"Parallel Execution Enabled: {not no_parallel}")
    print("--------------------------------------")
    print(f"Input File: {opt.input_file}")
    print(f"Output File: {opt.output_file}")
    # --- MODIFICATION END ---
    
    # 确保输出目录存在
    Path(opt.output_file).parent.mkdir(parents=True, exist_ok=True)
    
    input_dataset = json.load(open(opt.input_file, encoding="utf-8"))
    
    db_ids = [data["db_id"] for data in input_dataset]
    prompts = [data["prompt"] for data in input_dataset]
    
    results = llm_inference(
        model=model_name, # Use variable from .env
        prompts=prompts,
        db_ids=db_ids,
        api_url=api_url, # Use variable from .env
        api_key=api_key, # Use variable from .env
        parallel=not no_parallel # Use boolean variable from .env
    )

    with open(opt.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
