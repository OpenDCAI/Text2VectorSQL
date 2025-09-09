import argparse
import json
import re
import os
from dotenv import load_dotenv
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

def run_sql_synthesis(
    input_file: str,
    output_file: str,
    model_name: str,
    api_key: str,
    api_url: str,
    parallel: bool
):
    """
    Runs the main logic for SQL synthesis from a dataset.

    Args:
        input_file (str): Path to the input JSON file with prompts.
        output_file (str): Path to the output JSON file for results.
        model_name (str): Name of the LLM model to use.
        api_key (str): API key for the LLM service.
        api_url (str): Base URL for the LLM API.
        parallel (bool): Flag to enable or disable parallel execution.
    """
    # Add a check to ensure required variables are set
    if not api_key or not model_name:
        raise ValueError("Error: api_key and model_name must be provided.")

    print("--- Running Synthesis with Configuration ---")
    print(f"Model: {model_name}")
    print(f"API URL: {api_url}")
    print(f"Parallel Execution Enabled: {parallel}")
    print(f"Input File: {input_file}")
    print(f"Output File: {output_file}")
    print("------------------------------------------")
    
    # 确保输出目录存在
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    input_dataset = json.load(open(input_file, encoding="utf-8"))
    
    db_ids = [data["db_id"] for data in input_dataset]
    prompts = [data["prompt"] for data in input_dataset]
    
    results = llm_inference(
        model=model_name,
        prompts=prompts,
        db_ids=db_ids,
        api_url=api_url,
        api_key=api_key,
        parallel=parallel
    )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nSynthesis complete. Results saved to {output_file}")


if __name__ == '__main__':
    # This block now handles command-line parsing and environment variable loading,
    # then calls the main logic function with the collected configuration.
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="./prompts/sql_synthesis_prompts.json", 
                        help="Input JSON file with prompts")
    parser.add_argument("--output_file", type=str, default="./results/sql_synthesis.json", 
                        help="Output JSON file for results")
    opt = parser.parse_args()

    # Load configuration from environment variables
    api_key_env = os.getenv("API_KEY")
    api_url_env = os.getenv("BASE_URL")
    model_name_env = os.getenv("LLM_MODEL_NAME")
    
    # Handle the boolean value for NO_PARALLEL from environment variables
    no_parallel_str = os.getenv("NO_PARALLEL", "false").lower()
    parallel_execution = not (no_parallel_str == 'true')

    # Call the main function with the loaded and parsed configuration
    run_sql_synthesis(
        input_file=opt.input_file,
        output_file=opt.output_file,
        model_name=model_name_env,
        api_key=api_key_env,
        api_url=api_url_env,
        parallel=parallel_execution
    )
