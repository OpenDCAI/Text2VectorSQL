import argparse
import json
import os
import re
import time
import json_repair
import openai
from tqdm import tqdm
import random
from tenacity import retry, stop_after_attempt, wait_exponential

import concurrent.futures
from functools import partial

# 新增缓存目录常量
CACHE_DIR = "./cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def parse_response(response):
    domain_pattern = r'(?<=\[START_DOMAIN\])(.*?)(?=\[END_DOMAIN\])'
    scenario_pattern = r'(?<=\[START_SCENARIO\])(.*?)(?=\[END_SCENARIO\])'
    schema_pattern = r'(?<=\[START_DATABASE_SCHEMA\])(.*?)(?=\[END_DATABASE_SCHEMA\])'

    try:
        domain_match = re.search(domain_pattern, response, re.DOTALL)
        domain = domain_match.group(0).strip() if domain_match else None

        scenario_match = re.search(scenario_pattern, response, re.DOTALL)
        scenario = scenario_match.group(0).strip() if scenario_match else None

        schema_match = re.search(schema_pattern, response, re.DOTALL)
        schema = schema_match.group(0).strip() if schema_match else None
        schema_dict = json_repair.loads(schema)
        schema = json.dumps(schema_dict, indent=2, ensure_ascii=False)

        return domain, scenario, schema
    except Exception as e:
        print(response)
        print(f"length: {len(response)}")
        print("Parsing Exception:", str(e))
        return None, None, None


def get_cache_filename(prompt, model):
    """生成基于提示内容和模型名称的唯一缓存文件名"""
    import hashlib
    prompt_hash = hashlib.md5(prompt.encode('utf-8')).hexdigest()
    return f"{CACHE_DIR}/{model}_{prompt_hash}.json"

def load_from_cache(prompt, model):
    """尝试从缓存加载结果"""
    cache_file = get_cache_filename(prompt, model)
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def save_to_cache(prompt, model, result):
    """保存结果到缓存"""
    cache_file = get_cache_filename(prompt, model)
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

def llm_inference_openai(model, prompts, api_key, api_url=None, max_tokens=10240, temperature=0.7, max_workers=32):
    """
    改进后的推理函数，带有缓存机制和多线程并行处理
    
    Args:
        model: 模型名称
        prompts: 提示列表
        api_key: OpenAI API密钥
        api_url: API端点URL
        max_tokens: 最大token数
        temperature: 生成温度
        max_workers: 最大线程数
    """
    client = openai.OpenAI(
        api_key=api_key,
        base_url=api_url.rstrip('/') if api_url else "https://api.openai.com"
    )

    @retry(stop=stop_after_attempt(3), 
           wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_response(prompt):
        try:
            # 先检查缓存
            cached = load_from_cache(prompt, model)
            if cached:
                return cached["generated_content"]["response"]
                
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates database schemas."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            content = response.choices[0].message.content
            return content
        except Exception as e:
            print(f"生成响应失败: {str(e)}")
            raise

    def process_prompt(prompt):
        try:
            # 检查是否已有完整结果缓存
            cached_result = load_from_cache(prompt, model)
            if cached_result and all(cached_result["generated_content"].get(k) for k in ["response", "domain", "scenario", "schema"]):
                return cached_result
                
            response = generate_response(prompt)
            domain, scenario, schema = parse_response(response)
            
            if all([domain, scenario, schema]):
                result = {
                    "prompt": prompt,
                    "generated_content": {
                        "response": response,
                        "domain": domain,
                        "scenario": scenario,
                        "schema": schema
                    }
                }
                save_to_cache(prompt, model, result)
                return result
            else:
                print(f"无效响应格式 - 提示: {prompt[:50]}...")
                return None
                
        except Exception as e:
            print(f"处理失败 - 提示: {prompt[:50]}... 错误: {str(e)}")
            return None

    results = []
    
    # 使用线程池并行处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 使用partial固定除prompt外的其他参数
        process_fn = partial(process_prompt)
        
        # 使用tqdm显示进度
        futures = {executor.submit(process_fn, prompt): prompt for prompt in prompts}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(prompts), desc="并行生成响应进度"):
            result = future.result()
            if result:
                results.append(result)
    
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--api_key", type=str)
    parser.add_argument("--api_url", type=str, default=None)
    parser.add_argument("--use_cache", type=bool, default=True, help="是否使用缓存")
    args = parser.parse_args()
    
    # 加载并抽样提示
    prompts = json.load(open("./prompts/prompts_schema_synthesis.json"))
    sample_size = int(len(prompts) * 0.1)
    test_prompts = random.sample(prompts, sample_size)
    
    # 执行推理
    results = llm_inference_openai(args.model, test_prompts, args.api_key, args.api_url)
    
    # 保存最终结果
    output_file = "./results/schema_synthesis.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
