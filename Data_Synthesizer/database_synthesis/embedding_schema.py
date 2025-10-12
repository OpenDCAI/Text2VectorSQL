import argparse
import concurrent.futures
import json
import os
import re
from functools import partial
from json_repair import json_repair
from tqdm import tqdm
import hashlib
from tenacity import retry, stop_after_attempt, wait_exponential
import openai

# 配置缓存目录
CACHE_DIR = "./cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_key(prompt, model):
    """生成基于提示内容和模型名称的唯一缓存键"""
    key_str = f"{model}_{prompt}"
    return hashlib.md5(key_str.encode('utf-8')).hexdigest()

def load_from_cache(cache_key):
    """从缓存加载数据"""
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def save_to_cache(cache_key, data):
    """保存数据到缓存"""
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def parse_response(response):
    """保持原有解析函数不变"""
    schema_pattern = r'```json\s*([\s\S]*?)\s*```'

    try:
        enhanced_schema_match = re.search(schema_pattern, response, re.DOTALL)
        enhanced_schema_str = enhanced_schema_match.group(0).strip() if enhanced_schema_match else None
        enhanced_schema_dict = json_repair.loads(enhanced_schema_str)
        return enhanced_schema_dict
    except Exception as e:
        print(response)
        print("Parsing Exception:", str(e))
        return None

def parse_prompt(prompt):
    """保持原有解析函数不变"""
    domain_pattern = r'(?<=\*\*Business Domain:\*\*)(.*?)(?=\*\*Business Scenario:\*\*)'
    scenario_pattern = r'(?<=\*\*Business Scenario:\*\*)(.*?)(?=\*\*Initial Database Schema:\*\*)'

    domain_match = re.search(domain_pattern, prompt, re.DOTALL)
    domain = domain_match.group(0).strip() if domain_match else None

    scenario_match = re.search(scenario_pattern, prompt, re.DOTALL)
    scenario = scenario_match.group(0).strip() if scenario_match else None

    return domain, scenario

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_response(client, model, prompt):
    """带重试机制的API调用函数（仅修改max_tokens）"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一个专业的数据库架构师，负责生成带有embedding列的数据库模式"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=10000  # 修改为10000
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API调用失败: {str(e)}")
        raise

def process_prompt(prompt, model, client):
    """处理单个提示的核心逻辑"""
    cache_key = get_cache_key(prompt, model)
    
    # 检查缓存
    cached = load_from_cache(cache_key)
    if cached:
        return cached
    
    # 生成响应
    response = generate_response(client, model, prompt)
    
    # 解析响应
    enhanced_schema = parse_response(response)
    if not enhanced_schema:
        return None
    
    # 解析提示
    domain, scenario = parse_prompt(prompt)
    
    # 构建结果
    result = {
        "prompt": prompt,
        "domain": domain,
        "scenario": scenario,
        "enhanced_schema": json.dumps(enhanced_schema, indent=2, ensure_ascii=False),
        "raw_response": response
    }
    
    # 保存到缓存
    save_to_cache(cache_key, result)
    return result

def llm_inference(model, prompts, api_key=None, api_url=None, max_workers=8):
    """
    多线程推理函数（仅修改max_tokens）
    """
    # 初始化OpenAI客户端
    client = openai.OpenAI(
        api_key=api_key,
        base_url=api_url.rstrip('/') if api_url else "https://api.openai.com/v1"
    )
    
    results = []
    
    # 使用线程池并行处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 创建部分函数固定参数
        process_fn = partial(process_prompt, model=model, client=client)
        
        # 提交所有任务
        futures = {executor.submit(process_fn, prompt): prompt for prompt in prompts}
        
        # 使用tqdm显示进度
        for future in tqdm(concurrent.futures.as_completed(futures), 
                          total=len(prompts), 
                          desc="处理进度"):
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                prompt = futures[future]
                print(f"处理失败 - 提示: {prompt[:50]}... 错误: {str(e)}")
    
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--api_key", type=str)
    parser.add_argument("--api_url", type=str)
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--limited_num", type=int, default=0)
    args = parser.parse_args()
    
    # 加载提示
    limited_prompts = json.load(open("./prompts/prompts_schema_embedding.json", encoding='utf-8'))
    
    #限制数据数目
    limited_number = args.limited_num
    if limited_number != 0:
        limited_prompts = limited_prompts[:limited_number] 

    # 执行推理
    results = llm_inference(
        model=args.model,
        prompts=limited_prompts,
        api_key=args.api_key,
        api_url=args.api_url,
        max_workers=args.max_workers
    )
    
    # 保存结果
    output_file = "./results/schema_embedding.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"处理完成，结果已保存到 {output_file}")
