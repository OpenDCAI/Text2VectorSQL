import argparse
import json
import re
from tqdm import tqdm
from openai import OpenAI
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import logging
import traceback

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 缓存装饰器
@lru_cache(maxsize=2000000)
def cached_llm_call(model: str, prompt: str, api_url: str, api_key: str) -> str:
    """
    Cached version of the LLM call to avoid redundant requests for same prompts.
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
        logger.error(f"Error calling LLM API: {str(e)}")
        return ""

def parse_response(response: str) -> dict:
    """
    解析大模型响应，提取JSON对象
    """
    # 尝试直接解析为JSON
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    
    # 尝试从代码块中提取JSON
    pattern = r"```json\s*(.*?)\s*```"
    json_blocks = re.findall(pattern, response, re.DOTALL)
    
    if json_blocks:
        try:
            return json.loads(json_blocks[-1].strip())
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析错误: {str(e)}")
    
    # 尝试提取纯JSON内容
    try:
        # 查找第一个{和最后一个}之间的内容
        start = response.find('{')
        end = response.rfind('}')
        if start != -1 and end != -1 and end > start:
            return json.loads(response[start:end+1])
    except Exception:
        pass
    
    logger.error("无法从响应中提取有效的JSON")
    return {}

def process_db_info(db_info: dict, model: str, api_url: str, api_key: str, prompt_template: str) -> dict:
    """
    处理单个数据库信息
    """
    # 生成提示
    prompt = prompt_template.format(dababase_schema=json.dumps(db_info, ensure_ascii=False))
    
    # 调用大模型
    response = cached_llm_call(model, prompt, api_url, api_key)
    
    # 解析响应
    parsed_response = parse_response(response)
    
    # 将解析结果添加到db_info
    db_info["column_alter"] = parsed_response
    
    return db_info

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="OpenAI model name")
    parser.add_argument("--api_url", type=str, default="", help="OpenAI API URL")
    parser.add_argument("--api_key", type=str, required=True, help="OpenAI API key")
    parser.add_argument("--no_parallel", action="store_true", help="Disable parallel execution")
    parser.add_argument("--input_file", type=str, default="./results/after_add_description_tables.json", 
                        help="Input JSON file")
    parser.add_argument("--output_file", type=str, default="./results/embedding_after_add_description_tables.json", 
                        help="Output JSON file for results")
    
    opt = parser.parse_args()
    logger.info(f"运行参数: {opt}")
    
    # 确保输出目录存在
    output_dir = Path(opt.output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 读取提示模板
    prompt_template_path = "./prompt_template/add_embedding_column_prompt.txt"
    try:
        with open(prompt_template_path, 'r', encoding='utf-8') as file:
            prompt_template = file.read()
        logger.info("提示词模版文件内容读取成功！")
    except Exception as e:
        logger.error(f"读取提示词模版文件时出错: {str(e)}")
        exit(1)
    
    # 加载输入数据
    try:
        with open(opt.input_file, encoding="utf-8") as f:
            input_dataset = json.load(f)
        logger.info(f"成功加载输入文件，共 {len(input_dataset)} 个数据库")
    except Exception as e:
        logger.error(f"加载输入文件失败: {str(e)}")
        exit(1)
    
    # 处理函数包装器
    def process_item(db_info):
        return process_db_info(
            db_info=db_info,
            model=opt.model,
            api_url=opt.api_url,
            api_key=opt.api_key,
            prompt_template=prompt_template
        )
    
    # 并行或顺序处理
    if not opt.no_parallel:  # 开启并行模式
        logger.info("使用并行处理模式...")
        results = []
        with ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as executor:
            # 创建任务列表
            futures = [executor.submit(process_item, db_info) for db_info in input_dataset]
            
            # 使用tqdm显示进度
            for future in tqdm(futures, total=len(input_dataset), desc="处理数据库"):
                try:
                    results.append(future.result())
                except Exception as e:
                    logger.error(f"处理数据库时出错: {str(e)}")
                    traceback.print_exc()
                    results.append(input_dataset[futures.index(future)])  # 保留原始数据
    
    else:  # 顺序处理
        logger.info("使用顺序处理模式...")
        results = []
        for db_info in tqdm(input_dataset, desc="处理数据库"):
            try:
                results.append(process_item(db_info))
            except Exception as e:
                logger.error(f"处理数据库 {db_info.get('db_id', '未知')} 时出错: {str(e)}")
                results.append(db_info)  # 保留原始数据
    
    # 保存结果
    try:
        with open(opt.output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"结果成功保存到 {opt.output_file}")
    except Exception as e:
        logger.error(f"保存结果失败: {str(e)}")
        # 尝试保存到临时文件
        temp_file = output_dir / "temp_results.json"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"临时结果已保存到 {temp_file}")
