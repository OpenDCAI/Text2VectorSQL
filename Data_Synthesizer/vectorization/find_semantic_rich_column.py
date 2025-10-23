import json
import re
from tqdm import tqdm
from openai import OpenAI
import httpx # 导入 httpx
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import logging
import traceback
from dotenv import load_dotenv

# --- 1. 配置日志 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 2. 核心函数 (保留了代理修复) ---

@lru_cache(maxsize=10000)
def cached_llm_call(model: str, prompt: str, api_url: str, api_key: str) -> str:
    """
    带有缓存的LLM调用，以避免对相同提示的重复请求。
    """
    # --- 关键修复：创建一个不信任系统环境变量（包括代理）的HTTP客户端 ---
    # trust_env=False 是一个更可靠的方法来确保httpx不使用任何系统级的代理设置。
    http_client = httpx.Client(trust_env=False)
    
    client = OpenAI(
        api_key=api_key,
        base_url=api_url if api_url else None,
        http_client=http_client # 将自定义的客户端传递给OpenAI
    )
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"调用LLM API时出错: {str(e)}")
        return ""

def parse_response(response: str) -> dict:
    """
    解析大模型响应，提取JSON对象。
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
    处理单个数据库信息。
    """
    prompt = prompt_template.format(dababase_schema=json.dumps(db_info, ensure_ascii=False))
    response = cached_llm_call(model, prompt, api_url, api_key)
    parsed_response = parse_response(response)
    db_info["semantic_rich_column"] = parsed_response
    return db_info

# --- 3. 主执行逻辑 (已更新参数) ---

def main_find_rich_semantic_column(model,api_key,api_url,input_file,output_file,no_parallel_str,prompt_template_path = "./prompt_templates/find_semantic_rich_column.txt"):
    """
    主函数，加载配置并执行处理流程。
    """

    no_parallel = no_parallel_str in ['true', '1', 't']
    
    config = {
        "model": model,
        "api_url": api_url,
        "api_key": "********", # 不在日志中显示密钥
        "no_parallel": no_parallel,
        "input_file": input_file,
        "output_file": output_file
    }
    print(f"加载的配置: {config}")
    
    # 确保输出目录存在
    output_dir = Path(output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 读取提示模板
    try:
        with open(prompt_template_path, 'r', encoding='utf-8') as file:
            prompt_template = file.read()
        logger.info("提示词模版文件内容读取成功！")
    except Exception as e:
        logger.error(f"读取提示词模版文件时出错: {str(e)}")
        exit(1)
    
    # 加载输入数据
    try:
        with open(input_file, encoding="utf-8") as f:
            input_dataset = json.load(f)
        logger.info(f"成功加载输入文件，共 {len(input_dataset)} 个数据库")
    except Exception as e:
        logger.error(f"加载输入文件失败: {str(e)}")
        exit(1)
    
    # 处理函数包装器
    def process_item(db_info):
        return process_db_info(
            db_info=db_info,
            model=model,
            api_url=api_url,
            api_key=api_key,
            prompt_template=prompt_template
        )
    
    # 并行或顺序处理
    results = []
    if not no_parallel:
        logger.info("使用并行处理模式...")
        with ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as executor:
            futures = [executor.submit(process_item, db_info) for db_info in input_dataset]
            for future in tqdm(futures, total=len(input_dataset), desc="处理数据库"):
                try:
                    results.append(future.result())
                except Exception as e:
                    logger.error(f"处理数据库时出错: {str(e)}")
                    traceback.print_exc()
                    failed_index = futures.index(future)
                    results.append(input_dataset[failed_index])
    else:
        logger.info("使用顺序处理模式...")
        for db_info in tqdm(input_dataset, desc="处理数据库"):
            try:
                results.append(process_item(db_info))
            except Exception as e:
                logger.error(f"处理数据库 {db_info.get('db_id', '未知')} 时出错: {str(e)}")
                results.append(db_info)
    
    # 保存结果
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"结果成功保存到 {output_file}")
        print(f"结果成功保存到 {output_file}")
    except Exception as e:
        logger.error(f"保存结果失败: {str(e)}")
        temp_file = output_dir / "temp_results.json"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"临时结果已保存到 {temp_file}")

if __name__ == '__main__':
    # 加载 .env 文件中的环境变量
    load_dotenv()
    logger.info("正在从 .env 文件加载配置...")

    # 从环境变量中读取配置 (已更新)
    # 必填参数
    model = os.getenv("LLM_MODEL_NAME")
    api_key = os.getenv("API_KEY")

    # 检查必填参数是否存在 (已更新)
    if not model or not api_key:
        missing_vars = []
        if not model: missing_vars.append("LLM_MODEL_NAME")
        if not api_key: missing_vars.append("API_KEY")
        logger.error(f"错误：以下必须的环境变量未在 .env 文件中设置: {', '.join(missing_vars)}")
        exit(1)
    # 选填参数（带默认值）(已更新)
    api_url = os.getenv("BASE_URL", "http://123.129.219.111:3000/v1")
    input_file = os.getenv("INPUT_FILE_FIND_SEMANTIC_RICH", "./results/enhanced_train_tables.json")
    output_file = os.getenv("OUTPUT_FILE_FIND_SEMANTIC_RICH", "./results/find_semantic_tables.json")
    # 处理布尔值参数
    no_parallel_str = os.getenv("NO_PARALLEL_FIND_SEMANTIC_RICH", "false").lower()
    main_find_rich_semantic_column(model,api_key,api_url,input_file,output_file,no_parallel_str)
