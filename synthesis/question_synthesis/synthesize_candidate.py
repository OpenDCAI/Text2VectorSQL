import argparse
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Dict, List

import openai
from tqdm import tqdm

# --- 配置日志记录 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 缓存实现 ---
CACHE_FILE = 'openai_cache.json'
CACHE = {}

def load_cache():
    """如果缓存文件存在，则从中加载缓存。"""
    global CACHE
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                CACHE = json.load(f)
                logging.info(f"从缓存文件中加载了 {len(CACHE)} 个项目。")
        except (json.JSONDecodeError, IOError) as e:
            logging.warning(f"无法加载缓存文件，将使用空缓存启动。错误: {e}")
            CACHE = {}
    else:
        logging.info("未找到缓存文件，将使用空缓存启动。")

def save_cache():
    """将当前缓存保存到文件。"""
    try:
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(CACHE, f, indent=2, ensure_ascii=False)
            logging.info(f"已将 {len(CACHE)} 个项目保存到缓存文件。")
    except IOError as e:
        logging.error(f"保存缓存文件失败: {e}")

def get_cache_key(question: str, original_sql: str) -> str:
    """根据问题和SQL创建一致的缓存键。"""
    return f"{question}|{original_sql}"

# --- OpenAI API 交互 ---
@lru_cache(maxsize=None)  # 用于当前运行的内存缓存
def get_sql_candidates_from_openai(client: openai.OpenAI, question: str, original_sql: str, num_candidates: int = 5) -> List[str]:
    """
    使用 OpenAI API 生成 SQL 候选项。
    此函数首先会检查持久化缓存。
    """
    cache_key = get_cache_key(question, original_sql)
    if cache_key in CACHE:
        logging.info(f"缓存命中: '{question[:50]}...'")
        return CACHE[cache_key]

    logging.info(f"缓存未命中，正在为问题调用API: '{question[:50]}...'")

    # 从 lembed 函数中提取原始搜索短语
    try:
        start_phrase = original_sql.split("lembed('all-MiniLM-L6-v2',")[1]
        original_phrase = start_phrase.split("')")[0].strip()
        # 移除可能存在的引号
        if original_phrase.startswith(('"', "'")) and original_phrase.endswith(('"', "'")):
            original_phrase = original_phrase[1:-1]
    except IndexError:
        logging.error(f"无法从SQL中解析原始短语: {original_sql}")
        return [original_sql]  # 如果解析失败，则返回原始SQL

    system_prompt = (
        "你是一个精通语义搜索和SQL的专家。你的任务是重写SQL查询中`lembed`函数内的搜索短语。"
        "给定用户的问题和原始SQL查询，你必须生成多个能够捕捉用户意图的替代短语。"
        "SQL查询的其余结构必须保持完全相同。"
        "只更改`lembed`函数的第二个参数。"
        "请以JSON数组字符串的形式提供输出，其中每个字符串都是一个完整的SQL查询。"
    )

    user_prompt = f"""
    原始问题: "{question}"
    原始SQL: "{original_sql}"
    原始搜索短语: "{original_phrase}"

    根据问题，生成 {num_candidates} 个替代的SQL查询。每个查询都应与原始SQL相同，除了 `lembed('all-MiniLM-L6-v2', '...')` 中的搜索短语。
    新的搜索短语在语义上应与原始问题中的用户意图相似或是其替代解释。

    请仅返回一个有效的JSON数组字符串。例如:
    {{
      "sql_candidate": [
        "SELECT ... FROM ... WHERE ... MATCH lembed('all-MiniLM-L6-v2', '新短语1')",
        "SELECT ... FROM ... WHERE ... MATCH lembed('all-MiniLM-L6-v2', '新短语2')",
        "SELECT ... FROM ... WHERE ... MATCH lembed('all-MiniLM-L6-v2', '新短语3')"
      ]
    }}
    """

    max_retries = 3
    retry_delay = 5  # 秒
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",  # 您也可以选择其他强大的模型
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.5,
            )
            content = response.choices[0].message.content
            result_data = json.loads(content)

            # 在返回的JSON对象中找到SQL查询列表
            sql_list = []
            if isinstance(result_data, dict) and "sql_candidate" in result_data:
                value = result_data["sql_candidate"]
                if isinstance(value, list) and all(isinstance(item, str) for item in value):
                    sql_list = value

            if not sql_list:
                raise ValueError("JSON响应中未包含一个名为 'sql_candidate' 的SQL字符串列表。")

            # 更新缓存
            CACHE[cache_key] = sql_list
            save_cache()  # 持久化到磁盘
            return sql_list

        except (openai.APIError, json.JSONDecodeError, ValueError) as e:
            logging.error(f"在第 {attempt + 1} 次尝试中发生错误 ('{question[:50]}...'): {e}")
            if attempt < max_retries - 1:
                logging.info(f"将在 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
            else:
                logging.error(f"在 {max_retries} 次尝试后未能为 '{question[:50]}...' 生成候选项。")
                return [original_sql]  # 返回原始SQL作为备用

    return [original_sql] # 正常情况下不应执行到这里

# --- 主要处理逻辑 ---
def process_item(item: Dict, client: openai.OpenAI, num_candidates: int) -> Dict:
    """
    处理单个JSON项目以添加 'sql_candidate' 字段。
    """
    question = item.get("question")
    sql = item.get("sql")

    if not question or not sql:
        item['sql_candidate'] = []
        return item

    # 生成候选SQL查询
    candidates = get_sql_candidates_from_openai(client, question, sql, num_candidates)
    item['sql_candidate'] = candidates
    return item

def main():
    """
    主函数：解析参数，读取文件，处理数据，并写入输出。
    """
    parser = argparse.ArgumentParser(description="使用OpenAI为向量搜索查询生成SQL候选项。")
    parser.add_argument("--input_file", type=str, default="results/question_and_sql_pairs.json", help="输入JSON文件的路径。")
    parser.add_argument("--output_file", type=str, default="results/candidate_sq.json", help="输出JSON文件的路径。")
    parser.add_argument("--api_key", type=str, required=True, help="您的OpenAI API密钥。")
    parser.add_argument("--base_url", type=str, default="http://123.129.219.111:3000/v1", help="OpenAI API的基础URL。")
    parser.add_argument("--num_candidates", type=int, default=5, help="为每个问题生成的SQL候选项数量。")
    parser.add_argument("--max_workers", type=int, default=24, help="用于API调用的最大并发线程数。")

    args = parser.parse_args()

    # 初始化OpenAI客户端
    try:
        client = openai.OpenAI(api_key=args.api_key, base_url=args.base_url)
    except Exception as e:
        logging.error(f"初始化OpenAI客户端失败: {e}")
        return

    # 加载数据
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"读取或解析输入文件 '{args.input_file}' 失败: {e}")
        return

    # 加载缓存
    load_cache()

    results = []
    # 使用ThreadPoolExecutor进行并发API调用
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # 创建未来任务列表
        futures = [executor.submit(process_item, item, client, args.num_candidates) for item in data]

        # 使用tqdm显示进度条
        for future in tqdm(futures, total=len(data), desc="正在生成SQL候选项"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logging.error(f"线程池中的一个任务失败: {e}")

    # 将结果写入输出文件
    try:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logging.info(f"已成功将更新后的数据写入 {args.output_file}")
    except IOError as e:
        logging.error(f"写入输出文件 '{args.output_file}' 失败: {e}")

if __name__ == "__main__":
    main()
