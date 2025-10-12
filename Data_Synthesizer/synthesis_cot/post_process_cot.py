import json
import re
import sqlite3
import sqlite_vec
import os
from tqdm import tqdm
from func_timeout import func_timeout, FunctionTimedOut
import multiprocessing as mp
import sys
import ijson
import random
import requests

# --- 与VLLM服务交互的辅助函数 ---

def get_embedding_from_server(text: str, server_url: str, model_name: str) -> list:
    """通过HTTP请求从本地VLLM服务获取嵌入向量"""
    payload = {"model": model_name, "texts": [text]}
    try:
        response = requests.post(server_url, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result["embeddings"][0]
    except requests.exceptions.RequestException as e:
        print(f"❌ 请求VLLM服务失败: {e}", file=sys.stderr)
        raise
    except (KeyError, IndexError) as e:
        print(f"❌ 解析VLLM响应失败: {result}", file=sys.stderr)
        raise

def preprocess_sql(sql: str, server_url: str, model_name: str, default_k=5) -> str:
    """
    [再次修正版]
    修正了JSON格式，使其直接输出向量数组 "[]" 而不是对象 "{"vector": []}"
    """
    
    # --- 阶段一: 处理所有 `lembed(...) AND k = ...` 的标准情况 ---
    pattern_with_k = re.compile(
        r"""
        (lembed\s*\(\s*[^,]+?,\s*(['"])(.*?)\2\s*\)) # 第1组: 完整的lembed(...)调用; 第3组: 需要嵌入的文本
        (\s+AND\s+k\s*=\s*(\d+))                    # 第4组: 完整的AND k=...部分; 第5组: k的值
        """,
        re.IGNORECASE | re.VERBOSE | re.DOTALL
    )

    def replacer_with_k(match):
        text_to_embed = match.group(3)
        and_k_clause = match.group(4) 
        
        vector = get_embedding_from_server(text_to_embed, server_url, model_name)
        # [修改] 直接将vector列表转为JSON数组字符串
        vector_json_array = json.dumps(vector)
        
        return f"'{vector_json_array}' {and_k_clause}"

    processed_sql = pattern_with_k.sub(replacer_with_k, sql)

    # --- 阶段二: 处理剩余的、不规范的 lembed(...) 调用 ---
    if 'lembed' in processed_sql.lower():
        k_from_limit_match = re.search(r'LIMIT\s+(\d+)', processed_sql, re.IGNORECASE)
        k_value = default_k
        if k_from_limit_match:
            k_value = int(k_from_limit_match.group(1))

        pattern_only_lembed = re.compile(
            r"lembed\s*\(\s*[^,]+?,\s*(['\"])(.*?)\1\s*\)",
            re.IGNORECASE | re.DOTALL
        )

        def replacer_only_lembed(match):
            text_to_embed = match.group(2)
            vector = get_embedding_from_server(text_to_embed, server_url, model_name)
            # [修改] 直接将vector列表转为JSON数组字符串
            vector_json_array = json.dumps(vector)
            
            if not k_from_limit_match:
                return f"'{vector_json_array}' AND k = {k_value}"
            else:
                return f"'{vector_json_array}'"
        
        processed_sql = pattern_only_lembed.sub(replacer_only_lembed, processed_sql)

    return processed_sql

def parse_response(response):
    """从模型的响应文本中提取最后一个 SQL 代码块"""
    pattern = r"```sql\s*(.*?)\s*```"
    sql_blocks = re.findall(pattern, response, re.DOTALL)
    if sql_blocks:
        return sql_blocks[-1].strip()
    return ""

def execute_processed_sql(data_idx, db_file, original_sql, processed_sql):
    """在指定的 SQLite 数据库上执行一个已经被处理好的 SQL 查询"""
    if not processed_sql:
        return data_idx, db_file, original_sql, None, 0
    
    try:
        conn = sqlite3.connect(db_file, timeout=10.0)
    except sqlite3.OperationalError as e:
        return data_idx, db_file, original_sql, f"DB Connect Error: {e}", 0

    cursor = conn.cursor()
    
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn) 
        cursor.execute(processed_sql)
        execution_res = frozenset(cursor.fetchall())
        conn.rollback()
        return data_idx, db_file, original_sql, execution_res, 1
    except Exception as e:
        error_message = f"SQL Error on idx {data_idx} [DB: {os.path.basename(db_file)}]: {e}\nFailing SQL (Original):\n---\n{original_sql}\n---"
        # print(error_message, file=sys.stderr) # 暂时注释掉，避免过多错误输出
        conn.rollback()
        return data_idx, db_file, original_sql, str(e), 0
    finally:
        conn.close()

def execute_sql_wrapper(data_idx, db_file, original_sql, processed_sql, timeout):
    """带超时的 SQL 执行包装器"""
    try:
        return func_timeout(timeout, execute_processed_sql, args=(data_idx, db_file, original_sql, processed_sql))
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        return data_idx, db_file, original_sql, "Timeout", 0
    except Exception as e:
        return data_idx, db_file, original_sql, f"Wrapper Exception: {e}", 0

def execute_sqls_parallel(db_files, original_sqls, processed_sqls, num_cpus=1, timeout=1):
    """并行执行 SQL 查询，并直接返回结果列表"""
    pool = mp.Pool(processes=num_cpus)
    tasks = []
    for data_idx, (db_file, original_sql, processed_sql) in enumerate(zip(db_files, original_sqls, processed_sqls)):
        if processed_sql is not None:
             tasks.append(pool.apply_async(execute_sql_wrapper, args=(data_idx, db_file, original_sql, processed_sql, timeout)))
    pool.close()
    pool.join()
    return [res.get() for res in tasks]

def load_json_file(file):
    """使用 ijson 流式加载大型 JSON 文件"""
    if not os.path.exists(file):
        print(f"Error: Cannot find input file {file}", file=sys.stderr)
        return None
    dataset = []
    with open(file, 'r', encoding='utf-8') as f:
        objects = ijson.items(f, 'item')
        for obj in objects:
            dataset.append(obj)
    return dataset

def post_process_cot(results_path, db_dir, output_dir, server_url, model_name):
    """主函数，负责整个后处理流程"""
    results = load_json_file(results_path)
    if not results:
        print(f"Error: Input file {results_path} is empty or invalid.", file=sys.stderr)
        sys.exit(1)

    sampling_num = len(results[0]["responses"])
    print("sampling_num:", sampling_num)

    major_voting_filter_num = 0
    major_voting_results = []
    process_batch_size = 10240

    for pred_idx in tqdm(range(0, len(results), process_batch_size), desc="Post-processing batches"):
        batch_cot_results = results[pred_idx: pred_idx + process_batch_size]
        batch_db_files, batch_original_sqls = [], []
        for cot_result in batch_cot_results:
            db_path = os.path.join(db_dir, cot_result["db_id"], cot_result["db_id"] + ".sqlite")
            db_path_final = os.path.join(db_dir, cot_result["db_id"], cot_result["db_id"] + "_final.sqlite")
            if os.path.exists(db_path_final):
                db_path = db_path_final
            batch_db_files.extend([db_path] * sampling_num)
            batch_original_sqls.extend([parse_response(response) for response in cot_result["responses"]])
        
        batch_processed_sqls = []
        for sql in tqdm(batch_original_sqls, desc="  Preprocessing SQLs in batch", leave=False):
            if not sql:
                batch_processed_sqls.append(None)
                continue
            try:
                processed_sql = preprocess_sql(sql, server_url, model_name)
                batch_processed_sqls.append(processed_sql)
            except Exception as e:
                # print(f"Failed to preprocess SQL: {sql[:100]}... Error: {e}", file=sys.stderr)
                batch_processed_sqls.append(None)
        
        execution_results = execute_sqls_parallel(batch_db_files, batch_original_sqls, batch_processed_sqls, 20, 3)
        execution_results = sorted(execution_results, key=lambda x: x[0])

        for data_idx, cot_result in enumerate(batch_cot_results):
            start_index = sampling_num * data_idx
            end_index = sampling_num * (data_idx + 1)
            execution_results_in_one_sample = [res for res in execution_results if start_index <= res[0] < end_index]

            major_voting_dict = {}
            sql_to_response_map = {parse_response(resp): resp for resp in cot_result["responses"]}

            for exec_result in execution_results_in_one_sample:
                if exec_result[-1] == 1:
                    exec_res_tuple = exec_result[-2]
                    original_sql = exec_result[2]
                    if original_sql in sql_to_response_map:
                        response = sql_to_response_map[original_sql]
                        major_voting_dict.setdefault(exec_res_tuple, []).append(response)

            valid_cot_num = sum(len(cots) for cots in major_voting_dict.values())
            
            if valid_cot_num < 3:
                major_voting_filter_num += 1
                continue
            
            if not major_voting_dict:
                major_voting_filter_num += 1
                continue

            voting_key = max(major_voting_dict, key=lambda k: len(major_voting_dict[k]))
            final_cot = random.choice(major_voting_dict[voting_key])

            major_voting_results.append({
                "db_id": cot_result["db_id"],
                "sql_complexity": cot_result["sql_complexity"],
                "question_style": cot_result["question_style"],
                "sql_explanation": cot_result["sql_explanation"],
                "question": cot_result["question"],
                "sql_candidate": cot_result["sql_candidate"],
                "external_knowledge": cot_result["external_knowledge"],
                "cot": final_cot,
                "sql": parse_response(final_cot)
            })

    print("major_voting_filter_num:", major_voting_filter_num)
    print("num of data samples (after execution-based major voting):", len(major_voting_results))
    
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "synthetic_text2sql_dataset.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(major_voting_results, f, ensure_ascii=False, indent=2)
    
    print(f"Final dataset saved to {output_file}")

if __name__ == "__main__":
    VLLM_SERVER_URL = "http://127.0.0.1:8000/embed"
    MODEL_IN_VLLM = "CLIP-ViT-B-32-laion2B-s34B-b79K" 

    COT_SYNTHESIS_PATH = "sqlite/results/wikipedia_multimodal/cot_synthesis.json"
    DB_DIRECTORY = "sqlite/results/wikipedia_multimodal/vector_databases_wiki"
    OUTPUT_DIRECTORY = "sqlite/results/wikipedia_multimodal"

    post_process_cot(
        results_path=COT_SYNTHESIS_PATH,
        db_dir=DB_DIRECTORY,
        output_dir=OUTPUT_DIRECTORY,
        server_url=VLLM_SERVER_URL,
        model_name=MODEL_IN_VLLM
    )
