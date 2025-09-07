import json
import re
import sqlite3
import sqlite_vec
import sqlite_lembed
import os
from tqdm import tqdm
from func_timeout import func_timeout, FunctionTimedOut
import multiprocessing as mp
import sys
import ijson
import random

def parse_response(response):
    """从模型的响应文本中提取最后一个 SQL 代码块"""
    pattern = r"```sql\s*(.*?)\s*```"
    sql_blocks = re.findall(pattern, response, re.DOTALL)
    if sql_blocks:
        last_sql = sql_blocks[-1].strip()
        return last_sql
    else:
        return ""

def execute_sql(data_idx, db_file, sql):
    """在指定的 SQLite 数据库上执行 SQL 查询"""
    if not sql:
        return data_idx, db_file, sql, None, 0
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    # load sqlite-vec
    conn.enable_load_extension(True)
    sqlite_vec.load(conn) 
    sqlite_lembed.load(conn)

    MODEL_NAME = 'all-MiniLM-L6-v2'
    MODEL_PATH = "/Users/yaodongwen/WorkPlacs/project/OmniSQL/data_synthesis/all-MiniLM-L6-v2.e4ce9877.q8_0.gguf"
    cursor.execute(
        "INSERT OR IGNORE INTO temp.lembed_models(name, model) SELECT ?, lembed_model_from_file(?)",
        (MODEL_NAME, MODEL_PATH)
    )
    conn.commit()
    try:
        cursor.execute(sql)
        execution_res = cursor.fetchall()
        execution_res = frozenset(execution_res)
        conn.rollback()
        return data_idx, db_file, sql, execution_res, 1
    except Exception:
        conn.rollback()
        return data_idx, db_file, sql, None, 0
    finally:
        conn.close()

def execute_sql_wrapper(data_idx, db_file, sql, timeout):
    """带超时的 SQL 执行包装器"""
    try:
        res = func_timeout(timeout, execute_sql, args=(data_idx, db_file, sql))
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        res = (data_idx, db_file, sql, "Timeout", 0)
    except Exception:
        res = (data_idx, db_file, sql, "Exception", 0)
    return res

execution_results = []
def execute_callback_execute_sqls(result):
    """多进程回调函数，用于收集执行结果"""
    execution_results.append(result)

def execute_sqls_parallel(db_files, sqls, num_cpus=1, timeout=1):
    """并行执行 SQL 查询"""
    pool = mp.Pool(processes=num_cpus)
    for data_idx, (db_file, sql) in enumerate(zip(db_files, sqls)):
        pool.apply_async(execute_sql_wrapper, args=(data_idx, db_file, sql, timeout), callback=execute_callback_execute_sqls)
    pool.close()
    pool.join()

def load_json_file(file):
    """使用 ijson 流式加载大型 JSON 文件"""
    dataset = []
    with open(file, 'r', encoding='utf-8') as f:
        objects = ijson.items(f, 'item')
        for obj in objects:
            dataset.append(obj)
    return dataset

if __name__ == "__main__":
    results = load_json_file("./results/cot_synthesis.json")
    if not results:
        print("Error: Input file ./results/cot_synthesis.json is empty or invalid.")
        sys.exit(1)

    sampling_num = len(results[0]["responses"])
    print("sampling_num:", sampling_num)

    major_voting_filter_num = 0
    major_voting_results = []
    process_batch_size = 10240

    for pred_idx in tqdm(range(0, len(results), process_batch_size), desc="Post-processing batches"):
        batch_cot_results = results[pred_idx: pred_idx + process_batch_size]

        batch_db_files = []
        batch_sqls = []
        execution_results = []
        for cot_result in batch_cot_results:
            db_path = os.path.join("./results/vector_databases_toy", cot_result["db_id"], cot_result["db_id"] + ".sqlite")
            batch_db_files.extend([db_path] * sampling_num)
            batch_sqls.extend([parse_response(response) for response in cot_result["responses"]])
        
        execute_sqls_parallel(batch_db_files, batch_sqls, 20, 2)
        execution_results = sorted(execution_results, key = lambda x: x[0])

        for data_idx, cot_result in enumerate(batch_cot_results):
            execution_results_in_one_sample = execution_results[sampling_num * data_idx: sampling_num * (data_idx + 1)]

            major_voting_dict = dict()
            for cot, execution_result in zip(cot_result["responses"], execution_results_in_one_sample):
                if execution_result[-1] == 0: # 标记为无效的 SQL
                    continue
                
                exec_res = execution_result[-2]
                if exec_res in major_voting_dict:
                    major_voting_dict[exec_res].append(cot)
                else:
                    major_voting_dict[exec_res] = [cot]
            
            valid_cot_num = sum(len(cot_list) for cot_list in major_voting_dict.values())
            
            # 关键的过滤逻辑
            if valid_cot_num < 3:
                # <--- 新增的调试信息
                print(f"  [DEBUG] Discarding sample for db '{cot_result['db_id']}'. Valid CoTs: {valid_cot_num} < 3.")
                major_voting_filter_num += 1
                continue
            
            voting_key = max(major_voting_dict, key = lambda k: len(major_voting_dict[k]))
            voting_cots = major_voting_dict[voting_key]
            final_cot = random.choice(voting_cots)

            major_voting_results.append({
                "db_id": cot_result["db_id"],
                "sql_complexity": cot_result["sql_complexity"],
                "question_style": cot_result["question_style"],
                "question": cot_result["question"],
                "external_knowledge": cot_result["external_knowledge"],
                "cot": final_cot,
                "sql": parse_response(final_cot)
            })

    print("major_voting_filter_num:", major_voting_filter_num)
    print("num of data samples (after execution-based major voting):", len(major_voting_results))
    
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "synthetic_text2sql_dataset.json"), "w", encoding="utf-8") as f:
        json.dump(major_voting_results, f, ensure_ascii=False, indent=2)
    
    print(f"Final dataset saved to {os.path.join(output_dir, 'synthetic_text2sql_dataset.json')}")
