import json
import re
import os
from tqdm import tqdm
from func_timeout import func_timeout, FunctionTimedOut
import multiprocessing as mp
import sys
import ijson
import random

# 1. 获取当前文件(my_script.py)的绝对路径
current_file_path = os.path.abspath(__file__)

# 2. 从当前文件路径计算出项目根目录(project_root)的路径
project_root_path = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))

# 3. 将项目根目录添加到 sys.path 的开头
if project_root_path not in sys.path:
    sys.path.insert(0, project_root_path)

# 导入执行引擎
from Execution_Engine.execution_engine import ExecutionEngine


def parse_response(response):
    """从模型的响应文本中提取最后一个 SQL 代码块"""
    pattern = r"```sql\s*(.*?)\s*```"
    sql_blocks = re.findall(pattern, response, re.DOTALL)
    if sql_blocks:
        return sql_blocks[-1].strip()
    return ""

def execute_with_engine(data_idx, db_id, original_sql, engine, db_type):
    """
    使用 ExecutionEngine 执行单个 SQL 查询。
    此函数将被 func_timeout 包装。
    """
    if not original_sql:
        return data_idx, db_id, original_sql, "Empty SQL", 0
    
    # 调用执行引擎
    result = engine.execute(sql=original_sql, db_type=db_type, db_identifier=db_id)

    # 将引擎的返回结果适配为后续逻辑所需的格式
    if result['status'] == 'success':
        # 将结果行（列表）转换为元组，以便可以放入 frozenset
        execution_res = frozenset(tuple(row) for row in result['data'])
        return data_idx, db_id, original_sql, execution_res, 1
    else:
        # 执行失败
        error_message = result.get('message', 'Unknown execution error')
        print(f"Execution error for db_id {db_id}, SQL: {original_sql}. Error: {error_message}")
        return data_idx, db_id, original_sql, error_message, 0

def execute_sqls_parallel(db_ids, original_sqls, engine, db_type, num_cpus=1, timeout=1):
    """
    使用 ExecutionEngine 并行执行 SQL 查询，并使用 Pool 自带的超时机制。
    """
    pool = mp.Pool(processes=num_cpus)
    tasks = []
    # 注意：这里我们直接把 execute_with_engine 提交给子进程
    for data_idx, (db_id, original_sql) in enumerate(zip(db_ids, original_sqls)):
        if original_sql:
             tasks.append(
                 (
                     (data_idx, db_id, original_sql), # 原始信息，用于处理超时或错误
                     pool.apply_async(execute_with_engine, args=(data_idx, db_id, original_sql, engine, db_type))
                 )
             )
    
    results = []
    for original_info, res in tqdm(tasks, desc="Executing SQLs"):
        data_idx, db_id, original_sql = original_info
        try:
            # 在这里设置超时！
            # res.get() 会等待子进程完成，如果超过 timeout 秒，就会抛出 multiprocessing.TimeoutError
            result = res.get(timeout=timeout)
            results.append(result)
        except mp.TimeoutError:
            print(f"Timeout error for db_id {db_id}, SQL: {original_sql}")
            results.append((data_idx, db_id, original_sql, "Timeout", 0))
        except Exception as e:
            # 捕获在子进程中发生的其他异常
            print(f"Wrapper Exception for db_id {db_id}, SQL: {original_sql}. Error: {e}")
            results.append((data_idx, db_id, original_sql, f"Wrapper Exception: {e}", 0))

    pool.close()
    pool.join()
    return results
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

def post_process_cot(results_path, output_dir, db_type="sqlite"):
    """主函数，负责整个后处理流程"""
    try:
        # 初始化执行引擎，确保配置文件路径正确
        engine = ExecutionEngine(config_path=os.path.join(project_root_path, "Execution_Engine", "engine_config.yaml"))
    except Exception as e:
        print(f"Failed to initialize engine: {e}")
        sys.exit(1)

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
        
        # 准备执行任务所需的数据
        batch_db_ids, batch_original_sqls = [], []
        for cot_result in batch_cot_results:
            db_id = cot_result["db_id"]

            if not db_id or not isinstance(db_id, str):
                print(f"Found invalid db_id: {db_id}. Skipping this entry.")
                # 你可以选择跳过这个无效数据
                # 这里我们简单地不把它加入执行列表
                continue
            
            batch_db_ids.extend([db_id] * sampling_num)
            batch_original_sqls.extend([parse_response(response) for response in cot_result["responses"]])
        
        # --- SQL 预处理步骤被移除 ---
        # 现在直接将原始 SQL 交给执行引擎
        engine.execute(sql=batch_original_sqls[0], db_type=db_type, db_identifier=batch_db_ids[0])

        # 使用执行引擎并行执行 SQL
        execution_results = execute_sqls_parallel(
            db_ids=batch_db_ids, 
            original_sqls=batch_original_sqls, 
            engine=engine,
            db_type=db_type,
            num_cpus=20, 
            timeout=3
        )
        execution_results = sorted(execution_results, key=lambda x: x[0])

        for data_idx, cot_result in enumerate(batch_cot_results):
            start_index = sampling_num * data_idx
            end_index = sampling_num * (data_idx + 1)
            execution_results_in_one_sample = [res for res in execution_results if start_index <= res[0] < end_index]

            major_voting_dict = {}
            sql_to_response_map = {parse_response(resp): resp for resp in cot_result["responses"]}

            for exec_result in execution_results_in_one_sample:
                # exec_result 格式: (data_idx, db_id, original_sql, execution_res, status)
                status = exec_result[-1]
                if status == 1: # 仅处理成功执行的查询
                    exec_res_tuple = exec_result[-2]
                    original_sql = exec_result[2]
                    if original_sql in sql_to_response_map:
                        response = sql_to_response_map[original_sql]
                        major_voting_dict.setdefault(exec_res_tuple, []).append(response)

            valid_cot_num = sum(len(cots) for cots in major_voting_dict.values())
            
            if valid_cot_num < 3:
                major_voting_filter_num += 1
                print(f"Valid COT num: {valid_cot_num} < 3, skip this sample.")
                continue
            
            if not major_voting_dict:
                major_voting_filter_num += 1
                print("No valid SQL executions, skip this sample.")
                continue

            voting_key = max(major_voting_dict, key=lambda k: len(major_voting_dict[k]))
            final_cot = random.choice(major_voting_dict[voting_key])

            if 'schema' in cot_result:
                major_voting_results.append({
                    "db_id": cot_result["db_id"],
                    "sql_complexity": cot_result["sql_complexity"],
                    "question_style": cot_result["question_style"],
                    "sql_explanation": cot_result.get("sql_explanation", ""),
                    "question": cot_result["question"],
                    "sql_candidate": cot_result["sql_candidate"],
                    "external_knowledge": cot_result.get("external_knowledge", ""),
                    "cot": final_cot,
                    "sql": parse_response(final_cot),
                    "schema": cot_result["schema"]
                })
            else:
                major_voting_results.append({
                    "db_id": cot_result["db_id"],
                    "sql_complexity": cot_result["sql_complexity"],
                    "question_style": cot_result["question_style"],
                    "sql_explanation": cot_result.get("sql_explanation", ""),
                    "question": cot_result["question"],
                    "sql_candidate": cot_result["sql_candidate"],
                    "external_knowledge": cot_result.get("external_knowledge", ""),
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
    # --- 配置区域 ---
    # 定义数据库类型，可以是 "sqlite", "postgresql", "clickhouse"
    # 这个值将传递给 ExecutionEngine
    DB_TYPE = "sqlite"

    # 定义文件路径
    COT_SYNTHESIS_PATH = "sqlite/results/wikipedia_multimodal/cot_synthesis.json"
    OUTPUT_DIRECTORY = "sqlite/results/wikipedia_multimodal"

    # 调用主函数
    post_process_cot(
        results_path=COT_SYNTHESIS_PATH,
        output_dir=OUTPUT_DIRECTORY,
        db_type=DB_TYPE
    )
