# sql_executor.py - Stage 1: SQL Execution with Integrated Service Management

import argparse
import json
import yaml
import sys
import os
import requests
import subprocess
import time
import shutil
import base64
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
# 导入 multiprocessing 和 queue 以实现沙箱化超时
import multiprocessing
from queue import Empty as QueueEmpty

# Adjust path to import from parent directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Execution_Engine.execution_engine import ExecutionEngine

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, bytes):
            return base64.b64encode(o).decode('ascii')
        return super().default(o)

# --- 新增: 沙箱化执行的核心逻辑 ---

def _execute_task(queue, engine_config_path, sql, db_type, db_identifier):
    """
    这是一个在独立的“工人”进程中运行的目标函数。
    它会执行SQL并将结果放入队列。
    """
    try:
        engine = ExecutionEngine(config_path=engine_config_path)
        result = engine.execute(sql=sql, db_type=db_type, db_identifier=db_identifier)
        queue.put(result)
    except Exception as e:
        # 捕获任何执行期间的异常并返回
        queue.put({"status": "error", "error": f"Execution failed in subprocess: {str(e)}"})

def execute_in_sandbox(engine_config_path, sql, db_type, db_identifier, timeout):
    """
    在沙箱化子进程中执行SQL，并强制执行超时。
    这是“监工”函数。
    """
    q = multiprocessing.Queue()
    # 创建“工人”进程
    p = multiprocessing.Process(
        target=_execute_task, 
        args=(q, engine_config_path, sql, db_type, db_identifier)
    )
    p.start() # “工人”开始工作

    try:
        # “监工”带有时限地等待结果
        result = q.get(block=True, timeout=timeout)
    except QueueEmpty:
        # 如果超时，队列为空，捕获异常
        result = {"status": "error", "error": f"Query timed out after {timeout} seconds."}
    finally:
        # 无论成功、失败还是超时，都确保终止并清理“工人”进程
        if p.is_alive():
            p.terminate() # 强制终止
            p.join()

    return result

# --- 原有函数保持不变 ---

def load_yaml_config(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_json_file(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_execution_results(results: list, path: str):
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    results.sort(key=lambda x: x['eval_case'].get('query_id', ''))
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, cls=CustomJSONEncoder)
    print(f"\n{len(results)} 条 SQL 执行结果已保存至 '{path}'")

def check_embedding_service(host="127.0.0.1", port=8000, timeout=5):
    try:
        response = requests.get(f"http://{host}:{port}/health", timeout=timeout)
        return response.status_code == 200
    except requests.RequestException:
        return False

# start_embedding_service, stop_embedding_service, check_service_status...
# 这些函数保持不变，为简洁起见此处省略，请保留您文件中的原样
def start_embedding_service(service_config=None, startup_timeout=120):
    host = service_config.get('host', '127.0.0.1')
    port = service_config.get('port', 8000)
    if check_embedding_service(host, port):
        print("Embedding Service already running")
        return True, None
    print("Starting Embedding Service...")
    try:
        service_dir = "../Embedding_Service"
        service_path = os.path.abspath(service_dir)
        if not os.path.exists(service_path):
            print(f"Service directory not found: {service_path}")
            return False, None
        temp_config = {
            'server': {'host': '0.0.0.0', 'port': port},
            'models': service_config.get('models', [])
        }
        temp_config_path = os.path.join(service_path, "temp_config.yaml")
        with open(temp_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(temp_config, f, default_flow_style=False, allow_unicode=True)
        print(f"Generated temporary config: {temp_config_path}")
        print(f"Service will run on {host}:{port}")
        cmd = ["python", "server.py", "--config", "temp_config.yaml"]
        process = subprocess.Popen(
            cmd, cwd=service_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        print(f"Service process started with PID: {process.pid}")
        start_time = time.time()
        print(f"Waiting for service to be ready (timeout: {startup_timeout}s)...")
        while time.time() - start_time < startup_timeout:
            if check_embedding_service(host, port):
                print("Embedding Service is ready")
                return True, process
            print(".", end="", flush=True)
            time.sleep(2)
        print(f"\nService startup timeout ({startup_timeout}s)")
        process.terminate()
        return False, None
    except Exception as e:
        print(f"Failed to start Embedding Service: {e}")
        return False, None

def stop_embedding_service(process, timeout=30):
    if not process:
        return True
    print("Stopping Embedding Service...")
    try:
        process.terminate()
        start_time = time.time()
        while time.time() - start_time < timeout:
            if process.poll() is not None:
                print("Embedding Service stopped")
                break
            time.sleep(1)
        else:
            print("Force killing service process...")
            process.kill()
        service_dir = "../Embedding_Service"
        temp_config_path = os.path.join(os.path.abspath(service_dir), "temp_config.yaml")
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
            print("Cleaned up temporary config file")
        return True
    except Exception as e:
        print(f"Error stopping service: {e}")
        return False

def check_service_status(config_path="evaluation_config.yaml"):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        service_config = config.get('embedding_service', {}) if config else {}
        host = service_config.get('host', '127.0.0.1') if service_config else '127.0.0.1'
        port = service_config.get('port', 8000) if service_config else 8000
        health_url = f"http://{host}:{port}/health"
        try:
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                status = response.json()
                print(f"✓ Embedding Service Status: {status.get('status', 'unknown')}")
                print(f"  - Host: {host}:{port}")
                print(f"  - Available Models: {status.get('models', [])}")
                return True
            else:
                print(f"✗ Service responded with status code: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"✗ Cannot connect to embedding service at {host}:{port}")
            print(f"  Error: {e}")
            print(f"\nService Configuration:")
            print(f"  - Host: {host}")
            print(f"  - Port: {port}")
            if service_config:
                print(f"  - Models: {service_config.get('models', [])}")
            return False
    except Exception as e:
        print(f"✗ Error checking service status: {e}")
        return False

def resolve_db_identifier(base_dir: str, db_identifier: str, db_type: str) -> str:
    if db_type == 'sqlite':
        if not os.path.isabs(db_identifier):
            db_identifier = os.path.abspath(os.path.join(base_dir, db_identifier))
        if os.path.isdir(db_identifier): 
            for file in os.listdir(db_identifier):
                if file.endswith('.sqlite') or file.endswith('.db'):
                    db_identifier = os.path.join(db_identifier, file)
                    break
        return db_identifier
    else:
        return db_identifier

# --- process_single_case 函数被重构 ---

def process_single_case(eval_case, engine_config_path, db_type, base_dir, query_timeout):
    """
    现在这个函数调用沙箱化执行器来处理每个SQL。
    """
    case_result = {
        "eval_case": eval_case,
        "eval_execution": None,
        "ground_truth_executions": [],
        "question": eval_case.get('question', ''),
        "schema": eval_case.get('schema', '')
    }
    
    try:
        db_identifier = eval_case.get('db_id')
        if not db_identifier:
            raise ValueError("Missing 'db_identifier' in evaluation case")
        
        resolved_db_identifier = resolve_db_identifier(base_dir, db_identifier, db_type)
        
        predicted_sql = eval_case.get('predicted_sql')
        if predicted_sql:
            # MODIFIED: 调用沙箱执行器
            eval_result = execute_in_sandbox(
                engine_config_path, predicted_sql, db_type, resolved_db_identifier, query_timeout
            )
            case_result['eval_execution'] = eval_result
        else:
            case_result['eval_execution'] = {"error": "No 'predicted_sql' found in evaluation case."}

        ground_truth_sqls = eval_case.get('sql_candidate', [])
        
        if ground_truth_sqls:
            for gt_sql in ground_truth_sqls:
                # MODIFIED: 对每个 ground truth SQL 也调用沙箱执行器
                gt_result = execute_in_sandbox(
                    engine_config_path, gt_sql, db_type, resolved_db_identifier, query_timeout
                )
                case_result['ground_truth_executions'].append({
                    "sql": gt_sql,
                    "execution": gt_result
                })
            
    except Exception as e:
        case_result['execution_error'] = str(e)
    
    return case_result

# --- main 函数被修改以加载和传递新配置 ---

def main():
    parser = argparse.ArgumentParser(description="Text2VectorSQL SQL Executor (Stage 1)")
    parser.add_argument("--config", default="evaluation_config.yaml", help="Path to the evaluation configuration YAML file.")
    parser.add_argument("--no-service-management", action="store_true", help="Disable automatic embedding service management")
    parser.add_argument("--no-cache", action="store_true", default=False, help="Force re-execution and ignore any existing cache.")
    args = parser.parse_args()

    print(f"Loading configuration from '{args.config}'...")
    try:
        config = load_yaml_config(args.config)
        db_type = config['db_type']
        base_dir = config.get('base_dir', '.')
        eval_data_file = config['eval_data_file']
        eval_data = load_json_file(eval_data_file)
        output_file = config['execution_results_file']
        num_workers = config.get('num_workers', 8)
        # 新增: 从配置中加载查询超时时间，默认为60秒
        query_timeout = config.get('query_timeout', 30)
        
        service_config = config.get('embedding_service', {})
        auto_manage = service_config.get('auto_manage', True) and not args.no_service_management
        service_host = service_config.get('host', '127.0.0.1')
        service_port = service_config.get('port', 8000)
        startup_timeout = service_config.get('startup_timeout', 120)
        
        engine_config_path = os.path.abspath(os.path.join(
            config['project_dir'], config['engine_config_path']
        ))
        
        print(f"Database type: {db_type.upper()}")
        print(f"Concurrent workers (processes): {num_workers}")
        print(f"SQL Query Timeout: {query_timeout} seconds") # 打印超时配置
        
    except Exception as e:
        print(f"Error loading configuration or input files: {e}")
        sys.exit(1)

    # Service management... (保持不变)
    service_process = None
    print(f"\nEmbedding Service management: {'Auto' if auto_manage else 'Manual'}")
    print(f"Service URL: http://{service_host}:{service_port}")
    if auto_manage:
        success, service_process = start_embedding_service(service_config, startup_timeout)
        if not success:
            proceed = input("Failed to start Embedding Service. Continue anyway? (y/N): ")
            if proceed.lower() != 'y': sys.exit(1)
    else:
        if not check_embedding_service(service_host, service_port):
            print("WARNING: Embedding Service not running and auto-management disabled...")
            proceed = input("Continue anyway? (y/N): ")
            if proceed.lower() != 'y': sys.exit(1)
        else:
            print("Embedding Service is running (externally managed)")

    try:
        dataset_name, _ = os.path.splitext(os.path.basename(eval_data_file))
        cache_dir = os.path.join("cache", db_type, dataset_name, "execution")
        os.makedirs(cache_dir, exist_ok=True)
        print(f"\nUsing cache directory for recovery: '{cache_dir}'")
        
        if args.no_cache:
            print("--- The --no-cache flag is set. Clearing cache and re-processing all items. ---")
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
            os.makedirs(cache_dir, exist_ok=True)

        cached_results = []
        processed_ids = set()
        if not args.no_cache:
            for filename in os.listdir(cache_dir):
                if filename.endswith('.json'):
                    try:
                        with open(os.path.join(cache_dir, filename), 'r', encoding='utf-8') as f:
                            result = json.load(f)
                            query_id = result['eval_case'].get('query_id')
                            if query_id:
                                cached_results.append(result)
                                processed_ids.add(query_id)
                    except (IOError, json.JSONDecodeError): continue
        
        if cached_results:
            print(f"Loaded {len(cached_results)} completed executions from cache.")

        items_to_process = [case for case in eval_data if case.get('query_id') not in processed_ids]
        print(f"Total cases: {len(eval_data)}, To be executed: {len(items_to_process)}")

        new_results = []
        if items_to_process:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                # MODIFIED: 传递 query_timeout 到工作函数
                future_to_case = {
                    executor.submit(process_single_case, case, engine_config_path, db_type, base_dir, query_timeout): case 
                    for case in items_to_process
                }
                
                with tqdm(total=len(items_to_process), desc="Executing SQL (in parallel)") as pbar:
                    for future in as_completed(future_to_case):
                        case = future_to_case[future]
                        try:
                            result = future.result()
                            new_results.append(result)
                            query_id = result['eval_case'].get('query_id')
                            if query_id:
                                with open(os.path.join(cache_dir, f"{query_id}.json"), 'w', encoding='utf-8') as f:
                                    json.dump(result, f, indent=2, ensure_ascii=False, cls=CustomJSONEncoder)
                        except Exception as e:
                            print(f"\nError processing case {case.get('query_id', 'N/A')}: {e}")
                        pbar.update(1)

        final_results = cached_results + new_results
        save_execution_results(final_results, output_file)
        print(f"\nExecution completed. {len(final_results)} cases processed in total.")
        print(f"Next step: Run evaluation using 'python evaluate_results.py' or 'python run_eval_pipeline.py --evaluate'")

    finally:
        if service_process and auto_manage:
            print("\nCleaning up Embedding Service...")
            stop_embedding_service(service_process)

if __name__ == "__main__":
    # 确保在Windows等不支持fork的系统上正常工作
    multiprocessing.freeze_support()
    main()
