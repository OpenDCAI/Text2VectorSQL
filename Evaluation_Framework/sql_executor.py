# sql_executor.py - Stage 1: SQL Execution with Integrated Service Management

import argparse
import json
import yaml
import sys
import os
import requests
import subprocess
import time
from tqdm import tqdm

# Adjust path to import from parent directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Execution_Engine.execution_engine import ExecutionEngine

def load_yaml_config(path: str) -> dict:
    """Loads a YAML configuration file."""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_json_file(path: str):
    """Loads a JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_execution_results(results: list, path: str):
    """Saves the SQL execution results to a JSON file."""
    # Get directory path
    dir_path = os.path.dirname(path)
    
    # Only create directory if path has a directory component
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSQL execution results saved to '{path}'")

def check_embedding_service(host="127.0.0.1", port=8000, timeout=5):
    """Check if embedding service is running."""
    try:
        response = requests.get(f"http://{host}:{port}/health", timeout=timeout)
        return response.status_code == 200
    except requests.RequestException:
        return False

def start_embedding_service(service_config=None, startup_timeout=120):
    """Start embedding service with configuration from evaluation config."""
    
    # Extract server config
    host = service_config.get('host', '127.0.0.1')
    port = service_config.get('port', 8000)
    
    if check_embedding_service(host, port):
        print("Embedding Service already running")
        return True, None
    
    print("Starting Embedding Service...")
    
    try:
        # Use fixed service directory
        service_dir = "../Embedding_Service"
        service_path = os.path.abspath(service_dir)
        if not os.path.exists(service_path):
            print(f"Service directory not found: {service_path}")
            return False, None
        
        # Generate temporary config file from evaluation config
        temp_config = {
            'server': {
                'host': '0.0.0.0',  # Service should bind to all interfaces
                'port': port
            },
            'models': service_config.get('models', [])
        }
        
        # Write temporary config file
        temp_config_path = os.path.join(service_path, "temp_config.yaml")
        with open(temp_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(temp_config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"Generated temporary config: {temp_config_path}")
        print(f"Service will run on {host}:{port}")
        
        # Start service process with temporary config
        cmd = ["python", "server.py", "--config", "temp_config.yaml"]
        process = subprocess.Popen(
            cmd,
            cwd=service_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        print(f"Service process started with PID: {process.pid}")
        
        # Wait for service to be ready
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
    """Stop embedding service process and cleanup temporary config."""
    if not process:
        return True
    
    print("Stopping Embedding Service...")
    try:
        process.terminate()
        
        # Wait for graceful shutdown
        start_time = time.time()
        while time.time() - start_time < timeout:
            if process.poll() is not None:
                print("Embedding Service stopped")
                break
            time.sleep(1)
        else:
            # Force kill if graceful shutdown failed
            print("Force killing service process...")
            process.kill()
        
        # Cleanup temporary config file
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
    """Check and display embedding service status from config."""
    try:
        # Load config to get service settings
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        service_config = config.get('embedding_service', {}) if config else {}
        host = service_config.get('host', '127.0.0.1') if service_config else '127.0.0.1'
        port = service_config.get('port', 8000) if service_config else 8000
        
        # Check service status with health endpoint
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
            
            # Provide service info from config
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
    """
    Resolve the full database identifier based on database type.
    
    For SQLite: Combine base_dir and relative db_identifier to create full file path
    For PostgreSQL/ClickHouse: Return db_identifier directly (it's the database name)
    
    :param base_dir: Base directory for database files (only used for SQLite)
    :param db_identifier: Database identifier (relative path for SQLite, database name for others)
    :param db_type: Database type ('sqlite', 'postgresql', 'clickhouse')
    :return: Full path for SQLite, or database name for PostgreSQL/ClickHouse
    """
    if db_type == 'sqlite':

        # For SQLite, resolve file path
        if not os.path.isabs(db_identifier):
            # Combine base_dir and relative db_identifier
            db_identifier= os.path.abspath(os.path.join(base_dir, db_identifier))
        if not os.path.isfile(db_identifier):
            for file in os.listdir(db_identifier):
                if file.endswith('.sqlite') :
                    db_identifier = os.path.join(db_identifier,file)
                    break
        #print(f"Using SQLite database file: {db_identifier}")
        return db_identifier
    else:
        # For PostgreSQL/ClickHouse, db_identifier is the database name
        return db_identifier

def main():
    """
    Stage 1: Execute SQL queries and save results.
    This script handles SQL execution with integrated embedding service management.
    """
    parser = argparse.ArgumentParser(description="Text2VectorSQL SQL Executor (Stage 1)")
    parser.add_argument(
        "--config",
        default="evaluation_config.yaml",
        help="Path to the evaluation configuration YAML file."
    )
    parser.add_argument(
        "--no-service-management",
        action="store_true",
        help="Disable automatic embedding service management"
    )
    args = parser.parse_args()

    # --- 1. Load Configuration ---
    print(f"Loading configuration from '{args.config}'...")
    try:
        config = load_yaml_config(args.config)
        db_type = config['db_type']
        base_dir = config.get('base_dir', '.')  # Default to current directory
        
        # 从单个文件加载评估数据
        eval_data = load_json_file(config['eval_data_file'])
        
        output_file = config['execution_results_file']
        
        # Get embedding service config
        service_config = config.get('embedding_service', {})
        auto_manage = service_config.get('auto_manage', True) and not args.no_service_management  # Default to True
        service_host = service_config.get('host', '127.0.0.1')
        service_port = service_config.get('port', 8000)
        startup_timeout = service_config.get('startup_timeout', 120)
        
        if db_type == 'sqlite':
            print(f"Database type: SQLite")
            print(f"Database base directory: {os.path.abspath(base_dir)}")
        else:
            print(f"Database type: {db_type.upper()}")
        
    except Exception as e:
        print(f"Error loading configuration or input files: {e}")
        sys.exit(1)

    # --- 2. Manage Embedding Service ---
    service_process = None
    
    print(f"Embedding Service management: {'Auto' if auto_manage else 'Manual'}")
    print(f"Service URL: http://{service_host}:{service_port}")
    
    if auto_manage:
        # Auto-management enabled (default behavior)
        success, service_process = start_embedding_service(
            service_config, startup_timeout
        )
        
        if not success:
            print("Failed to start Embedding Service")
            print("This may affect vector-based queries. Continue anyway? (y/N): ", end="")
            proceed = input()
            if proceed.lower() != 'y':
                sys.exit(1)
    else:
        # Auto-management disabled - check if service is running
        if check_embedding_service(service_host, service_port):
            print("Embedding Service is running (externally managed)")
        else:
            print("WARNING: Embedding Service not running and auto-management disabled")
            print("Vector-based queries may fail.")
            print("To enable auto-management, set 'auto_manage: true' in config or remove --no-service-management flag")
            proceed = input("Continue anyway? (y/N): ")
            if proceed.lower() != 'y':
                sys.exit(1)

    try:
        # --- 3. Initialize Execution Engine ---
        print("Initializing Execution Engine...")
        try:
            engine_config_path = os.path.abspath(os.path.join(
                config['project_dir'], config['engine_config_path']
            ))
            engine = ExecutionEngine(config_path=engine_config_path)
        except Exception as e:
            print(f"Error initializing ExecutionEngine: {e}")
            sys.exit(1)

        # --- 4. Execute SQL Queries ---
        execution_results = []
        print(f"Executing {len(eval_data)} evaluation cases...")
        
        for eval_case in tqdm(eval_data, desc="Executing SQL"):
            case_result = {
                "eval_case": eval_case,
                "eval_execution": None,
                "ground_truth_executions": [],
                "question": eval_case.get('question', ''),
                "schema": eval_case.get('schema', '')
            }
            
            try:
                # Get database identifier
                db_identifier = eval_case.get('db_id')
                if not db_identifier:
                    raise ValueError("Missing 'db_identifier' in evaluation case")
                
                # Resolve database identifier based on db_type
                resolved_db_identifier = resolve_db_identifier(base_dir, db_identifier, db_type)
                
                #print('-*-*-*'*10)
                #print(resolved_db_identifier)
                
                # Execute evaluation query from 'predicted_sql'
                predicted_sql = eval_case.get('predicted_sql')
                if predicted_sql:
                    eval_result = engine.execute(
                        sql=predicted_sql, 
                        db_type=db_type, 
                        db_identifier=resolved_db_identifier
                    )
                    case_result['eval_execution'] = eval_result
                else:
                    case_result['eval_execution'] = {"error": "No 'predicted_sql' found in evaluation case."}

                # Execute ground truth queries from 'ground_truth_sqls'
                ground_truth_sqls = eval_case.get('sql_candidate', [])
                
                if ground_truth_sqls:
                    # Ground truth should use the same database identifier
                    for gt_sql in ground_truth_sqls:
                        gt_result = engine.execute(
                            sql=gt_sql, 
                            db_type=db_type, 
                            db_identifier=resolved_db_identifier
                        )
                        case_result['ground_truth_executions'].append({
                            "sql": gt_sql,
                            "execution": gt_result
                        })
                else:
                    query_id = eval_case.get('query_id', 'N/A')
                    print(f"Warning: No 'ground_truth_sqls' list found for query_id '{query_id}'")
                    
            except Exception as e:
                case_result['execution_error'] = str(e)
                print(f"Error executing queries for {eval_case.get('query_id', 'unknown')}: {e}")
            
            execution_results.append(case_result)

        # --- 5. Save Execution Results ---
        save_execution_results(execution_results, output_file)
        print(f"Execution completed. {len(execution_results)} cases processed.")
        print(f"Use 'python evaluate_results.py' to compute metrics on these results.")
        print(f"Or use 'python run_eval_pipeline.py --evaluate' to run evaluation stage.")

    finally:
        # --- 6. Cleanup Embedding Service ---
        if service_process and auto_manage:
            print("\nCleaning up Embedding Service...")
            stop_embedding_service(service_process)

if __name__ == "__main__":
    main()
