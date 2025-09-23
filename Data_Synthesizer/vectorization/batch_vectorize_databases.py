import os
import sys
import logging
import json
import warnings
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer
import torchvision
from typing import Optional

# --- 关键修复 1：在所有torch相关操作前，强制设置多进程启动方式为'spawn' ---
# 这能从根本上解决顽固的'leaked semaphore'警告
try:
    torch.multiprocessing.set_start_method('spawn', force=True)
    print("Multiprocessing start method set to 'spawn'.")
except RuntimeError:
    pass # 忽略如果已经被设置的错误

# --- 关键修复 2：屏蔽良性的PyTorch性能警告 ---
warnings.filterwarnings("ignore", message=".*Torch was not compiled with memory efficient attention.*")

# --- Early Setup: Logging Configuration ---
os.makedirs("logging", exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='logging/out.log', filemode='w')
torchvision.disable_beta_transforms_warning()

try:
    from .vector_database_generate import generate_database_script, build_vector_database
except ImportError as e:
    logging.critical(f"Import Error: {e}. Please ensure vector_database_generate.py is accessible.")
    sys.exit(1)

from dotenv import load_dotenv
load_dotenv()

SOURCE_DB_ROOT = os.getenv("SOURCE_DB_ROOT")
SQL_SCRIPT_DIR = os.getenv("SQL_SCRIPT_DIR")
VECTOR_DB_ROOT = os.getenv("VECTOR_DB_ROOT")
TABLE_JSON_PATH = os.getenv("TABLE_JSON_PATH")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
model_path = os.getenv("model_path", "/mnt/b_public/data/yaodongwen/model")
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def load_completion_status(status_file):
    if os.path.exists(status_file):
        try:
            with open(status_file, 'r', encoding='utf-8') as f: return json.load(f)
        except (json.JSONDecodeError, TypeError): return {}
    return {}

def save_completion_status(status_file, completed_dbs_dict):
    with open(status_file, 'w', encoding='utf-8') as f: json.dump(completed_dbs_dict, f, indent=2)

def find_database_file(base_path: str, db_id: str) -> Optional[str]:
    """
    Finds a database file, checking for .sqlite and then .db extensions.
    Returns the full path if found, otherwise None.
    """
    # Check for .sqlite first
    path_sqlite = os.path.join(base_path, f"{db_id}.sqlite")
    if os.path.exists(path_sqlite):
        return path_sqlite
    
    # If not found, check for .db
    path_db = os.path.join(base_path, f"{db_id}.db")
    if os.path.exists(path_db):
        return path_db
        
    # If neither exists
    return None

def main_batch_vectorize_databases(SOURCE_DB_ROOT: str, SQL_SCRIPT_DIR: str, VECTOR_DB_ROOT: str, TABLE_JSON_PATH: str, EMBEDDING_MODEL_NAME: str, cache_model_path: str):
    logging.info("--- Starting Batch Database Vectorization ---")
    if not all([SOURCE_DB_ROOT, SQL_SCRIPT_DIR, VECTOR_DB_ROOT, TABLE_JSON_PATH]):
        logging.critical("One or more required configurations are missing.")
        sys.exit(1)
    
    os.makedirs(SQL_SCRIPT_DIR, exist_ok=True)
    os.makedirs(VECTOR_DB_ROOT, exist_ok=True)
    status_file_path = os.path.join(VECTOR_DB_ROOT, "processing_status.json")
    
    db_targets = []
    if os.path.exists(SOURCE_DB_ROOT):
        for db_id in os.listdir(SOURCE_DB_ROOT):
            db_dir = os.path.join(SOURCE_DB_ROOT, db_id)
            if os.path.isdir(db_dir):
                # Call the helper function to find the database path
                db_path = find_database_file(db_dir, db_id)
                
                # If the helper found a file (either .sqlite or .db), its path will be returned
                if db_path:
                    db_targets.append({'id': db_id, 'path': db_path})

    if not db_targets:
        # A more precise warning message
        logging.warning(f"No valid database files (.sqlite or .db) found in the subdirectories of '{SOURCE_DB_ROOT}'.")
        return
    logging.info(f"Found {len(db_targets)} database(s).")

    # --- 阶段一：生成所有需要的SQL文件 (管理GPU进程池) ---
    completed_dbs = load_completion_status(status_file_path)
    dbs_needing_sql = [target for target in db_targets if completed_dbs.get(target['id']) not in ['sql_generated', 'db_built']]
    
    if dbs_needing_sql:
        logging.info(f"--- Phase 1: Generating SQL for {len(dbs_needing_sql)} database(s) ---")
        model = SentenceTransformer(EMBEDDING_MODEL_NAME, device='cpu', cache_folder=cache_model_path)
        pool = None
        try:
            if torch.cuda.is_available():
                logging.info("CUDA is available, starting multi-GPU process pool...")
                pool = model.start_multi_process_pool()
                logging.info("Multi-GPU process pool started.")
            
            for target in tqdm(dbs_needing_sql, desc="Generating SQL"):
                db_id, source_db_path = target['id'], target['path']
                sql_script_path = os.path.join(SQL_SCRIPT_DIR, f"{db_id}_vector.sql")
                try:
                    generate_database_script(db_path=source_db_path, output_file=sql_script_path, embedding_model=model, pool=pool, table_json_path=TABLE_JSON_PATH)
                    completed_dbs[db_id] = 'sql_generated'
                    save_completion_status(status_file_path, completed_dbs)
                except Exception as e:
                     logging.error(f"Error generating SQL for '{db_id}': {e}", exc_info=True)
        finally:
            if pool:
                model.stop_multi_process_pool(pool)
                logging.info("Multi-GPU process pool stopped after SQL generation.")
    else:
        logging.info("--- Phase 1: All SQL scripts already generated. ---")

    # --- 阶段二：从SQL文件构建数据库 (无GPU进程) ---
    completed_dbs = load_completion_status(status_file_path)
    dbs_to_build = [target for target in db_targets if completed_dbs.get(target['id']) == 'sql_generated']
    if dbs_to_build:
        logging.info(f"--- Phase 2: Building {len(dbs_to_build)} database(s) ---")
        for target in tqdm(dbs_to_build, desc="Building Databases"):
            db_id = target['id']
            sql_script_path = os.path.join(SQL_SCRIPT_DIR, f"{db_id}_vector.sql")
            final_db_path = os.path.join(VECTOR_DB_ROOT, db_id, f"{db_id}.sqlite")
            os.makedirs(os.path.dirname(final_db_path), exist_ok=True)
            try:
                build_vector_database(SQL_FILE=sql_script_path, DB_FILE=final_db_path)
                completed_dbs[db_id] = 'db_built'
                save_completion_status(status_file_path, completed_dbs)
            except Exception as e:
                logging.error(f"Error building database for '{db_id}': {e}", exc_info=True)
    else:
        logging.info("--- Phase 2: No new databases to build. ---")

    logging.info("--- Batch Vectorization Process Completed ---")

if __name__ == '__main__':
    main_batch_vectorize_databases(
        SOURCE_DB_ROOT=SOURCE_DB_ROOT,
        SQL_SCRIPT_DIR=SQL_SCRIPT_DIR,
        VECTOR_DB_ROOT=VECTOR_DB_ROOT,
        TABLE_JSON_PATH=TABLE_JSON_PATH
    )
