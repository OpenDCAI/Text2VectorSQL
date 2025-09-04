# batch_vectorize_databases.py (最终修正版)

import os
import sys
import logging
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer
import torchvision
import json

# --- Early Setup: Logging Configuration ---
os.makedirs("logging", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='logging/out.log',
    filemode='w'  # 修改为 'w'，每次运行清空日志，方便查看本次运行信息
)

# --- Main Application ---
torchvision.disable_beta_transforms_warning()

try:
    from vector_database_generate import generate_database_script, build_vector_database
except ImportError as e:
    logging.critical(f"Import Error: {e}. Please ensure vector_database_generate.py is accessible.")
    sys.exit(1)

# --- Configuration ---
from dotenv import load_dotenv
load_dotenv()

# --- Configuration ---
SOURCE_DB_ROOT = os.getenv("SOURCE_DB_ROOT") # 直接从.env读取，不再设置易出错的默认值
SQL_SCRIPT_DIR = os.getenv("SQL_SCRIPT_DIR")
VECTOR_DB_ROOT = os.getenv("VECTOR_DB_ROOT")
TABLE_JSON_PATH = os.getenv("TABLE_JSON_PATH")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
model_path = os.getenv("model_path", "/mnt/b_public/data/yaodongwen/model")

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def load_completion_status(status_file):
    if os.path.exists(status_file):
        try:
            with open(status_file, 'r', encoding='utf-8') as f:
                content = json.load(f)
                if isinstance(content, list):
                    logging.warning("检测到旧版状态文件格式，将进行转换。")
                    return {db_id: "sql_generated" for db_id in content}
                return content
        except (json.JSONDecodeError, TypeError):
            logging.warning(f"状态文件 '{status_file}' 格式不正确或为空，将重新开始。")
            return {}
    return {}

def save_completion_status(status_file, completed_dbs_dict):
    with open(status_file, 'w', encoding='utf-8') as f:
        json.dump(completed_dbs_dict, f, indent=2)

def main():
    logging.info("--- Starting Batch Database Vectorization ---")

    # --- 关键检查：在开始时就验证所有必要的路径配置 ---
    if not all([SOURCE_DB_ROOT, SQL_SCRIPT_DIR, VECTOR_DB_ROOT, TABLE_JSON_PATH]):
        logging.critical("一个或多个必要的路径配置 (SOURCE_DB_ROOT, SQL_SCRIPT_DIR, VECTOR_DB_ROOT, TABLE_JSON_PATH) 未在 .env 文件中设置。程序退出。")
        sys.exit(1)
    
    os.makedirs(SQL_SCRIPT_DIR, exist_ok=True)
    os.makedirs(VECTOR_DB_ROOT, exist_ok=True)
    logging.info(f"Intermediate SQL scripts will be saved to: {SQL_SCRIPT_DIR}")
    logging.info(f"Final vector databases will be saved to: {VECTOR_DB_ROOT}")

    status_file_path = os.path.join(VECTOR_DB_ROOT, "processing_status.json")
    completed_dbs = load_completion_status(status_file_path)
    logging.info(f"已加载状态文件，发现 {len(completed_dbs)} 个有记录的数据库。")

    if not os.path.exists(TABLE_JSON_PATH):
        logging.error(f"Critical Error: The table info file was not found at '{TABLE_JSON_PATH}'")
        return

    model = None
    pool = None
    try:
        # --- 数据库发现逻辑 (已修复) ---
        db_targets = []
        if not os.path.exists(SOURCE_DB_ROOT):
            logging.error(f"源数据库目录 '{SOURCE_DB_ROOT}' 不存在！请检查 .env 配置。")
            return
            
        logging.info(f"正在从 '{SOURCE_DB_ROOT}' 发现数据库...")
        for item_name in os.listdir(SOURCE_DB_ROOT):
            full_path = os.path.join(SOURCE_DB_ROOT, item_name)
            # 情况1: 项目是一个目录 (例如 spider/database/concert_singer)
            if os.path.isdir(full_path):
                db_id = item_name
                db_file_path = os.path.join(full_path, f"{db_id}.sqlite")
                if not os.path.exists(db_file_path):
                    db_file_path = os.path.join(full_path, f"{db_id}.db")
                
                if os.path.exists(db_file_path):
                    db_targets.append({'id': db_id, 'path': db_file_path})
            # 情况2: 项目是一个数据库文件 (例如 train/arxiv.db)
            elif os.path.isfile(full_path) and item_name.endswith(('.sqlite', '.db')):
                db_id = os.path.splitext(item_name)[0]
                db_targets.append({'id': db_id, 'path': full_path})
        
        if not db_targets:
            logging.warning(f"在 '{SOURCE_DB_ROOT}' 中未发现任何有效的数据库。程序退出。")
            return
            
        dbs_to_process = [target for target in db_targets if completed_dbs.get(target['id']) != 'db_built']
        
        if not dbs_to_process:
            logging.info("所有数据库都已处理完毕，程序退出。")
            return

        dbs_needing_sql = [target for target in dbs_to_process if completed_dbs.get(target['id']) != 'sql_generated']
        if dbs_needing_sql:
            logging.info(f"需要为 {len(dbs_needing_sql)} 个数据库生成SQL，开始加载嵌入模型...")
            model = SentenceTransformer(EMBEDDING_MODEL_NAME, device='cpu', cache_folder=model_path)
            logging.info("Embedding model loaded.")
            if torch.cuda.is_available():
                pool = model.start_multi_process_pool()
                logging.info("Multi-GPU process pool started.")
        
        logging.info(f"共发现 {len(db_targets)} 个数据库，将处理其中 {len(dbs_to_process)} 个。")

        for target in tqdm(db_targets, desc="Overall Progress", unit="db", position=0, file=sys.stdout):
            db_id = target['id']
            source_db_path = target['path']
            
            if completed_dbs.get(db_id) == 'db_built':
                continue

            logging.info(f"--- Processing database: {db_id} ---")
            sql_script_path = os.path.join(SQL_SCRIPT_DIR, f"{db_id}_vector.sql")
            final_db_dir = os.path.join(VECTOR_DB_ROOT, db_id)
            final_db_path = os.path.join(final_db_dir, f"{db_id}.sqlite")
            os.makedirs(final_db_dir, exist_ok=True)

            try:
                if completed_dbs.get(db_id) != 'sql_generated':
                    logging.info(f"Step 1/2: Generating SQL for '{db_id}'...")
                    generate_database_script(
                        db_path=source_db_path, output_file=sql_script_path,
                        embedding_model=model, pool=pool, table_json_path=TABLE_JSON_PATH
                    )
                    completed_dbs[db_id] = 'sql_generated'
                    save_completion_status(status_file_path, completed_dbs)
                
                logging.info(f"Step 2/2: Building vector DB for '{db_id}'...")
                build_vector_database(SQL_FILE=sql_script_path, DB_FILE=final_db_path)
                completed_dbs[db_id] = 'db_built'
                save_completion_status(status_file_path, completed_dbs)

            except Exception as e:
                logging.error(f"Error processing '{db_id}': {e}", exc_info=True)
                continue

        logging.info("--- Batch Vectorization Process Completed ---")
    except Exception as e:
        logging.critical(f"A critical error occurred: {e}", exc_info=True)
    finally:
        if pool:
            model.stop_multi_process_pool(pool)
            logging.info("Process pool stopped.")

if __name__ == '__main__':
    main()
