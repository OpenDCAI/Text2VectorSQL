import os
import sys
import logging
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer
import torchvision
import json

# --- Early Setup: Logging Configuration ---
# Create logging directory if it doesn't exist
os.makedirs("logging", exist_ok=True)

# Configure logging to write to a file, overwriting it on each run (filemode='w')
# All print statements will be replaced by logging calls, so they go to this file.
# The tqdm progress bar, which prints to stderr by default, will remain on the console.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='logging/out.log',
    filemode='a'
)

# --- Main Application ---

# Shield against torchvision Beta version warning
torchvision.disable_beta_transforms_warning()

try:
    from vector_database_generate import generate_database_script, build_vector_database
except ImportError as e:
    # Log critical error and exit if essential modules are missing
    logging.critical(f"Import Error: {e}. Please ensure vector_database_generate.py is accessible.")
    sys.exit(1)

# --- Configuration ---
from dotenv import load_dotenv

# 加载.env文件
load_dotenv()

# --- Configuration ---
SOURCE_DB_ROOT = os.getenv("SOURCE_DB_ROOT", "spider_data/database")
SQL_SCRIPT_DIR = os.getenv("SQL_SCRIPT_DIR", "results/vector_sql_spider1")
VECTOR_DB_ROOT = os.getenv("VECTOR_DB_ROOT", "results/vector_databases_spider1")
TABLE_JSON_PATH = os.getenv("TABLE_JSON_PATH", "./results/spider_json/embedding_after_add_description_tables_spider.json")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
model_path = os.getenv("model_path", "/mnt/b_public/data/yaodongwen/model")

# Set environment variable for Hugging Face model download mirror
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


# --- 关键修改：更新状态文件处理逻辑 ---
def load_completion_status(status_file):
    """加载已完成数据库的状态字典 {db_id: status}。"""
    if os.path.exists(status_file):
        try:
            with open(status_file, 'r', encoding='utf-8') as f:
                content = json.load(f)
                # 兼容旧的列表格式和新的字典格式
                if isinstance(content, list):
                    logging.warning("检测到旧版状态文件格式，将进行转换。")
                    return {db_id: "sql_generated" for db_id in content}
                return content
        except (json.JSONDecodeError, TypeError):
            logging.warning(f"状态文件 '{status_file}' 格式不正确或为空，将重新开始。")
            return {}
    return {}

def save_completion_status(status_file, completed_dbs_dict):
    """保存已完成数据库的状态字典。"""
    with open(status_file, 'w', encoding='utf-8') as f:
        json.dump(completed_dbs_dict, f, indent=2)
# --- 修改结束 ---


def main():
    """
    Main function to orchestrate the batch vectorization of databases.
    All standard output is redirected to 'logging/out.log', only progress bars are shown in console.
    """
    logging.info("--- Starting Batch Database Vectorization ---")
    os.makedirs(SQL_SCRIPT_DIR, exist_ok=True)
    os.makedirs(VECTOR_DB_ROOT, exist_ok=True)
    logging.info(f"Intermediate SQL scripts will be saved to: {SQL_SCRIPT_DIR}")
    logging.info(f"Final vector databases will be saved to: {VECTOR_DB_ROOT}")

    # --- 关键修改：使用新的状态文件逻辑 ---
    status_file_path = os.path.join(VECTOR_DB_ROOT, "processing_status.json")
    completed_dbs = load_completion_status(status_file_path)
    logging.info(f"已加载状态文件，发现 {len(completed_dbs)} 个有记录的数据库。")
    # --- 修改结束 ---

    if not os.path.exists(TABLE_JSON_PATH):
        logging.error(f"Critical Error: The table info file was not found at '{TABLE_JSON_PATH}'")
        return

    model = None
    pool = None
    try:
        # --- 模型加载只在需要时进行 ---
        # 检查是否所有数据库都已完成，如果完成则无需加载模型
        db_ids = [name for name in os.listdir(SOURCE_DB_ROOT) if os.path.isdir(os.path.join(SOURCE_DB_ROOT, name))]
        
        # 筛选出需要处理的数据库
        dbs_to_process = [db_id for db_id in db_ids if completed_dbs.get(db_id) != 'db_built']
        
        if not dbs_to_process:
            logging.info("所有数据库都已处理完毕，程序退出。")
            return

        # 筛选出需要生成SQL的数据库，只有当这个列表不为空时才加载模型
        dbs_needing_sql = [db_id for db_id in dbs_to_process if completed_dbs.get(db_id) != 'sql_generated']
        if dbs_needing_sql:
            logging.info(f"需要为 {len(dbs_needing_sql)} 个数据库生成SQL，开始加载嵌入模型...")
            model = SentenceTransformer(
                EMBEDDING_MODEL_NAME,
                device='cpu',
                cache_folder=model_path
            )
            logging.info("Embedding model loaded to CPU memory.")

            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                target_devices = [f'cuda:{i}' for i in range(gpu_count)]
                logging.info(f"Successfully identified {gpu_count} CUDA-enabled GPU(s).")
                logging.info(f"Preparing to start multi-process pool on devices: {target_devices}")
                pool = model.start_multi_process_pool(target_devices=target_devices)
                logging.info("Multi-GPU process pool started successfully.")
            else:
                logging.warning("No CUDA-enabled GPU detected. Running in single-process CPU mode.")
        else:
            logging.info("所有必需的SQL文件均已生成，无需加载嵌入模型。")
        # --- 模型加载优化结束 ---
            
        logging.info(f"共发现 {len(db_ids)} 个数据库，将处理其中 {len(dbs_to_process)} 个。")

        # This main progress bar will be visible in the console
        for db_id in tqdm(db_ids, desc="Overall Progress", unit="db", position=0, file=sys.stdout):
            
            db_status = completed_dbs.get(db_id)
            
            # --- 关键修改：分阶段检查状态 ---
            if db_status == 'db_built':
                logging.info(f"Skipping '{db_id}': Marked as 'db_built' in status file.")
                continue
            # --- 修改结束 ---

            logging.info(f"--- Processing database: {db_id} ---")

            source_db_path = os.path.join(SOURCE_DB_ROOT, db_id, f"{db_id}.sqlite")
            sql_script_path = os.path.join(SQL_SCRIPT_DIR, f"{db_id}_vector.sql")
            final_db_dir = os.path.join(VECTOR_DB_ROOT, db_id)
            final_db_path = os.path.join(final_db_dir, f"{db_id}.sqlite")
            
            os.makedirs(final_db_dir, exist_ok=True)

            if not os.path.exists(source_db_path):
                logging.warning(f"Skipping '{db_id}': Source file not found at '{source_db_path}'")
                continue

            try:
                # --- Stage 1: Generate SQL Script ---
                if db_status != 'sql_generated':
                    logging.info(f"Step 1/2: Generating SQL script for '{db_id}'...")
                    if not model:
                        logging.error(f"无法为 '{db_id}' 生成SQL，因为嵌入模型未加载。")
                        continue
                    generate_database_script(
                        db_path=source_db_path,
                        output_file=sql_script_path,
                        embedding_model=model,
                        pool=pool,
                        table_json_path=TABLE_JSON_PATH
                    )
                    logging.info(f"Successfully generated SQL script: {sql_script_path}")
                    
                    # 更新状态为第一阶段完成
                    completed_dbs[db_id] = 'sql_generated'
                    save_completion_status(status_file_path, completed_dbs)
                    logging.info(f"Marked '{db_id}' as 'sql_generated' in status file.")
                else:
                    logging.info(f"Step 1/2: Skipping SQL script generation for '{db_id}' (already complete).")


                # --- Stage 2: Build Vector Database ---
                logging.info(f"Step 2/2: Building vector database for '{db_id}'...")
                build_vector_database(
                    SQL_FILE=sql_script_path,
                    DB_FILE=final_db_path
                )
                logging.info(f"Successfully created vector database: {final_db_path}")

                # 更新状态为第二阶段完成
                completed_dbs[db_id] = 'db_built'
                save_completion_status(status_file_path, completed_dbs)
                logging.info(f"Marked '{db_id}' as 'db_built' in status file.")

            except Exception as e:
                logging.error(f"An error occurred while processing '{db_id}': {e}", exc_info=True)
                continue

        logging.info("--- Batch Vectorization Process Completed ---")

    except Exception as e:
        logging.critical(f"A critical error occurred in the main process. Error: {e}", exc_info=True)
    finally:
        if pool:
            logging.info("Stopping multi-GPU process pool...")
            model.stop_multi_process_pool(pool)
            logging.info("Process pool stopped.")

if __name__ == '__main__':
    main()
