# batch_vectorize_databases.py (最终版)

import os, sys, logging, json
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer
import torchvision

os.makedirs("logging", exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='logging/out.log', filemode='w')
torchvision.disable_beta_transforms_warning()

try:
    from vector_database_generate import generate_database_script, build_vector_database
except ImportError as e:
    logging.critical(f"Import Error: {e}.")
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

def main():
    logging.info("--- Starting Batch Database Vectorization ---")
    if not all([SOURCE_DB_ROOT, SQL_SCRIPT_DIR, VECTOR_DB_ROOT, TABLE_JSON_PATH]):
        logging.critical("One or more required configurations are missing in .env. Please check (SOURCE_DB_ROOT, SQL_SCRIPT_DIR, VECTOR_DB_ROOT, TABLE_JSON_PATH).")
        sys.exit(1)
    
    os.makedirs(SQL_SCRIPT_DIR, exist_ok=True)
    os.makedirs(VECTOR_DB_ROOT, exist_ok=True)
    status_file_path = os.path.join(VECTOR_DB_ROOT, "processing_status.json")
    completed_dbs = load_completion_status(status_file_path)
    
    model, pool = None, None
    try:
        db_targets = [{'id': os.path.splitext(f)[0], 'path': os.path.join(SOURCE_DB_ROOT, f)} for f in os.listdir(SOURCE_DB_ROOT) if f.endswith(('.sqlite', '.db'))]
        if not db_targets: return
            
        dbs_to_process = [target for target in db_targets if completed_dbs.get(target['id']) != 'db_built']
        if not dbs_to_process:
            logging.info("All databases are already processed.")
            return

        if any(completed_dbs.get(target['id']) != 'sql_generated' for target in dbs_to_process):
            model = SentenceTransformer(EMBEDDING_MODEL_NAME, device='cpu', cache_folder=model_path)
            if torch.cuda.is_available(): pool = model.start_multi_process_pool()

        for target in tqdm(db_targets, desc="Overall Progress"):
            db_id, source_db_path = target['id'], target['path']
            if completed_dbs.get(db_id) == 'db_built': continue

            logging.info(f"--- Processing database: {db_id} ---")
            sql_script_path = os.path.join(SQL_SCRIPT_DIR, f"{db_id}_vector.sql")
            final_db_path = os.path.join(VECTOR_DB_ROOT, db_id, f"{db_id}.sqlite")
            
            try:
                if completed_dbs.get(db_id) != 'sql_generated':
                    logging.info(f"Step 1/2: Generating SQL for '{db_id}'...")
                    generate_database_script(db_path=source_db_path, output_file=sql_script_path, embedding_model=model, pool=pool, table_json_path=TABLE_JSON_PATH)
                    completed_dbs[db_id] = 'sql_generated'
                    save_completion_status(status_file_path, completed_dbs)

                logging.info(f"Step 2/2: Building vector DB for '{db_id}'...")
                # 恢复为简单的调用
                build_vector_database(SQL_FILE=sql_script_path, DB_FILE=final_db_path)
                completed_dbs[db_id] = 'db_built'
                save_completion_status(status_file_path, completed_dbs)
            except Exception as e:
                logging.error(f"Error processing '{db_id}': {e}", exc_info=True)
                continue
    finally:
        if pool: model.stop_multi_process_pool(pool)
        logging.info("--- Batch Vectorization Process Completed ---")

if __name__ == '__main__':
    main()
