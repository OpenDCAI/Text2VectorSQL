# batch_vectorize_databases.py (最终版)

import os, sys, logging, json
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer, models
from transformers import AutoConfig
import torchvision
from typing import Optional

os.makedirs("logging", exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='logging/out.log', filemode='w')
torchvision.disable_beta_transforms_warning()

try:
    from .vector_database_generate import generate_database_script, build_vector_database
except ImportError as e:
    logging.critical(f"Import Error: {e}.")
    sys.exit(1)

from dotenv import load_dotenv
load_dotenv()

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

def load_universal_sentence_transformer(model_name: str, cache_folder: str, device: str = 'cpu') -> SentenceTransformer:
    """
    加载一个 SentenceTransformer 模型，能自动兼容处理标准的 Transformer 模型和 CLIP 模型。

    Args:
        model_name (str): 需要加载的模型名称或路径 (例如 "all-MiniLM-L6-v2" 或 "openai/clip-vit-base-patch32")。
        cache_folder (str): 用于缓存下载的模型的文件夹路径。
        device (str, optional): 加载模型的设备 ('cpu', 'cuda', etc.)。默认为 'cpu'。

    Returns:
        SentenceTransformer: 初始化完成的模型实例。
    """
    try:
        config = AutoConfig.from_pretrained(model_name, cache_dir=cache_folder)
        is_clip_model = 'clip' in getattr(config, 'model_type', '').lower()
    except Exception as e:
        logging.warning(f"无法加载模型 '{model_name}' 的配置. 将基于模型名称进行判断。错误: {e}")
        is_clip_model = 'clip' in model_name.lower()

    if is_clip_model:
        logging.info(f"正在以 CLIP 模型方式加载: '{model_name}'")
        clip_model_wrapper = models.CLIPModel(model_name)
        
        # 外层的 SentenceTransformer 会处理缓存，所以这里 cache_folder 参数是必须的
        model = SentenceTransformer(modules=[clip_model_wrapper], device=device, cache_folder=cache_folder)
    else:
        logging.info(f"正在以标准模型方式加载: '{model_name}'")
        model = SentenceTransformer(model_name, device=device, cache_folder=cache_folder)
        
    return model

def main_batch_vectorize_databases(
        SOURCE_DB_ROOT,
        SQL_SCRIPT_DIR,
        VECTOR_DB_ROOT,
        TABLE_JSON_PATH,
        EMBEDDING_MODEL_NAME,
        model_path
    ):
    logging.info("--- Starting Batch Database Vectorization ---")
    if not all([SOURCE_DB_ROOT, SQL_SCRIPT_DIR, VECTOR_DB_ROOT, TABLE_JSON_PATH]):
        logging.critical("One or more required configurations are missing in main_batch_vectorize_databases. Please check (SOURCE_DB_ROOT, SQL_SCRIPT_DIR, VECTOR_DB_ROOT, TABLE_JSON_PATH).")
        sys.exit(1)
    
    if not os.path.exists(SOURCE_DB_ROOT):
        print(f"error: no source db: {SOURCE_DB_ROOT}")

    os.makedirs(SQL_SCRIPT_DIR, exist_ok=True)
    os.makedirs(VECTOR_DB_ROOT, exist_ok=True)
    status_file_path = os.path.join(VECTOR_DB_ROOT, "processing_status.json")
    completed_dbs = load_completion_status(status_file_path)
    
    model, pool = None, None
    try:
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
            print(f"error: there is not file in {db_targets}")
            return
            
        dbs_to_process = [target for target in db_targets if completed_dbs.get(target['id']) != 'db_built']
        if not dbs_to_process:
            logging.info("All databases are already processed.")
            return

        if any(completed_dbs.get(target['id']) != 'sql_generated' for target in dbs_to_process):
            model = load_universal_sentence_transformer(EMBEDDING_MODEL_NAME, model_path)
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
    SOURCE_DB_ROOT = "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/sqlite/train/arxiv"
    SQL_SCRIPT_DIR = "./vector_sql"
    VECTOR_DB_ROOT = "./vector_databases"
    TABLE_JSON_PATH = '/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/sqlite/results/arxiv/find_semantic_tables.json'
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
    model_path = "/mnt/b_public/data/yaodongwen/model"
    main_batch_vectorize_databases(
        SOURCE_DB_ROOT,
        SQL_SCRIPT_DIR,
        VECTOR_DB_ROOT,
        TABLE_JSON_PATH,
        EMBEDDING_MODEL_NAME,
        model_path
    )
