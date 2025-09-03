import os
import sys
import logging
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer
import torchvision

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
    filemode='w'
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

    if not os.path.exists(TABLE_JSON_PATH):
        logging.error(f"Critical Error: The table info file was not found at '{TABLE_JSON_PATH}'")
        return

    model = None
    pool = None
    try:
        logging.info(f"Loading embedding model: '{EMBEDDING_MODEL_NAME}'...")
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

        try:
            db_ids = [name for name in os.listdir(SOURCE_DB_ROOT) if os.path.isdir(os.path.join(SOURCE_DB_ROOT, name))]
            if not db_ids:
                logging.error(f"No databases found in the source directory: {SOURCE_DB_ROOT}")
                return
        except FileNotFoundError:
            logging.error(f"Source database directory not found: {SOURCE_DB_ROOT}")
            return
            
        logging.info(f"Found {len(db_ids)} databases to process.")

        # This main progress bar will be visible in the console
        for db_id in tqdm(db_ids, desc="Overall Progress", unit="db", position=0, file=sys.stdout):
            logging.info(f"--- Processing database: {db_id} ---")

            source_db_path = os.path.join(SOURCE_DB_ROOT, db_id, f"{db_id}.sqlite")
            sql_script_path = os.path.join(SQL_SCRIPT_DIR, f"{db_id}_vector.sql")
            
            final_db_dir = os.path.join(VECTOR_DB_ROOT, db_id)
            os.makedirs(final_db_dir, exist_ok=True)
            final_db_path = os.path.join(final_db_dir, f"{db_id}.sqlite")

            if not os.path.exists(source_db_path):
                logging.warning(f"Skipping '{db_id}': Source file not found at '{source_db_path}'")
                continue

            try:
                logging.info(f"Step 1/2: Generating SQL script for '{db_id}'...")
                generate_database_script(
                    db_path=source_db_path,
                    output_file=sql_script_path,
                    embedding_model=model,
                    pool=pool,
                    table_json_path=TABLE_JSON_PATH
                )
                logging.info(f"Successfully generated SQL script: {sql_script_path}")

                logging.info(f"Step 2/2: Building vector database for '{db_id}'...")
                build_vector_database(
                    SQL_FILE=sql_script_path,
                    DB_FILE=final_db_path
                )
                logging.info(f"Successfully created vector database: {final_db_path}")

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
