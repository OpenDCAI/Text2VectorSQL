import os
import json
import sqlite3
import shelve
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# --- Configuration ---
# Load environment variables from .env file
load_dotenv()

# VLLM and API settings
VLLM_API_URL = os.getenv("VLLM_API_URL")
VLLM_MODEL_NAME = os.getenv("VLLM_MODEL_NAME")
MAX_WORKERS = int(os.getenv("MAX_WORKERS", 16))
DB_BATCH_SIZE = 500 # How many rows to fetch from DB at a time
LLM_BATCH_SIZE = 16 # How many rows to send to LLM in a single request
MODEL_CONTEXT_LIMIT = 32768 # Ensure this matches your VLLM startup parameter
TOKEN_SAFETY_MARGIN = 4096 # A safety buffer

# File and directory paths
INPUT_JSON_PATH = "results/add_description_table_spider.json"
DB_BASE_PATH = "spider_data/database"
CACHE_FILE_PATH = os.getenv("CACHE_FILE_PATH", "cache/vllm_disk_cache")

# --- Database Utilities ---
def get_db_connection(db_path):
    """Establishes a connection to the SQLite database."""
    try:
        conn = sqlite3.connect(db_path, timeout=30) # Increase connection timeout
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        print(f"Error connecting to database {db_path}: {e}")
        return None

def get_primary_key_columns(cursor, table_name):
    """Finds the primary key column(s) for a table."""
    try:
        cursor.execute(f"PRAGMA table_info(`{table_name}`)")
        columns = cursor.fetchall()
        pk_cols = [row['name'] for row in columns if row['pk'] > 0]
        if not pk_cols and columns:
            # Fallback for tables without an explicit PRIMARY KEY
            cursor.execute(f"SELECT name FROM pragma_table_info('{table_name}') WHERE `notnull` = 1")
            not_null_cols = [row['name'] for row in cursor.fetchall()]
            if not_null_cols:
                return not_null_cols
            return [columns[0]['name']] # Last resort
        return pk_cols
    except sqlite3.Error as e:
        print(f"Warning: Could not determine primary key for {table_name}: {e}")
        return []


def add_description_column(db_path, table_name, new_column_name):
    """Adds a new text column to a table if it doesn't already exist."""
    conn = get_db_connection(db_path)
    if not conn: return
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info(`{table_name}`)")
            existing_columns = [row['name'] for row in cursor.fetchall()]
            if new_column_name not in existing_columns:
                cursor.execute(f'ALTER TABLE `{table_name}` ADD COLUMN `{new_column_name}` TEXT')
    except sqlite3.Error as e:
        print(f"Error adding column to {table_name}: {e}")
    finally:
        if conn:
            conn.close()

# --- LLM Interaction ---
def generate_single_row_description(client, schema_info, row_dict):
    """Generates a description for a single row, used as a fallback."""
    prompt = f"""You are a data enrichment expert. Your task is to generate a concise, natural-language description for the following row from a database table.
**Table Schema:**
- Table Name: {schema_info['table_name']}
- Columns: {', '.join(schema_info['columns'])}
**Row Data (JSON):**
{json.dumps(dict(row_dict), indent=2, ensure_ascii=False)}
Based on the schema and data, generate a single-sentence description.
Generated Description:"""
    try:
        response = client.chat.completions.create(
            model=VLLM_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2, max_tokens=200)
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"\nSingle row API call failed: {e}")
        return None

def generate_batch_descriptions(client, schema_info, batch_of_rows):
    """Generates descriptions for a batch of rows with dynamic token calculation and retries."""
    row_jsons = [json.dumps(dict(row), ensure_ascii=False) for row in batch_of_rows]
    prompt = f"""You are a data enrichment expert. Your task is to generate a concise, natural-language description for each row in the provided JSON array.
**Table Schema:**
- Table Name: {schema_info['table_name']}
- Columns: {', '.join(schema_info['columns'])}
**Row Data (JSON Array):**
{json.dumps(row_jsons, indent=2, ensure_ascii=False)}
Based on the schema and data, generate a description for EACH row. Return your response as a single JSON object with one key "descriptions" which is a JSON array of strings. The output array MUST have the same number of elements as the input array."""
    
    prompt_tokens = len(prompt) // 3 # A more conservative estimate for tokens
    available_tokens = MODEL_CONTEXT_LIMIT - prompt_tokens - TOKEN_SAFETY_MARGIN
    if available_tokens <= 0: return [None] * len(batch_of_rows)
    max_output_tokens = min(available_tokens, 200 * len(batch_of_rows))

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=VLLM_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2, max_tokens=max_output_tokens, response_format={"type": "json_object"})
            response_data = json.loads(response.choices[0].message.content)
            descriptions = response_data.get("descriptions")
            if isinstance(descriptions, list) and len(descriptions) == len(batch_of_rows):
                return descriptions
            time.sleep(2 * (attempt + 1))
        except Exception as e:
            if "context length" in str(e).lower(): break
            print(f"\nLLM batch API call failed on attempt {attempt + 1}: {e}. Retrying...")
            time.sleep(5 * (attempt + 1))
    return [None] * len(batch_of_rows)

# --- Main Processing Logic ---
def process_table(db_id, table_name, table_meta, client, cache):
    """Orchestrates the entire process for a single table using batching and frequent commits."""
    db_path = os.path.join(DB_BASE_PATH, db_id, f"{db_id}.sqlite")
    if not os.path.exists(db_path): return

    new_column_name = table_meta['name']
    add_description_column(db_path, table_name, new_column_name)

    conn = get_db_connection(db_path)
    if not conn: return
    
    try:
        cursor = conn.cursor()
        pk_cols = get_primary_key_columns(cursor, table_name)
        if not pk_cols:
            print(f"Skipping table {table_name} as no suitable primary key could be determined.")
            return

        cursor.execute(f"SELECT COUNT(*) FROM `{table_name}` WHERE `{new_column_name}` IS NULL")
        total_rows_to_process = cursor.fetchone()[0]
        if total_rows_to_process == 0: return

        with tqdm(total=total_rows_to_process, desc=f"  - Table '{table_name}'", leave=False) as pbar:
            last_batch_pks = None # Safety check for infinite loops
            while True:
                query = f"SELECT * FROM `{table_name}` WHERE `{new_column_name}` IS NULL LIMIT ?"
                cursor.execute(query, (DB_BATCH_SIZE,))
                db_batch = cursor.fetchall()
                if not db_batch: break

                # **STUCK LOOP DETECTOR**
                current_batch_pks = { "_".join(map(str, [r[c] for c in pk_cols])) for r in db_batch }
                if current_batch_pks == last_batch_pks:
                    print(f"\nError: Stuck in a loop on table '{table_name}'. The same batch is being fetched repeatedly. This likely means database UPDATES are failing. Aborting this table.")
                    break
                last_batch_pks = current_batch_pks

                all_columns = list(db_batch[0].keys())
                schema_info = {'table_name': table_name, 'columns': all_columns}
                
                rows_to_send_to_llm = []
                for row in db_batch:
                    pk_str = "_".join(map(str, [row[col] for col in pk_cols]))
                    cache_key = f"{db_id}|{table_name}|{pk_str}"
                    try:
                        if cache_key not in cache:
                            rows_to_send_to_llm.append(row)
                        else:
                            pbar.update(1)
                    except (pickle.UnpicklingError, EOFError):
                        rows_to_send_to_llm.append(row)
                
                if not rows_to_send_to_llm: continue

                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    llm_batches = [rows_to_send_to_llm[i:i + LLM_BATCH_SIZE] for i in range(0, len(rows_to_send_to_llm), LLM_BATCH_SIZE)]
                    future_to_batch = {executor.submit(generate_batch_descriptions, client, schema_info, batch): batch for batch in llm_batches}

                    for future in as_completed(future_to_batch):
                        original_batch = future_to_batch[future]
                        descriptions = future.result()
                        
                        update_params = []
                        # Check if the batch failed and needs individual retries
                        if any(d is None for d in descriptions):
                            for row in original_batch:
                                single_desc = generate_single_row_description(client, schema_info, row)
                                if single_desc:
                                    pk_values = [row[col] for col in pk_cols]
                                    pk_str = "_".join(map(str, pk_values))
                                    cache_key = f"{db_id}|{table_name}|{pk_str}"
                                    cache[cache_key] = single_desc
                                    update_params.append(tuple([single_desc] + pk_values))
                        else: # Batch was successful
                            for i, row in enumerate(original_batch):
                                pk_values = [row[col] for col in pk_cols]
                                pk_str = "_".join(map(str, pk_values))
                                cache_key = f"{db_id}|{table_name}|{pk_str}"
                                cache[cache_key] = descriptions[i]
                                update_params.append(tuple([descriptions[i]] + pk_values))
                        
                        # **CRITICAL CHANGE: Commit after each small LLM batch**
                        if update_params:
                            update_cursor = conn.cursor()
                            where_clause = " AND ".join([f"`{col}` = ?" for col in pk_cols])
                            sql = f"UPDATE `{table_name}` SET `{new_column_name}` = ? WHERE {where_clause}"
                            update_cursor.executemany(sql, update_params)
                            conn.commit()
                        
                        pbar.update(len(original_batch))
    except sqlite3.Error as e:
        print(f"An error occurred while processing {table_name}: {e}")
    finally:
        if conn:
            conn.close()

def is_database_complete(db_schema):
    """Checks if a database has any remaining work to be done."""
    db_id = db_schema['db_id']
    db_path = os.path.join(DB_BASE_PATH, db_id, f"{db_id}.sqlite")
    if not os.path.exists(db_path) or 'new_description' not in db_schema:
        return True # Nothing to process, so it's "complete"

    conn = get_db_connection(db_path)
    if not conn:
        return True # Can't connect, treat as complete to avoid errors

    try:
        cursor = conn.cursor()
        for table_name, table_meta in db_schema['new_description'].items():
            new_column_name = table_meta['name']
            
            # Check if column exists first
            cursor.execute(f"PRAGMA table_info(`{table_name}`)")
            if new_column_name not in [row['name'] for row in cursor.fetchall()]:
                return False # Column doesn't exist, so it needs processing

            # Check for NULL values efficiently
            cursor.execute(f"SELECT 1 FROM `{table_name}` WHERE `{new_column_name}` IS NULL LIMIT 1")
            if cursor.fetchone():
                return False # Found at least one NULL row, so not complete
        
        return True # No NULL rows found in any relevant table
    except sqlite3.Error:
        return False # If there's a DB error, better to assume it needs processing
    finally:
        if conn:
            conn.close()

def main():
    """Main function to orchestrate the entire process."""
    print("🚀 Starting database description population script...")
    
    client = OpenAI(base_url=VLLM_API_URL, api_key="vllm", timeout=600.0)
    
    try:
        with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
            all_db_schemas = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Input file not found at '{INPUT_JSON_PATH}'")
        return

    schemas_to_process = [s for s in all_db_schemas if s.get('add_description') == "true"]
    print(f"Found {len(schemas_to_process)} databases marked for potential enhancement.")

    # **CRITICAL CHANGE: Pre-flight check to find databases that actually need work**
    truly_pending_schemas = []
    print("Performing pre-flight check to find incomplete databases...")
    for db_schema in tqdm(schemas_to_process, desc="Checking database status"):
        if not is_database_complete(db_schema):
            truly_pending_schemas.append(db_schema)

    print(f"Found {len(truly_pending_schemas)} databases that require processing.")

    os.makedirs(os.path.dirname(CACHE_FILE_PATH), exist_ok=True)
    with shelve.open(CACHE_FILE_PATH) as cache:
        # The main progress bar now only wraps the pending schemas
        for db_schema in tqdm(truly_pending_schemas, desc="Processing Databases"):
            db_id = db_schema['db_id']
            if 'new_description' in db_schema:
                for table_name, table_meta in db_schema['new_description'].items():
                    process_table(db_id, table_name, table_meta, client, cache)

    print("\n✅ All tasks completed!")

if __name__ == "__main__":
    main()
