import os
import json
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# --- Configuration ---
# Load environment variables from .env file
load_dotenv()

BASE_URL = os.getenv("VLLM_API_URL")
API_KEY = os.getenv("API_KEY")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")
MAX_WORKERS = int(os.getenv("MAX_WORKERS", 8))
CACHE_FILE_PATH = os.getenv("CACHE_FILE_PATH", "results/llm_cache_spider.json")
INPUT_JSON_PATH = "results/enhanced_train_tables_spider.json"
OUTPUT_JSON_PATH = "results/add_description_table_spider.json"

# --- Caching ---
def load_cache():
    """Loads the LLM response cache from a file."""
    if os.path.exists(CACHE_FILE_PATH):
        try:
            with open(CACHE_FILE_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}

def save_cache(cache):
    """Saves the LLM response cache to a file."""
    try:
        os.makedirs(os.path.dirname(CACHE_FILE_PATH), exist_ok=True)
        with open(CACHE_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=4)
    except IOError as e:
        print(f"Error saving cache: {e}")

# --- LLM Interaction (MODIFIED FOR VLLM COMPATIBILITY) ---
def get_llm_judgment(client, db_id, table_name, table_info, cache):
    """
    Asks the LLM to judge if a table is suitable for a description column.
    Uses a cache to avoid re-processing.
    This version is compatible with both OpenAI and VLLM endpoints.
    """
    cache_key = f"{db_id}|{table_name}"
    if cache_key in cache:
        return cache[cache_key]

    prompt = f"""
    You are a database schema analysis expert. Your task is to determine if a table is a good candidate for adding a new text column called 'description'.

    A table is a **good candidate** if it represents core entities that could have a natural language description (e.g., products, movies, employees, customers).
    A table is a **poor candidate** if it's a linking/junction table (many-to-many relationships), a log table, or primarily stores simple relationships or numerical data without a clear core entity to describe.

    Analyze the following table schema and its sample data:

    **Table Name:** `{table_name}`

    **Columns:**
    {json.dumps([col[1] for col in table_info['columns']], indent=2)}

    **Sample Data (up to 2 rows):**
    {json.dumps(table_info['samples'], indent=2)}

    Based on this information, is this table a good candidate for adding a 'description' column?
    Respond ONLY with a valid JSON object with two keys:
    1. "is_suitable": boolean (true or false)
    2. "reason": A brief explanation for your decision.
    """

    result_json = None
    
    # MODIFICATION: Try to use JSON mode first (for OpenAI API).
    # If it fails, fall back to standard text completion (for VLLM).
    try:
        # --- Attempt 1: JSON Mode (Ideal for OpenAI) ---
        response = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a database schema analysis expert. Respond ONLY with a valid JSON object."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        result_str = response.choices[0].message.content
        result_json = json.loads(result_str)

    except Exception as e:
        # --- Attempt 2: Standard Mode (Fallback for VLLM) ---
        print(f"\n⚠️ Warning: JSON mode failed for '{table_name}' (as expected for most VLLM models). Retrying in standard mode. Error: {e}")
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a database schema analysis expert. Respond ONLY with a valid JSON object."},
                    {"role": "user", "content": prompt}
                ],
                # Do NOT include response_format here
                temperature=0.1,
            )
            result_str = response.choices[0].message.content
            
            # MODIFICATION: Extract JSON from the raw text response, which might include markdown.
            # This makes parsing more robust.
            json_match = re.search(r'\{.*\}', result_str, re.DOTALL)
            if json_match:
                clean_json_str = json_match.group(0)
                result_json = json.loads(clean_json_str)
            else:
                raise ValueError("No JSON object found in the response string.")

        except Exception as fallback_e:
            # If both attempts fail, return an error.
            return {"is_suitable": False, "reason": f"LLM API call failed in both modes. Fallback error: {str(fallback_e)}"}

    # --- Final Validation and Caching ---
    if result_json and "is_suitable" in result_json and "reason" in result_json:
        cache[cache_key] = result_json
        return result_json
    else:
        return {"is_suitable": False, "reason": "Invalid response format or content from LLM."}


def process_database_schema(db_schema, client, cache):
    """
    Processes a single database schema, analyzing all its tables in parallel.
    """
    db_id = db_schema['db_id']
    table_names = db_schema['table_names_original']
    
    suitable_tables = {}
    
    tasks = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for i, table_name in enumerate(table_names):
            table_info = {
                "columns": [col for col in db_schema['column_names_original'] if col[0] == i],
                "samples": db_schema.get('table_samples', {}).get(table_name, [])
            }
            tasks.append(executor.submit(get_llm_judgment, client, db_id, table_name, table_info, cache))

        for future in as_completed(tasks):
            # This part is intentionally left simple as we process results later.
            pass

    for table_name in table_names:
        cache_key = f"{db_id}|{table_name}"
        result = cache.get(cache_key)
        if result and result.get('is_suitable'):
            suitable_tables[table_name] = {
                "name": f"{table_name}_description",
                "reason": result.get('reason', 'No reason provided.')
            }

    if suitable_tables:
        db_schema['add_description'] = "true"
        db_schema['new_description'] = suitable_tables
    else:
        db_schema['add_description'] = "false"

    return db_schema


def main():
    """
    Main function to orchestrate the entire process.
    """
    print("🚀 Starting database schema analysis...")

    # MODIFICATION: Handle the API key for VLLM, which often can be empty or a placeholder.
    client = OpenAI(
        api_key=API_KEY or "EMPTY", # Use a placeholder if API_KEY is not set
        base_url=BASE_URL
    )

    try:
        with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
            all_db_schemas = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Input file not found at '{INPUT_JSON_PATH}'")
        return
    except json.JSONDecodeError:
        print(f"❌ Error: Could not decode JSON from '{INPUT_JSON_PATH}'")
        return

    cache = load_cache()
    print(f"Loaded {len(cache)} items from cache.")

    updated_schemas = []
    
    progress_bar = tqdm(total=len(all_db_schemas), desc="Processing Databases")

    for db_schema in all_db_schemas:
        updated_schema = process_database_schema(db_schema, client, cache)
        updated_schemas.append(updated_schema)
        save_cache(cache)
        progress_bar.update(1)

    progress_bar.close()

    try:
        os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
        with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(updated_schemas, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Analysis complete! Results saved to '{OUTPUT_JSON_PATH}'")
    except IOError as e:
        print(f"❌ Error writing output file: {e}")


if __name__ == "__main__":
    main()
