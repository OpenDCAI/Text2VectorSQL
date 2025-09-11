import os
import json
import sqlite3
import sqlite_vec  # Import the sqlite_vec library
from tqdm import tqdm
from dotenv import load_dotenv

# --- Configuration from .env ---
load_dotenv()

# Read the variables using os.getenv()
VECTOR_DB_ROOT = os.getenv("VECTOR_DB_ROOT_GENERATE_SCHEMA")
ORIGINAL_SCHEMA_PATH = os.getenv("ORIGINAL_SCHEMA_PATH")
OUTPUT_DIR = os.getenv("OUTPUT_DIR_GENERATE_SCHEMA")
OUTPUT_JSON_PATH = os.getenv("OUTPUT_JSON_PATH_GENERATE_SCHEMA")

def generate_schema_for_db(db_id, db_path, original_schema):
    """
    Connects to a single vector database, loads the vec extension,
    inspects its schema, and returns a dictionary in the BIRD format.
    """
    new_schema = {
        "db_id": db_id,
        "table_names_original": [],
        "table_names": [],
        "column_names_original": [[-1, "*"]],
        "column_names": [[-1, "*"]],
        "column_types": ["text"],
        "primary_keys": original_schema.get("primary_keys", []),
        "foreign_keys": original_schema.get("foreign_keys", [])
    }

    try:
        print(f"""processing da_path: {db_path}""")
        conn = sqlite3.connect(db_path)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY rowid")
        table_names = [row[0] for row in cursor.fetchall()]

        new_schema["table_names_original"] = table_names
        new_schema["table_names"] = table_names
        
        table_name_to_idx = {name: i for i, name in enumerate(table_names)}

        for table_name in table_names:
            table_idx = table_name_to_idx[table_name]
            cursor.execute(f'PRAGMA table_xinfo("{table_name}");')
            columns_info = cursor.fetchall()

            for col in columns_info:
                col_name = col[1]
                col_type = col[2].upper()
                if col[5] != 0:
                    continue
                new_schema["column_names_original"].append([table_idx, col_name])
                new_schema["column_names"].append([table_idx, col_name])
                if 'FLOAT' in col_type or '[' in col_type:
                    new_schema["column_types"].append("text")
                else:
                    new_schema["column_types"].append(col_type.lower())
        
        conn.close()
        return new_schema
    except sqlite3.Error as e:
        print(f"  [ERROR] Could not process database {db_id}: {e}")
        return None

def main():
    """
    Main function to find vector databases, generate their schemas,
    and write the result to a new JSON file.
    """
    print(f"Starting schema generation from vector databases in: {VECTOR_DB_ROOT}")

    try:
        with open(ORIGINAL_SCHEMA_PATH, 'r', encoding='utf-8') as f:
            original_schemas_list = json.load(f)
        original_schemas = {item['db_id']: item for item in original_schemas_list}
        print(f"Loaded {len(original_schemas)} original schemas for reference.")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"✖ Could not load original schema file '{ORIGINAL_SCHEMA_PATH}': {e}")
        return

    all_new_schemas = []
    
    # --- MODIFICATION START ---
    # Discover all potential database targets (either in subdirs or directly in the root)
    db_targets = []
    try:
        if not os.path.exists(VECTOR_DB_ROOT):
            raise FileNotFoundError
        
        for item_name in os.listdir(VECTOR_DB_ROOT):
            full_path = os.path.join(VECTOR_DB_ROOT, item_name)
            
            # Case 1: Item is a directory (original logic)
            if os.path.isdir(full_path):
                db_id = item_name
                # Check for both .sqlite and .db extensions for flexibility
                db_path_sqlite = os.path.join(full_path, f"{db_id}.sqlite")
                db_path_db = os.path.join(full_path, f"{db_id}.db")
                if os.path.exists(db_path_sqlite):
                    db_targets.append({'id': db_id, 'path': db_path_sqlite})
                elif os.path.exists(db_path_db):
                    db_targets.append({'id': db_id, 'path': db_path_db})

            # Case 2: Item is a database file directly in the root (new logic)
            elif os.path.isfile(full_path) and item_name.endswith(('.sqlite', '.db')):
                db_id = os.path.splitext(item_name)[0]
                db_targets.append({'id': db_id, 'path': full_path})

    except FileNotFoundError:
        print(f"✖ Vector database directory not found: {VECTOR_DB_ROOT}")
        return
    # --- MODIFICATION END ---

    print(f"Found {len(db_targets)} potential databases. Processing...")

    for target in tqdm(db_targets, desc="Processing Databases"):
        db_id = target['id']
        db_path = target['path']
        
        # This check is now implicitly handled by the discovery logic, but kept for safety
        if not os.path.exists(db_path):
            print(f"  [WARN] Skipping '{db_id}': database file not found at {db_path}")
            continue
        
        if db_id not in original_schemas:
            print(f"  [WARN] Skipping '{db_id}': no matching entry in original schema file.")
            continue
            
        original_schema = original_schemas[db_id]
        new_schema_data = generate_schema_for_db(db_id, db_path, original_schema)

        if new_schema_data:
            all_new_schemas.append(new_schema_data)

    try:
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        
        # Ensure OUTPUT_JSON_PATH is a full path
        full_output_path = os.path.join(OUTPUT_DIR, OUTPUT_JSON_PATH)
        
        with open(full_output_path, 'w', encoding='utf-8') as f:
            json.dump(all_new_schemas, f, indent=2, ensure_ascii=False)
        print(f"\n✔ Successfully created '{full_output_path}' with {len(all_new_schemas)} database schemas.")
    except IOError as e:
        print(f"\n✖ Failed to write output file: {e}")

if __name__ == '__main__':
    main()
