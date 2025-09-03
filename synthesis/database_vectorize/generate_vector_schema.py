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
    # Initialize the new schema dictionary.
    # FIX: The 'column_alter' field is now included again.
    new_schema = {
        "db_id": db_id,
        "table_names_original": [],
        "table_names": [],
        "column_names_original": [[-1, "*"]],
        "column_names": [[-1, "*"]],
        "column_types": ["text"],  # Type for the wildcard '*'
        "primary_keys": original_schema.get("primary_keys", []),
        "foreign_keys": original_schema.get("foreign_keys", []),
        "column_alter": original_schema.get("column_alter", {}) # Re-added this field
    }

    try:
        conn = sqlite3.connect(db_path)
        
        # Load the sqlite_vec extension to understand virtual tables
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        
        cursor = conn.cursor()

        # Get all table names, preserving their order
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY rowid")
        table_names = [row[0] for row in cursor.fetchall()]

        new_schema["table_names_original"] = table_names
        new_schema["table_names"] = table_names
        
        table_name_to_idx = {name: i for i, name in enumerate(table_names)}

        # Iterate through each table to get its columns
        for table_name in table_names:
            table_idx = table_name_to_idx[table_name]
            
            # Use PRAGMA table_xinfo for better virtual table support
            cursor.execute(f'PRAGMA table_xinfo("{table_name}");')
            columns_info = cursor.fetchall()

            for col in columns_info:
                col_name = col[1]
                col_type = col[2].upper()

                # Don't add hidden columns from the virtual table
                if col[5] != 0:  # 'hidden' column in table_xinfo result
                    continue

                new_schema["column_names_original"].append([table_idx, col_name])
                
                # For the new schema, the "pretty" name is the same as the actual column name
                new_schema["column_names"].append([table_idx, col_name])

                # Normalize vector types to 'text' for consistency
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

    # Load original schema file to a dictionary for easy lookup
    try:
        with open(ORIGINAL_SCHEMA_PATH, 'r', encoding='utf-8') as f:
            original_schemas_list = json.load(f)
        original_schemas = {item['db_id']: item for item in original_schemas_list}
        print(f"Loaded {len(original_schemas)} original schemas for reference.")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"✖ Could not load original schema file '{ORIGINAL_SCHEMA_PATH}': {e}")
        return

    all_new_schemas = []
    
    # Discover all database directories
    try:
        db_ids = [name for name in os.listdir(VECTOR_DB_ROOT) if os.path.isdir(os.path.join(VECTOR_DB_ROOT, name))]
    except FileNotFoundError:
        print(f"✖ Vector database directory not found: {VECTOR_DB_ROOT}")
        return

    print(f"Found {len(db_ids)} potential database directories. Processing...")

    for db_id in tqdm(db_ids, desc="Processing Databases"):
        db_path = os.path.join(VECTOR_DB_ROOT, db_id, f"{db_id}.sqlite")
        
        if not os.path.exists(db_path):
            print(f"  [WARN] Skipping '{db_id}': database file not found at {db_path}")
            continue
        
        if db_id not in original_schemas:
            print(f"  [WARN] Skipping '{db_id}': no matching entry in original schema file.")
            continue
            
        original_schema = original_schemas[db_id]
        
        # Generate the new schema by inspecting the live database
        new_schema_data = generate_schema_for_db(db_id, db_path, original_schema)

        if new_schema_data:
            all_new_schemas.append(new_schema_data)

    # Write the final list of schemas to the output JSON file
    try:
        # 检查目录是否存在，不存在则创建
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)  # 可以创建多级目录
        global OUTPUT_JSON_PATH
        OUTPUT_JSON_PATH = os.path.join(OUTPUT_DIR, OUTPUT_JSON_PATH)
        with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(all_new_schemas, f, indent=2, ensure_ascii=False)
        print(f"\n✔ Successfully created '{OUTPUT_JSON_PATH}' with {len(all_new_schemas)} database schemas.")
    except IOError as e:
        print(f"\n✖ Failed to write output file: {e}")


if __name__ == '__main__':
    main()
