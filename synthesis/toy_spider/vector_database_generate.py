# vector_database_generate.py (已修复索引问题)

import sqlite3
import datetime
import traceback
import json
from tqdm import tqdm
import logging
from sentence_transformers import SentenceTransformer
import sqlite_vec
from io import StringIO
import os
import re

def sanitize_identifier(identifier: str) -> str:
    s = str(identifier)
    s = s.replace(' ', '_').replace('(', '').replace(')', '')
    s = re.sub(r'[^a-zA-Z0-9_]', '_', s)
    if not re.match(r'^[a-zA-Z]', s):
        s = 'fld_' + s
    if s.lower() == 'distance':
        s = 'distance_val'
    return s

def type_convert(original_type):
    if not original_type:
        return 'TEXT'
    original_type = original_type.upper()
    type_keyword = original_type.split('(')[0]
    if "INT" in type_keyword or "NUMBER" in type_keyword:
        return 'INTEGER'
    if "REAL" in type_keyword or "NUMERIC" in type_keyword or "DECIMAL" in type_keyword or "FLOAT" in type_keyword or "DOUBLE" in type_keyword:
        return 'FLOAT'
    return 'TEXT'

def create_virtual_table_ddl(conn, table_name, db_info, vec_dim):
    cursor = conn.cursor()
    cursor.execute(f'PRAGMA table_info("{table_name}");')
    original_columns_info = cursor.fetchall()
    column_definitions = []
    for col_info in original_columns_info:
        original_col_name, original_type = col_info[1], col_info[2]
        sanitized_col_name = sanitize_identifier(original_col_name)
        new_type = type_convert(original_type)
        column_definitions.append(f'  {sanitized_col_name} {new_type}')
    if table_name in db_info.get("semantic_rich_column", {}):
        for col_info in db_info["semantic_rich_column"][table_name]:
            original_col_name = col_info.get('column_name')
            if original_col_name in [c[1] for c in original_columns_info]:
                sanitized_col_name = sanitize_identifier(original_col_name)
                column_definitions.append(f'  {sanitized_col_name}_embedding float[{vec_dim}]')
    columns_str = ",\n".join(column_definitions)
    return f"CREATE VIRTUAL TABLE \"{table_name}\" USING vec0(\n{columns_str}\n);"

def generate_embeddings_parallel(model, texts, batch_size=128, pool=None):
    if pool:
        return model.encode_multi_process(texts, pool=pool, batch_size=batch_size)
    else:
        return model.encode(texts, batch_size=batch_size, show_progress_bar=False)

def export_to_single_sql_file(db_path, output_file, db_info, embedding_model, pool=None):
    PROCESSING_BATCH_SIZE = 5000 
    conn = sqlite3.connect(db_path)
    vec_dim = embedding_model.get_sentence_embedding_dimension()
    sql_buffer = StringIO()
    
    def sql_format_value(val):
        if val is None:
            return "NULL"
        if isinstance(val, bytes):
            return f"X'{val.hex()}'"
        if isinstance(val, str):
            return f"'{val.replace('\'', '\'\'')}'"
        return str(val)

    try:
        sql_buffer.write(f"-- SQLite Database Export\n-- Exported at: {datetime.datetime.now()}\n-- Source: {db_path}\n\n")
        sql_buffer.write("PRAGMA foreign_keys = OFF;\nBEGIN TRANSACTION;\n\n")
        
        objects = conn.execute("SELECT type, name, tbl_name, sql FROM sqlite_master WHERE type IN ('table', 'view', 'trigger', 'index') AND name NOT LIKE 'sqlite_%' ORDER BY CASE type WHEN 'table' THEN 1 WHEN 'view' THEN 4 ELSE 3 END").fetchall()
        
        created_tables = []
        table_objects = [obj for obj in objects if obj[0] == 'table' and obj[1] and obj[3]]
        
        db_info["can_convert_virtual"] = {}
        for _, name, _, sql in table_objects:
            sql_buffer.write(f"-- Table: {name}\n")
            db_info["can_convert_virtual"][name] = False
            if name in db_info.get("semantic_rich_column", {}):
                try:
                    table_info_cursor = conn.execute(f"PRAGMA table_info('{name}')")
                    columns_info = table_info_cursor.fetchall()
                    original_col_names = [c[1] for c in columns_info]
                    semantic_cols_from_json = db_info["semantic_rich_column"].get(name, [])
                    valid_semantic_cols = [col for col in semantic_cols_from_json if col.get('column_name') in original_col_names]
                    if valid_semantic_cols and (len(columns_info) + len(valid_semantic_cols) <= 16):
                        ddl = create_virtual_table_ddl(conn, name, db_info, vec_dim)
                        db_info["can_convert_virtual"][name] = True
                    else:
                        ddl = sql + ';'
                except Exception as e:
                    logging.warning(f"Could not get column info for table '{name}' ({e}), using original schema.")
                    ddl = sql + ';'
            else:
                ddl = sql + ';'
            sql_buffer.write(ddl + "\n\n")
            created_tables.append(name)
            
        sql_buffer.write("\n-- DATA INSERTION --\n\n")

        for table in tqdm(created_tables, desc="Processing tables"):
            is_virtual_table = db_info["can_convert_virtual"].get(table, False)
            cursor = conn.execute(f'SELECT * FROM "{table}"')
            original_col_names = [desc[0] for desc in cursor.description]
            
            all_cols_list = []
            embedding_col_names = []
            if is_virtual_table:
                sanitized_cols = [sanitize_identifier(c) for c in original_col_names]
                embedding_cols_info = [c for c in db_info.get("semantic_rich_column", {}).get(table, []) if c.get('column_name') in original_col_names]
                embedding_col_names = [col['column_name'] for col in embedding_cols_info]
                sanitized_embedding_cols = [f"{sanitize_identifier(c)}_embedding" for c in embedding_col_names]
                all_cols_list = sanitized_cols + sanitized_embedding_cols
            else:
                all_cols_list = original_col_names
                
            quoted_cols = ', '.join([f'"{c}"' for c in all_cols_list])
            insert_header = f'INSERT INTO "{table}" ({quoted_cols}) VALUES '
            
            first_batch_for_table = True
            while True:
                batch_data = cursor.fetchmany(PROCESSING_BATCH_SIZE)
                if not batch_data: break

                embedding_data = {}
                if is_virtual_table and embedding_col_names:
                    for col_name in embedding_col_names:
                        col_idx = original_col_names.index(col_name)
                        col_values = [str(row[col_idx]) if row[col_idx] is not None else "" for row in batch_data]
                        embeddings = generate_embeddings_parallel(embedding_model, col_values, pool=pool)
                        embedding_data[col_name] = ['[' + ', '.join(map(str, emb.tolist())) + ']' for emb in embeddings]
                
                rows_for_insert = []
                for i, row in enumerate(batch_data):
                    values = [sql_format_value(val) for val in row]
                    if is_virtual_table:
                        for col_name in embedding_col_names:
                            values.append(f"'{embedding_data[col_name][i]}'")
                    rows_for_insert.append("(" + ", ".join(values) + ")")
                
                if rows_for_insert:
                    # For subsequent batches in the same table, use a comma. For the first, use the header.
                    separator = ",\n" if not first_batch_for_table else insert_header
                    sql_buffer.write(separator)
                    sql_buffer.write(",\n".join(rows_for_insert))
                    first_batch_for_table = False
            
            if not first_batch_for_table:
                sql_buffer.write(";\n\n")

        # --- 关键修复：过滤掉属于虚拟表的索引 ---
        sql_buffer.write("\n-- ADDITIONAL DATABASE OBJECTS (Views, Triggers, and valid Indexes) --\n\n")
        other_objects = [obj for obj in objects if obj[0] != 'table' and obj[3]]
        for obj_type, _, table_name, sql in other_objects:
            if obj_type == 'index':
                # Check if the table this index belongs to was converted to a virtual table
                if db_info.get("can_convert_virtual", {}).get(table_name, False):
                    logging.info(f"Skipping index for table '{table_name}' because it was converted to a virtual table.")
                    continue  # Skip this index!
            
            # For all other objects (views, triggers) or for valid indexes, write the SQL
            sql_buffer.write(sql + ";\n\n")
        # --- 修复结束 ---
        
        sql_buffer.write("COMMIT;\nPRAGMA foreign_keys = ON;\n")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(sql_buffer.getvalue())
        
        return True, f"Successfully exported {len(created_tables)} tables to {output_file}"
    except Exception as e:
        return False, f"Export failed: {str(e)}\n{traceback.format_exc()}"
    finally:
        if conn:
            conn.close()

def generate_database_script(db_path, output_file, embedding_model, table_json_path, pool=None):
    try:
        with open(table_json_path, 'r', encoding='utf-8') as f:
            db_infos = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"✖ Error reading or parsing JSON file at {table_json_path}: {e}")
        return

    logging.info(f"Exporting database to {output_file}...")
    db_id = os.path.splitext(os.path.basename(db_path))[0]
    target_db_info = next((info for info in db_infos if info.get("db_id") == db_id), None)
    if not target_db_info:
        logging.warning(f"✖ Could not find configuration for db_id '{db_id}' in {table_json_path}. Will proceed without semantic info.")
        target_db_info = {} # Proceed with an empty info dict

    success, message = export_to_single_sql_file(db_path, output_file, target_db_info, embedding_model, pool=pool)
    if not success:
        logging.error(f"✖ Export failed for {db_id}!\n{message}")

def process_sql_file(sql_file_path, db_path):
    conn = None
    try:
        if os.path.exists(db_path):
            os.remove(db_path)
        conn = sqlite3.connect(db_path)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)

        cursor = conn.cursor()
        cursor.execute("PRAGMA synchronous = OFF;")
        cursor.execute("PRAGMA journal_mode = MEMORY;")
        
        with open(sql_file_path, 'r', encoding='utf-8') as f:
            sql_script = f.read()
        
        logging.info(f"Importing SQL from {sql_file_path}...")
        cursor.executescript(sql_script)
        logging.info("SQL script executed.")
        
        conn.commit()
        logging.info("Transaction committed successfully.")
        
    except sqlite3.Error as e:
        logging.error(f"Failed to process SQL script {sql_file_path}. Error: {e}", exc_info=True)
        raise
    finally:
        if conn:
            conn.close()

def build_vector_database(SQL_FILE, DB_FILE):
    try:
        process_sql_file(SQL_FILE, DB_FILE)
    except Exception:
        logging.error(f"❌ SQL import failed! Check logs for specific SQLite errors.")
        raise
