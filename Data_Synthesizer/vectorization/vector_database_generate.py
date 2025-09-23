import os
import sqlite3
import json
import logging
from typing import List, Dict, Any, Optional
import sqlite_vec

# --- 健壮性提升：定义一个唯一的、不会与数据冲突的SQL命令分隔符 ---
SQL_DELIMITER = "\n-- VEC_SQL_SEPARATOR --\n"

# --- 核心辅助函数 (保持不变) ---

def sql_format_value(val: Any, column_type: str) -> str:
    """
    根据列的数据类型，将Python值安全地格式化为SQL字面量。
    - 为None或空字符串等空值提供类型安全的默认值。
    - 正确处理数值、字符串、布尔值和字节。
    """
    ct = column_type.upper()

    is_null_like = val is None or (isinstance(val, str) and val.strip() == '')

    if is_null_like:
        if 'INT' in ct or 'BOOL' in ct:
            return "0"
        if 'REAL' in ct or 'FLOAT' in ct or 'DOUBLE' in ct:
            return "0.0"
        return "''"

    if isinstance(val, bool):
        return "1" if val else "0"
    
    if isinstance(val, bytes):
        return f"X'{val.hex()}'"

    if 'INT' in ct:
        try:
            return str(int(float(val)))
        except (ValueError, TypeError):
            return "0"
    if 'REAL' in ct or 'FLOAT' in ct or 'DOUBLE' in ct:
        try:
            return str(float(val))
        except (ValueError, TypeError):
            return "0.0"

    clean_val = str(val).replace("'", "''").replace('\n', ' ').replace('\r', ' ')
    return f"'{clean_val}'"

def escape_sql_string(val: str) -> str:
    return str(val).replace("'", "''")

# --- 主要功能函数 (已修改) ---

def generate_database_script(db_path: str, output_file: str, embedding_model, pool, table_json_path: str):
    logging.info(f"Generating SQL script for database: {db_path}")

    db_id = os.path.basename(os.path.dirname(db_path))
    with open(table_json_path, 'r', encoding='utf-8') as f:
        schema_data = json.load(f)

    db_schema_info = next((item for item in schema_data if item['db_id'] == db_id), None)
    
    if not db_schema_info:
        logging.error(f"Could not find schema for db_id '{db_id}' in {table_json_path}")
        return

    table_names = db_schema_info.get('table_names_original', [])
    columns_original = db_schema_info.get('column_names_original', [])
    column_types = db_schema_info.get('column_types', [])

    if columns_original and columns_original[0][0] == -1:
        aligned_columns_original = columns_original[1:]
        aligned_column_types = column_types[1:]
    else:
        aligned_columns_original = columns_original
        aligned_column_types = column_types

    if len(aligned_columns_original) != len(aligned_column_types):
        logging.error(f"Schema mismatch for {db_id}: Found {len(aligned_columns_original)} columns but {len(aligned_column_types)} types after alignment.")
        return

    unified_columns_info = []
    for i, col_details in enumerate(aligned_columns_original):
        unified_columns_info.append([col_details[0], col_details[1], aligned_column_types[i]])

    schema_map = {}
    for i, table_name in enumerate(table_names):
        table_columns_map = {
            col[1].lower(): col[2]
            for col in unified_columns_info if col[0] == i
        }
        schema_map[table_name.lower()] = table_columns_map

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    embedding_dim = embedding_model.get_sentence_embedding_dimension()

    # 使用 'w' 模式清空并写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for table_name in table_names:
            table_name_lower = table_name.lower()
            if table_name_lower not in schema_map: continue

            try:
                table_schema = schema_map.get(table_name_lower, {})
                
                # --- 【核心修复】检查表是否包含任何 TEXT 列 ---
                text_columns_in_table = [
                    col_name for col_name, col_type in table_schema.items() if 'TEXT' in col_type.upper()
                ]

                # 只有当表包含TEXT列时，才创建 VIRTUAL TABLE
                if text_columns_in_table:
                    # 生成 VIRTUAL TABLE DDL
                    modified_create_sql = f"CREATE VIRTUAL TABLE \"{table_name}\" USING vec0(\n"
                    column_definitions = []
                    for col_name, col_type in table_schema.items():
                        column_definitions.append(f"  {col_name} {col_type}")
                        # 仅为TEXT列添加嵌入
                        if 'TEXT' in col_type.upper():
                            column_definitions.append(f"  {col_name}_embedding float[{embedding_dim}]")
                    
                    modified_create_sql += ",\n".join(column_definitions)
                    modified_create_sql += "\n);"
                    f.write(modified_create_sql + SQL_DELIMITER)
                else:
                    # 对于没有TEXT列的表，创建普通TABLE
                    logging.info(f"Table '{table_name}' has no TEXT columns. Creating as a regular table.")
                    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
                    original_sql_row = cursor.fetchone()
                    if original_sql_row and original_sql_row[0]:
                        f.write(original_sql_row[0] + ";" + SQL_DELIMITER)
                    else:
                        logging.warning(f"Could not find original CREATE statement for table '{table_name}'. Skipping DDL.")
                        continue

                cursor.execute(f'SELECT * FROM "{table_name}"')
                col_names = [description[0] for description in cursor.description]
                
                all_rows = cursor.fetchall()
                if not all_rows: continue

                # 仅当表是虚拟表时才准备嵌入
                if text_columns_in_table:
                    texts_to_embed, text_col_indices = [], []
                    for i, name in enumerate(col_names):
                        # 使用原始 schema_map 来检查类型
                        if 'TEXT' in schema_map.get(table_name_lower, {}).get(name.lower(), '').upper():
                            text_col_indices.append(i)
                    
                    for row in all_rows:
                        for idx in text_col_indices:
                            texts_to_embed.append(str(row[idx]))
                    
                    embeddings = []
                    if texts_to_embed:
                        embeddings = embedding_model.encode_multi_process(texts_to_embed, pool=pool) if pool else embedding_model.encode(texts_to_embed)
                    
                    embedding_iter = iter(embeddings)

                for row in all_rows:
                    formatted_values = []
                    new_col_names = []
                    
                    for i, val in enumerate(row):
                        col_name = col_names[i]
                        new_col_names.append(col_name)
                        col_type = table_schema.get(col_name.lower(), 'TEXT')
                        formatted_values.append(sql_format_value(val, col_type))
                    
                    # 仅为虚拟表添加嵌入向量值
                    if text_columns_in_table:
                        for idx in text_col_indices:
                            col_name = col_names[idx]
                            new_col_names.append(f"{col_name}_embedding")
                            embedding_vector = next(embedding_iter)
                            formatted_values.append(f"'{str(embedding_vector.tolist())}'")
                    
                    columns_str = ", ".join([f"{name}" for name in new_col_names])
                    values_str = ", ".join(formatted_values)
                    # 使用唯一的SQL分隔符
                    f.write(f"INSERT INTO \"{table_name}\" ({columns_str}) VALUES ({values_str});" + SQL_DELIMITER)
                
            except Exception as e:
                logging.error(f"Error processing table '{table_name}' in db '{db_id}': {e}", exc_info=True)

    conn.close()
    logging.info(f"Successfully generated SQL script: {output_file}")


def build_vector_database(SQL_FILE: str, DB_FILE: str):
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
    
    conn = None
    try:
        conn = sqlite3.connect(DB_FILE)
        
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        logging.info("Successfully loaded sqlite-vec extension.")

        cursor = conn.cursor()

        with open(SQL_FILE, 'r', encoding='utf-8') as f:
            sql_script = f.read()

        # --- 【健壮性提升】使用自定义分隔符来分割命令 ---
        statements = [stmt.strip() for stmt in sql_script.split(SQL_DELIMITER) if stmt.strip()]
        
        logging.info(f"Importing SQL from {SQL_FILE} using robust custom delimiter...")
        for statement in statements:
            try:
                # 每个 'statement' 已经是完整的命令，可以直接执行
                cursor.execute(statement)
            except sqlite3.OperationalError as e:
                logging.error(f"Failed to execute statement: {statement[:200]}...")
                logging.error(f"SQL execution failed: {e}")
                raise e

        conn.commit()
        logging.info(f"Successfully built database: {DB_FILE}")

    except Exception as e:
        logging.error(f"An error occurred during database build: {e}")
        raise e
    finally:
        if conn:
            conn.close()
            logging.info("Database connection closed.")
