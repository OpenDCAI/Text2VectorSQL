# vector_database_generate.py (修改后)

import sqlite3
import datetime
import traceback
import json
import concurrent.futures
from tqdm import tqdm
import logging
from sentence_transformers import SentenceTransformer
import sqlite_vec
from io import StringIO
import os
import re

# ... (从 sanitize_identifier 到 generate_database_script 的所有函数保持不变) ...
# ... (此处省略了未改动的代码，请保留你文件中的原有部分) ...
# =================================================================
# 1. 清理标识符的辅助函数 (保持不变)
# =================================================================
def sanitize_identifier(identifier: str) -> str:
    """
    清理并安全地引用 SQL 标识符。

    处理逻辑：
    1. 将输入转换为字符串，以防传入非字符串类型。
    2. 替换空格、括号及其他所有非字母、非数字、非下划线的字符为下划线。
    3. 检查清理后的标识符是否以字母开头，如果不是，则添加 'fld_' 前缀。
    4. 检查是否为 SQLite-VEC 的保留关键字 'distance'，如果是则重命名。
    5. 用双引号包裹最终结果，使其成为一个安全的 SQL 标识符。
    """
    # 确保输入是字符串
    s = str(identifier)
    
    # 替换空格和括号
    s = s.replace(' ', '_').replace('(', '').replace(')', '')
    
    # 替换所有非字母、非数字、非下划线的字符为下划线
    # 这个表达式会正确处理 Unicode 字符（如 ñ），将它们替换为 _
    # 如果你想保留 ñ 这样的字符，可以使用 re.sub(r'[^\w]', '_', s)
    s = re.sub(r'[^a-zA-Z0-9_]', '_', s)
    
    # 如果标识符不以字母开头，则添加前缀。
    # 这样可以同时处理以数字、下划线或其他符号开头的情况。
    if not re.match(r'^[a-zA-Z]', s):
        s = 'fld_' + s
        
    # 检查是否为 sqlite-vec 的保留关键字 'distance' (不区分大小写)
    if s.lower() == 'distance':
        s = 'distance_val'  # 重命名冲突列

    return s

# 将sqlite-vec不支持的类型变成其支持的float，integer和text
def type_convert(original_type):
    original_type = original_type.upper()
    unsupported_types = {'DATE', 'YEAR', 'DATETIME', 'TIMESTAMP', 'BOOLEAN'}
    type_keyword = original_type.split('(')[0]
    new_type = 'TEXT'
    if 'char' in type_keyword.lower() or 'bool' in type_keyword.lower() or type_keyword in unsupported_types: 
        new_type = 'TEXT'
    elif 'real' in type_keyword.lower() or "numeric" in type_keyword.lower() or "decimal" in type_keyword.lower(): 
        new_type = 'FLOAT'
    elif "int" in type_keyword.lower() or "number" in type_keyword.lower(): 
        new_type = 'INTEGER'
    elif "blob" in type_keyword.lower():
        new_type = 'TEXT'
    else:
        new_type = 'TEXT'
        logging.error(f"no type to convert: {original_type} in type_convert")
    return new_type

# =================================================================
# 2. 生成虚拟表的函数 (保持不变)
# =================================================================
def create_virtual_table_ddl(conn, table_name, db_info, vec_dim):
    cursor = conn.cursor()
    cursor.execute(f'PRAGMA table_info("{table_name}");')
    original_columns_info = cursor.fetchall()

    column_definitions = []

    for col_info in original_columns_info:
        original_col_name, original_type = col_info[1], col_info[2].upper()
        sanitized_col_name = sanitize_identifier(original_col_name)
        new_type = type_convert(original_type)

        column_definitions.append(f'  {sanitized_col_name} {new_type}')

    if table_name in db_info.get("semantic_rich_column", {}):
        for col_info in db_info["semantic_rich_column"][table_name]:
            original_col_name = col_info.get('column_name')
            # --- 关键修复：只为实际存在的列添加嵌入定义 ---
            if original_col_name in [c[1] for c in original_columns_info]:
                sanitized_col_name = sanitize_identifier(original_col_name)
                column_definitions.append(f'  {sanitized_col_name}_embedding float[{vec_dim}]')

    columns_str = ",\n".join(column_definitions)
    return f"CREATE VIRTUAL TABLE \"{table_name}\" USING vec0(\n{columns_str}\n);"

# =================================================================
# 3. 核心修改：嵌入生成和主导出函数 (保持不变)
# =================================================================
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
    buffer_size = 0
    BUFFER_FLUSH_SIZE = 10 * 1024 * 1024

    def flush_buffer():
        nonlocal buffer_size
        if buffer_size > 0:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(sql_buffer.getvalue())
            sql_buffer.seek(0)
            sql_buffer.truncate()
            buffer_size = 0
            
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"-- SQLite Database Export\n-- Exported at: {datetime.datetime.now()}\n-- Source: {db_path}\n\n")
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
                    valid_semantic_cols = [
                        col for col in semantic_cols_from_json if col.get('column_name') in original_col_names
                    ]

                    if valid_semantic_cols:
                        column_num = len(columns_info)
                        new_column_num = len(valid_semantic_cols)
                        
                        if column_num + new_column_num <= 16:
                            ddl = create_virtual_table_ddl(conn, name, db_info, vec_dim)
                            db_info["can_convert_virtual"][name] = True
                        else:
                            ddl = sql + ';'
                    else:
                        ddl = sql + ';'
                except Exception as e:
                    logging.info(f"警告：无法获取表 '{name}' 的列信息 ({e})，将使用原始表结构。")
                    ddl = sql + ';'
            else:
                ddl = sql + ';'
                
            sql_buffer.write(ddl + "\n\n")
            created_tables.append(name)

        flush_buffer()
        sql_buffer.write("\n-- DATA INSERTION --\n\n")

        table_pbar = tqdm(created_tables, desc="Processing tables")
        for table in table_pbar:
            is_virtual_table = db_info["can_convert_virtual"].get(table, False)
            cursor = conn.execute(f'SELECT * FROM "{table}"')
            original_col_names = [desc[0] for desc in cursor.description]
            
            if is_virtual_table:
                sanitized_cols = [sanitize_identifier(c) for c in original_col_names]
                embedding_cols_info = db_info.get("semantic_rich_column", {}).get(table, [])
                
                valid_embedding_cols_info = [
                    col for col in embedding_cols_info if col['column_name'] in original_col_names
                ]
                for col in embedding_cols_info:
                    if col['column_name'] not in original_col_names:
                        logging.warning(f"Skipping semantic column '{col['column_name']}' in table '{table}': Column not found in the database.")

                embedding_col_names = [col['column_name'] for col in valid_embedding_cols_info]
                sanitized_embedding_cols = [f"{sanitize_identifier(c)}_embedding" for c in embedding_col_names]
                all_cols_list = sanitized_cols + sanitized_embedding_cols
            else:
                all_cols_list = original_col_names
                embedding_col_names = []
                
            quoted_cols = ', '.join([f'"{c}"' for c in all_cols_list])
            insert_header = f'INSERT OR IGNORE INTO "{table}" ({quoted_cols}) VALUES\n'

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
                cur = conn.execute(f"PRAGMA table_info('{table}')")
                column_types = {info[1]: info[2].lower() for info in cur.fetchall()}

                for i, row in enumerate(batch_data):
                    values = []
                    for j, val in enumerate(row):
                        col_name = original_col_names[j]
                        col_type = type_convert(column_types.get(col_name, "")).lower()
                        if val is None or val == '' or val =='NULL' or val == 'nil' or val == 0 or val == 0.0:
                            if "int" in col_type: values.append("0")
                            elif "float" in col_type: values.append("0.0")
                            elif "text" in col_type: values.append("''")
                            else: values.append("'NULL'")
                        else:
                            try:
                                if isinstance(val, bytes): values.append(f"X'{val.hex()}'")
                                else:
                                    val_str = str(val).replace('\x00', '')

                                    if "float" in col_type:
                                        values.append(str(float(val_str)))
                                    elif "int" in col_type:
                                        values.append(str(int(val_str)))
                                    else: 
                                        escaped_val = val_str.replace("'", "''")
                                        values.append(f"'{escaped_val}'")
                                        if "text" not in col_type:
                                            logging.error(f'''col_name:{col_name} col_type:{col_type}''')
                            except Exception as e:
                                logging.error(f"Export failed: {str(e)}\n{traceback.format_exc()}")
                                logging.error(f'''val:{val} col_name:{col_name} col_type:{col_type}''')
                                continue
                    
                    if is_virtual_table:
                        for col_name in embedding_col_names:
                            values.append(f"'{embedding_data[col_name][i]}'")
                    rows_for_insert.append("(" + ", ".join(values) + ")")

                write_batch_size = 10
                for i in range(0, len(rows_for_insert), write_batch_size):
                    batch = rows_for_insert[i:i+write_batch_size]
                    sql_buffer.write(insert_header)
                    sql_buffer.write(",\n".join(batch))
                    sql_buffer.write(";\n\n")
                    buffer_size += sql_buffer.tell()
                    if buffer_size > BUFFER_FLUSH_SIZE: flush_buffer()
        
        # 刷新之前批处理的数据
        flush_buffer() 

        # 写入剩余的数据库对象和 COMMIT 命令
        sql_buffer.write("\n-- ADDITIONAL DATABASE OBJECTS --\n\n")
        other_objects = [obj for obj in objects if obj[0] != 'table' and obj[3]]
        for _, _, _, sql in other_objects:
            sql_buffer.write(sql + ";\n\n")
        sql_buffer.write("COMMIT;\nPRAGMA foreign_keys = ON;\n")

        # --- 关键修复：不再依赖复杂的 flush_buffer() 逻辑 ---
        # 直接获取缓冲区中所有剩余的内容，并一次性写入文件末尾。
        final_chunk = sql_buffer.getvalue()
        if final_chunk:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(final_chunk)
        # --- 修复结束 ---
        
        return True, f"Successfully exported {len(created_tables)} tables to {output_file}"
    except Exception as e:
        return False, f"Export failed: {str(e)}\n{traceback.format_exc()}"
    finally:
        conn.close()


def generate_database_script(db_path, output_file, embedding_model, table_json_path="./results/embedding_train_tables.json", pool=None):
    try:
        with open(table_json_path, 'r', encoding='utf-8') as f:
            db_infos = json.load(f)
    except FileNotFoundError:
        logging.info(f"✖ Error: JSON file not found at {table_json_path}")
        return
    except json.JSONDecodeError:
        logging.info(f"✖ Error: Could not decode JSON from {table_json_path}")
        return

    logging.info(f"Exporting database to {output_file}...")
    
    base_name = os.path.basename(db_path)
    db_id = os.path.splitext(base_name)[0]
    target_db_info = next((info for info in db_infos if info.get("db_id") == db_id), None)
    if not target_db_info:
        logging.info(f"✖ Error: Could not find configuration for db_id '{db_id}' in {table_json_path}")
        return

    success, message = export_to_single_sql_file(db_path, output_file, target_db_info, embedding_model, pool=pool)

    if success:
        pass
    else:
        logging.info(f"✖ Export failed for {db_id}!\n{message}")

# =================================================================
# 4. 构建向量数据库的函数 (已修改)
# =================================================================
def process_sql_file(sql_file_path, db_path):
    if os.path.exists(db_path):
        os.remove(db_path)
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)

    try:
        cursor = conn.cursor()
        cursor.execute("PRAGMA synchronous = OFF;")
        cursor.execute("PRAGMA journal_mode = MEMORY;")
        cursor.execute("PRAGMA cache_size = -100000;")

        with open(sql_file_path, 'r', encoding='utf-8') as f:
            sql_script = f.read()
        
        logging.info(f"Importing SQL from {sql_file_path}...")
        cursor.executescript(sql_script)
        logging.info("SQL script imported successfully.")
        
    except Exception as e:
        # --- 关键修改：发生错误时不删除文件，而是记录日志并重新抛出异常 ---
        logging.error(f"Failed to process SQL script {sql_file_path} into {db_path}. The incomplete file is kept for debugging. Error: {e}", exc_info=True)
        # if os.path.exists(db_path): os.remove(db_path) # <--- 注释掉这一行
        if conn:
            conn.close()
        raise
    finally:
        if conn: 
            conn.close()

def build_vector_database(SQL_FILE, DB_FILE):
    try:
        import sqlite_vec
    except ImportError:
        logging.info("Fatal Error: 'sqlite_vec' library not installed. Please install with 'pip install sqlite-vec'")
        exit()
    try:
        process_sql_file(SQL_FILE, DB_FILE)
    except Exception as e:
        logging.info(f"❌ SQL import failed! The original exception will be re-raised.")
        raise
