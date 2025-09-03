import sqlite3
import datetime
import traceback
import json
import re
import concurrent.futures
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from io import StringIO

# =================================================================
# 1. 移除标识符引号的函数 (保持不变, 功能正确)
# =================================================================
def remove_identifier_quotes(sql_str):
    """
    一个更强大的函数，用于删除 CREATE TABLE 语句中
    列定义和约束里作为标识符的引号 (` " ')。
    """
    # 模式1: 处理列定义。
    pattern1 = r'(?P<prefix>^\s*|[(,]\s*)(?P<quote>[`"\'])(?P<col_name>\w+)(?P=quote)'
    replacement1 = r'\g<prefix>\g<col_name>'
    sql_str = re.sub(pattern1, replacement1, sql_str, flags=re.MULTILINE)

    # 模式2: 处理约束中的列名。
    pattern2 = r'(?P<prefix>\w+\s*\()(?P<quote>[`"\'])(?P<col_name>\w+)(?P=quote)(?P<suffix>\))'
    replacement2 = r'\g<prefix>\g<col_name>\g<suffix>'
    sql_str = re.sub(pattern2, replacement2, sql_str, flags=re.IGNORECASE)
    
    return sql_str

# =================================================================
# 2. 调整表约束的函数 (保持不变, 功能正确)
# =================================================================
def adjust_table_constraints(sql_str):
    """调整表约束"""
    lines = sql_str.splitlines()
    new_lines = []
    column_lines = {}
    found_constraints = False
    in_create_table = False
    last_column_line = -1

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.upper().startswith("CREATE TABLE"): in_create_table = True; new_lines.append(line); continue
        if in_create_table and stripped == ");": in_create_table = False; new_lines.append(line); continue
        if in_create_table:
            col_match = re.match(r'^\s*(\w+)\s', stripped)
            if not stripped.upper().startswith(("PRIMARY KEY", "FOREIGN KEY")) and col_match:
                col_name = col_match.group(1); column_lines[col_name] = len(new_lines); last_column_line = len(new_lines); new_lines.append(line); continue
            if stripped.upper().startswith("PRIMARY KEY"):
                found_constraints = True
                single_match = re.match(r'PRIMARY KEY\s*\(\s*(\w+)\s*\)\s*,?\s*$', stripped, re.IGNORECASE)
                if single_match:
                    col_name = single_match.group(1)
                    if col_name in column_lines:
                        idx = column_lines[col_name]; old_line = new_lines[idx]
                        if old_line.rstrip().endswith(','): new_lines[idx] = old_line.rstrip()[:-1] + ' PRIMARY KEY,'
                        else: new_lines[idx] = old_line.rstrip() + ' PRIMARY KEY'
                continue
            if stripped.upper().startswith("FOREIGN KEY"): found_constraints = True; continue
        new_lines.append(line)
    if found_constraints and last_column_line >= 0:
        last_line = new_lines[last_column_line]
        if last_line.strip().endswith(','): new_lines[last_column_line] = re.sub(r',\s*$', '', last_line)
    return '\n'.join(new_lines)

# =================================================================
# 3. 【核心修正】修正生成虚拟表的函数，确保列名无引号
# =================================================================
def create_virtual_table_ddl(conn, table_name, db_info, vec_dim):
    """通过 PRAGMA table_info 获取准确的表结构，并从零开始构建无引号列名的 CREATE VIRTUAL TABLE 语句"""
    cursor = conn.cursor()
    cursor.execute(f'PRAGMA table_info("{table_name}");')
    original_columns_info = cursor.fetchall()
    column_definitions = []
    unsupported_types = {'DATE', 'DATETIME', 'TIMESTAMP', 'BOOLEAN'}
    for col_info in original_columns_info:
        col_name, original_type = col_info[1], col_info[2].upper()
        type_keyword = original_type.split('(')[0]
        new_type = 'TEXT' if type_keyword in unsupported_types else (original_type if original_type else 'TEXT')
        # 【修正点】直接拼接列名和类型，不添加任何引号
        column_definitions.append(f'  {col_name} {new_type}')
    if table_name in db_info.get("column_alter", {}):
        for col_info in db_info["column_alter"][table_name]:
            col_name = col_info.get('column_name')
            if col_name in [c[1] for c in original_columns_info]:
                # 【修正点】embedding列名也不加引号
                column_definitions.append(f'  {col_name}_embedding float[{vec_dim}]')
    columns_str = ",\n".join(column_definitions)
    # 表名可以带引号，这是安全的
    return f"CREATE VIRTUAL TABLE \"{table_name}\" USING vec0(\n{columns_str}\n);"

# =================================================================
# 4. 其他辅助函数和主导出函数
# =================================================================
def generate_embeddings_parallel(model, texts, batch_size=128):
    """并行生成嵌入向量"""
    embeddings = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]; futures.append(executor.submit(model.encode, batch, show_progress_bar=False))
        for future in concurrent.futures.as_completed(futures): embeddings.extend(future.result())
    return embeddings

def export_to_single_sql_file(db_path, output_file, db_info, embedding_model):
    """综合所有修复的最终导出函数"""
    conn = sqlite3.connect(db_path)
    vec_dim = embedding_model.get_sentence_embedding_dimension()
    sql_buffer = StringIO(); buffer_size = 0; BUFFER_FLUSH_SIZE = 10 * 1024 * 1024
    def flush_buffer():
        nonlocal buffer_size
        if buffer_size > 0:
            with open(output_file, 'a', encoding='utf-8') as f: f.write(sql_buffer.getvalue())
            sql_buffer.seek(0); sql_buffer.truncate(); buffer_size = 0
    try:
        with open(output_file, 'w', encoding='utf-8') as f: f.write(f"-- SQLite Database Export\n-- Exported at: {datetime.datetime.now()}\n-- Source: {db_path}\n\n")
        sql_buffer.write("PRAGMA foreign_keys = OFF;\nBEGIN TRANSACTION;\n\n")
        objects = conn.execute("SELECT type, name, tbl_name, sql FROM sqlite_master WHERE type IN ('table', 'view', 'trigger', 'index') AND name NOT LIKE 'sqlite_%' ORDER BY CASE type WHEN 'table' THEN 1 WHEN 'view' THEN 4 ELSE 3 END").fetchall()
        
        created_tables = []
        table_objects = [obj for obj in objects if obj[0] == 'table' and obj[1] and obj[3]]
        
        for _, name, _, sql in table_objects:
            sql_buffer.write(f"-- Table: {name}\n")
            if name in db_info.get("column_alter", {}):
                ddl = create_virtual_table_ddl(conn, name, db_info, vec_dim)
            else:
                ddl = adjust_table_constraints(remove_identifier_quotes(sql + ';'))
            sql_buffer.write(ddl + "\n\n")
            created_tables.append(name)

        flush_buffer()
        sql_buffer.write("\n-- DATA INSERTION --\n\n")
        table_pbar = tqdm(created_tables, desc="Processing tables")
        for table in table_pbar:
            is_virtual_table = table in db_info.get("column_alter", {})
            cursor = conn.execute(f'SELECT * FROM "{table}" LIMIT 0'); col_names = [desc[0] for desc in cursor.description]
            embedding_cols = [col_info['column_name'] for col_info in db_info.get("column_alter", {}).get(table, [])] if is_virtual_table else []
            data = conn.execute(f'SELECT * FROM "{table}"').fetchall()
            if not data: continue
            
            # INSERT 语句的列名需要带引号，以处理包含空格或关键字的列名，这是标准做法
            all_cols_list = col_names + [f"{col}_embedding" for col in embedding_cols] if is_virtual_table else col_names
            quoted_cols = ', '.join([f'"{c}"' for c in all_cols_list])
            insert_header = f'INSERT INTO "{table}" ({quoted_cols}) VALUES\n'
            
            embedding_data = {}
            if is_virtual_table and embedding_cols:
                with tqdm(total=len(embedding_cols), desc=f"Generating embeddings for {table}", leave=False) as pbar:
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        futures = {}
                        for col in embedding_cols:
                            col_idx = col_names.index(col); col_values = [str(row[col_idx]) if row[col_idx] is not None else "" for row in data]
                            futures[executor.submit(generate_embeddings_parallel, embedding_model, col_values)] = col
                        for future in concurrent.futures.as_completed(futures):
                            col = futures[future]; embeddings = future.result()
                            embedding_data[col] = ['[' + ', '.join(map(str, emb.tolist())) + ']' for emb in embeddings]; pbar.update(1)
            
            rows = []
            for i, row in enumerate(data):
                values = []
                for val in row:
                    if val is None: values.append("NULL")
                    elif isinstance(val, (int, float)): values.append(str(val))
                    elif isinstance(val, bytes): values.append(f"X'{val.hex()}'")
                    else: values.append("'" + str(val).replace("'", "''") + "'")
                if is_virtual_table:
                    for col in embedding_cols: values.append(f"'{embedding_data[col][i]}'")
                rows.append("(" + ", ".join(values) + ")")

            # 【核心要求】限制 INSERT 语句大小，每 10 行为一个独立的 INSERT
            batch_size = 10
            for i in range(0, len(rows), batch_size):
                batch = rows[i:i+batch_size]
                sql_buffer.write(insert_header)
                sql_buffer.write(",\n".join(batch))
                sql_buffer.write(";\n\n")
                buffer_size += sum(len(s) for s in batch)
                if buffer_size > BUFFER_FLUSH_SIZE: flush_buffer()
        
        flush_buffer()
        sql_buffer.write("\n-- ADDITIONAL DATABASE OBJECTS --\n\n")
        other_objects = [obj for obj in objects if obj[0] != 'table' and obj[3]]
        for _, _, _, sql in other_objects:
             sql_buffer.write(sql + ";\n\n")
        sql_buffer.write("COMMIT;\nPRAGMA foreign_keys = ON;\n"); flush_buffer()
        return True, f"Successfully exported {len(created_tables)} tables to {output_file}"
    except Exception as e: return False, f"Export failed: {str(e)}\n{traceback.format_exc()}"
    finally: conn.close()

# =================================================================
# 5. 使用示例的主函数部分，保持不变
# =================================================================
if __name__ == "__main__":
    db_path = "train/train_databases/european_football_1/european_football_1.sqlite"
    output_file = "complete_database_export.sql"
    table_json_path = "./results/embedding_train_tables.json"
    try:
        with open(table_json_path, 'r', encoding='utf-8') as f: db_infos = json.load(f)
    except FileNotFoundError: print(f"✖ Error: JSON file not found at {table_json_path}"); exit()
    except json.JSONDecodeError: print(f"✖ Error: Could not decode JSON from {table_json_path}"); exit()
    EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    print(f"Exporting database to {output_file}...")
    success, message = export_to_single_sql_file(db_path, output_file, db_infos[0], EMBEDDING_MODEL)
    if success:
        print(f"✔ {message}")
        print("Import command:")
        print(f"  sqlite3 new_database.db < {output_file}")
    else:
        print(f"✖ Export failed!\n{message}")
