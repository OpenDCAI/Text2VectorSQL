# vector_database_generate.py (最终修复版 v4)

import sqlite3, datetime, traceback, json, os, re
import math
from tqdm import tqdm
import logging
from sentence_transformers import SentenceTransformer, models
import sqlite_vec
from io import StringIO

def sanitize_identifier(s: str) -> str:
    s = str(s).replace(' ', '_').replace('(', '').replace(')', '')
    s = re.sub(r'[^a-zA-Z0-9_]', '_', s)
    if not re.match(r'^[a-zA-Z]', s): s = 'fld_' + s
    if s.lower() == 'distance': s = 'distance_val'
    return s

def type_convert(t: str) -> str:
    if not t: return 'TEXT'
    t = t.upper().split('(')[0]
    if "INT" in t or "NUMBER" in t: return 'INTEGER'
    if any(k in t for k in ["REAL", "NUMERIC", "DECIMAL", "FLOAT", "DOUBLE"]): return 'FLOAT'
    return 'TEXT'

def get_model_dim(model):
    if isinstance(model[0], models.CLIPModel):
        print("检测到是 CLIP 模型，正在从配置中读取维度...")
        embedding_dim = model[0].model.text_model.config.projection_dim
    else:
        print("检测到是标准模型，使用官方API获取维度...")
        embedding_dim = model.get_sentence_embedding_dimension()

    if embedding_dim is not None:
        print(f"从配置中确定模型维度是: {embedding_dim}")
        return embedding_dim
    else:
        try:
            dummy_embedding = model.encode("test")
            return dummy_embedding.shape[-1]
        except Exception as e:
            raise RuntimeError(f"严重错误：两种方法均无法确定模型维度: {e}")

def create_virtual_table_ddl(conn, table_name, db_info, vec_dim):
    cursor = conn.cursor()
    cursor.execute(f'PRAGMA table_info("{table_name}");')
    cols = []
    original_cols = {c[1] for c in cursor.fetchall()}
    cursor.execute(f'PRAGMA table_info("{table_name}");')
    for _, name, type, _, _, _ in cursor.fetchall():
        cols.append(f'  {sanitize_identifier(name)} {type_convert(type)}')
    if table_name in db_info.get("semantic_rich_column", {}):
        for col_info in db_info["semantic_rich_column"][table_name]:
            if col_info.get('column_name') in original_cols:
                cols.append(f'  {sanitize_identifier(col_info["column_name"])}_embedding float[{vec_dim}]')
    columns_str = ",\n".join(cols)
    return f"CREATE VIRTUAL TABLE \"{table_name}\" USING vec0(\n{columns_str}\n)"

def generate_embeddings_parallel(model, texts, batch_size=128, pool=None):
    return model.encode_multi_process(texts, pool=pool, batch_size=batch_size) if pool else model.encode(texts, batch_size=batch_size, show_progress_bar=False)

# --- 核心修复函数 v4 (最终版) ---
def sql_format_value(val, is_virtual=False, col_type='TEXT'):
    """
    根据目标列类型严格格式化 Python 值为 SQL 字符串。
    - 对 inf 和 NaN 的处理修改为返回 0 或 0.0
    - 对无法转换为数字的脏数据，强制返回 0 或 0.0
    """
    col_type_upper = col_type.upper()

    if val is None:
        if is_virtual:
            if 'INTEGER' in col_type_upper: return "0"
            if any(k in col_type_upper for k in ["REAL", "FLOAT", "DOUBLE", "DECIMAL", "NUMERIC"]): return "0.0"
            return "''"
        else:
            return "NULL"

    # --- 关键修改：重构数字处理逻辑 ---
    is_numeric_type = 'INTEGER' in col_type_upper or any(k in col_type_upper for k in ["REAL", "FLOAT", "DOUBLE", "DECIMAL", "NUMERIC"])

    if is_numeric_type:
        # 如果目标是数字类型，则尝试转换
        try:
            # 对于空字符串或仅包含非数字字符的字符串，float()会失败
            if isinstance(val, str) and not val.strip():
                 numeric_val = 0.0
            else:
                 numeric_val = float(val)

            if math.isinf(numeric_val) or math.isnan(numeric_val):
                return "0" if 'INTEGER' in col_type_upper else "0.0"
            
            if 'INTEGER' in col_type_upper:
                return str(int(numeric_val))
            else:
                return str(numeric_val)
        except (ValueError, TypeError):
            # 如果转换失败（例如, 遇到 '$' 或 'cases'），则返回适合虚拟表的默认值
            if is_virtual:
                return "0" if 'INTEGER' in col_type_upper else "0.0"
            else:
                return "NULL"

    # 如果不是数字类型，则按原样处理
    if isinstance(val, bytes):
        return f"X'{val.hex()}'"
    
    return "'" + str(val).replace("'", "''") + "'"


def export_to_single_sql_file(db_path, output_file, db_info, embedding_model, pool=None):
    SQL_DELIMITER = "\n--<SQL_COMMAND_SEPARATOR>--\n"

    conn = sqlite3.connect(db_path)
    conn.text_factory = lambda b: b.decode('utf-8', errors='ignore')
    
    sql_buffer = StringIO()
    try:
        sql_buffer.write(f"-- DB Export\n-- {datetime.datetime.now()}\n\n")
        sql_buffer.write("BEGIN TRANSACTION;" + SQL_DELIMITER)

        objects = conn.execute("SELECT type, name, tbl_name, sql FROM sqlite_master WHERE type IN ('table', 'view', 'trigger', 'index') AND name NOT LIKE 'sqlite_%' ORDER BY type='table' DESC").fetchall()
        table_objects = [obj for obj in objects if obj[0] == 'table' and obj[3]]
        virtual_tables = set()

        for _, name, _, sql in table_objects:
            if name in db_info.get("semantic_rich_column", {}):
                cursor = conn.cursor()
                cursor.execute(f"PRAGMA table_info('{name}')")
                cols_info = cursor.fetchall()
                sem_cols = [c for c in db_info["semantic_rich_column"][name] if c.get('column_name') in {i[1] for i in cols_info}]
                if sem_cols and (len(cols_info) + len(sem_cols) <= 16):
                    sql_buffer.write(create_virtual_table_ddl(conn, name, db_info, get_model_dim(embedding_model)) + ";" + SQL_DELIMITER)
                    virtual_tables.add(name)
                else:
                    sql_buffer.write(sql + ";" + SQL_DELIMITER)
            else:
                sql_buffer.write(sql + ";" + SQL_DELIMITER)

        for _, table, _, _ in tqdm(table_objects, desc="Exporting data"):
            is_virtual = table in virtual_tables
            cursor = conn.execute(f'SELECT * FROM "{table}"')
            o_cols = [d[0] for d in cursor.description]
            
            type_cursor = conn.cursor()
            type_cursor.execute(f'PRAGMA table_info("{table}")')
            col_types = {row[1]: type_convert(row[2]) for row in type_cursor.fetchall()}

            a_cols, e_cols = [], []
            if is_virtual:
                s_cols = [sanitize_identifier(c) for c in o_cols]
                e_info = [c for c in db_info.get("semantic_rich_column", {}).get(table, []) if c.get('column_name') in o_cols]
                e_cols = [c['column_name'] for c in e_info]
                s_e_cols = [f"{sanitize_identifier(c)}_embedding" for c in e_cols]
                a_cols = s_cols + s_e_cols
            else:
                a_cols = o_cols
            header = f'INSERT INTO "{table}" ({", ".join(f"`{c}`" for c in a_cols)}) VALUES '
            
            while True:
                try:
                    batch = cursor.fetchmany(5000)
                except sqlite3.OperationalError as e:
                    logging.warning(f"Skipping batch for table '{table}' due to fetch error: {e}")
                    break

                if not batch: break
                e_data = {}
                if is_virtual and e_cols:
                    for col in e_cols:
                        c_idx = o_cols.index(col)
                        c_vals = [str(r[c_idx]) if r[c_idx] is not None else "" for r in batch]
                        embs = generate_embeddings_parallel(embedding_model, c_vals, pool=pool)
                        e_data[col] = ['[' + ', '.join(map(str, e.tolist())) + ']' for e in embs]
                
                rows = []
                for i, row in enumerate(batch):
                    vals = [sql_format_value(v, is_virtual=is_virtual, col_type=col_types.get(o_cols[j], 'TEXT')) for j, v in enumerate(row)]
                    
                    if is_virtual:
                        for col in e_cols: vals.append(f"'{e_data[col][i]}'")
                    rows.append("(" + ", ".join(vals) + ")")
                
                if rows:
                    sql_buffer.write(header + ",\n".join(rows) + ";" + SQL_DELIMITER)

        for type, _, tbl, sql in [o for o in objects if o[0] != 'table' and o[3]]:
            if type == 'index' and tbl in virtual_tables: continue
            sql_buffer.write(sql + ";" + SQL_DELIMITER)

        sql_buffer.write("COMMIT;" + SQL_DELIMITER)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(sql_buffer.getvalue())
        return True
    finally:
        if conn: conn.close()

def generate_database_script(db_path, output_file, embedding_model, table_json_path, pool=None):
    try:
        with open(table_json_path, 'r', encoding='utf-8') as f:
            db_infos = json.load(f)
    except Exception as e:
        logging.error(f"Error reading JSON {table_json_path}: {e}")
        return
    db_id = os.path.splitext(os.path.basename(db_path))[0]
    target_info = next((i for i in db_infos if i.get("db_id") == db_id), {})
    if not export_to_single_sql_file(db_path, output_file, target_info, embedding_model, pool=pool):
        logging.error(f"Export failed for {db_id}!")

def build_vector_database(SQL_FILE, DB_FILE):
    SQL_DELIMITER = "\n--<SQL_COMMAND_SEPARATOR>--\n"
    
    conn = None
    try:
        db_dir = os.path.dirname(DB_FILE)
        os.makedirs(db_dir, exist_ok=True)
        if os.path.exists(DB_FILE): os.remove(DB_FILE)
            
        conn = sqlite3.connect(DB_FILE)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        cursor = conn.cursor()
        logging.info("Database connection opened and sqlite-vec extension loaded.")
        
        logging.info(f"Importing SQL from {SQL_FILE} using robust custom delimiter...")
        with open(SQL_FILE, 'r', encoding='utf-8') as f:
            sql_script = f.read()
        
        statements = sql_script.split(SQL_DELIMITER)
        
        for statement in tqdm(statements, desc="Executing SQL Statements"):
            statement = statement.strip()
            if not statement: continue
            
            try:
                cursor.execute(statement)
            except sqlite3.Error as e:
                logging.error(f"Failed to execute statement: {statement[:500]}...")
                raise e

        logging.info("✅ SQL script processed successfully.")

    except sqlite3.Error as e:
        logging.error(f"SQL execution failed: {e}", exc_info=True)
        raise
    finally:
        if conn:
            conn.close()
            logging.info("Database connection closed.")
