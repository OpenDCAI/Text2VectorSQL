import sqlite3, datetime, traceback, json, os, re
from tqdm import tqdm
import logging
from sentence_transformers import SentenceTransformer
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

def sql_format_value(val):
    if val is None: return "NULL"
    if isinstance(val, (int, float)): return str(val)
    if isinstance(val, bytes): return f"X'{val.hex()}'"
    return "'" + str(val).replace("'", "''") + "'"

def export_to_single_sql_file(db_path, output_file, db_info, embedding_model, pool=None):
    SQL_DELIMITER = "\n--<SQL_COMMAND_SEPARATOR>--\n"
    conn = sqlite3.connect(db_path)
    sql_buffer = StringIO()
    try:
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
                    sql_buffer.write(create_virtual_table_ddl(conn, name, db_info, embedding_model.get_sentence_embedding_dimension()) + ";" + SQL_DELIMITER)
                    virtual_tables.add(name)
                else: sql_buffer.write(sql + ";" + SQL_DELIMITER)
            else: sql_buffer.write(sql + ";" + SQL_DELIMITER)
        
        for _, table, _, _ in tqdm(table_objects, desc="Exporting data"):
            is_virtual = table in virtual_tables
            cursor = conn.execute(f'SELECT * FROM "{table}"')
            o_cols = [d[0] for d in cursor.description]
            a_cols, e_cols = [], []
            if is_virtual:
                s_cols = [sanitize_identifier(c) for c in o_cols]
                e_info = [c for c in db_info.get("semantic_rich_column", {}).get(table, []) if c.get('column_name') in o_cols]
                e_cols = [c['column_name'] for c in e_info]
                s_e_cols = [f"{sanitize_identifier(c)}_embedding" for c in e_cols]
                a_cols = s_cols + s_e_cols
            else: a_cols = o_cols
            header = f'INSERT INTO "{table}" ({", ".join(f"`{c}`" for c in a_cols)}) VALUES '
            while True:
                batch = cursor.fetchmany(5000)
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
                    vals = ["''" if v is None else sql_format_value(v) for v in row] if is_virtual else [sql_format_value(v) for v in row]
                    if is_virtual:
                        for col in e_cols: vals.append(f"'{e_data[col][i]}'")
                    rows.append("(" + ", ".join(vals) + ")")
                if rows: sql_buffer.write(header + ",\n".join(rows) + ";" + SQL_DELIMITER)

        for type, _, tbl, sql in [o for o in objects if o[0] != 'table' and o[3]]:
            if type == 'index' and tbl in virtual_tables: continue
            sql_buffer.write(sql + ";" + SQL_DELIMITER)
        
        sql_buffer.write("COMMIT;" + SQL_DELIMITER)
        with open(output_file, 'w', encoding='utf-8') as f: f.write(sql_buffer.getvalue())
        return True
    finally:
        if conn: conn.close()

def generate_database_script(db_path, output_file, embedding_model, table_json_path, pool=None):
    try:
        with open(table_json_path, 'r', encoding='utf-8') as f: db_infos = json.load(f)
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
