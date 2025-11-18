import argparse
import os
import sqlite3
import re
import struct
import glob
import logging
from contextlib import contextmanager
from datetime import datetime
### --- NEW --- ###
from concurrent.futures import ProcessPoolExecutor, as_completed
### --- NEW --- ###

# Attempt to import dependencies, if they fail, log a warning
try:
    import psycopg2
    from psycopg2.extras import execute_batch
except ImportError:
    psycopg2 = None
    execute_batch = None

try:
    from clickhouse_driver import Client
except ImportError:
    Client = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

BATCH_SIZE = 10000 # You can adjust this value based on your available memory

# ... (cleanup_postgres_db 和 cleanup_clickhouse_db 函数保持不变) ...
def cleanup_postgres_db(args, db_name):
    """Connects to the PostgreSQL server and drops a specific database."""
    conn_admin = None
    try:
        logging.info("  🧹 Cleaning up failed PostgreSQL database '%s'...", db_name)
        conn_admin = psycopg2.connect(
            host=args.host, port=args.port, user=args.user, password=args.password, dbname='postgres'
        )
        conn_admin.autocommit = True
        with conn_admin.cursor() as cur:
            cur.execute(f'DROP DATABASE IF EXISTS "{db_name}" WITH (FORCE);')
        logging.info("  ✔ Cleanup successful.")
    except Exception as e:
        logging.error("  🔴 Error during cleanup: %s", e)
        logging.error("     You may need to manually drop the database '%s'.", db_name)
    finally:
        if conn_admin:
            conn_admin.close()

def cleanup_clickhouse_db(args, db_name):
    """Connects to the ClickHouse server and drops a specific database."""
    client = None
    try:
        logging.info("  🧹 Cleaning up failed ClickHouse database '%s'...", db_name)
        client = Client(host=args.host, port=args.port, user=args.user, password=args.password)
        client.execute(f'DROP DATABASE IF EXISTS `{db_name}`')
        logging.info("  ✔ Cleanup successful.")
    except Exception as e:
        logging.error("  🔴 Error during cleanup: %s", e)
        logging.error("     You may need to manually drop the database '%s'.", db_name)
    finally:
        if client:
            client.disconnect()

# ... (coerce_value, translate_type_*, translate_schema_*, find_database_files, sqlite_conn, get_sqlite_schema, get_vec_info 保持不变) ...
# [此处省略了所有未修改的函数，以节省空间]
# [您原始脚本中的 coerce_value...get_vec_info 函数应放在这里]
def coerce_value(value, target_type_str):
    """
    根据目标数据库的列类型字符串，将 Python 值强制转换为更具体的类型，
    使转换过程更加鲁棒。
    """
    if value is None:
        return None

    target_type_str = target_type_str.lower()
    if 'nullable' in target_type_str:
        target_type_str = re.sub(r'nullable\((.+?)\)', r'\1', target_type_str)

    # --- 整数转换 ---
    if any(int_type in target_type_str for int_type in ['int', 'bigint', 'smallint', 'tinyint']):
        try:
            return int(value)
        except (ValueError, TypeError):
            # logging.warning("  🟡 Cannot convert '%s' (type: %s) to an integer. Inserting NULL.", value, type(value).__name__)
            return None

    # --- 浮点数/数值转换 ---
    if ('array' not in target_type_str) and ('vector' not in target_type_str) and any(float_type in target_type_str for float_type in ['float', 'double', 'real', 'numeric', 'decimal']):
        try:
            return float(value)
        except (ValueError, TypeError):
            # logging.warning("  🟡 Cannot convert '%s' (type: %s) to a float. Inserting NULL.", value, type(value).__name__)
            return None
    
    # --- 日期/时间转换 ---
    is_date_type = 'date' in target_type_str and 'datetime' not in target_type_str
    is_datetime_type = 'datetime' in target_type_str or 'timestamp' in target_type_str
    if isinstance(value, str) and (is_date_type or is_datetime_type):
        datetime_formats = [
            '%Y-%m-%d %H:%M:%S.%f',  # 带毫秒
            '%Y-%m-%d %H:%M:%S',    # 带秒
            '%Y-%m-%dT%H:%M:%S.%f', # ISO 8601 格式
            '%Y-%m-%dT%H:%M:%S',   # ISO 8601 格式
            '%Y-%m-%d %H:%M',       # 不带秒
            '%Y-%m-%d',
        ]
        
        for fmt in datetime_formats:
            try:
                return datetime.strptime(value, fmt)
            except (ValueError, TypeError):
                continue # If this format fails, try the next one

        # logging.warning("  🟡 Could not parse string '%s' into any known date/time format. Inserting NULL.", value)
        return None

    # --- 布尔转换 ---
    if 'bool' in target_type_str:
        if isinstance(value, int):
            return value != 0
        if isinstance(value, str):
            val_lower = value.lower()
            if val_lower in ('true', 't', '1', 'yes', 'y'):
                return True
            if val_lower in ('false', 'f', '0', 'no', 'n'):
                return False
        return bool(value)

    # --- 字符串回退 ---
    if any(str_type in target_type_str for str_type in ['string', 'text', 'char', 'clob']):
        if not isinstance(value, str):
            return str(value)

    # 如果没有应用特定的转换，返回原始值
    return value

# --- Schema Translation Module ---

def translate_type_for_postgres(sqlite_type):
    """Maps SQLite data types to PostgreSQL types."""
    sqlite_type = sqlite_type.upper()
    if 'INT' in sqlite_type:
        return 'BIGINT'
    if 'CHAR' in sqlite_type or 'TEXT' in sqlite_type or 'CLOB' in sqlite_type:
        return 'TEXT'
    if 'BLOB' in sqlite_type:
        return 'BYTEA'
    if 'REAL' in sqlite_type or 'FLOA' in sqlite_type or 'DOUB' in sqlite_type:
        return 'DOUBLE PRECISION'
    if 'NUMERIC' in sqlite_type or 'DECIMAL' in sqlite_type:
        return 'NUMERIC'
    return 'TEXT' # Default fallback

def translate_schema_for_postgres(create_sql):
    """Translates a SQLite CREATE TABLE statement (including vec0 virtual tables) to a PostgreSQL compatible format."""
    is_virtual_vec_table = 'USING VEC0' in create_sql.upper() and 'VIRTUAL' in create_sql.upper()
    create_sql = create_sql.replace('`', '"')

    if is_virtual_vec_table:
        table_name_match = re.search(
            r'CREATE\s+VIRTUAL\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?\s*([`"\[]\S+[`"\]]|\S+)',
            create_sql, flags=re.IGNORECASE
        )
        if not table_name_match:
            raise ValueError(f"Could not parse table name from vec0 SQL: {create_sql}")
        table_name = table_name_match.group(1)

        columns_part_match = re.search(r'USING\s+vec0\s*\((.*)\)', create_sql, re.IGNORECASE | re.DOTALL)
        if not columns_part_match:
            raise ValueError(f"Could not parse column definitions from vec0 SQL: {create_sql}")
        columns_part = columns_part_match.group(1)

        columns_defs = re.split(r',(?![^\(]*\))', columns_part)
        new_defs = []
        for col_def in columns_defs:
            col_def = col_def.strip()
            if not col_def:
                continue

            vec_col_match = re.match(r'([`"\[]?\w+[`"\]]?)\s+float\s*\[\s*(\d+)\s*\]', col_def, re.IGNORECASE)
            if vec_col_match:
                col_name = vec_col_match.group(1).replace('`', '"')
                if '"' not in col_name:
                    col_name = f'"{col_name}"'
                dimension = int(vec_col_match.group(2))
                new_defs.append(f'{col_name} vector({dimension})')
            else:
                col_match = re.match(r'([`"\[]?\w+[`"\]]?)\s+([A-Z]+(?:\(\d+(?:,\d+)?\))?)(.*)', col_def, flags=re.IGNORECASE)
                if col_match:
                    col_name, col_type, constraints = col_match.groups()
                    col_name = col_name.replace('`', '"')
                    if '"' not in col_name:
                        col_name = f'"{col_name}"'
                    pg_type = translate_type_for_postgres(col_type)
                    new_defs.append(f'{col_name} {pg_type}{constraints}')
                else:
                    new_defs.append(col_def)
        return f"CREATE TABLE {table_name} ({', '.join(new_defs)})"

    create_sql = re.sub(r'\bAUTOINCREMENT\b', '', create_sql, flags=re.IGNORECASE)
    create_sql = re.sub(r'INTEGER\s+PRIMARY\s+KEY', 'INTEGER', create_sql, flags=re.IGNORECASE)
    try:
        table_name_match = re.search(r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?\s*([`"\[]\S+[`"\]]|\S+)\s*(\(.*\))', create_sql, flags=re.IGNORECASE | re.DOTALL)
        if not table_name_match:
            return create_sql
        table_name, defs_part = table_name_match.groups()
        defs_content = defs_part.strip()[1:-1]
        defs_list = re.split(r',(?![^\(]*\))', defs_content)
        new_defs = []
        for definition in defs_list:
            definition = definition.strip()
            if definition.upper().startswith(('PRIMARY KEY', 'FOREIGN KEY', 'UNIQUE', 'CONSTRAINT', 'CHECK')):
                continue
            col_match = re.match(r'([`"\[]?\w+[`"\]]?)\s+([A-Z]+(?:\(\d+(?:,\d+)?\))?)(.*)', definition, flags=re.IGNORECASE)
            if col_match:
                col_name, col_type, constraints = col_match.groups()
                if '"' not in col_name:
                    col_name = f'"{col_name}"'
                pg_type = translate_type_for_postgres(col_type)
                new_defs.append(f'{col_name} {pg_type}{constraints}')
            else:
                new_defs.append(definition)
        return f"CREATE TABLE {table_name} ({', '.join(new_defs)})"
    except Exception:
        return re.sub(r'\bINTEGER\b', 'BIGINT', create_sql, flags=re.IGNORECASE)


def translate_type_for_clickhouse(sqlite_type):
    """Maps SQLite data types to ClickHouse types."""
    sqlite_type = sqlite_type.upper()
    if 'INT' in sqlite_type: return 'Int64'
    if 'CHAR' in sqlite_type or 'TEXT' in sqlite_type or 'CLOB' in sqlite_type: return 'String'
    if 'BLOB' in sqlite_type: return 'String'
    if 'REAL' in sqlite_type or 'FLOA' in sqlite_type or 'DOUB' in sqlite_type: return 'Float64'
    if 'NUMERIC' in sqlite_type or 'DECIMAL' in sqlite_type: return 'Decimal(38, 6)'
    if 'DATE' in sqlite_type: return 'Date'
    if 'DATETIME' in sqlite_type: return 'DateTime'
    return 'String'

def translate_schema_for_clickhouse(create_sql):
    """Translates a SQLite CREATE TABLE statement (including vec0 virtual tables) to a ClickHouse compatible format."""
    create_sql = re.sub(r'--.*', '', create_sql)
    table_name_match = re.search(
        r'CREATE\s+(?:VIRTUAL\s+)?TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?\s*([`"\[]\S+[`"\]]|[^\s(]+)',
        create_sql, flags=re.IGNORECASE
    )
    if not table_name_match:
        raise ValueError(f"Could not parse table name from SQL: {create_sql}")
    table_name = table_name_match.group(1).strip('`"[]\'')
    lines, indices = [], []
    columns_part_match = re.search(r'\((.*)\)', create_sql, re.DOTALL)
    if not columns_part_match:
        raise ValueError(f"Could not parse column definitions from SQL: {create_sql}")
    columns_part = columns_part_match.group(1)
    columns_defs = re.split(r',(?![^\(]*\))', columns_part)
    for col_def in columns_defs:
        col_def = col_def.strip()
        if not col_def or col_def.upper().startswith(('PRIMARY KEY', 'FOREIGN KEY', 'UNIQUE', 'CONSTRAINT', 'CHECK')):
            continue
        vec_col_match = re.match(r'([`"\[]?\w+[`"\]]?)\s+float\s*\[\s*(\d+)\s*\]', col_def, re.IGNORECASE)
        if vec_col_match:
            col_name = vec_col_match.group(1).strip('`"')
            dimension = int(vec_col_match.group(2))
            lines.append(f'`{col_name}` Array(Float32)')
            indices.append(
                f"INDEX `idx_vec_{col_name}` `{col_name}` TYPE vector_similarity('hnsw', 'L2Distance', {dimension}) GRANULARITY 1000"
            )
        else:
            parts = re.split(r'\s+', col_def, 2)
            col_name = parts[0].strip('`"')
            if len(parts) > 1:
                sqlite_type = parts[1].split('(')[0]
                ch_type = translate_type_for_clickhouse(sqlite_type)
                if 'NOT NULL' not in col_def.upper():
                    ch_type = f'Nullable({ch_type})'
                lines.append(f'`{col_name}` {ch_type}')
    all_definitions = lines + indices
    create_table_ch = f"CREATE TABLE `{table_name}` (\n    "
    create_table_ch += ',\n    '.join(all_definitions)
    create_table_ch += f"\n) ENGINE = MergeTree() ORDER BY tuple();"
    return create_table_ch

# --- Database File Search ---

def find_database_files(source_path):
    """Recursively searches a folder for database files (.db, .sqlite, .sqlite3)."""
    database_files = []
    if os.path.isfile(source_path):
        if source_path.lower().endswith(('.db', '.sqlite', '.sqlite3')):
            database_files.append(source_path)
        else:
            logging.error("✖ File '%s' is not a supported database format (.db, .sqlite, .sqlite3).", source_path)
    elif os.path.isdir(source_path):
        logging.info("🔍 Searching for database files in folder '%s'...", source_path)
        patterns = ['**/*.db', '**/*.sqlite', '**/*.sqlite3']
        for pattern in patterns:
            files = glob.glob(os.path.join(source_path, pattern), recursive=True)
            database_files.extend(files)
        database_files = sorted(list(set(database_files)))
        if not database_files:
            logging.warning("✖ No database files found in folder '%s'.", source_path)
        else:
            logging.info("✔ Found %d database files:", len(database_files))
            for i, db_file in enumerate(database_files, 1):
                logging.info("  %d. %s", i, db_file)
    else:
        logging.error("✖ Path '%s' does not exist or is not accessible.", source_path)
    return database_files

# --- Database Connection & Operations ---

@contextmanager
def sqlite_conn(db_path):
    """Context manager for SQLite connection with vec0 extension loading."""
    conn = sqlite3.connect(db_path)
    try:
        conn.enable_load_extension(True)
        import sqlite_vec
        sqlite_vec.load(conn)
        logging.info("✔ sqlite-vec extension loaded successfully.")
    except Exception as e:
        logging.warning("🟡 Failed to load sqlite-vec extension: %s. This can be ignored if the DB has no vector tables.", e)
    try:
        yield conn
    finally:
        conn.close()

def get_sqlite_schema(conn):
    """Gets all table names and their CREATE statements from a SQLite database."""
    cursor = conn.cursor()
    cursor.execute("SELECT name, sql FROM sqlite_master WHERE (type='table' OR type='view') AND name NOT LIKE 'sqlite_%' AND sql IS NOT NULL;")
    schema = cursor.fetchall()
    if not schema:
        raise RuntimeError("No tables found in the source SQLite database.")
    return schema

def get_vec_info(conn):
    """Scans the database for sqlite-vec tables and extracts vector column info."""
    vec_info = {}
    cursor = conn.cursor()
    cursor.execute("SELECT name, sql FROM sqlite_master WHERE sql LIKE '%USING vec0%';")
    vec_tables = cursor.fetchall()
    for table_name, create_sql in vec_tables:
        match = re.search(r'USING\s+vec0\s*\((.*)\)', create_sql, re.IGNORECASE | re.DOTALL)
        if not match:
            continue
        vec_info[table_name] = {'columns': []}
        defs_part = match.group(1).strip()
        columns_defs = re.split(r',(?![^\(]*\))', defs_part)
        for col_def in columns_defs:
            vec_col_match = re.match(r'([`"\[]?\w+[`"\]]?)\s+float\s*\[\s*(\d+)\s*\]', col_def.strip(), re.IGNORECASE)
            if vec_col_match:
                col_name = vec_col_match.group(1).strip('`"[]')
                dimension = int(vec_col_match.group(2))
                vec_info[table_name]['columns'].append({'name': col_name, 'dim': dimension})
    return vec_info

# ... (migrate_to_postgres 和 migrate_to_clickhouse 函数保持不变) ...
# [此处省略了未修改的 migrate_to_postgres 和 migrate_to_clickhouse]
# [您原始脚本中的 migrate_to_postgres 和 migrate_to_clickhouse 应放在这里]

def migrate_to_postgres(args, db_name, source_db_path):
    """Performs migration to PostgreSQL (with sqlite-vec support)."""
    if not psycopg2:
        logging.error("psycopg2-binary is not installed. Please run 'pip install psycopg2-binary'.")
        return
    if not tqdm:
        logging.warning("tqdm is not installed. Progress bars will not be shown. Please run 'pip install tqdm'.")
    logging.info("\n🚀 Starting migration to PostgreSQL...")
    conn_admin, target_conn = None, None
    try:
        logging.info("Connecting to server to create database...")
        conn_admin = psycopg2.connect(
            host=args.host, port=args.port, user=args.user, password=args.password, dbname='postgres'
        )
        conn_admin.autocommit = True
        with conn_admin.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
            if cur.fetchone():
                logging.info("Database '%s' already exists.", db_name)
            else:
                logging.info("Creating database: %s", db_name)
                cur.execute(f'CREATE DATABASE "{db_name}"')
                logging.info("✔ Database '%s' created successfully.", db_name)
        
        logging.info("Connecting to new database '%s'...", db_name)
        with psycopg2.connect(host=args.host, port=args.port, user=args.user, password=args.password, dbname=db_name) as target_conn:
            with target_conn.cursor() as target_cur:
                logging.info("  - Enabling pgvector extension (if not already enabled)...")
                target_cur.execute('CREATE EXTENSION IF NOT EXISTS vector;')
                logging.info("  ✔ pgvector extension is ready.")

                with sqlite_conn(source_db_path) as source_conn:
                    vec_tables_info = get_vec_info(source_conn)
                    if vec_tables_info:
                        logging.info("✔ Detected %d sqlite-vec tables: %s", len(vec_tables_info), ', '.join(vec_tables_info.keys()))

                    source_cur = source_conn.cursor()
                    tables_schema = get_sqlite_schema(source_conn)

                    for table_name, create_sql in tables_schema:
                        if table_name == 'sqlite_sequence' or table_name.endswith(('_info', '_chunks', '_rowids')) or any(suffix in table_name for suffix in ('_metadatatext', '_metadatachunks', '_vector_chunks')):
                            continue
                        
                        table_name_quoted = f'"{table_name.strip("`[]")}"'
                        logging.info("\n-- Processing table: %s --", table_name_quoted)
                        logging.info("  - Translating Schema...")
                        pg_create_sql = translate_schema_for_postgres(create_sql)
                        pg_create_sql = re.sub(
                            r'CREATE\s+TABLE\s+([`"\[]?\S+[`"\]]?)',
                            f'CREATE TABLE {table_name_quoted}',
                            pg_create_sql, count=1, flags=re.IGNORECASE
                        )
                        logging.info("  - Creating table %s in target database...", table_name_quoted)
                        target_cur.execute(f'DROP TABLE IF EXISTS {table_name_quoted} CASCADE;')
                        target_cur.execute(pg_create_sql)
                        
                        logging.info("  - Preparing to extract data from SQLite...")
                        source_cur.execute(f'SELECT COUNT(*) FROM `{table_name.strip("`[]")}`')
                        total_rows = source_cur.fetchone()[0]
                        if total_rows == 0:
                            logging.info("  - Table %s is empty, skipping data insertion.", table_name_quoted)
                            continue

                        source_cur.execute(f'SELECT * FROM `{table_name.strip("`[]")}`')
                        original_column_names = [desc[0] for desc in source_cur.description]
                        column_names = list(original_column_names)

                        try:
                            rowid_index = column_names.index('rowid')
                            logging.info("  - Detected and removing implicit 'rowid' column.")
                            del column_names[rowid_index]
                        except ValueError:
                            rowid_index = -1
                        
                        logging.info("  - Fetching PostgreSQL target table schema...")
                        target_cur.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = %s ORDER BY ordinal_position;", (table_name.strip('`"[]'),))
                        target_column_info = {name.lower(): data_type for name, data_type in target_cur.fetchall()}
                        try:
                            ordered_target_types = [target_column_info[col.lower()] for col in column_names]
                        except KeyError as e:
                            logging.error("  🔴 Column '%s' does not match between source data and target schema.", e.args[0])
                            raise ValueError(f"Column mismatch, cannot continue migrating table {table_name}")

                        vec_info_for_table = vec_tables_info.get(table_name)
                        vector_column_indices = {}
                        if vec_info_for_table:
                            logging.info("  - Decoding sqlite-vec vector data...")
                            vector_column_indices = {i: item['dim'] for i, name in enumerate(column_names) for item in vec_info_for_table['columns'] if item['name'] == name}
                        
                        logging.info("  - Starting batch insert of %d rows into PostgreSQL...", total_rows)
                        # 在并行工作进程中禁用tqdm，因为它会搞乱主进度条
                        # progress_bar = tqdm(total=total_rows, desc=f"  📤 Migrating {table_name}", unit="rows") if tqdm else None
                        while True:
                            rows_batch = source_cur.fetchmany(BATCH_SIZE)
                            if not rows_batch: break
                            
                            processed_rows = [list(row) for row in rows_batch]
                            if rowid_index != -1:
                                original_rowid_index = original_column_names.index('rowid')
                                for row in processed_rows:
                                    del row[original_rowid_index]

                            if vec_info_for_table:
                                for row in processed_rows:
                                    for i, dim in vector_column_indices.items():
                                        blob = row[i]
                                        if isinstance(blob, bytes):
                                            num_floats = len(blob) // 4
                                            unpacked_data = list(struct.unpack(f'<{num_floats}f', blob))
                                            if num_floats != dim:
                                                logging.warning("  🟡 Vector dimension mismatch in table '%s'. Expected %d, got %d. Padding/truncating.", table_name, dim, num_floats)
                                                unpacked_data = (unpacked_data + [0.0] * dim)[:dim]
                                            row[i] = str(unpacked_data)
                            
                            final_data = []
                            for row in processed_rows:
                                coerced_row = [
                                    value if 'vector' in ordered_target_types[i].lower() else coerce_value(value, ordered_target_types[i])
                                    for i, value in enumerate(row)
                                ]
                                final_data.append(tuple(coerced_row))
                            
                            insert_query = 'INSERT INTO {} ({}) VALUES ({})'.format(
                                table_name_quoted,
                                ', '.join([f'"{c}"' for c in column_names]),
                                ', '.join(['%s'] * len(column_names))
                            )
                            execute_batch(target_cur, insert_query, final_data, page_size=1000)
                            # if progress_bar: progress_bar.update(len(rows_batch))
                        # if progress_bar: progress_bar.close()
                        logging.info("  ✔ Table %s data migration complete.", table_name_quoted)
            target_conn.commit()
            logging.info("\n🎉 All tables migrated successfully!")

    except (Exception, psycopg2.Error) as e:
        if target_conn: target_conn.rollback()
        cleanup_postgres_db(args, db_name)
        raise e
    finally:
        if conn_admin: conn_admin.close()
        if target_conn and not target_conn.closed: target_conn.close()


def migrate_to_clickhouse(args, db_name, source_db_path):
    """Performs migration to ClickHouse (with sqlite-vec support)."""
    if not Client:
        logging.error("clickhouse-driver is not installed. Please run 'pip install clickhouse-driver'.")
        return
    if not tqdm:
        logging.warning("tqdm is not installed. Progress bars will not be shown. Please run 'pip install tqdm'.")
    logging.info("\n🚀 Starting migration to ClickHouse...")
    client = None
    try:
        logging.info("Connecting to server to create database...")
        with Client(host=args.host, port=args.port, user=args.user, password=args.password) as admin_client:
            logging.info("Creating database (if not exists): %s", db_name)
            admin_client.execute(f'CREATE DATABASE IF NOT EXISTS `{db_name}`')
            logging.info("✔ Database '%s' is ready.", db_name)

        logging.info("Connecting to new database '%s'...", db_name)
        client = Client(host=args.host, port=args.port, user=args.user, password=args.password, database=db_name)

        with sqlite_conn(source_db_path) as source_conn:
            vec_tables_info = get_vec_info(source_conn)
            if vec_tables_info:
                logging.info("✔ Detected %d sqlite-vec tables: %s", len(vec_tables_info), ', '.join(vec_tables_info.keys()))

            source_cur = source_conn.cursor()
            tables_schema = get_sqlite_schema(source_conn)

            for table_name, create_sql in tables_schema:
                if table_name == 'sqlite_sequence' or table_name.endswith(('_info', '_chunks', '_rowids')) or any(suffix in table_name for suffix in ('_metadatatext', '_metadatachunks', '_vector_chunks')):
                    continue
                logging.info("\n-- Processing table: %s --", table_name)
                logging.info("  - Translating Schema...")
                ch_create_sql = translate_schema_for_clickhouse(create_sql)
                logging.info("  - Creating table '%s' in target database...", table_name)
                client.execute(f"DROP TABLE IF EXISTS `{table_name}`")
                client.execute(ch_create_sql)
                
                logging.info("  - Fetching ClickHouse target table schema...")
                target_column_types = {name: type_str for name, type_str in client.execute(f"SELECT name, type FROM system.columns WHERE database = '{db_name}' AND table = '{table_name}'")}
                
                logging.info("  - Preparing to extract data from SQLite...")
                source_cur.execute(f'SELECT COUNT(*) FROM `{table_name.strip("`[]")}`')
                total_rows = source_cur.fetchone()[0]
                if total_rows == 0:
                    logging.info("  - Table '%s' is empty, skipping data insertion.", table_name)
                    continue

                source_cur.execute(f'SELECT * FROM `{table_name.strip("`[]")}`')
                original_column_names = [desc[0] for desc in source_cur.description]
                column_names = list(original_column_names)

                try:
                    rowid_index = column_names.index('rowid')
                    logging.info("  - Detected and removing implicit 'rowid' column.")
                    del column_names[rowid_index]
                except ValueError:
                    rowid_index = -1
                
                vec_info_for_table = vec_tables_info.get(table_name)
                vector_column_indices = {}
                if vec_info_for_table:
                    logging.info("  - Decoding sqlite-vec vector data...")
                    vector_column_indices = {i: item['dim'] for i, name in enumerate(column_names) for item in vec_info_for_table['columns'] if item['name'] == name}

                logging.info("  - Starting batch insert of %d rows into ClickHouse...", total_rows)
                # 在并行工作进程中禁用tqdm，因为它会搞乱主进度条
                # progress_bar = tqdm(total=total_rows, desc=f"  📤 Migrating {table_name}", unit="rows") if tqdm else None
                while True:
                    rows_batch = source_cur.fetchmany(BATCH_SIZE)
                    if not rows_batch: break
                    
                    processed_rows = [list(row) for row in rows_batch]
                    if rowid_index != -1:
                        original_rowid_index = original_column_names.index('rowid')
                        for row in processed_rows: del row[original_rowid_index]
                    
                    if vec_info_for_table:
                        for row in processed_rows:
                            for i, dim in vector_column_indices.items():
                                blob = row[i]
                                if isinstance(blob, bytes):
                                    num_floats = len(blob) // 4
                                    if num_floats == dim:
                                        row[i] = list(struct.unpack(f'<{dim}f', blob))
                                    else:
                                        logging.warning("  🟡 Vector dimension mismatch in table '%s'. Expected %d, got %d. Padding/truncating.", table_name, dim, num_floats)
                                        unpacked_data = list(struct.unpack(f'<{num_floats}f', blob))
                                        row[i] = (unpacked_data + [0.0] * dim)[:dim]

                    final_data = []
                    for row in processed_rows:
                        coerced_row = [coerce_value(row[i], target_column_types.get(col_name, 'String')) for i, col_name in enumerate(column_names)]
                        final_data.append(tuple(coerced_row))
                    
                    if not final_data: continue
                    insert_statement = f"INSERT INTO `{table_name}` ({', '.join([f'`{c}`' for c in column_names])}) VALUES"
                    client.execute(insert_statement, final_data, types_check=True)
                    # if progress_bar: progress_bar.update(len(rows_batch))
                # if progress_bar: progress_bar.close()
                logging.info("  ✔ Table '%s' data migration complete.", table_name)
        logging.info("\n🎉 All tables migrated successfully!")

    except Exception as e:
        cleanup_clickhouse_db(args, db_name)
        raise e
    finally:
        if client: client.disconnect()

### --- NEW --- ###
def run_migration_task(db_file, db_name, target, task_args, pg_args, ch_args):
    """
    A self-contained worker function to be run in a process pool.
    It handles one DB file and returns a status tuple.
    
    NOTE: Logging from this function will be interleaved in the console,
    which is expected behavior in a parallel setup.
    """
    pg_success, ch_success = None, None
    pg_error, ch_error = None, None

    logging.info("======================================================================")
    logging.info("🔄 [Worker] Processing: %s (Target DB: %s)", os.path.basename(db_file), db_name)
    logging.info("======================================================================")

    if target == 'postgresql':
        try:
            migrate_to_postgres(task_args, db_name, db_file)
            pg_success = True
            logging.info("✅ [Worker] SUCCESS: PostgreSQL migration for %s", db_name)
        except Exception as e:
            pg_success = False
            pg_error = str(e)
            logging.error("❌ [Worker] FAILED: PostgreSQL migration for %s: %s", db_name, e)
    
    elif target == 'clickhouse':
        try:
            migrate_to_clickhouse(task_args, db_name, db_file)
            ch_success = True
            logging.info("✅ [Worker] SUCCESS: ClickHouse migration for %s", db_name)
        except Exception as e:
            ch_success = False
            ch_error = str(e)
            logging.error("❌ [Worker] FAILED: ClickHouse migration for %s: %s", db_name, e)
    
    elif target == 'both':
        # PostgreSQL
        try:
            logging.info("  -> [Worker] Attempting PostgreSQL for %s...", db_name)
            migrate_to_postgres(pg_args, db_name, db_file)
            pg_success = True
            logging.info("  ✔ [Worker] SUCCESS: PostgreSQL for %s", db_name)
        except Exception as e:
            pg_success = False
            pg_error = str(e)
            logging.error("  ❌ [Worker] FAILED: PostgreSQL for %s: %s", db_name, e)
        
        # ClickHouse
        try:
            logging.info("  -> [Worker] Attempting ClickHouse for %s...", db_name)
            migrate_to_clickhouse(ch_args, db_name, db_file)
            ch_success = True
            logging.info("  ✔ [Worker] SUCCESS: ClickHouse for %s", db_name)
        except Exception as e:
            ch_success = False
            ch_error = str(e)
            logging.error("  ❌ [Worker] FAILED: ClickHouse for %s: %s", db_name, e)

    return (db_name, pg_success, pg_error, ch_success, ch_error)


### --- DELETED --- ###
# The `migrate_to_both_backends` function is no longer needed,
# its logic is now inside `run_migration_task` and the `main` function's result processing.


### --- MODIFIED --- ###
def main():
    # --- Setup Logging ---
    logging.basicConfig(
        ### --- MODIFIED --- ###
        level=logging.INFO, # Changed from WARNING to INFO to see progress
        format='%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # --- Check for missing optional dependencies ---
    if psycopg2 is None:
        logging.warning("psycopg2-binary not found. PostgreSQL migration will not be available.")
    if Client is None:
        logging.warning("clickhouse-driver not found. ClickHouse migration will not be available.")
    if tqdm is None:
        logging.info("tqdm not found. Progress bars will not be displayed.")

    parser = argparse.ArgumentParser(
        description="Migrate SQLite databases (with sqlite-vec support) to PostgreSQL, ClickHouse, or both. Supports single files or batch folder processing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--source', default='/mnt/DataFlow/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/sqlite/results/spider/vector_databases', help="Source SQLite database file or folder path.")
    parser.add_argument('--target', default='both', choices=['postgresql', 'clickhouse', 'both'], help="Target database type. 'both' migrates to both backends.")
    parser.add_argument('--host', help="Target DB host (for single-target mode).")
    parser.add_argument('--port', type=int, help="Target DB port (for single-target mode).")
    parser.add_argument('--user', help="Target DB username (for single-target mode).")
    parser.add_argument('--password', help="Target DB password (for single-target mode).")
    
    ### --- NEW --- ###
    # A safe default for workers: min(32, cpu_count + 4). 
    # For DB work, more workers than cores can be useful due to I/O wait.
    # But with 100+ CPUs, we cap at a reasonable number to avoid overwhelming the target DBs.
    # 你可以根据你的100+CPU环境，把 32 调整得更高，比如 64 或 96
    default_workers = min(32, (os.cpu_count() or 1) + 4)
    parser.add_argument('--workers', type=int, default=default_workers, help="Number of parallel migration processes to run.")

    pg_group = parser.add_argument_group('PostgreSQL Options (for --target=both)')
    pg_group.add_argument('--pg-host', default='localhost', help="[PostgreSQL] Host.")
    pg_group.add_argument('--pg-port', type=int, default=5432, help="[PostgreSQL] Port.")
    pg_group.add_argument('--pg-user', default='postgres', help="[PostgreSQL] User.")
    pg_group.add_argument('--pg-password', default='postgres', help="[PostgreSQL] Password.")

    ch_group = parser.add_argument_group('ClickHouse Options (for --target=both)')
    ch_group.add_argument('--ch-host', default='localhost', help="[ClickHouse] Host.")
    ch_group.add_argument('--ch-port', type=int, default=9000, help="[ClickHouse] Port.")
    ch_group.add_argument('--ch-user', default='default', help="[ClickHouse] User.")
    ch_group.add_argument('--ch-password', default='', help="[ClickHouse] Password.")
    
    args = parser.parse_args()

    if not os.path.exists(args.source):
        logging.error("✖ Source path '%s' does not exist.", args.source)
        return

    database_files = find_database_files(args.source)
    if not database_files:
        logging.warning("✖ No migratable database files were found.")
        return

    ### --- MODIFIED --- ###
    # This section is completely rewritten for parallelism.
    
    logging.info("🚀 Starting parallel migration for %d files using %d workers...", len(database_files), args.workers)

    # Prepare arguments for 'both' mode
    pg_args = argparse.Namespace(host=args.pg_host, port=args.pg_port, user=args.pg_user, password=args.pg_password)
    ch_args = argparse.Namespace(host=args.ch_host, port=args.ch_port, user=args.ch_user, password=args.ch_password)
    
    # Prepare arguments for 'single' mode
    task_args = None
    if args.target != 'both':
        if args.host is None: args.host = 'localhost'
        if args.port is None: args.port = 5432 if args.target == 'postgresql' else 9000
        if args.user is None: args.user = 'postgres' if args.target == 'postgresql' else 'default'
        if args.password is None: args.password = 'postgres' if args.target == 'postgresql' else ''
        task_args = args # Pass the main args namespace

    # Dictionaries to store results
    successful_dbs = []
    pg_failures = []
    ch_failures = []
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for db_file in database_files:
            db_name = os.path.splitext(os.path.basename(db_file))[0]
            db_name = re.sub(r'[^a-zA-Z0-9_]', '_', str(db_name)).lower()
            
            # Submit the task to the pool
            future = executor.submit(
                run_migration_task, 
                db_file, 
                db_name, 
                args.target, 
                task_args,  # For single-target mode
                pg_args,    # For 'both' mode
                ch_args     # For 'both' mode
            )
            futures.append(future)

        logging.info("All %d migration tasks submitted to the pool. Waiting for completion...", len(futures))
        
        # Setup tqdm progress bar
        progress_bar = None
        if tqdm:
            progress_bar = tqdm(total=len(futures), desc="Migrating Databases", unit="db")
        
        # Process results as they complete
        for future in as_completed(futures):
            try:
                (db_name, pg_success, pg_error, ch_success, ch_error) = future.result()

                if args.target == 'postgresql':
                    if pg_success:
                        successful_dbs.append(db_name)
                    else:
                        pg_failures.append(db_name)
                elif args.target == 'clickhouse':
                    if ch_success:
                        successful_dbs.append(db_name)
                    else:
                        ch_failures.append(db_name)
                elif args.target == 'both':
                    if pg_success and ch_success:
                        successful_dbs.append(db_name)
                    if not pg_success:
                        pg_failures.append(db_name)
                    if not ch_success:
                        ch_failures.append(db_name)
            
            except Exception as e:
                # This catches errors in the worker process *itself* before our try/except
                logging.error("A worker process failed unexpectedly: %s", e)
            
            if progress_bar:
                progress_bar.update(1)

        if progress_bar:
            progress_bar.close()

    # --- Final Report ---
    total_files = len(database_files)
    success_count = len(successful_dbs)
    
    logging.info("\n======================================================================")
    logging.info("🎯 Final Migration Report (Mode: %s)", args.target)
    logging.info("======================================================================")
    logging.info("  Total files processed: %d", total_files)

    if args.target == 'both':
        logging.info("  Fully successful (migrated to both backends): %d", success_count)
        logging.info("  Partially or fully failed: %d", total_files - success_count)
        if successful_dbs:
            logging.info("\n  List of fully successful databases:")
            for db in successful_dbs: logging.info("    - %s", db)
        if pg_failures:
            logging.warning("\n  List of databases that FAILED PostgreSQL migration:")
            for db in pg_failures: logging.warning("    - %s", db)
        if ch_failures:
            logging.warning("\n  List of databases that FAILED ClickHouse migration:")
            for db in ch_failures: logging.warning("    - %s", db)
    else:
        logging.info("  Successful: %d", success_count)
        logging.info("  Failed: %d", total_files - success_count)
        if successful_dbs:
            logging.info("\n  List of successful databases:")
            for db in successful_dbs: logging.info("    - %s", db)
        failures = pg_failures if args.target == 'postgresql' else ch_failures
        if failures:
            logging.warning("\n  List of databases that FAILED migration:")
            for db in failures: logging.warning("    - %s", db)

    total_failures = len(pg_failures) + len(ch_failures)
    if total_failures > 0:
        logging.warning("\n⚠️  %d migration tasks did not complete successfully. Please review the logs above.", total_failures)
    else:
        logging.info("\n🎉 All database migration tasks completed successfully!")


if __name__ == '__main__':
    main()
