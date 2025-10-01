import argparse
import os
import sqlite3
import re
import struct
import glob
from contextlib import contextmanager
from datetime import datetime

# 尝试导入依赖，如果失败则给出提示
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

def cleanup_postgres_db(args, db_name):
    """Connects to the PostgreSQL server and drops a specific database."""
    conn_admin = None
    try:
        print(f"  🧹 Cleaning up failed PostgreSQL database '{db_name}'...")
        conn_admin = psycopg2.connect(
            host=args.host, port=args.port, user=args.user, password=args.password, dbname='postgres'
        )
        conn_admin.autocommit = True
        with conn_admin.cursor() as cur:
            # WITH (FORCE) is available from PostgreSQL 13+ and is very useful
            # for terminating any lingering connections from the failed script.
            cur.execute(f'DROP DATABASE IF EXISTS "{db_name}" WITH (FORCE);')
        print(f"  ✔ Cleanup successful.")
    except Exception as e:
        print(f"  🔴 Error during cleanup: {e}")
        print(f"     You may need to manually drop the database '{db_name}'.")
    finally:
        if conn_admin:
            conn_admin.close()

def cleanup_clickhouse_db(args, db_name):
    """Connects to the ClickHouse server and drops a specific database."""
    client = None
    try:
        print(f"  🧹 Cleaning up failed ClickHouse database '{db_name}'...")
        client = Client(host=args.host, port=args.port, user=args.user, password=args.password)
        client.execute(f'DROP DATABASE IF EXISTS `{db_name}`')
        print(f"  ✔ Cleanup successful.")
    except Exception as e:
        print(f"  🔴 Error during cleanup: {e}")
        print(f"     You may need to manually drop the database '{db_name}'.")
    finally:
        if client:
            client.disconnect()

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
            # print("  🟡 警告: 无法将 '{}' (类型: {}) 转换为整数。将插入 NULL。".format(value, type(value).__name__))
            return None

    # --- 浮点数/数值转换 ---
    if ('array' not in target_type_str) and ('vector' not in target_type_str) and any(float_type in target_type_str for float_type in ['float', 'double', 'real', 'numeric', 'decimal']):
        try:
            return float(value)
        except (ValueError, TypeError):
            print(target_type_str)
            # print("  🟡 警告: 无法将 '{}' (类型: {}) 转换为浮点数。将插入 NULL。".format(value, type(value).__name__))
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

        # print("  🟡 警告: 无法将字符串 '{}' 解析为任何已知的日期/时间格式。将插入 NULL。".format(value))
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

# --- Schema 翻译模块 ---

def translate_type_for_postgres(sqlite_type):
    """将 SQLite 数据类型映射到 PostgreSQL 类型"""
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
    return 'TEXT' # 默认回退

def translate_schema_for_postgres(create_sql):
    """将 SQLite 的 CREATE TABLE 语句（包括 vec0 虚拟表）翻译为 PostgreSQL 兼容格式"""
    # 检查是否为 sqlite-vec 虚拟表
    is_virtual_vec_table = 'USING VEC0' in create_sql.upper() and 'VIRTUAL' in create_sql.upper()

    # 统一处理引号
    create_sql = create_sql.replace('`', '"')

    if is_virtual_vec_table:
        # 从 "CREATE VIRTUAL TABLE" 中提取表名
        table_name_match = re.search(
            r'CREATE\s+VIRTUAL\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?\s*([`"\[]\S+[`"\]]|\S+)',
            create_sql, flags=re.IGNORECASE
        )
        if not table_name_match:
            raise ValueError(f"无法从 vec0 SQL 中解析表名: {create_sql}")
        table_name = table_name_match.group(1)

        # 从 "USING vec0(...)" 中提取列定义
        columns_part_match = re.search(r'USING\s+vec0\s*\((.*)\)', create_sql, re.IGNORECASE | re.DOTALL)
        if not columns_part_match:
            raise ValueError(f"无法从 vec0 SQL 中解析列定义: {create_sql}")
        columns_part = columns_part_match.group(1)

        columns_defs = re.split(r',(?![^\(]*\))', columns_part)
        new_defs = []
        for col_def in columns_defs:
            col_def = col_def.strip()
            if not col_def:
                continue

            # 匹配向量列，例如 "col_embedding float[384]"
            vec_col_match = re.match(r'([`"\[]?\w+[`"\]]?)\s+float\s*\[\s*(\d+)\s*\]', col_def, re.IGNORECASE)
            if vec_col_match:
                col_name = vec_col_match.group(1).replace('`', '"')
                if '"' not in col_name:
                    col_name = f'"{col_name}"'
                dimension = int(vec_col_match.group(2))
                # 翻译为 pgvector 类型
                new_defs.append(f'{col_name} vector({dimension})')
            else:
                # 处理虚拟表定义中的普通列
                col_match = re.match(r'([`"\[]?\w+[`"\]]?)\s+([A-Z]+(?:\(\d+(?:,\d+)?\))?)(.*)', col_def, flags=re.IGNORECASE)
                if col_match:
                    col_name, col_type, constraints = col_match.groups()
                    col_name = col_name.replace('`', '"')
                    if '"' not in col_name:
                        col_name = f'"{col_name}"'
                    pg_type = translate_type_for_postgres(col_type)
                    new_defs.append(f'{col_name} {pg_type}{constraints}')
                else:
                    new_defs.append(col_def) # 回退

        # 构建标准的 CREATE TABLE 语句
        return f"CREATE TABLE {table_name} ({', '.join(new_defs)})"

    # --- 对标准表的现有处理逻辑 ---
    create_sql = re.sub(r'\bAUTOINCREMENT\b', '', create_sql, flags=re.IGNORECASE)
    create_sql = re.sub(r'INTEGER\s+PRIMARY\s+KEY', 'INTEGER', create_sql, flags=re.IGNORECASE)

    try:
        table_name_match = re.search(r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?\s*([`"\[]\S+[`"\]]|\S+)\s*(\(.*\))', create_sql, flags=re.IGNORECASE | re.DOTALL)
        if not table_name_match:
            return create_sql
            
        table_name = table_name_match.group(1)
        defs_part = table_name_match.group(2)
        
        defs_content = defs_part.strip()[1:-1]
        
        defs_list = re.split(r',(?![^\(]*\))', defs_content)
        new_defs = []
        for definition in defs_list:
            definition = definition.strip()
            
            if definition.upper().startswith(('PRIMARY KEY', 'FOREIGN KEY', 'UNIQUE', 'CONSTRAINT', 'CHECK')):
                # new_defs.append(definition)
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
    """将 SQLite 数据类型映射到 ClickHouse 类型"""
    sqlite_type = sqlite_type.upper()
    if 'INT' in sqlite_type: return 'Int64'
    if 'CHAR' in sqlite_type or 'TEXT' in sqlite_type or 'CLOB' in sqlite_type: return 'String'
    if 'BLOB' in sqlite_type: return 'String' # 默认为 String，向量 BLOB 会被特殊处理
    if 'REAL' in sqlite_type or 'FLOA' in sqlite_type or 'DOUB' in sqlite_type: return 'Float64'
    if 'NUMERIC' in sqlite_type or 'DECIMAL' in sqlite_type: return 'Decimal(38, 6)'
    if 'DATE' in sqlite_type: return 'Date'
    if 'DATETIME' in sqlite_type: return 'DateTime'
    return 'String'

def translate_schema_for_clickhouse(create_sql):
    """将 SQLite 的 CREATE TABLE 语句（包括 vec0 虚拟表）翻译为 ClickHouse 兼容格式"""
    create_sql = re.sub(r'--.*', '', create_sql)

    table_name_match = re.search(
        # r'CREATE\s+(?:VIRTUAL\s+)?TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?\s*([`"\[]\S+[`"\]]|\S+)',
        r'CREATE\s+(?:VIRTUAL\s+)?TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?\s*([`"\[]\S+[`"\]]|[^\s(]+)',
        create_sql, flags=re.IGNORECASE
    )
    if not table_name_match:
        raise ValueError(f"无法从 SQL 中解析表名: {create_sql}")
        
    table_name = table_name_match.group(1).strip('`"[]\'')

    lines = []
    indices = []

    columns_part_match = re.search(r'\((.*)\)', create_sql, re.DOTALL)
    if not columns_part_match:
        raise ValueError(f"无法从 SQL 中解析列定义: {create_sql}")
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

    create_table_ch = f"CREATE TABLE `{table_name}` (\n    "
    all_definitions = lines + indices
    create_table_ch += ',\n    '.join(all_definitions)
    create_table_ch += f"\n) ENGINE = MergeTree() ORDER BY tuple();"
    
    return create_table_ch


# --- 数据库文件搜索 ---

def find_database_files(source_path):
    """递归搜索文件夹中的数据库文件（.db, .sqlite, .sqlite3）"""
    database_files = []
    
    if os.path.isfile(source_path):
        # 如果是单个文件，检查是否为数据库文件
        if source_path.lower().endswith(('.db', '.sqlite', '.sqlite3')):
            database_files.append(source_path)
        else:
            print(f"✖ 错误：文件 '{source_path}' 不是支持的数据库文件格式（.db, .sqlite, .sqlite3）")
    elif os.path.isdir(source_path):
        # 如果是文件夹，递归搜索所有数据库文件
        print(f"🔍 正在搜索文件夹 '{source_path}' 中的数据库文件...")
        
        # 使用 glob 递归搜索所有支持的数据库文件
        patterns = ['**/*.db', '**/*.sqlite', '**/*.sqlite3']
        for pattern in patterns:
            files = glob.glob(os.path.join(source_path, pattern), recursive=True)
            database_files.extend(files)
        
        # 去重並排序
        database_files = sorted(list(set(database_files)))
        
        if not database_files:
            print(f"✖ 在文件夹 '{source_path}' 中未找到任何数据库文件")
        else:
            print(f"✔ 找到 {len(database_files)} 个数据库文件:")
            for i, db_file in enumerate(database_files, 1):
                print(f"  {i}. {db_file}")
    else:
        print(f"✖ 错误：路径 '{source_path}' 不存在或无法访问")
    
    return database_files

# --- 数据库连接与操作 ---

@contextmanager
def sqlite_conn(db_path):
    conn = sqlite3.connect(db_path)
    try:
        conn.enable_load_extension(True)
        import sqlite_vec
        # import sqlite_lembed # 迁移时通常不需要加载 lembed
        sqlite_vec.load(conn)
        # sqlite_lembed.load(conn)
        print("✔ sqlite-vec 扩展已成功加载。")
    except Exception as e:
        print(f"🟡 警告: 加载 sqlite-vec 扩展失败: {e}。如果数据库不含向量表，可忽略此消息。")

    try:
        yield conn
    finally:
        conn.close()

def get_sqlite_schema(conn):
    """获取 SQLite 数据库中所有表的名称和 CREATE 语句"""
    cursor = conn.cursor()
    cursor.execute("SELECT name, sql FROM sqlite_master WHERE (type='table' OR type='view') AND name NOT LIKE 'sqlite_%' AND sql IS NOT NULL;")
    schema = cursor.fetchall()
    if not schema:
        raise RuntimeError("源 SQLite 数据库中没有找到任何表。")
    return schema

def get_vec_info(conn):
    """扫描数据库以查找 sqlite-vec 表并提取向量列信息"""
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


def migrate_to_postgres(args, db_name, source_db_path):
    """执行到 PostgreSQL 的迁移（支持 sqlite-vec）"""
    if not psycopg2:
        print("错误：psycopg2-binary 未安装。请运行 'pip install psycopg2-binary'")
        return
    if not tqdm:
        print("错误：tqdm 未安装。请运行 'pip install tqdm'")
        return

    print(f"\n🚀 开始迁移到 PostgreSQL...")
    conn_admin = None
    target_conn = None
    try:
        # 1. 连接到 'postgres' 数据库来创建新数据库
        print("正在连接到服务器以创建数据库...")
        conn_admin = psycopg2.connect(
            host=args.host, port=args.port, user=args.user, password=args.password, dbname='postgres'
        )
        conn_admin.autocommit = True
        with conn_admin.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
            if cur.fetchone():
                print(f"数据库 '{db_name}' 已存在。")
            else:
                print(f"正在创建数据库: {db_name}")
                cur.execute(f'CREATE DATABASE "{db_name}"')
                print(f"✔ 数据库 '{db_name}' 创建成功。")

        # 2. 连接到新创建的数据库进行迁移
        print(f"正在连接到新数据库 '{db_name}'...")
        with psycopg2.connect(host=args.host, port=args.port, user=args.user, password=args.password, dbname=db_name) as target_conn:
            with target_conn.cursor() as target_cur:
                # 启用 pgvector 扩展
                print("  - 正在启用 pgvector 扩展 (如果尚未启用)...")
                target_cur.execute('CREATE EXTENSION IF NOT EXISTS vector;')
                print("  ✔ pgvector 扩展已就绪。")

                with sqlite_conn(source_db_path) as source_conn:
                    vec_tables_info = get_vec_info(source_conn)
                    if vec_tables_info:
                        print(f"✔ 检测到 {len(vec_tables_info)} 个 sqlite-vec 表: {', '.join(vec_tables_info.keys())}")

                    source_cur = source_conn.cursor()
                    tables_schema = get_sqlite_schema(source_conn)

                    for table_name, create_sql in tables_schema:
                        if table_name == 'sqlite_sequence' or table_name.endswith(('_info', '_chunks', '_rowids')) or any(suffix in table_name for suffix in ('_metadatatext', '_metadatachunks', '_vector_chunks')):
                            continue

                        table_name_quoted = f'"{table_name.strip("`[]")}"'
                        
                        print(f"\n-- 正在处理表: {table_name_quoted} --")

                        print("  - 翻译 Schema...")
                        pg_create_sql = translate_schema_for_postgres(create_sql)
                        
                        pg_create_sql = re.sub(
                            r'CREATE\s+TABLE\s+([`"\[]?\S+[`"\]]?)', 
                            f'CREATE TABLE {table_name_quoted}', 
                            pg_create_sql, 
                            count=1, 
                            flags=re.IGNORECASE
                        )

                        is_virtual = False
                        if 'USING vec0' in create_sql:
                            is_virtual = True
                            pg_create_sql = re.sub(r'USING\s+vec0', '', pg_create_sql, flags=re.IGNORECASE | re.DOTALL)
                            pg_create_sql = re.sub(r'VIRTUAL\s+TABLE', 'TABLE', pg_create_sql, flags=re.IGNORECASE)
                            pg_create_sql = re.sub(r'float\s*\[\s*(\d+)\s*\]', r'vector(\1)', pg_create_sql, flags=re.IGNORECASE)

                        print(f"  - 正在目标数据库中创建表 {table_name_quoted}...")
                        target_cur.execute(f'DROP TABLE IF EXISTS {table_name_quoted} CASCADE;')
                        target_cur.execute(pg_create_sql)

                        print("  - 正在从 SQLite 准备批量提取数据...")
                        source_cur.execute(f'SELECT COUNT(*) FROM `{table_name.strip("`[]")}`')
                        total_rows = source_cur.fetchone()[0]
                        
                        if total_rows == 0:
                            print(f"  - 表 {table_name_quoted} 为空，跳过数据插入。")
                            continue

                        source_cur.execute(f'SELECT * FROM `{table_name.strip("`[]")}`')
                        original_column_names = [desc[0] for desc in source_cur.description]
                        column_names = list(original_column_names)

                        try:
                            rowid_index = column_names.index('rowid')
                            print("  - 检测到并移除隐式的 'rowid' 列。")
                            del column_names[rowid_index]
                        except ValueError:
                            rowid_index = -1

                        print("  - 正在获取 PostgreSQL 目标表 Schema...")
                        target_cur.execute("""
                            SELECT column_name, data_type 
                            FROM information_schema.columns 
                            WHERE table_name = %s
                            ORDER BY ordinal_position;
                        """, (table_name.strip('`"[]'),))
                        
                        target_column_info = {name.lower(): data_type for name, data_type in target_cur.fetchall()}
                        
                        try:
                            ordered_target_types = [target_column_info[col.lower()] for col in column_names]
                        except KeyError as e:
                            print(f"  🔴 错误: 列 '{e.args[0]}' 在源数据和目标 Schema 之间不匹配。")
                            raise ValueError(f"列不匹配，无法继续迁移表 {table_name}")

                        vec_info_for_table = vec_tables_info.get(table_name)
                        if vec_info_for_table:
                             print("  - 正在解码 sqlite-vec 向量数据...")
                             vector_column_indices = {
                                i: item['dim'] 
                                for i, name in enumerate(column_names) 
                                for item in vec_info_for_table['columns'] if item['name'] == name
                            }

                        print(f"  - 开始批量插入 {total_rows} 条数据到 PostgreSQL...")

                        with tqdm(total=total_rows, desc=f"  📤 Migrating {table_name}", unit="rows") as pbar:
                            while True:
                                rows_batch = source_cur.fetchmany(BATCH_SIZE)
                                if not rows_batch:
                                    break
                                
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
                                                    print(f"  🟡 警告: 在表 '{table_name}' 中发现不匹配的向量维度。预期 {dim}，得到 {num_floats}。将填充或截断。")
                                                    unpacked_data = (unpacked_data + [0.0] * dim)[:dim]
                                                row[i] = str(unpacked_data)

                                final_data = []
                                for row in processed_rows:
                                    coerced_row = []
                                    for i, value in enumerate(row):
                                        target_type = ordered_target_types[i]
                                        if 'vector' in target_type.lower():
                                            coerced_row.append(value)
                                        else:
                                            coerced_value = coerce_value(value, target_type)
                                            coerced_row.append(coerced_value)
                                    final_data.append(tuple(coerced_row))

                                insert_query = 'INSERT INTO {} ({}) VALUES ({})'.format(
                                    table_name_quoted,
                                    ', '.join([f'"{c}"' for c in column_names]),
                                    ', '.join(['%s'] * len(column_names))
                                )
                                
                                execute_batch(target_cur, insert_query, final_data, page_size=1000)
                                
                                pbar.update(len(rows_batch))
                        
                        print(f"  ✔ 表 {table_name_quoted} 数据迁移完成。")

            target_conn.commit()
            print("\n🎉 所有表迁移成功！")

    except (Exception, psycopg2.Error) as e:
        if target_conn:
            target_conn.rollback()
        cleanup_postgres_db(args, db_name)
        raise e
    finally:
        if conn_admin:
            conn_admin.close()
        if target_conn and not target_conn.closed:
            target_conn.close()

def migrate_to_clickhouse(args, db_name, source_db_path):
    """执行到 ClickHouse 的迁移（支持 sqlite-vec、移除 rowid 并强制类型转换）"""
    if not Client:
        print("错误：clickhouse-driver 未安装。请运行 'pip install clickhouse-driver'")
        return
    if not tqdm:
        print("错误：tqdm 未安装。请运行 'pip install tqdm'")
        return
        
    print(f"\n🚀 开始迁移到 ClickHouse...")
    
    client = None
    try:
        print("正在连接到服务器以创建数据库...")
        with Client(host=args.host, port=args.port, user=args.user, password=args.password) as admin_client:
            print(f"正在创建数据库 (如果不存在): {db_name}")
            admin_client.execute(f'CREATE DATABASE IF NOT EXISTS `{db_name}`')
            print(f"✔ 数据库 '{db_name}' 已就绪。")

        print(f"正在连接到新数据库 '{db_name}'...")
        client = Client(host=args.host, port=args.port, user=args.user, password=args.password, database=db_name)

        with sqlite_conn(source_db_path) as source_conn:
            vec_tables_info = get_vec_info(source_conn)
            if vec_tables_info:
                print(f"✔ 检测到 {len(vec_tables_info)} 个 sqlite-vec 表: {', '.join(vec_tables_info.keys())}")

            source_cur = source_conn.cursor()
            tables_schema = get_sqlite_schema(source_conn)

            for table_name, create_sql in tables_schema:
                if table_name == 'sqlite_sequence' or table_name.endswith(('_info', '_chunks', '_rowids')) or any(suffix in table_name for suffix in ('_metadatatext', '_metadatachunks', '_vector_chunks')):
                    continue

                print(f"\n-- 正在处理表: {table_name} --")

                print("  - 翻译 Schema...")
                ch_create_sql = translate_schema_for_clickhouse(create_sql)
                
                print(f"  - 正在目标数据库中创建表 '{table_name}'...")
                client.execute(f"DROP TABLE IF EXISTS `{table_name}`")
                client.execute(ch_create_sql)
                
                print("  - 正在获取 ClickHouse 目标表 Schema...")
                target_column_types = {
                    name: type_str for name, type_str in client.execute(
                        f"SELECT name, type FROM system.columns WHERE database = '{db_name}' AND table = '{table_name}'"
                    )
                }

                print("  - 正在从 SQLite 准备批量提取数据...")
                source_cur.execute(f'SELECT COUNT(*) FROM `{table_name.strip("`[]")}`')
                total_rows = source_cur.fetchone()[0]

                if total_rows == 0:
                    print(f"  - 表 '{table_name}' 为空，跳过数据插入。")
                    continue

                source_cur.execute(f'SELECT * FROM `{table_name.strip("`[]")}`')
                original_column_names = [desc[0] for desc in source_cur.description]
                column_names = list(original_column_names)

                try:
                    rowid_index = column_names.index('rowid')
                    print("  - 检测到并移除隐式的 'rowid' 列。")
                    del column_names[rowid_index]
                except ValueError:
                    rowid_index = -1

                vec_info_for_table = vec_tables_info.get(table_name)
                if vec_info_for_table:
                    print("  - 正在解码 sqlite-vec 向量数据...")
                    vector_column_indices = {
                        i: item['dim'] 
                        for i, name in enumerate(column_names) 
                        for item in vec_info_for_table['columns'] if item['name'] == name
                    }

                print(f"  - 开始批量插入 {total_rows} 条数据到 ClickHouse...")

                with tqdm(total=total_rows, desc=f"  📤 Migrating {table_name}", unit="rows") as pbar:
                    while True:
                        rows_batch = source_cur.fetchmany(BATCH_SIZE)
                        if not rows_batch:
                            break
                        
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
                                        if num_floats == dim:
                                            row[i] = list(struct.unpack(f'<{dim}f', blob))
                                        else:
                                            print(f"  🟡 警告: 在表 '{table_name}' 中发现不匹配的向量维度。预期 {dim}，得到 {num_floats}。将填充或截断。")
                                            unpacked_data = list(struct.unpack(f'<{num_floats}f', blob))
                                            row[i] = (unpacked_data + [0.0] * dim)[:dim]

                        final_data = []
                        for row in processed_rows:
                            coerced_row = []
                            for i, col_name in enumerate(column_names):
                                target_type = target_column_types.get(col_name, 'String')
                                value = row[i]
                                coerced_value = coerce_value(value, target_type)
                                coerced_row.append(coerced_value)
                            final_data.append(tuple(coerced_row))

                        if not final_data:
                            continue
                            
                        insert_statement = f"INSERT INTO `{table_name}` ({', '.join([f'`{c}`' for c in column_names])}) VALUES"
                        client.execute(insert_statement, final_data, types_check=True)
                        
                        pbar.update(len(rows_batch))
                
                print(f"  ✔ 表 '{table_name}' 数据迁移完成。")
        print("\n🎉 所有表迁移成功！")

    except Exception as e:
        cleanup_clickhouse_db(args, db_name)
        raise e
    finally:
        if client:
            client.disconnect()

def migrate_to_both_backends(database_files, pg_args, ch_args):
    """
    Orchestrates the migration of SQLite databases to BOTH PostgreSQL and ClickHouse.

    This function scans a source path for SQLite files, then for each file, it
    attempts to migrate it first to PostgreSQL and then to ClickHouse using
    their respective migration functions. Only if a database is successfully
    migrated to *both* backends will its name be added to the returned list.

    Args:
        database_files (List): The list of SQLite files.
        pg_args (argparse.Namespace): An object containing connection arguments
                                     for PostgreSQL (host, port, user, password).
        ch_args (argparse.Namespace): An object containing connection arguments
                                     for ClickHouse (host, port, user, password).

    Returns:
        list: A list of strings, where each string is the name of a database
              that was successfully migrated to both PostgreSQL and ClickHouse.
    """
    print("🚀 Starting migration to both PostgreSQL and ClickHouse backends.")
    
    if not database_files:
        print("✖ No database files found. Exiting.")
        return []

    successful_migrations = []
    pg_failures, ch_failures = [], []

    for i, db_file in enumerate(database_files, 1):
        print(f"\n{'='*70}")
        print(f"🔄 Processing file {i}/{len(database_files)}: {os.path.basename(db_file)}")
        print(f"{'='*70}")

        db_name = os.path.splitext(os.path.basename(db_file))[0]
        db_name = re.sub(r'[^a-zA-Z0-9_]', '_', db_name).lower()
        
        pg_success = False
        ch_success = False

        # --- Attempt PostgreSQL Migration ---
        try:
            print(f"\n  -> Attempting migration to PostgreSQL (DB: {db_name})...")
            migrate_to_postgres(pg_args, db_name, db_file)
            pg_success = True
            print(f"  ✔ Successfully migrated '{db_name}' to PostgreSQL.")
        except Exception as e:
            print(f"  ❌ FAILED to migrate '{db_name}' to PostgreSQL.")
            print(f"     Error: {e}")
            # The cleanup function is already called within migrate_to_postgres on failure

        # --- Attempt ClickHouse Migration ---
        try:
            print(f"\n  -> Attempting migration to ClickHouse (DB: {db_name})...")
            migrate_to_clickhouse(ch_args, db_name, db_file)
            ch_success = True
            print(f"  ✔ Successfully migrated '{db_name}' to ClickHouse.")
        except Exception as e:
            print(f"  ❌ FAILED to migrate '{db_name}' to ClickHouse.")
            print(f"     Error: {e}")
            # The cleanup function is already called within migrate_to_clickhouse on failure

        # --- Final Check and Reporting ---
        print("\n  -- Summary for this file --")
        if not pg_success:
            pg_failures.append(db_name)
        if not ch_success:
            ch_failures.append(db_name)
        if pg_success and ch_success:
            successful_migrations.append(db_name)
            print(f"  ✅ SUCCESS: '{db_name}' was migrated to BOTH backends.")
        else:
            print(f"  ⚠️ INCOMPLETE: Migration for '{db_name}' failed on at least one backend.")
            print(f"     PostgreSQL: {'Success' if pg_success else 'Failed'}")
            print(f"     ClickHouse: {'Success' if ch_success else 'Failed'}")

    return successful_migrations, pg_failures, ch_failures

def main():
    parser = argparse.ArgumentParser(
        description="一键将 SQLite 数据库（支持 sqlite-vec）迁移到 PostgreSQL、ClickHouse 或两者。支持单个文件或文件夹批量迁移。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # --- Source and Target Arguments ---
    parser.add_argument('--source', default='/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/sqlite/results/spider', help="源 SQLite 数据库文件路径或文件夹路径。")
    parser.add_argument('--target', default='both', choices=['postgresql', 'clickhouse', 'both'], help="目标数据库类型。选择 'both' 将迁移到两个后端。")

    # --- Generic/Single Target Arguments (for backward compatibility) ---
    parser.add_argument('--host', help="目标数据库主机地址 (用于单目标模式)。")
    parser.add_argument('--port', type=int, help="目标数据库端口 (用于单目标模式)。")
    parser.add_argument('--user', help="目标数据库用户名 (用于单目标模式)。")
    parser.add_argument('--password', help="目标数据库密码 (用于单目标模式)。")

    # --- PostgreSQL Specific Arguments (for 'both' mode) ---
    pg_group = parser.add_argument_group('PostgreSQL Options (for --target=both)')
    pg_group.add_argument('--pg-host', default='localhost', help="[PostgreSQL] 主机地址。")
    pg_group.add_argument('--pg-port', type=int, default=5432, help="[PostgreSQL] 端口。")
    pg_group.add_argument('--pg-user', default='postgres', help="[PostgreSQL] 用户名。")
    pg_group.add_argument('--pg-password', default='postgres', help="[PostgreSQL] 密码。")

    # --- ClickHouse Specific Arguments (for 'both' mode) ---
    ch_group = parser.add_argument_group('ClickHouse Options (for --target=both)')
    ch_group.add_argument('--ch-host', default='localhost', help="[ClickHouse] 主机地址。")
    ch_group.add_argument('--ch-port', type=int, default=9000, help="[ClickHouse] 端口。")
    ch_group.add_argument('--ch-user', default='default', help="[ClickHouse] 用户名。")
    ch_group.add_argument('--ch-password', default='', help="[ClickHouse] 密码。")
    
    args = parser.parse_args()

    if not os.path.exists(args.source):
        print(f"✖ 错误：源路径 '{args.source}' 不存在。")
        return

    database_files = find_database_files(args.source)
    if not database_files:
        print("✖ 未找到任何可迁移的数据库文件。")
        return

    success_count = 0
    error_count = 0

    # --- Execution Logic ---
    if args.target == 'both':
        # Create separate config namespaces for each database
        pg_args = argparse.Namespace(host=args.pg_host, port=args.pg_port, user=args.pg_user, password=args.pg_password)
        ch_args = argparse.Namespace(host=args.ch_host, port=args.ch_port, user=args.ch_user, password=args.ch_password)
        
        successful_dbs, pg_failures, ch_failures = migrate_to_both_backends(database_files, pg_args, ch_args)
        
        success_count = len(successful_dbs)
        error_count = len(database_files) - success_count
        
        print(f"\n{'='*70}")
        print("🎯 最终迁移报告 (模式: both)")
        print(f"{'='*70}")
        print(f"  总共处理的文件数: {len(database_files)}")
        print(f"  完全成功 (迁移到两个后端): {success_count}")
        print(f"  部分或完全失败: {error_count}")
        if successful_dbs:
            print("\n  成功迁移的数据库名列表:")
            for db in successful_dbs:
                print(f"    - {db}")
        if pg_failures:
            print("\n  PostgreSQL 迁移失败的数据库名列表:")
            for db in pg_failures:
                print(f"    - {db}")
        if ch_failures:
            print("\n  ClickHouse 迁移失败的数据库名列表:")
            for db in ch_failures:
                print(f"    - {db}")
        print(f"{'='*70}")

    else: # Logic for single target migration
        # Populate single-target args if not provided
        if args.host is None:
            args.host = 'localhost'
        if args.port is None:
            args.port = 5432 if args.target == 'postgresql' else 9000
        if args.user is None:
            args.user = 'postgres' if args.target == 'postgresql' else 'default'
        if args.password is None:
            args.password = 'postgres' if args.target == 'postgresql' else ''
        
        print(f"\n📊 迁移任务:")
        print(f"  源路径: {args.source}")
        print(f"  目标类型: {args.target}")
        print(f"  目标主机: {args.host}:{args.port}")
        print(f"  找到数据库文件: {len(database_files)} 个")

        for i, db_file in enumerate(database_files, 1):
            print(f"\n{'='*60}")
            print(f"🔄 正在处理数据库 {i}/{len(database_files)}: {os.path.basename(db_file)}")
            print(f"{'='*60}")
            
            try:
                db_name = os.path.splitext(os.path.basename(db_file))[0]
                db_name = re.sub(r'[^a-zA-Z0-9_]', '_', str(db_name)).lower()
                print(f"目标数据库名: {db_name}")
                
                if args.target == 'postgresql':
                    migrate_to_postgres(args, db_name, db_file)
                elif args.target == 'clickhouse':
                    migrate_to_clickhouse(args, db_name, db_file)
                
                success_count += 1
                print(f"✅ 数据库 {i}/{len(database_files)} 迁移成功: {os.path.basename(db_file)}")
                
            except Exception as e:
                error_count += 1
                print(f"❌ 数据库 {i}/{len(database_files)} 迁移失败: {os.path.basename(db_file)}")
                print(f"   错误详情: {e}")
                import traceback
                traceback.print_exc()
        
        # Display final statistics for single mode
        print(f"\n{'='*60}")
        print(f"🎯 迁移完成统计 (模式: {args.target}):")
        print(f"   总文件数: {len(database_files)}")
        print(f"   成功: {success_count}")
        print(f"   失败: {error_count}")
        print(f"{'='*60}")

    if error_count > 0:
        print(f"⚠️  有 {error_count} 个数据库迁移任务未能完全成功，请检查上述错误信息。")
    else:
        print("🎉 所有数据库迁移任务均已成功！")

# The rest of the script (all other functions) remains the same.
# You just need to replace the original main() with this new version
# and add the migrate_to_both_backends() function anywhere in the file.

if __name__ == '__main__':
    main()