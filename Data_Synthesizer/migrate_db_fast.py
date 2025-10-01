import argparse
import os
import re
import struct
import glob
import csv
import sys
import io
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

def coerce_value(value, target_type_str):
    """
    Coerces a Python value to a more specific type based on the target DB's column type string.
    This version tries multiple common date/time formats for increased flexibility.
    """
    if value is None or value == '':
        return None

    target_type_str = target_type_str.lower()
    
    # --- Date/DateTime Conversion ---
    is_date_type = 'date' in target_type_str and 'datetime' not in target_type_str
    is_datetime_type = 'datetime' in target_type_str or 'timestamp' in target_type_str
    if isinstance(value, str) and (is_date_type or is_datetime_type):
        # Define formats to try, from most specific to least specific
        datetime_formats = [
            '%Y-%m-%d %H:%M:%S.%f', # With fractional seconds
            '%Y-%m-%d %H:%M:%S',   # With seconds
            '%Y-%m-%d %H:%M',      # << NEW: Without seconds
            '%Y-%m-%d'
        ]
        
        for fmt in datetime_formats:
            try:
                return datetime.strptime(value, fmt)
            except (ValueError, TypeError):
                continue # If this format fails, try the next one

        # If we are here, all parsing attempts have failed.
        print(f"  🟡 警告: 无法将字符串 '{value}' 解析为任何已知的日期/时间格式。将插入 NULL。")
        return None

    # --- Boolean Conversion ---
    if 'bool' in target_type_str and isinstance(value, int):
        return value != 0

    # --- ClickHouse Vector (String to List) Conversion ---
    if 'array(float32)' in target_type_str and isinstance(value, str):
        try:
            if value.startswith('[') and value.endswith(']'):
                return [] if value == '[]' else [float(x) for x in value.strip('[]').split(',')]
            else:
                return []
        except (ValueError, TypeError):
            return []

    # --- Fallback for general strings ---
    if 'string' in target_type_str or 'text' in target_type_str:
        if not isinstance(value, str):
            return str(value)

    return value

# --- Schema 翻译模块 (此部分保持不变) ---

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
                    col_name = '"' + col_name + '"'
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
                        col_name = '"' + col_name + '"'
                    pg_type = translate_type_for_postgres(col_type)
                    new_defs.append(f'{col_name} {pg_type}{constraints}')
                else:
                    new_defs.append(col_def) # 回退

        # 构建标准的 CREATE TABLE 语句
        return f"CREATE TABLE {table_name} ({', '.join(new_defs)})"

    # --- 对标准表的现有处理逻辑 ---
    create_sql = re.sub(r'\bAUTOINCREMENT\b', '', create_sql, flags=re.IGNORECASE)
    create_sql = re.sub(r'INTEGER\s+PRIMARY\s+KEY', '', create_sql, flags=re.IGNORECASE)

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
            
            if 'FOREIGN KEY' in definition.upper():
                # print(f'  - 忽略外键约束: {definition}')
                continue

            if definition.upper().startswith(('PRIMARY KEY', 'FOREIGN KEY', 'UNIQUE', 'CONSTRAINT', 'CHECK')):
                # new_defs.append(definition)
                continue

            col_match = re.match(r'([`"\[]?\w+[`"\]]?)\s+([A-Z]+(?:\(\d+(?:,\d+)?\))?)(.*)', definition, flags=re.IGNORECASE)
            
            if col_match:
                col_name, col_type, constraints = col_match.groups()
                if '"' not in col_name:
                    col_name = '"' + col_name + '"'
                if 'SERIAL PRIMARY KEY' in definition.upper():
                    new_defs.append(definition)
                else:
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
    if 'BLOB' in sqlite_type: return 'String'
    if 'REAL' in sqlite_type or 'FLOA' in sqlite_type or 'DOUB' in sqlite_type: return 'Float64'
    if 'NUMERIC' in sqlite_type or 'DECIMAL' in sqlite_type: return 'Decimal(38, 6)'
    if 'DATE' in sqlite_type: return 'Date'
    if 'DATETIME' in sqlite_type: return 'DateTime'
    return 'String'

def translate_schema_for_clickhouse(create_sql):
    """将 SQLite 的 CREATE TABLE 语句（包括 vec0 虚拟表）翻译为 ClickHouse 兼容格式"""
    table_name_match = re.search(
        r'CREATE\s+(?:VIRTUAL\s+)?TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?\s*([`"\[]\S+[`"\]]|\S+)',
        create_sql, flags=re.IGNORECASE
    )
    if not table_name_match:
        raise ValueError(f"无法从 SQL 中解析表名: {create_sql}")
        
    table_name = table_name_match.group(1).strip('`"[]')

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


# --- 新增: SQL 文件解析模块 ---

def parse_sql_value(value_str):
    csv.field_size_limit(sys.maxsize)
    """将 SQL 字面量字符串转换为 Python 类型"""
    value_str = value_str.strip()
    if value_str.upper() == 'NULL':
        return None
    if value_str.startswith("'") and value_str.endswith("'"):
        # 去除引号并处理 SQL 中的转义单引号 ('')
        return value_str[1:-1].replace("''", "'")
    try:
        return int(value_str)
    except ValueError:
        try:
            return float(value_str)
        except ValueError:
            # 对于不能转换为数字的，作为字符串返回 (例如 BLOB 字面量 x'...')
            return value_str

def parse_sql_file(file_path):
    """解析 .sql 文件，提取 CREATE TABLE 语句和 INSERT 数据"""
    print(f"  - 正在解析 SQL 文件: {os.path.basename(file_path)}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    commands = content.split('--<SQL_COMMAND_SEPARATOR>--')
    db_structure = {}

    for cmd in commands:
        cmd = cmd.strip()
        if not cmd or cmd.upper().startswith('BEGIN') or cmd.upper().startswith('COMMIT'):
            continue

        # 匹配 CREATE TABLE 语句
        create_match = re.search(
            r'CREATE\s+(?:VIRTUAL\s+)?TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?\s*([`"\[]\S+[`"\]]|\S+)',
            cmd, re.IGNORECASE
        )
        if create_match:
            table_name = create_match.group(1).strip('`"[]')
            if table_name not in db_structure:
                db_structure[table_name] = {'create_sql': cmd, 'columns': None, 'data': []}
            else:
                db_structure[table_name]['create_sql'] = cmd
            continue

        # 匹配 INSERT INTO 语句
        insert_match = re.search(
            r'INSERT\s+INTO\s+([`"\[]\S+[`"\]]|\S+)\s*(?:\((.*?)\))?\s+VALUES\s+(.*)',
            cmd, re.IGNORECASE | re.DOTALL
        )
        if insert_match:
            table_name = insert_match.group(1).strip('`"[]')
            columns_str = insert_match.group(2)
            values_multiline = insert_match.group(3).strip().rstrip(';')

            if table_name not in db_structure:
                 raise RuntimeError(f"在 CREATE TABLE 语句之前找到了表 '{table_name}' 的 INSERT 语句")

            if columns_str and not db_structure[table_name]['columns']:
                columns = [c.strip().strip('`"') for c in columns_str.split(',')]
                db_structure[table_name]['columns'] = columns

            # 解析 VALUES (...) , (...) , (...)
            # 这是一个简化的解析器，适用于常见格式
            value_tuples_str = re.split(r'\)\s*,\s*\(', values_multiline)
            for i, val_tuple in enumerate(value_tuples_str):
                row_str = val_tuple.strip()
                if i == 0: row_str = row_str.lstrip('(')
                if i == len(value_tuples_str) - 1: row_str = row_str.rstrip(')')
                
                # 使用 csv 模块来安全地分割值，处理带逗号的字符串
                f = io.StringIO(row_str)
                reader = csv.reader(f, quotechar="'", escapechar='\\', skipinitialspace=True)
                try:
                    # 将解析出的字符串值转换为正确的 Python 类型
                    parsed_values = [parse_sql_value(v) for v in next(reader)]
                    db_structure[table_name]['data'].append(parsed_values)
                except StopIteration:
                    print(f"  🟡 警告: 在表 '{table_name}' 中发现一个空的或格式错误的行。")

    print(f"  ✔ SQL 文件解析完成，找到 {len(db_structure)} 个表。")
    return db_structure


# --- 数据库文件搜索 (已修改为搜索 .sql) ---

def find_database_files(source_path):
    """递归搜索文件夹中的 SQL 文件 (.sql)"""
    database_files = []
    
    if os.path.isfile(source_path):
        if source_path.lower().endswith('.sql'):
            database_files.append(source_path)
        else:
            print(f"✖ 错误：文件 '{source_path}' 不是支持的 SQL 文件格式 (.sql)")
    elif os.path.isdir(source_path):
        print(f"🔍 正在搜索文件夹 '{source_path}' 中的 SQL 文件...")
        files = glob.glob(os.path.join(source_path, '**/*.sql'), recursive=True)
        database_files.extend(files)
        
        database_files = sorted(list(set(database_files)))
        
        if not database_files:
            print(f"✖ 在文件夹 '{source_path}' 中未找到任何 SQL 文件")
        else:
            print(f"✔ 找到 {len(database_files)} 个 SQL 文件:")
            for i, db_file in enumerate(database_files, 1):
                print(f"  {i}. {db_file}")
    else:
        print(f"✖ 错误：路径 '{source_path}' 不存在或无法访问")
    
    return database_files


# --- 数据库操作 (已修改为基于 SQL 文件) ---
def migrate_to_postgres(args, db_name, source_sql_path):
    """执行到 PostgreSQL 的迁移 (基于 .sql 文件)"""
    if not psycopg2:
        print("错误：psycopg2-binary 未安装。请运行 'pip install psycopg2-binary'")
        return

    print(f"\n🚀 开始迁移到 PostgreSQL...")
    conn_admin = None
    target_conn = None
    try:
        db_structure = parse_sql_file(source_sql_path)
        if not db_structure:
            print("✖ SQL 文件为空或无法解析，迁移中止。")
            return

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

        print(f"正在连接到新数据库 '{db_name}'...")
        with psycopg2.connect(host=args.host, port=args.port, user=args.user, password=args.password, dbname=db_name) as target_conn:
            with target_conn.cursor() as target_cur:
                print("  - 正在启用 pgvector 扩展 (如果尚未启用)...")
                target_cur.execute('CREATE EXTENSION IF NOT EXISTS vector;')
                print("  ✔ pgvector 扩展已就绪。")

                for table_name, table_info in db_structure.items():
                    create_sql = table_info['create_sql']
                    column_names = table_info['columns']
                    rows = table_info['data']
                    
                    table_name_quoted = f'"{table_name}"'
                    
                    print(f"\n-- 正在处理表: {table_name_quoted} --")

                    print("  - 翻译 Schema...")
                    pg_create_sql = translate_schema_for_postgres(create_sql)
                    
                    pg_create_sql = re.sub(
                        r'CREATE\s+(?:VIRTUAL\s+)?TABLE\s+([`"\[]?\S+[`"\]]?)', 
                        f'CREATE TABLE {table_name_quoted}', 
                        pg_create_sql, 
                        count=1, 
                        flags=re.IGNORECASE
                    )
                    
                    print(f"  - 正在目标数据库中创建表 {table_name_quoted}...")
                    target_cur.execute(f'DROP TABLE IF EXISTS {table_name_quoted} CASCADE;')
                    target_cur.execute(pg_create_sql)
                    
                    if not rows:
                        print(f"  - 表 {table_name_quoted} 为空，跳过数据插入。")
                        continue
                    
                    if not column_names:
                         raise ValueError(f"表 '{table_name}' 缺少列定义，无法插入数据。")
                    
                    print("  - 正在根据目标 Schema 强制转换数据类型...")
                    target_cur.execute(
                        "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = %s", (table_name,)
                    )
                    target_column_types = dict(target_cur.fetchall())

                    final_data = []
                    for row in rows:
                        coerced_row = [
                            coerce_value(value, target_column_types.get(col_name, ''))
                            for col_name, value in zip(column_names, row)
                        ]
                        final_data.append(tuple(coerced_row))
                    
                    insert_query = 'INSERT INTO {} ({}) VALUES ({});'.format(
                        table_name_quoted,
                        ', '.join([f'"{c}"' for c in column_names]),
                        ', '.join(['%s'] * len(column_names))
                    )
                    
                    print(f"  - 批量插入 {len(final_data)} 条数据到 PostgreSQL...")
                    execute_batch(target_cur, insert_query, final_data, page_size=1000)
                    print(f"  ✔ 表 {table_name_quoted} 数据迁移完成。")
            
            target_conn.commit()
            print("\n🎉 所有表迁移成功！")

    except (Exception, psycopg2.Error) as e:
        if target_conn:
            target_conn.rollback()
        print(f"\n✖ 迁移过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if conn_admin:
            conn_admin.close()
        if target_conn and not target_conn.closed:
            target_conn.close()

def migrate_to_clickhouse(args, db_name, source_sql_path):
    """执行到 ClickHouse 的迁移 (基于 .sql 文件)"""
    if not Client:
        print("错误：clickhouse-driver 未安装。请运行 'pip install clickhouse-driver'")
        return
        
    print(f"\n🚀 开始迁移到 ClickHouse...")
    
    client = None
    try:
        db_structure = parse_sql_file(source_sql_path)
        if not db_structure:
            print("✖ SQL 文件为空或无法解析，迁移中止。")
            return

        with Client(host=args.host, port=args.port, user=args.user, password=args.password) as admin_client:
            print(f"正在创建数据库 (如果不存在): {db_name}")
            admin_client.execute(f'CREATE DATABASE IF NOT EXISTS `{db_name}`')
            print(f"✔ 数据库 '{db_name}' 已就绪。")

        client = Client(host=args.host, port=args.port, user=args.user, password=args.password, database=db_name)

        for table_name, table_info in db_structure.items():
            create_sql = table_info['create_sql']
            column_names = table_info['columns']
            rows = table_info['data']

            print(f"\n-- 正在处理表: {table_name} --")

            print("  - 翻译 Schema...")
            ch_create_sql = translate_schema_for_clickhouse(create_sql)
            
            print(f"  - 正在目标数据库中创建表 '{table_name}'...")
            client.execute(f"DROP TABLE IF EXISTS `{table_name}`")
            client.execute(ch_create_sql)
            
            if not rows:
                print(f"  - 表 '{table_name}' 为空，跳过数据插入。")
                continue

            if not column_names:
                 raise ValueError(f"表 '{table_name}' 缺少列定义，无法插入数据。")

            print("  - 正在根据目标 Schema 强制转换数据类型...")
            target_column_types = {
                name: type_str for name, type_str in client.execute(
                    f"SELECT name, type FROM system.columns WHERE database = '{db_name}' AND table = '{table_name}'"
                )
            }
            
            final_data = []
            for row in rows:
                coerced_row = [
                    coerce_value(value, target_column_types.get(col_name, ''))
                    for col_name, value in zip(column_names, row)
                ]
                final_data.append(tuple(coerced_row))
                
            print(f"  - 批量插入 {len(final_data)} 条数据到 ClickHouse...")
            insert_statement = f"INSERT INTO `{table_name}` ({', '.join([f'`{c}`' for c in column_names])}) VALUES"
            client.execute(insert_statement, final_data, types_check=True)
            print(f"  ✔ 表 '{table_name}' 数据迁移完成。")
            
        print("\n🎉 所有表迁移成功！")

    except Exception as e:
        print(f"\n✖ 迁移过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if client:
            client.disconnect()

def main():
    parser = argparse.ArgumentParser(description="一键将 .sql 数据库导出文件迁移到 PostgreSQL 或 ClickHouse。支持单个文件或文件夹批量迁移。")
    
    # --- 修改: 更新默认源路径和帮助文本 ---
    parser.add_argument('--source', default='/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/sqlite/results/wikipedia_multimodal/vector_sql/', help="源 .sql 文件路径或包含 .sql 文件的文件夹路径。")

    # parser.add_argument('--target', default='postgresql', choices=['postgresql', 'clickhouse'], help="目标数据库类型。")
    # parser.add_argument('--host', default='localhost', help="目标数据库主机地址。")
    # parser.add_argument('--user', default='postgres', help="postgres")
    # parser.add_argument('--password', default='postgres', help="目标数据库密码。")
    # parser.add_argument('--port', type=int, help="目标数据库端口 (PostgreSQL 默认为 5432, ClickHouse 默认为 9000)。")

    parser.add_argument('--target', default='clickhouse', choices=['postgresql', 'clickhouse'], help="目标数据库类型。")
    parser.add_argument('--host', default='localhost', help="目标数据库主机地址。")
    parser.add_argument('--user', default='default', help="postgres")
    parser.add_argument('--password', default='', help="目标数据库密码。")
    parser.add_argument('--port', type=int, help="目标数据库端口 (PostgreSQL 默认为 5432, ClickHouse 默认为 9000)。")

    args = parser.parse_args()

    # --- 逻辑保持不变 ---
    if args.port is None:
        args.port = 5432 if args.target == 'postgresql' else 9000
    
    if args.user is None:
        args.user = 'postgres' if args.target == 'postgresql' else 'default'

    if not os.path.exists(args.source):
        print(f"✖ 错误：源路径 '{args.source}' 不存在。")
        return

    database_files = find_database_files(args.source)
    
    if not database_files:
        print("✖ 未找到任何可迁移的 .sql 文件。")
        return

    print(f"\n📊 迁移统计:")
    print(f"  源路径: {args.source}")
    print(f"  目标类型: {args.target}")
    print(f"  目标主机: {args.host}:{args.port}")
    print(f"  找到 SQL 文件: {len(database_files)} 个")

    success_count = 0
    error_count = 0
    
    for i, sql_file in enumerate(database_files, 1):
        print(f"\n{'='*60}")
        print(f"🔄 正在处理文件 {i}/{len(database_files)}: {os.path.basename(sql_file)}")
        print(f"{'='*60}")
        
        try:
            # 为每个文件生成唯一的数据库名称
            db_name = os.path.splitext(os.path.basename(sql_file))[0]
            db_name = re.sub(r'[^a-zA-Z0-9_]', '_', str(db_name))
            db_name = db_name.lower()
            
            print(f"目标数据库名: {db_name}")
            
            if args.target == 'postgresql':
                migrate_to_postgres(args, db_name, sql_file)
            elif args.target == 'clickhouse':
                migrate_to_clickhouse(args, db_name, sql_file)
            
            success_count += 1
            print(f"✅ 文件 {i}/{len(database_files)} 迁移成功: {os.path.basename(sql_file)}")
            
        except Exception as e:
            error_count += 1
            print(f"❌ 文件 {i}/{len(database_files)} 迁移失败: {os.path.basename(sql_file)}")
            print(f"   错误详情: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"🎯 迁移完成统计:")
    print(f"   总文件数: {len(database_files)}")
    print(f"   成功: {success_count}")
    print(f"   失败: {error_count}")
    print(f"{'='*60}")
    
    if error_count > 0:
        print(f"⚠️  有 {error_count} 个文件迁移失败，请检查上述错误信息。")
    else:
        print("🎉 所有文件迁移成功！")

if __name__ == '__main__':
    main()