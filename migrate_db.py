import argparse
import os
import sqlite3
import re
import struct
from contextlib import contextmanager

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
    is_virtual_vec_table = 'USING vec0' in create_sql.upper() and 'VIRTUAL' in create_sql.upper()

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
                dimension = int(vec_col_match.group(2))
                # 翻译为 pgvector 类型
                new_defs.append(f'{col_name} vector({dimension})')
            else:
                # 处理虚拟表定义中的普通列
                col_match = re.match(r'([`"\[]?\w+[`"\]]?)\s+([A-Z]+(?:\(\d+(?:,\d+)?\))?)(.*)', col_def, flags=re.IGNORECASE)
                if col_match:
                    col_name, col_type, constraints = col_match.groups()
                    col_name = col_name.replace('`', '"')
                    pg_type = translate_type_for_postgres(col_type)
                    new_defs.append(f'{col_name} {pg_type}{constraints}')
                else:
                    new_defs.append(col_def) # 回退

        # 构建标准的 CREATE TABLE 语句
        return f"CREATE TABLE {table_name} ({', '.join(new_defs)})"

    # --- 对标准表的现有处理逻辑 ---
    create_sql = re.sub(r'\bAUTOINCREMENT\b', '', create_sql, flags=re.IGNORECASE)
    create_sql = re.sub(r'INTEGER\s+PRIMARY\s+KEY', 'SERIAL PRIMARY KEY', create_sql, flags=re.IGNORECASE)

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
                print(f'  - 忽略外键约束: {definition}')
                continue

            if definition.upper().startswith(('PRIMARY', 'FOREIGN', 'UNIQUE', 'CONSTRAINT', 'CHECK')):
                new_defs.append(definition)
                continue

            col_match = re.match(r'([`"\[]?\w+[`"\]]?)\s+([A-Z]+(?:\(\d+(?:,\d+)?\))?)(.*)', definition, flags=re.IGNORECASE)
            
            if col_match:
                col_name, col_type, constraints = col_match.groups()
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
    if 'BLOB' in sqlite_type: return 'String' # 默认为 String，向量 BLOB 会被特殊处理
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
        if not col_def or col_def.upper().startswith(('PRIMARY', 'FOREIGN', 'UNIQUE', 'CONSTRAINT', 'CHECK')):
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


def migrate_to_postgres(args, db_name):
    """执行到 PostgreSQL 的迁移（支持 sqlite-vec）"""
    if not psycopg2:
        print("错误：psycopg2-binary 未安装。请运行 'pip install psycopg2-binary'")
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

                with sqlite_conn(args.source) as source_conn:
                    vec_tables_info = get_vec_info(source_conn)
                    if vec_tables_info:
                        print(f"✔ 检测到 {len(vec_tables_info)} 个 sqlite-vec 表: {', '.join(vec_tables_info.keys())}")

                    source_cur = source_conn.cursor()
                    tables_schema = get_sqlite_schema(source_conn)

                    for table_name, create_sql in tables_schema:
                        # 跳过 sqlite-vec 内部表和系统表
                        if table_name == 'sqlite_sequence' or table_name.endswith(('_info', '_chunks', '_rowids')) or any(suffix in table_name for suffix in ('_metadatatext', '_metadatachunks', '_vector_chunks')):
                            continue

                        table_name_quoted = f'"{table_name.strip("`[]")}"'
                        
                        print(f"\n-- 正在处理表: {table_name_quoted} --")

                        print("  - 翻译 Schema...")
                        pg_create_sql = translate_schema_for_postgres(create_sql)

                        print("  - 从 SQLite 中提取数据...")
                        source_cur.execute(f'SELECT * FROM `{table_name.strip("`[]")}`')
                        
                        original_column_names = [desc[0] for desc in source_cur.description]
                        rows = source_cur.fetchall()

                        column_names = list(original_column_names)
                        processed_rows = [list(row) for row in rows]
                        
                        # 确保最终语句中的表名是带引号的，以防万一
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
                            for column_name in column_names:
                                if f' "{column_name}" ' not in pg_create_sql:
                                    pg_create_sql = pg_create_sql.replace(f' {column_name} ', f' "{column_name}" ')
                        pg_create_sql = pg_create_sql + ';'

                        print(f"  - 正在目标数据库中创建表 {table_name_quoted}...")
                        target_cur.execute(f'DROP TABLE IF EXISTS {table_name_quoted} CASCADE;')
                        target_cur.execute(pg_create_sql)

                        if not rows:
                            print(f"  - 表 {table_name_quoted} 为空，跳过数据插入。")
                            continue
                        
                        # 移除虚拟表可能包含的隐式 'rowid' 列
                        try:
                            rowid_index = column_names.index('rowid')
                            print("  - 检测到并移除隐式的 'rowid' 列。")
                            del column_names[rowid_index]
                            for row in processed_rows:
                                del row[rowid_index]
                        except ValueError:
                            pass # 'rowid' 列不存在

                        # 如果是向量表，解码向量数据
                        vec_info_for_table = vec_tables_info.get(table_name)
                        if vec_info_for_table:
                            print("  - 正在解码 sqlite-vec 向量数据...")
                            vector_column_indices = {
                                i: item['dim'] 
                                for i, name in enumerate(column_names) 
                                for item in vec_info_for_table['columns'] if item['name'] == name
                            }
                            for row in processed_rows:
                                for i, dim in vector_column_indices.items():
                                    blob = row[i]
                                    if isinstance(blob, bytes):
                                        num_floats = len(blob) // 4
                                        unpacked_data = list(struct.unpack(f'<{num_floats}f', blob))
                                        if num_floats != dim:
                                            print(f"  🟡 警告: 在表 '{table_name}' 中发现不匹配的向量维度。预期 {dim}，得到 {num_floats}。将填充或截断。")
                                            unpacked_data = (unpacked_data + [0.0] * dim)[:dim]
                                        # 转换为 pgvector 接受的字符串格式 '[1.2,3.4,...]'
                                        row[i] = str(unpacked_data)

                        # 构建带有显式列名的 INSERT 语句
                        # insert_query = f'INSERT INTO {table_name_quoted} ({", ".join([f'"{c}"' for c in column_names])}) VALUES ({", ".join(["%s"] * len(column_names))})'
                        # rewrite without f-string
                        insert_query = 'INSERT INTO {} ({}) VALUES ({})'.format(
                            table_name_quoted,
                            ', '.join([f'"{c}"' for c in column_names]),
                            ', '.join(['%s'] * len(column_names))
                        )
                        insert_query = insert_query + ';'
                        
                        print(f"  - 批量插入 {len(processed_rows)} 条数据到 PostgreSQL...")
                        execute_batch(target_cur, insert_query, processed_rows, page_size=1000)
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

def migrate_to_clickhouse(args, db_name):
    """执行到 ClickHouse 的迁移（支持 sqlite-vec、移除 rowid 并强制类型转换）"""
    if not Client:
        print("错误：clickhouse-driver 未安装。请运行 'pip install clickhouse-driver'")
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

        with sqlite_conn(args.source) as source_conn:
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

                print("  - 从 SQLite 中提取数据...")
                source_cur.execute(f'SELECT * FROM `{table_name.strip("`[]")}`')
                
                original_column_names = [desc[0] for desc in source_cur.description]
                rows = source_cur.fetchall()
                
                column_names = list(original_column_names)
                
                try:
                    rowid_index = column_names.index('rowid')
                    print("  - 检测到并移除隐式的 'rowid' 列。")
                    del column_names[rowid_index]
                    processed_rows = [list(row) for row in rows]
                    for row in processed_rows:
                        del row[rowid_index]
                except ValueError:
                    processed_rows = [list(row) for row in rows]

                vec_info_for_table = vec_tables_info.get(table_name)
                if vec_info_for_table:
                    print("  - 正在解码 sqlite-vec 向量数据...")
                    vector_column_indices = {
                        i: item['dim'] 
                        for i, name in enumerate(column_names) 
                        for item in vec_info_for_table['columns'] if item['name'] == name
                    }
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
                
                print("  - 正在根据目标 Schema 强制转换数据类型...")
                final_data = []
                for row in processed_rows:
                    coerced_row = list(row)
                    for i, col_name in enumerate(column_names):
                        target_type = target_column_types.get(col_name)
                        value = coerced_row[i]
                        
                        if target_type and 'String' in target_type and not isinstance(value, str) and value is not None:
                            coerced_row[i] = str(value)
                            
                    final_data.append(tuple(coerced_row))

                if not final_data:
                    print(f"  - 表 '{table_name}' 为空，跳过数据插入。")
                    continue
                    
                print(f"  - 批量插入 {len(final_data)} 条数据到 ClickHouse...")
                insert_statement = f"INSERT INTO `{table_name}` ({', '.join([f'`{c}`' for c in column_names])}) VALUES"
                client.execute(insert_statement, final_data)
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
    parser = argparse.ArgumentParser(description="一键将 SQLite 数据库（支持 sqlite-vec）迁移到 PostgreSQL 或 ClickHouse。")
    
    parser.add_argument('--source', default='/mnt/b_public/data/wangzr/Text2VectorSQL/synthesis/toy_spider/results/vector_databases_toy/musical/musical.sqlite', help="源 SQLite 数据库文件路径 (.db 或 .sqlite)。")
    parser.add_argument('--target', default='postgresql', choices=['postgresql', 'clickhouse'], help="目标数据库类型。")
    parser.add_argument('--host', default='localhost', help="目标数据库主机地址。")
    parser.add_argument('--user', help="postgres")
    parser.add_argument('--password', default='postgres', help="目标数据库密码。")
    parser.add_argument('--port', type=int, help="目标数据库端口 (PostgreSQL 默认为 5432, ClickHouse 默认为 9000)。")
    
    args = parser.parse_args()

    if args.port is None:
        args.port = 5432 if args.target == 'postgresql' else 9000
    
    if args.user is None:
        args.user = 'postgres' if args.target == 'postgresql' else 'default'


    if not os.path.exists(args.source):
        print(f"✖ 错误：源文件 '{args.source}' 不存在。")
        return

    db_name = os.path.splitext(os.path.basename(args.source))[0]
    db_name = re.sub(r'[^a-zA-Z0-9_]', '_', str(db_name))
    db_name = db_name.lower()

    print(f"源文件: {args.source}")
    print(f"目标类型: {args.target}")
    print(f"目标主机: {args.host}:{args.port}")
    print(f"自动创建/使用的数据库名: {db_name}")

    if args.target == 'postgresql':
        migrate_to_postgres(args, db_name)
    elif args.target == 'clickhouse':
        migrate_to_clickhouse(args, db_name)

if __name__ == '__main__':
    main()