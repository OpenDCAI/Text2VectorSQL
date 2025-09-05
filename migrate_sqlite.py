import argparse
import os
import sqlite3
import re
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
    """将 SQLite 的 CREATE TABLE 语句翻译为 PostgreSQL 兼容格式"""
    # 统一将 SQLite 的引用符号 ` 替换为 PostgreSQL 的 "
    create_sql = create_sql.replace('`', '"')
    
    # 移除 AUTOINCREMENT 关键字，PostgreSQL 使用 SERIAL/BIGSERIAL
    create_sql = re.sub(r'\bAUTOINCREMENT\b', '', create_sql, flags=re.IGNORECASE)
    # 将 INTEGER PRIMARY KEY 转换为 SERIAL PRIMARY KEY
    create_sql = re.sub(r'INTEGER\s+PRIMARY\s+KEY', 'SERIAL PRIMARY KEY', create_sql, flags=re.IGNORECASE)

    # 从 CREATE 语句中提取列定义部分
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
    if 'BLOB' in sqlite_type: return 'String'
    if 'REAL' in sqlite_type or 'FLOA' in sqlite_type or 'DOUB' in sqlite_type: return 'Float64'
    if 'NUMERIC' in sqlite_type or 'DECIMAL' in sqlite_type: return 'Decimal(38, 6)'
    if 'DATE' in sqlite_type: return 'Date'
    if 'DATETIME' in sqlite_type: return 'DateTime'
    return 'String'

def translate_schema_for_clickhouse(create_sql):
    """将 SQLite 的 CREATE TABLE 语句翻译为 ClickHouse 兼容格式"""
    table_name_match = re.search(
        r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?\s*([`"\[]\S+[`"\]]|\S+)\s*\(',
        create_sql, flags=re.IGNORECASE
    )
    if not table_name_match:
        raise ValueError(f"无法从 SQL 中解析表名: {create_sql}")
        
    table_name = table_name_match.group(1).strip('`"[]')

    lines = []
    columns_part_match = re.search(r'\((.*)\)', create_sql, re.DOTALL)
    if not columns_part_match:
        raise ValueError(f"无法从 SQL 中解析列定义: {create_sql}")
    columns_part = columns_part_match.group(1)
    
    columns_defs = re.split(r',(?![^\(]*\))', columns_part)

    for col_def in columns_defs:
        col_def = col_def.strip()
        if not col_def or col_def.upper().startswith(('PRIMARY', 'FOREIGN', 'UNIQUE', 'CONSTRAINT', 'CHECK')):
            continue

        parts = re.split(r'\s+', col_def, 2)
        col_name = parts[0].strip('`"')
        
        if len(parts) > 1:
            sqlite_type = parts[1].split('(')[0]
            ch_type = translate_type_for_clickhouse(sqlite_type)
            if 'NOT NULL' not in col_def.upper():
                ch_type = f'Nullable({ch_type})'
            lines.append(f'`{col_name}` {ch_type}')

    create_table_ch = f"CREATE TABLE `{table_name}` (\n    "
    create_table_ch += ',\n    '.join(lines)
    create_table_ch += f"\n) ENGINE = MergeTree() ORDER BY tuple();"
    
    return create_table_ch


# --- 数据库连接与操作 ---

@contextmanager
def sqlite_conn(db_path):
    conn = sqlite3.connect(db_path)
    try:
        yield conn
    finally:
        conn.close()

def get_sqlite_schema(conn):
    """获取 SQLite 数据库中所有表的名称和 CREATE 语句"""
    cursor = conn.cursor()
    cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    schema = cursor.fetchall()
    if not schema:
        raise RuntimeError("源 SQLite 数据库中没有找到任何表。")
    return schema

def migrate_to_postgres(args, db_name):
    """执行到 PostgreSQL 的迁移"""
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
            cur.execute(f"SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
            if cur.fetchone():
                print(f"数据库 '{db_name}' 已存在。")
            else:
                print(f"正在创建数据库: {db_name}")
                cur.execute(f"CREATE DATABASE {db_name}")
                print(f"✔ 数据库 '{db_name}' 创建成功。")

        # 2. 连接到新创建的数据库进行迁移
        print(f"正在连接到新数据库 '{db_name}'...")
        with psycopg2.connect(host=args.host, port=args.port, user=args.user, password=args.password, dbname=db_name) as target_conn:
            with target_conn.cursor() as target_cur:
                with sqlite_conn(args.source) as source_conn:
                    source_cur = source_conn.cursor()
                    tables_schema = get_sqlite_schema(source_conn)

                    print(f"共找到 {len(tables_schema)} 个表需要迁移。")

                    for table_name, create_sql in tables_schema:
                        table_name_quoted = f'"{table_name.strip("`[]")}"'
                        
                        print(f"\n-- 正在处理表: {table_name_quoted} --")

                        print("  - 翻译 Schema...")
                        pg_create_sql = translate_schema_for_postgres(create_sql)
                        # 确保最终语句中的表名是带引号的
                        pg_create_sql = re.sub(
                            r'CREATE\s+TABLE\s+([`"\[]?\S+[`"\]]?)', 
                            f'CREATE TABLE {table_name_quoted}', 
                            pg_create_sql, 
                            count=1, 
                            flags=re.IGNORECASE
                        )

                        print(f"  - 正在目标数据库中创建表 {table_name_quoted}...")
                        target_cur.execute(f'DROP TABLE IF EXISTS {table_name_quoted} CASCADE;')
                        target_cur.execute(pg_create_sql)

                        print("  - 从 SQLite 中提取数据...")
                        source_cur.execute(f'SELECT * FROM `{table_name.strip("`[]")}`')
                        data = source_cur.fetchall()

                        if not data:
                            print(f"  - 表 {table_name_quoted} 为空，跳过数据插入。")
                            continue

                        cols_count = len(source_cur.description)
                        insert_query = f'INSERT INTO {table_name_quoted} VALUES ({", ".join(["%s"] * cols_count)})'
                        
                        print(f"  - 批量插入 {len(data)} 条数据到 PostgreSQL...")
                        execute_batch(target_cur, insert_query, data, page_size=1000)
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
        # 'with' 语句会自动关闭 target_conn，但为保险起见可以添加
        if target_conn and not target_conn.closed:
            target_conn.close()

def migrate_to_clickhouse(args, db_name):
    """执行到 ClickHouse 的迁移"""
    if not Client:
        print("错误：clickhouse-driver 未安装。请运行 'pip install clickhouse-driver'")
        return
        
    print(f"\n🚀 开始迁移到 ClickHouse...")
    
    client = None
    try:
        # 1. 连接到默认数据库，仅用于创建新库
        print("正在连接到服务器以创建数据库...")
        with Client(host=args.host, port=args.port, user=args.user, password=args.password) as admin_client:
            print(f"正在创建数据库 (如果不存在): {db_name}")
            admin_client.execute(f'CREATE DATABASE IF NOT EXISTS `{db_name}`')
            print(f"✔ 数据库 '{db_name}' 已就绪。")

        # 2. 连接到新创建的数据库，执行所有表操作
        print(f"正在连接到新数据库 '{db_name}'...")
        client = Client(host=args.host, port=args.port, user=args.user, password=args.password, database=db_name)

        with sqlite_conn(args.source) as source_conn:
            source_cur = source_conn.cursor()
            tables_schema = get_sqlite_schema(source_conn)

            print(f"共找到 {len(tables_schema)} 个表需要迁移。")

            for table_name, create_sql in tables_schema:
                print(f"\n-- 正在处理表: {table_name} --")

                print("  - 翻译 Schema...")
                ch_create_sql = translate_schema_for_clickhouse(create_sql)
                
                print(f"  - 正在目标数据库中创建表 '{table_name}'...")
                client.execute(f"DROP TABLE IF EXISTS `{table_name}`")
                client.execute(ch_create_sql)
                
                print("  - 从 SQLite 中提取数据...")
                source_cur.execute(f"SELECT * FROM `{table_name}`")
                data = source_cur.fetchall()

                if not data:
                    print(f"  - 表 '{table_name}' 为空，跳过数据插入。")
                    continue
                    
                print(f"  - 批量插入 {len(data)} 条数据到 ClickHouse...")
                # ClickHouse driver 的 execute 方法本身支持批量插入
                column_names = [desc[0] for desc in source_cur.description]
                insert_statement = f"INSERT INTO `{table_name}` ({', '.join([f'`{c}`' for c in column_names])}) VALUES"
                client.execute(insert_statement, data)
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
    parser = argparse.ArgumentParser(description="一键将 SQLite 数据库迁移到 PostgreSQL 或 ClickHouse。")
    
    parser.add_argument('--source', required=True, help="源 SQLite 数据库文件路径 (.db 或 .sqlite)。")
    parser.add_argument('--target', required=True, choices=['postgresql', 'clickhouse'], help="目标数据库类型。")
    parser.add_argument('--host', default='localhost', help="目标数据库主机地址。")
    parser.add_argument('--user', default='default', help="目标数据库用户名。")
    parser.add_argument('--password', default='', help="目标数据库密码。")
    parser.add_argument('--port', type=int, help="目标数据库端口 (PostgreSQL 默认为 5432, ClickHouse 默认为 9000)。")
    
    args = parser.parse_args()

    if args.port is None:
        args.port = 5432 if args.target == 'postgresql' else 9000

    if not os.path.exists(args.source):
        print(f"✖ 错误：源文件 '{args.source}' 不存在。")
        return

    db_name = os.path.splitext(os.path.basename(args.source))[0]
    db_name = re.sub(r'[^a-zA-Z0-9_]', '_', db_name)
    db_name = db_name.lower() # PostgreSQL 数据库名通常为小写

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

# python migrate_sqlite.py   --source /mnt/b_public/data/wangzr/Text2VectorSQL/synthesis/toy_spider/train/toy_spider/station_weather/station_weather.sqlite   --target clickhouse   --host localhost   --port 9000   --user default   --password ''
# python migrate_sqlite.py   --source /mnt/b_public/data/wangzr/Text2VectorSQL/synthesis/toy_spider/train/toy_spider/browser_web/browser_web.sqlite   --target postgresql   --host localhost   --port 5432   --user postgres   --password 'postgres'