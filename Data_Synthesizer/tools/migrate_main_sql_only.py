import json
import logging
import argparse
import os
import re
import sys
from collections import defaultdict
import psycopg2
# [--- 修改 ---] 导入两个不同的 ClickHouse 客户端
from clickhouse_driver import Client as Client_Native  # 用于 ClickHouse (Port 9000)
import clickhouse_connect  # 用于 MyScale (Port 8123)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# [--- 修改 ---] 尝试导入 MyScale 转换器
try:
    from migrate_sql import SQLiteToPostgreSQLConverter, SQLiteToClickHouseConverter, SQLiteToMyScaleConverter
    from Execution_Engine.execution_engine import ExecutionEngine, TimeoutError
except ImportError:
    print("Could not import necessary modules. Please ensure the project structure is correct.")
    print("Assuming dummy classes for demonstration purposes.")
    # Dummy classes to allow the script to run without the actual modules
    class SQLiteToPostgreSQLConverter:
        def __init__(self, sql): self.sql = sql
        def convert(self): return f"-- PG Converted: {self.sql}", "level_1"
    class SQLiteToClickHouseConverter:
        def __init__(self, sql): self.sql = sql
        def convert(self): return f"-- CH Converted: {self.sql}", "level_2"
    # [--- 新增 ---] 为 MyScale 添加一个虚拟类
    class SQLiteToMyScaleConverter:
        def __init__(self, sql): self.sql = sql
        def convert(self): return f"-- MyScale Converted: {self.sql}", "level_2"
    class ExecutionEngine:
        def __init__(self, config_path): pass
        def execute(self, sql, db_type, db_identifier): return {'status': 'success'}


# --- Configuration Area ---
# Input/Output File Configuration
INPUT_SQL_FILE_PATH = '/mnt/DataFlow/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/sqlite/results/synthesis_data/candidate_sql.json'

# Generate output paths dynamically from the input path
base_path, _ = os.path.splitext(INPUT_SQL_FILE_PATH)
OUTPUT_LITE_SUCCESS_PATH = f"{base_path}_lite.json"
OUTPUT_PG_SUCCESS_PATH = f"{base_path}_pg.json"
OUTPUT_CH_SUCCESS_PATH = f"{base_path}_ch.json"
# [--- 新增 ---] MyScale 输出路径
OUTPUT_MS_SUCCESS_PATH = f"{base_path}_ms.json"

# Execution Engine Configuration File
ENGINE_CONFIG_PATH = os.path.join(project_root, 'Execution_Engine', 'engine_config.yaml')
# --- Configuration Area End ---

def get_database_schema(db_type, conn_args, db_name):
    """
    Connects to a database and retrieves its schema.
    
    [--- 修改 ---] 此函数现在支持 'myscale'
    """
    schema_str = ""
    tables = defaultdict(list)

    try:
        if db_type == 'postgresql':
            conn = psycopg2.connect(host=conn_args.host, port=conn_args.port, user=conn_args.user, password=conn_args.password, dbname=db_name)
            query = """
                SELECT table_name, column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = 'public'
                ORDER BY table_name, ordinal_position;
            """
            with conn.cursor() as cur:
                cur.execute(query)
                for row in cur.fetchall():
                    table, column, dtype = row
                    if dtype == 'USER-DEFINED':
                        dtype = 'vector'
                    tables[table].append(f"{column} {dtype}")
            conn.close()

        elif db_type == 'clickhouse':
            # [--- 修改 ---] 使用 Native 客户端 (Port 9000)
            client = Client_Native(host=conn_args.host, port=conn_args.port, user=conn_args.user, password=conn_args.password, database=db_name)
            query = f"SELECT table, name, type FROM system.columns WHERE database = '{db_name}' ORDER BY table, position;"
            result = client.execute(query)
            for row in result:
                table, column, dtype = row
                tables[table].append(f"`{column}` {dtype}")

        # [--- 新增 ---] MyScale 的 schema 获取逻辑
        elif db_type == 'myscale':
            # MyScale 使用 HTTP 客户端 (Port 8123)
            client = clickhouse_connect.get_client(
                host=conn_args.host,
                port=conn_args.port,
                user=conn_args.user,
                password=conn_args.password,
                database=db_name
            )
            query = f"SELECT table, name, type FROM system.columns WHERE database = '{db_name}' ORDER BY table, position;"
            result = client.query(query)
            for row in result.result_rows:
                table, column, dtype = row
                tables[table].append(f"`{column}` {dtype}")
            client.close()

        # Format the schema string
        for table_name, columns in tables.items():
            schema_str += f"CREATE TABLE {table_name} (\n  "
            schema_str += ",\n  ".join(columns)
            schema_str += "\n);\n"
        return schema_str.strip()

    except Exception as e:
        logging.error(f"Failed to retrieve schema for {db_type} database '{db_name}': {e}")
        return ""


def get_pg_ch_common_databases(pg_args, ch_args):
    """
    [--- 修改 ---] 此函数现在只查找 PG 和 CH 的交集
    """
    pg_dbs = set()
    ch_dbs = set()

    # Retrieve databases from PostgreSQL
    try:
        conn_pg = psycopg2.connect(host=pg_args.host, port=pg_args.port, user=pg_args.user, password=pg_args.password, dbname='postgres')
        conn_pg.autocommit = True
        with conn_pg.cursor() as cur_pg:
            cur_pg.execute("SELECT datname FROM pg_database;")
            pg_system_dbs = {'template0', 'template1', 'postgres'}
            for record in cur_pg.fetchall():
                db_name = record[0]
                if db_name not in pg_system_dbs:
                    pg_dbs.add(db_name)
        conn_pg.close()
        logging.info("Successfully retrieved database list from PostgreSQL.")
    except Exception as e:
        logging.error("Failed to connect to PostgreSQL or retrieve database list: %s", e)
        return []

    # Retrieve databases from ClickHouse
    try:
        # [--- 修改 ---] 使用 Native 客户端 (Port 9000)
        client_ch = Client_Native(host=ch_args.host, port=ch_args.port, user=ch_args.user, password=ch_args.password)
        ch_system_dbs = {'system', 'default', 'INFORMATION_SCHEMA', 'information_schema'}
        result = client_ch.execute('SHOW DATABASES')
        for row in result:
            db_name = row[0]
            if db_name not in ch_system_dbs:
                ch_dbs.add(db_name)
        logging.info("Successfully retrieved database list from ClickHouse.")
    except Exception as e:
        logging.error("Failed to connect to ClickHouse or retrieve database list: %s", e)
        return []

    return list(pg_dbs.intersection(ch_dbs))

# [--- 新增 ---] 用于获取 MyScale 数据库列表的函数
def get_myscale_databases(ms_args):
    """
    Connects to MyScale to find and return a list of databases.
    """
    ms_dbs = set()
    try:
        # MyScale 使用 HTTP 客户端 (Port 8123)
        client_ms = clickhouse_connect.get_client(
            host=ms_args.host,
            port=ms_args.port,
            user=ms_args.user,
            password=ms_args.password
        )
        ms_system_dbs = {'system', 'default', 'INFORMATION_SCHEMA', 'information_schema'}
        result = client_ms.query('SHOW DATABASES')
        for row in result.result_rows:
            db_name = row[0]
            if db_name not in ms_system_dbs:
                ms_dbs.add(db_name)
        client_ms.close()
        logging.info("Successfully retrieved database list from MyScale.")
    except Exception as e:
        logging.error("Failed to connect to MyScale or retrieve database list: %s", e)
        return []
    return list(ms_dbs)


def process_item_for_backend(item, db_id, target_db, engine, schema, ConverterClass):
    """
    (此函数无需修改，它已经是通用的)
    """
    logging.info(f"  -> [{target_db.upper()}] Processing item...")
    processed_item = item.copy()
    
    # 1. Convert and execute the main 'sql' query
    original_sql = item.get('sql')
    if not original_sql:
        logging.warning(f"  -> [{target_db.upper()}] Item lacks 'sql' field, skipping.")
        return None

    try:
        converter = ConverterClass(original_sql)
        converted_sql, integration_level = converter.convert()
        
        # [--- 关键 ---] 
        # 你的 ExecutionEngine 已经支持 'myscale'
        # 所以这里我们什么都不用改
        result = engine.execute(sql=converted_sql, db_type=target_db, db_identifier=db_id)

        if result.get('status') == 'success':
            logging.info(f"  ✅ [{target_db.upper()}] Main SQL validation successful.")
            processed_item['sql'] = converted_sql
            processed_item['integration_level'] = integration_level
        else:
            logging.warning(f"  ❌ [{target_db.upper()}] Main SQL execution failed: {result.get('message', 'Unknown error')}")
            return None
    except Exception as e:
        logging.error(f"  -> [{target_db.upper()}] Error processing main SQL: {e}", exc_info=False)
        return None

    # 2. Convert all 'sql_candidate' queries
    new_candidates = []
    for candidate_sql in item.get('sql_candidate', []):
        try:
            cand_converter = ConverterClass(candidate_sql)
            converted_cand_sql, _ = cand_converter.convert()
            new_candidates.append(converted_cand_sql)
        except Exception as e:
            logging.warning(f"  -> [{target_db.upper()}] Failed to convert candidate SQL. Storing failed query. Error: {e}")
            new_candidates.append(f"-- CONVERSION FAILED for {target_db.upper()}: {e}\n{candidate_sql}")
    
    processed_item['sql_candidate'] = new_candidates
    processed_item['db_type'] = target_db
    processed_item['schema'] = schema

    # 3. **NEW**: Delete specified fields for PG and CH outputs
    keys_to_delete = ['sql_explanation', 'input', 'database_note_prompt']
    for key in keys_to_delete:
        processed_item.pop(key, None)
        
    return processed_item


def main():
    """
    Main execution function.
    """
    logging.info("--- 步骤 1: 识别所有后端中的数据库 ---")
    # [--- 修改 ---] PG 和 CH 使用本地的、用于验证的数据库
    pg_args = argparse.Namespace(host='localhost', port=5432, user='postgres', password='postgres')
    ch_args = argparse.Namespace(host='localhost', port=9000, user='default', password='') # Port 9000 (Native)
    
    # [--- 新增 ---] MyScale 使用你的公网IP (Port 8123 (HTTP))
    ms_args = argparse.Namespace(host='8.140.37.123', port=8123, user='default', password='') 
    
    common_dbs_pg_ch = get_pg_ch_common_databases(pg_args, ch_args)
    common_dbs_ms = get_myscale_databases(ms_args)
    
    # [--- 修改 ---] 
    # 我们将处理存在于 (PG 和 CH) 或 (MyScale) 中的任何数据库
    all_available_dbs = set(common_dbs_pg_ch).union(set(common_dbs_ms))

    if not all_available_dbs:
        logging.warning("在所有后端中都没有找到可用的数据库。无法继续处理查询。")
        return

    logging.info("✔ PG/CH 通用数据库: %s", common_dbs_pg_ch)
    logging.info("✔ MyScale 可用数据库: %s", common_dbs_ms)
    logging.info("--- 步骤 1 完成 ---")
    logging.info("\n--- 步骤 2: 开始处理、转换和执行SQL查询 ---")

    try:
        # [--- 关键 ---] 
        # 你的 ExecutionEngine 已经从 engine_config.yaml 中
        # 加载了 PG, CH 和 MyScale 的连接配置。
        engine = ExecutionEngine(config_path=ENGINE_CONFIG_PATH)
    except Exception as e:
        logging.error(f"无法初始化 ExecutionEngine: {e}")
        return

    try:
        with open(INPUT_SQL_FILE_PATH, 'r', encoding='utf-8') as f:
            all_queries = json.load(f)
    except Exception as e:
        logging.error(f"读取或解析输入文件 '{INPUT_SQL_FILE_PATH}' 失败: {e}")
        return

    lite_queries = []
    pg_successful_queries = []
    ch_successful_queries = []
    ms_successful_queries = [] # [--- 新增 ---]
    
    pg_schema_cache = {}
    ch_schema_cache = {}
    ms_schema_cache = {} # [--- 新增 ---]

    total_queries = len(all_queries)
    logging.info("共加载 %s 条查询。", total_queries)

    for i, item in enumerate(all_queries):
        db_id = item.get('db_id')
        logging.info("处理第 %s/%s 条查询 | 数据库: %s", i + 1, total_queries, db_id)

        if not item.get('sql') or not db_id:
            logging.warning("记录缺少 'sql' 或 'db_id'，跳过。")
            continue

        if db_id not in all_available_dbs:
            logging.warning("数据库 '%s' 在任何后端都不可用，跳过此查询。", db_id)
            continue
            
        pg_item = None
        ch_item = None
        ms_item = None
            
        # [--- 修改 ---] 仅当 db_id 存在于 PG/CH 对中时才处理它们
        if db_id in common_dbs_pg_ch:
            # Fetch and cache PG schema
            if db_id not in pg_schema_cache:
                logging.info(f"  -> 正在为 '{db_id}' 获取 PostgreSQL schema...")
                pg_schema_cache[db_id] = get_database_schema('postgresql', pg_args, db_id)
            # Fetch and cache CH schema
            if db_id not in ch_schema_cache:
                logging.info(f"  -> 正在为 '{db_id}' 获取 ClickHouse schema...")
                ch_schema_cache[db_id] = get_database_schema('clickhouse', ch_args, db_id)
            
            # Process for PostgreSQL
            pg_item = process_item_for_backend(
                item, db_id, 'postgresql', engine, pg_schema_cache.get(db_id, ""), SQLiteToPostgreSQLConverter
            )
            # Process for ClickHouse
            ch_item = process_item_for_backend(
                item, db_id, 'clickhouse', engine, ch_schema_cache.get(db_id, ""), SQLiteToClickHouseConverter
            )

            if pg_item:
                pg_successful_queries.append(pg_item)
            if ch_item:
                ch_successful_queries.append(ch_item)

        # [--- 新增 ---] 仅当 db_id 存在于 MyScale 中时才处理它
        if db_id in common_dbs_ms:
            # Fetch and cache MyScale schema
            if db_id not in ms_schema_cache:
                logging.info(f"  -> 正在为 '{db_id}' 获取 MyScale schema...")
                ms_schema_cache[db_id] = get_database_schema('myscale', ms_args, db_id)
            
            # Process for MyScale
            ms_item = process_item_for_backend(
                item, db_id, 'myscale', engine, ms_schema_cache.get(db_id, ""), SQLiteToMyScaleConverter
            )
            
            if ms_item:
                ms_successful_queries.append(ms_item)

        # [--- 修改 ---] Lite item 逻辑保持不变
        lite_item = item.copy()
        lite_item['db_type'] = 'sqlite'
        
        if pg_item and ch_item:
            try:
                pg_level_num = int(pg_item.get('integration_level', '0'))
                ch_level_num = int(ch_item.get('integration_level', '0'))
                avg_level = (pg_level_num + ch_level_num) / 2.0
                lite_item['integration_level'] = avg_level
            except (ValueError, TypeError) as e:
                logging.warning(f"  -> Error calculating average integration level: {e}")

        lite_queries.append(lite_item)


    logging.info("\n--- 步骤 2 完成 ---")
    logging.info("\n--- 步骤 3: 保存结果 ---")

    # Save results to their respective files
    output_files = {
        OUTPUT_LITE_SUCCESS_PATH: (lite_queries, "Lite"),
        OUTPUT_PG_SUCCESS_PATH: (pg_successful_queries, "PG"),
        OUTPUT_CH_SUCCESS_PATH: (ch_successful_queries, "CH"),
        OUTPUT_MS_SUCCESS_PATH: (ms_successful_queries, "MyScale") # [--- 新Z---]
    }

    for path, (data, label) in output_files.items():
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            msg = f"✔ [{label}] 成功将 {len(data)} 条查询写入到: {path}"
            logging.info(msg)
            print(msg)
        except Exception as e:
            logging.error(f"写入 [{label}] 结果文件失败: %s", e)

    logging.info("--- 全部流程结束 ---")


if __name__ == '__main__':
    # [--- 修改 ---] 将日志级别改为 INFO，以便看到更多过程信息
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()
