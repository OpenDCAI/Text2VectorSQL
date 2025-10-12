import json
import logging
import argparse
import os
import re
import sys
from collections import defaultdict
import psycopg2
from clickhouse_driver import Client

# Assuming the project structure is correct for imports
# You may need to adjust this path if the script is moved
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from migrate_sql import SQLiteToPostgreSQLConverter, SQLiteToClickHouseConverter
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
    class ExecutionEngine:
        def __init__(self, config_path): pass
        def execute(self, sql, db_type, db_identifier): return {'status': 'success'}


# --- Configuration Area ---
# Input/Output File Configuration
INPUT_SQL_FILE_PATH = '/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/sqlite/results/bird/input_llm.json'

# Generate output paths dynamically from the input path
base_path, _ = os.path.splitext(INPUT_SQL_FILE_PATH)
OUTPUT_LITE_SUCCESS_PATH = f"{base_path}_lite.json"
OUTPUT_PG_SUCCESS_PATH = f"{base_path}_pg.json"
OUTPUT_CH_SUCCESS_PATH = f"{base_path}_ch.json"

# Execution Engine Configuration File
ENGINE_CONFIG_PATH = 'Execution_Engine/engine_config.yaml'
# --- Configuration Area End ---

def get_database_schema(db_type, conn_args, db_name):
    """
    Connects to a database and retrieves its schema.

    Args:
        db_type (str): The type of the database ('postgresql' or 'clickhouse').
        conn_args (argparse.Namespace): Connection arguments.
        db_name (str): The name of the database to inspect.

    Returns:
        str: A string representing the database schema, or an empty string on failure.
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
            client = Client(host=conn_args.host, port=conn_args.port, user=conn_args.user, password=conn_args.password, database=db_name)
            query = f"SELECT table, name, type FROM system.columns WHERE database = '{db_name}' ORDER BY table, position;"
            result = client.execute(query)
            for row in result:
                table, column, dtype = row
                tables[table].append(f"`{column}` {dtype}")

        # Format the schema string
        for table_name, columns in tables.items():
            schema_str += f"CREATE TABLE {table_name} (\n  "
            schema_str += ",\n  ".join(columns)
            schema_str += "\n);\n"
        return schema_str.strip()

    except Exception as e:
        logging.error(f"Failed to retrieve schema for {db_type} database '{db_name}': {e}")
        return ""


def get_common_databases(pg_args, ch_args):
    """
    Connects to PostgreSQL and ClickHouse to find and return a list of
    databases that exist in both systems.
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
        client_ch = Client(host=ch_args.host, port=ch_args.port, user=ch_args.user, password=ch_args.password)
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

def process_item_for_backend(item, db_id, target_db, engine, schema, ConverterClass):
    """
    Converts and validates an entire item (main SQL and candidates) for a specific backend.
    
    Args:
        item (dict): The original query item from the input file.
        db_id (str): The database identifier.
        target_db (str): The target database type ('postgresql' or 'clickhouse').
        engine (ExecutionEngine): The execution engine instance.
        schema (str): The schema for the target database.
        ConverterClass: The converter class to use for migration.

    Returns:
        dict or None: The processed item if successful, otherwise None.
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
    logging.info("--- 步骤 1: 识别 PostgreSQL 和 ClickHouse 中的通用数据库 ---")
    pg_args = argparse.Namespace(host='localhost', port=5432, user='postgres', password='postgres')
    ch_args = argparse.Namespace(host='localhost', port=9000, user='default', password='')
    common_dbs = get_common_databases(pg_args, ch_args)

    if not common_dbs:
        logging.warning("在 PostgreSQL 和 ClickHouse 之间没有找到通用数据库。无法继续处理查询。")
        return

    logging.info("✔ 通用数据库识别完成: %s", common_dbs)
    logging.info("--- 步骤 1 完成 ---")
    logging.info("\n--- 步骤 2: 开始处理、转换和执行SQL查询 ---")

    try:
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
    
    pg_schema_cache = {}
    ch_schema_cache = {}

    total_queries = len(all_queries)
    logging.info("共加载 %s 条查询。", total_queries)

    for i, item in enumerate(all_queries):
        db_id = item.get('db_id')
        logging.info("处理第 %s/%s 条查询 | 数据库: %s", i + 1, total_queries, db_id)

        if not item.get('sql') or not db_id:
            logging.warning("记录缺少 'sql' 或 'db_id'，跳过。")
            continue

        if db_id not in common_dbs:
            logging.warning("数据库 '%s' 未在两个后端都存在，跳过此查询。", db_id)
            continue
            
        # Fetch and cache schemas if not already present
        if db_id not in pg_schema_cache:
            logging.info(f"  -> 正在为 '{db_id}' 获取 PostgreSQL schema...")
            pg_schema_cache[db_id] = get_database_schema('postgresql', pg_args, db_id)
        if db_id not in ch_schema_cache:
            logging.info(f"  -> 正在为 '{db_id}' 获取 ClickHouse schema...")
            ch_schema_cache[db_id] = get_database_schema('clickhouse', ch_args, db_id)
        
        # Process for PostgreSQL and ClickHouse first to get their integration levels
        pg_item = process_item_for_backend(
            item, db_id, 'postgresql', engine, pg_schema_cache.get(db_id, ""), SQLiteToPostgreSQLConverter
        )
        ch_item = process_item_for_backend(
            item, db_id, 'clickhouse', engine, ch_schema_cache.get(db_id, ""), SQLiteToClickHouseConverter
        )

        if pg_item:
            pg_successful_queries.append(pg_item)
        if ch_item:
            ch_successful_queries.append(ch_item)

        # **NEW**: Prepare the SQLite item with the averaged integration_level
        lite_item = item.copy()
        lite_item['db_type'] = 'sqlite'
        
        # Calculate average integration level only if both migrations were successful
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
        OUTPUT_CH_SUCCESS_PATH: (ch_successful_queries, "CH")
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
    logging.basicConfig(level=logging.ERROR, format='[%(levelname)s] %(message)s')
    main()