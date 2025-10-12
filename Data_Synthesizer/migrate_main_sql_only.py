import json
import logging
import argparse
import os
import re
import sys
import psycopg2
from clickhouse_driver import Client

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# From your provided script, assuming these modules exist and are correct
from migrate_sql import SQLiteToPostgreSQLConverter, SQLiteToClickHouseConverter
from Execution_Engine.execution_engine import ExecutionEngine, TimeoutError

# --- Configuration Area ---
# Logging Configuration
logging.basicConfig(level=logging.ERROR, format='[%(levelname)s] %(message)s')

# Input/Output File Configuration
# Note: SOURCE_DB_PATH is no longer used for migration but kept for context if needed.
SOURCE_DB_PATH = '/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/sqlite/results/toy_spider/'
INPUT_SQL_FILE_PATH = '/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/sqlite/results/toy_spider/question_and_sql_pairs.json'
OUTPUT_SUCCESSFUL_QUERIES_PATH = './successful_queries.json'

# Execution Engine Configuration File
ENGINE_CONFIG_PATH = 'Execution_Engine/engine_config.yaml'
# --- Configuration Area End ---

def get_common_databases(pg_args, ch_args):
    """
    Connects to PostgreSQL and ClickHouse to find and return a list of
    databases that exist in both systems.

    Args:
        pg_args (argparse.Namespace): Connection arguments for PostgreSQL.
        ch_args (argparse.Namespace): Connection arguments for ClickHouse.

    Returns:
        list: A list of database names present in both backends.
    """
    pg_dbs = set()
    ch_dbs = set()

    # Retrieve databases from PostgreSQL
    try:
        conn_pg = psycopg2.connect(
            host=pg_args.host,
            port=pg_args.port,
            user=pg_args.user,
            password=pg_args.password,
            dbname='postgres'  # Connect to a default DB to query the catalog
        )
        conn_pg.autocommit = True
        with conn_pg.cursor() as cur_pg:
            cur_pg.execute("SELECT datname FROM pg_database;")
            # Filter out system databases
            pg_system_dbs = {'template0', 'template1', 'postgres'}
            for record in cur_pg.fetchall():
                db_name = record[0]
                if db_name not in pg_system_dbs:
                    pg_dbs.add(db_name)
        conn_pg.close()
        logging.info("Successfully retrieved database list from PostgreSQL.")
    except Exception as e:
        logging.error("Failed to connect to PostgreSQL or retrieve database list: %s", e)
        return []  # Return empty on failure

    # Retrieve databases from ClickHouse
    try:
        client_ch = Client(
            host=ch_args.host,
            port=ch_args.port,
            user=ch_args.user,
            password=ch_args.password
        )
        # Filter out system databases
        ch_system_dbs = {'system', 'default', 'INFORMATION_SCHEMA', 'information_schema'}
        result = client_ch.execute('SHOW DATABASES')
        for row in result:
            db_name = row[0]
            if db_name not in ch_system_dbs:
                ch_dbs.add(db_name)
        logging.info("Successfully retrieved database list from ClickHouse.")
    except Exception as e:
        logging.error("Failed to connect to ClickHouse or retrieve database list: %s", e)
        return []  # Return empty on failure

    # Calculate and return the intersection
    common_dbs = list(pg_dbs.intersection(ch_dbs))
    return common_dbs

def main():
    """
    Main execution function:
    1. Identifies databases that exist on both PostgreSQL and ClickHouse.
    2. For that common set of databases, converts and executes queries.
    3. Retains and saves only the queries that execute successfully on both backends.
    """
    logging.info("--- 步骤 1: 识别 PostgreSQL 和 ClickHouse 中的通用数据库 ---")

    # Set database connection parameters (hardcoded for script simplicity)
    pg_args = argparse.Namespace(host='localhost', port=5432, user='postgres', password='postgres')
    ch_args = argparse.Namespace(host='localhost', port=9000, user='default', password='')

    # Find databases that exist on both backends
    common_dbs = get_common_databases(pg_args, ch_args)

    if not common_dbs:
        logging.warning("在 PostgreSQL 和 ClickHouse 之间没有找到通用数据库。无法继续处理查询。")
        return

    logging.info("✔ 通用数据库识别完成。在两个后端都存在的数据库: %s", common_dbs)
    logging.info("--- 步骤 1 完成 ---")

    # ---

    logging.info("\n--- 步骤 2: 开始处理、转换和执行SQL查询 ---")

    # Initialize Execution Engine
    try:
        engine = ExecutionEngine(config_path=ENGINE_CONFIG_PATH)
    except Exception as e:
        logging.error("无法初始化 ExecutionEngine: %s", e)
        logging.error("请确保 'engine_config.yaml' 配置文件存在且格式正确。")
        return

    # Load original SQL queries
    try:
        with open(INPUT_SQL_FILE_PATH, 'r', encoding='utf-8') as f:
            all_queries = json.load(f)
    except FileNotFoundError:
        logging.error("输入SQL文件未找到: %s", INPUT_SQL_FILE_PATH)
        return
    except json.JSONDecodeError:
        logging.error("无法解析JSON文件: %s", INPUT_SQL_FILE_PATH)
        return

    successful_queries_data = []

    total_queries = len(all_queries)
    logging.info("共加载 %s 条查询。", total_queries)

    for i, item in enumerate(all_queries):
        db_id = item.get('db_id')
        original_sql = item.get('sql')
        pattern = r"'all-MiniLM-L6-v2'|\"all-MiniLM-L6-v2\"|all-MiniLM-L6-v2"
        replacement = "'all-MiniLM-L6-v2'"
        original_sql = re.sub(pattern, replacement, original_sql)
        pattern = r"'laion/CLIP-ViT-B-32-laion2B-s34B-b79K'|\"laion/CLIP-ViT-B-32-laion2B-s34B-b79K\"|laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
        replacement = "'laion/CLIP-ViT-B-32-laion2B-s34B-b79K'"
        original_sql = re.sub(pattern, replacement, original_sql)

        # print(f"\n{'='*40}")
        logging.info("处理第 %s/%s 条查询 | 数据库: %s", i + 1, total_queries, db_id)

        if not original_sql or not db_id:
            logging.warning("记录缺少 'sql' 或 'db_id'，跳过。")
            continue

        # Check if the database is in the list of common databases
        if db_id not in common_dbs:
            logging.warning("数据库 '%s' 未在两个后端都存在，跳过此查询。", db_id)
            continue

        pg_success = False
        ch_success = False
        converted_pg_sql = ""
        converted_ch_sql = ""

        try:
            # 1. Convert to PostgreSQL
            logging.info("  -> 转换为 PostgreSQL 语法...")
            pg_converter = SQLiteToPostgreSQLConverter(original_sql)
            converted_pg_sql, pg_integration_level = pg_converter.convert()
            logging.info("  -> [PG] 转换成功。")

            # 2. Execute PostgreSQL Query
            logging.info("  -> 正在 PostgreSQL 上执行...")
            pg_result = engine.execute(
                sql=converted_pg_sql,
                db_type='postgresql',
                db_identifier=db_id
            )
            if pg_result.get('status') == 'success':
                pg_success = True
                logging.info("  -> [PG] 执行成功！返回 %s 行。", pg_result.get('row_count', 'N/A'))
            else:
                logging.warning("  -> [PG] 执行失败: %s", pg_result.get('message', '未知错误'))
                logging.debug("失败的PG SQL:\n%s", converted_pg_sql)

            # Only attempt CH if PG was successful to save time
            if pg_success:
                # 3. Convert to ClickHouse
                logging.info("  -> 转换为 ClickHouse 语法...")
                ch_converter = SQLiteToClickHouseConverter(original_sql)
                converted_ch_sql, ch_integration_level = ch_converter.convert()
                logging.info("  -> [CH] 转换成功。")

                # 4. Execute ClickHouse Query
                logging.info("  -> 正在 ClickHouse 上执行...")
                ch_result = engine.execute(
                    sql=converted_ch_sql,
                    db_type='clickhouse',
                    db_identifier=db_id
                )
                if ch_result.get('status') == 'success':
                    ch_success = True
                    logging.info("  -> [CH] 执行成功！返回 %s 行。", ch_result.get('row_count', 'N/A'))
                else:
                    logging.warning("  -> [CH] 执行失败: %s", ch_result.get('message', '未知错误'))
                    logging.debug("失败的CH SQL:\n%s", converted_ch_sql)

        except (TimeoutError, ValueError) as e:
            logging.error("处理过程中发生错误 (超时或值错误): %s", e)
        except Exception as e:
            logging.error("发生意外错误: %s", e, exc_info=True)

        # 5. Check and save the result
        if pg_success and ch_success:
            logging.info("✅ 查询在两个后端均成功执行！已保存。")
            item["sqlite_sql"] = original_sql
            item["postgresql_sql"] = converted_pg_sql
            item["clickhouse_sql"] = converted_ch_sql
            item["integration_level"] = (pg_integration_level + ch_integration_level) / 2
            successful_queries_data.append(item)
        else:
            logging.warning("❌ 此查询未在所有后端成功执行，已丢弃。")

        # print(f"{'='*40}")

    logging.info("\n--- 步骤 2 完成 ---")

    # ---

    logging.info("\n--- 步骤 3: 保存结果 ---")
    try:
        with open(OUTPUT_SUCCESSFUL_QUERIES_PATH, 'w', encoding='utf-8') as f:
            json.dump(successful_queries_data, f, indent=4, ensure_ascii=False)
        logging.info("✔ 成功将 %s 条查询写入到: %s", len(successful_queries_data), OUTPUT_SUCCESSFUL_QUERIES_PATH)
    except Exception as e:
        logging.error("写入结果文件失败: %s", e)

    logging.info("--- 全部流程结束 ---")


if __name__ == '__main__':
    main()