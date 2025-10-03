import json
import logging
import argparse
import os
import re

# 从您提供的脚本中导入必要的模块和类
from migrate_db import find_database_files, migrate_to_both_backends
from migrate_sql import SQLiteToPostgreSQLConverter, SQLiteToClickHouseConverter
from Execution_Engine.execution_engine import ExecutionEngine, TimeoutError

# --- 配置区 ---
# 日志配置
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# 输入/输出文件配置
SOURCE_DB_PATH = '/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/sqlite/results/toy_spider/'
INPUT_SQL_FILE_PATH = '/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/sqlite/results/toy_spider/question_and_sql_pairs.json'
OUTPUT_SUCCESSFUL_QUERIES_PATH = './successful_queries.json'

# 执行引擎配置文件
ENGINE_CONFIG_PATH = 'Execution_Engine/engine_config.yaml'
# --- 配置区结束 ---

def main():
    """
    主执行函数：
    1. 迁移数据库到 PostgreSQL 和 ClickHouse。
    2. 对迁移成功的数据库，转换并执行查询。
    3. 仅保留在两个后端都执行成功的查询。
    """
    logging.info("--- 步骤 1: 开始数据库迁移 ---")

    # 为数据库迁移设置参数 (硬编码以简化脚本，也可从命令行读取)
    pg_args = argparse.Namespace(host='localhost', port=5432, user='postgres', password='postgres')
    ch_args = argparse.Namespace(host='localhost', port=9000, user='default', password='')

    if not os.path.exists(SOURCE_DB_PATH):
        logging.error(f"源数据库路径不存在: {SOURCE_DB_PATH}")
        return

    database_files = find_database_files(SOURCE_DB_PATH)
    if not database_files:
        logging.error(f"在 {SOURCE_DB_PATH} 中未找到数据库文件。")
        return

    # 执行双后端迁移
    successful_dbs, _, _ = migrate_to_both_backends(database_files, pg_args, ch_args)

    if not successful_dbs:
        logging.warning("没有任何数据库被成功迁移到两个后端。无法继续处理查询。")
        return

    logging.info(f"✔ 数据库迁移完成。成功迁移到两个后端的数据库: {successful_dbs}")
    logging.info("--- 步骤 1 完成 ---")

    # ---
    
    logging.info("\n--- 步骤 2: 开始处理、转换和执行SQL查询 ---")
    
    # 初始化执行引擎
    try:
        engine = ExecutionEngine(config_path=ENGINE_CONFIG_PATH)
    except Exception as e:
        logging.error(f"无法初始化 ExecutionEngine: {e}")
        logging.error("请确保 'engine_config.yaml' 配置文件存在且格式正确。")
        return

    # 加载原始SQL查询
    try:
        with open(INPUT_SQL_FILE_PATH, 'r', encoding='utf-8') as f:
            all_queries = json.load(f)
    except FileNotFoundError:
        logging.error(f"输入SQL文件未找到: {INPUT_SQL_FILE_PATH}")
        return
    except json.JSONDecodeError:
        logging.error(f"无法解析JSON文件: {INPUT_SQL_FILE_PATH}")
        return

    successful_queries_data = []
    
    total_queries = len(all_queries)
    logging.info(f"共加载 {total_queries} 条查询。")

    for i, item in enumerate(all_queries):
        db_id = item.get('db_id')
        original_sql = item.get('sql')
        pattern = r"'all-MiniLM-L6-v2'|\"all-MiniLM-L6-v2\"|all-MiniLM-L6-v2"
        replacement = "'all-MiniLM-L6-v2'"
        original_sql = re.sub(pattern, replacement, original_sql)
        pattern = r"'laion/CLIP-ViT-B-32-laion2B-s34B-b79K'|\"laion/CLIP-ViT-B-32-laion2B-s34B-b79K\"|laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
        replacement = "'laion/CLIP-ViT-B-32-laion2B-s34B-b79K'"
        original_sql = re.sub(pattern, replacement, original_sql)

        print(f"\n{'='*40}")
        logging.info(f"处理第 {i+1}/{total_queries} 条查询 | 数据库: {db_id}")

        if not original_sql or not db_id:
            logging.warning("记录缺少 'sql' 或 'db_id'，跳过。")
            continue

        # 检查数据库是否在成功迁移的列表中
        if db_id not in successful_dbs:
            logging.warning(f"数据库 '{db_id}' 未在两个后端都迁移成功，跳过此查询。")
            continue

        pg_success = False
        ch_success = False
        converted_pg_sql = ""
        converted_ch_sql = ""

        try:
            # 1. 转换为 PostgreSQL
            logging.info("  -> 转换为 PostgreSQL 语法...")
            pg_converter = SQLiteToPostgreSQLConverter(original_sql)
            converted_pg_sql = pg_converter.convert()
            logging.info("  -> [PG] 转换成功。")

            # 2. 执行 PostgreSQL 查询
            logging.info("  -> 正在 PostgreSQL 上执行...")
            pg_result = engine.execute(
                sql=converted_pg_sql,
                db_type='postgresql',
                db_identifier=db_id
            )
            if pg_result.get('status') == 'success':
                pg_success = True
                logging.info(f"  -> [PG] 执行成功！返回 {pg_result.get('row_count', 'N/A')} 行。")
            else:
                logging.warning(f"  -> [PG] 执行失败: {pg_result.get('message', '未知错误')}")
                logging.debug(f"失败的PG SQL:\n{converted_pg_sql}")


            # 只有在 PG 成功后才尝试 CH，以节省时间
            if pg_success:
                # 3. 转换为 ClickHouse
                logging.info("  -> 转换为 ClickHouse 语法...")
                ch_converter = SQLiteToClickHouseConverter(original_sql)
                converted_ch_sql = ch_converter.convert()
                logging.info("  -> [CH] 转换成功。")

                # 4. 执行 ClickHouse 查询
                logging.info("  -> 正在 ClickHouse 上执行...")
                ch_result = engine.execute(
                    sql=converted_ch_sql,
                    db_type='clickhouse',
                    db_identifier=db_id
                )
                if ch_result.get('status') == 'success':
                    ch_success = True
                    logging.info(f"  -> [CH] 执行成功！返回 {ch_result.get('row_count', 'N/A')} 行。")
                else:
                    logging.warning(f"  -> [CH] 执行失败: {ch_result.get('message', '未知错误')}")
                    logging.debug(f"失败的CH SQL:\n{converted_ch_sql}")

        except (TimeoutError, ValueError) as e:
            logging.error(f"处理过程中发生错误 (超时或值错误): {e}")
        except Exception as e:
            logging.error(f"发生意外错误: {e}", exc_info=True)

        # 5. 检查并保存结果
        if pg_success and ch_success:
            logging.info(f"✅ 查询在两个后端均成功执行！已保存。")
            successful_queries_data.append({
                "question": item.get("question"),
                "db_id": db_id,
                "sqlite_sql": original_sql,
                "postgresql_sql": converted_pg_sql,
                "clickhouse_sql": converted_ch_sql
            })
        else:
            logging.warning(f"❌ 此查询未在所有后端成功执行，已丢弃。")
        
        print(f"{'='*40}")

    logging.info("\n--- 步骤 2 完成 ---")

    # ---

    logging.info("\n--- 步骤 3: 保存结果 ---")
    try:
        with open(OUTPUT_SUCCESSFUL_QUERIES_PATH, 'w', encoding='utf-8') as f:
            json.dump(successful_queries_data, f, indent=4, ensure_ascii=False)
        logging.info(f"✔ 成功将 {len(successful_queries_data)} 条查询写入到: {OUTPUT_SUCCESSFUL_QUERIES_PATH}")
    except Exception as e:
        logging.error(f"写入结果文件失败: {e}")
        
    logging.info("--- 全部流程结束 ---")


if __name__ == '__main__':
    main()