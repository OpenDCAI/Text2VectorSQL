import json
import logging
import argparse
import os
import re
import sys
from collections import defaultdict
import psycopg2
from clickhouse_driver import Client as Client_Native
import clickhouse_connect
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from migrate_sql import SQLiteToPostgreSQLConverter, SQLiteToClickHouseConverter, SQLiteToMyScaleConverter
    from Execution_Engine.execution_engine import ExecutionEngine, TimeoutError
except ImportError:
    print("Could not import necessary modules. Using dummy classes.")
    class SQLiteToPostgreSQLConverter:
        def __init__(self, sql): self.sql = sql
        def convert(self): return f"-- PG Converted: {self.sql}", "level_1"
    class SQLiteToClickHouseConverter:
        def __init__(self, sql): self.sql = sql
        def convert(self): return f"-- CH Converted: {self.sql}", "level_2"
    class SQLiteToMyScaleConverter:
        def __init__(self, sql): self.sql = sql
        def convert(self): return f"-- MyScale Converted: {self.sql}", "level_2"
    class ExecutionEngine:
        def __init__(self, config_path): pass
        def execute(self, sql, db_type, db_identifier): return {'status': 'success'}

# --- Configuration Area ---
# 可以在这里定义默认要处理的数据集列表
DEFAULT_DATASETS = ["arxiv", "spider", "bird"] 

# Execution Engine Configuration File
ENGINE_CONFIG_PATH = os.path.join(project_root, 'Execution_Engine', 'engine_config.yaml')
# --- Configuration Area End ---

def get_paths(dataset_name):
    """根据数据集名称动态生成路径"""
    input_path = f'/mnt/DataFlow/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/sqlite/results/{dataset_name}/candidate_sql.json'
    
    # 确保输出目录存在
    base_dir_pg = f'/mnt/DataFlow/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/postgresql/results/{dataset_name}'
    base_dir_ch = f'/mnt/DataFlow/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/clickhouse/results/{dataset_name}'
    base_dir_ms = f'/mnt/DataFlow/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/myscale/results/{dataset_name}'
    
    os.makedirs(base_dir_pg, exist_ok=True)
    os.makedirs(base_dir_ch, exist_ok=True)
    os.makedirs(base_dir_ms, exist_ok=True)

    base_path, _ = os.path.splitext(input_path)
    
    return {
        "input": input_path,
        "lite": f"{base_path}_lite.json",
        "pg": os.path.join(base_dir_pg, 'candidate_sql.json'),
        "ch": os.path.join(base_dir_ch, 'candidate_sql.json'),
        "ms": os.path.join(base_dir_ms, 'candidate_sql.json')
    }

def get_database_schema(db_type, conn_args, db_name):
    """Connects to a database and retrieves its schema."""
    schema_str = ""
    tables = defaultdict(list)
    try:
        if db_type == 'postgresql':
            conn = psycopg2.connect(host=conn_args.host, port=conn_args.port, user=conn_args.user, password=conn_args.password, dbname=db_name)
            query = "SELECT table_name, column_name, data_type FROM information_schema.columns WHERE table_schema = 'public' ORDER BY table_name, ordinal_position;"
            with conn.cursor() as cur:
                cur.execute(query)
                for row in cur.fetchall():
                    table, column, dtype = row
                    if dtype == 'USER-DEFINED': dtype = 'vector'
                    tables[table].append(f"{column} {dtype}")
            conn.close()
        elif db_type == 'clickhouse':
            client = Client_Native(host=conn_args.host, port=conn_args.port, user=conn_args.user, password=conn_args.password, database=db_name)
            query = f"SELECT table, name, type FROM system.columns WHERE database = '{db_name}' ORDER BY table, position;"
            result = client.execute(query)
            for row in result:
                tables[row[0]].append(f"`{row[1]}` {row[2]}")
        elif db_type == 'myscale':
            client = clickhouse_connect.get_client(host=conn_args.host, port=conn_args.port, user=conn_args.user, password=conn_args.password, database=db_name)
            query = f"SELECT table, name, type FROM system.columns WHERE database = '{db_name}' ORDER BY table, position;"
            result = client.query(query)
            for row in result.result_rows:
                tables[row[0]].append(f"`{row[1]}` {row[2]}")
            client.close()

        for table_name, columns in tables.items():
            schema_str += f"CREATE TABLE {table_name} (\n  " + ",\n  ".join(columns) + "\n);\n"
        return schema_str.strip()
    except Exception:
        return ""

def get_pg_ch_common_databases(pg_args, ch_args):
    """Get common DBs between PG and CH"""
    pg_dbs, ch_dbs = set(), set()
    try:
        conn_pg = psycopg2.connect(host=pg_args.host, port=pg_args.port, user=pg_args.user, password=pg_args.password, dbname='postgres')
        with conn_pg.cursor() as cur:
            cur.execute("SELECT datname FROM pg_database;")
            pg_dbs = {row[0] for row in cur.fetchall()} - {'template0', 'template1', 'postgres'}
        conn_pg.close()
    except Exception as e: logging.error(f"PG List Error: {e}")

    try:
        client_ch = Client_Native(host=ch_args.host, port=ch_args.port, user=ch_args.user, password=ch_args.password)
        ch_dbs = {row[0] for row in client_ch.execute('SHOW DATABASES')} - {'system', 'default', 'INFORMATION_SCHEMA', 'information_schema'}
    except Exception as e: logging.error(f"CH List Error: {e}")

    return list(pg_dbs.intersection(ch_dbs))

def get_myscale_databases(ms_args):
    """Get MyScale DBs"""
    try:
        client_ms = clickhouse_connect.get_client(host=ms_args.host, port=ms_args.port, user=ms_args.user, password=ms_args.password)
        dbs = {row[0] for row in client_ms.query('SHOW DATABASES').result_rows} - {'system', 'default', 'INFORMATION_SCHEMA', 'information_schema'}
        client_ms.close()
        return list(dbs)
    except Exception as e:
        logging.error(f"MyScale List Error: {e}")
        return []

def process_item_for_backend(item, db_id, target_db, engine, schema, ConverterClass):
    """Generic single item processor"""
    processed_item = item.copy()
    original_sql = item.get('sql')
    if not original_sql: return None

    try:
        converter = ConverterClass(original_sql)
        converted_sql, integration_level = converter.convert()
        
        # 执行 SQL
        result = engine.execute(sql=converted_sql, db_type=target_db, db_identifier=db_id)

        if result.get('status') == 'success':
            processed_item['sql'] = converted_sql
            processed_item['integration_level'] = integration_level
            processed_item['execution_status'] = 'success' # 标记成功
        else:
            # --- 修复点开始 ---
            # 即使失败，也保留这个条目，并记录错误，方便你在 JSON 中看到原因
            # 如果你只想要成功的，至少在这里加个日志打印
            error_msg = result.get('error', 'Unknown Error')
            logging.warning(f"[{target_db}] SQL执行失败 DB:{db_id}\nSQL: {converted_sql}\nErr: {error_msg}")
            
            # 策略 A: 仍然返回 item，但在里面标记错误（建议调试用）
            processed_item['sql'] = converted_sql
            processed_item['execution_status'] = 'failed'
            processed_item['error_message'] = str(error_msg)
            processed_item['integration_level'] = '0'
            
            # 策略 B: 如果你坚决只想要成功的，保持 return None，但必须看上面的 logging.warning
            # return None 
            # --- 修复点结束 ---
            
    except Exception as e:
        logging.error(f"[{target_db}] 转换或执行异常: {e}")
        # 同上，建议返回带有错误信息的 item
        processed_item['execution_status'] = 'exception'
        processed_item['error_message'] = str(e)
        # return None

    # 处理候选 SQL (保持原样，但建议也增加异常捕获日志)
    new_candidates = []
    for candidate_sql in item.get('sql_candidate', []):
        try:
            cand_converter = ConverterClass(candidate_sql)
            converted, _ = cand_converter.convert()
            new_candidates.append(converted)
        except Exception as e:
            new_candidates.append(f"-- FAILED: {e}\n{candidate_sql}")
    
    processed_item['sql_candidate'] = new_candidates
    processed_item['db_type'] = target_db
    processed_item['schema'] = schema
    for key in ['sql_explanation', 'input', 'database_note_prompt']:
        processed_item.pop(key, None)
        
    return processed_item

def process_item_parallel(item, engine_config_path, common_dbs_pg_ch, common_dbs_ms, pg_schema, ch_schema, ms_schema):
    """Worker function"""
    try:
        engine = ExecutionEngine(config_path=engine_config_path)
    except Exception:
        return (item.copy(), None, None, None)

    db_id = item.get('db_id')
    pg_item, ch_item, ms_item = None, None, None
    
    if db_id in common_dbs_pg_ch:
        if pg_schema: pg_item = process_item_for_backend(item, db_id, 'postgresql', engine, pg_schema, SQLiteToPostgreSQLConverter)
        if ch_schema: ch_item = process_item_for_backend(item, db_id, 'clickhouse', engine, ch_schema, SQLiteToClickHouseConverter)
    
    if db_id in common_dbs_ms and ms_schema:
        ms_item = process_item_for_backend(item, db_id, 'myscale', engine, ms_schema, SQLiteToMyScaleConverter)

    lite_item = item.copy()
    lite_item['db_type'] = 'sqlite'
    
    if pg_item and ch_item:
        try:
            pg_lvl = int(pg_item.get('integration_level', '0'))
            ch_lvl = int(ch_item.get('integration_level', '0'))
            lite_item['integration_level'] = (pg_lvl + ch_lvl) / 2.0
        except Exception: pass

    return (lite_item, pg_item, ch_item, ms_item)

def process_one_dataset(dataset_name, args, common_dbs_pg_ch, common_dbs_ms, all_available_dbs, pg_args, ch_args, ms_args):
    """
    处理单个数据集的核心逻辑
    """
    logging.info(f"\n{'='*20} 开始处理数据集: {dataset_name} {'='*20}")
    
    paths = get_paths(dataset_name)
    input_file = paths["input"]

    if not os.path.exists(input_file):
        logging.error(f"找不到输入文件: {input_file}，跳过此数据集。")
        return

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            all_queries = json.load(f)
    except Exception as e:
        logging.error(f"读取 {dataset_name} 失败: {e}")
        return

    total_queries = len(all_queries)
    logging.info(f"[{dataset_name}] 加载了 {total_queries} 条查询。")

    # 识别当前数据集需要的 DB
    relevant_db_ids = {item.get('db_id') for item in all_queries if item.get('db_id')}
    dbs_to_cache = all_available_dbs.intersection(relevant_db_ids)
    
    logging.info(f"[{dataset_name}] 需要缓存 {len(dbs_to_cache)} 个相关数据库 Schema...")

    pg_schema_cache = {}
    ch_schema_cache = {}
    ms_schema_cache = {}

    # 缓存 Schema
    cache_iterator = tqdm(dbs_to_cache, desc=f"[{dataset_name}] Caching Schemas") if tqdm else dbs_to_cache
    for db_id in cache_iterator:
        if db_id in common_dbs_pg_ch:
            if db_id not in pg_schema_cache:
                pg_schema_cache[db_id] = get_database_schema('postgresql', pg_args, db_id)
            if db_id not in ch_schema_cache:
                ch_schema_cache[db_id] = get_database_schema('clickhouse', ch_args, db_id)
        if db_id in common_dbs_ms:
            if db_id not in ms_schema_cache:
                ms_schema_cache[db_id] = get_database_schema('myscale', ms_args, db_id)

    # 准备并行处理
    lite_queries = []
    pg_successful_queries = []
    ch_successful_queries = []
    ms_successful_queries = []

    logging.info(f"[{dataset_name}] 启动并行处理 (Workers: {args.workers})...")
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for item in all_queries:
            db_id = item.get('db_id')
            if not item.get('sql') or not db_id or db_id not in all_available_dbs:
                continue
            
            futures.append(executor.submit(
                process_item_parallel,
                item,
                ENGINE_CONFIG_PATH,
                common_dbs_pg_ch,
                common_dbs_ms,
                pg_schema_cache.get(db_id),
                ch_schema_cache.get(db_id),
                ms_schema_cache.get(db_id)
            ))
        
        results_iterator = as_completed(futures)
        if tqdm:
            results_iterator = tqdm(results_iterator, total=len(futures), desc=f"[{dataset_name}] Validating SQL")

        for future in results_iterator:
            try:
                lite, pg, ch, ms = future.result()
                if lite: lite_queries.append(lite)
                if pg: pg_successful_queries.append(pg)
                if ch: ch_successful_queries.append(ch)
                if ms: ms_successful_queries.append(ms)
            except Exception as e:
                logging.error(f"Task failed: {e}")

    # 保存结果
    output_map = {
        paths["lite"]: (lite_queries, "Lite"),
        paths["pg"]: (pg_successful_queries, "PG"),
        paths["ch"]: (ch_successful_queries, "CH"),
        paths["ms"]: (ms_successful_queries, "MyScale")
    }

    for p, (data, label) in output_map.items():
        try:
            with open(p, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            logging.info(f"✔ [{dataset_name} - {label}] 保存 {len(data)} 条记录到 {p}")
        except Exception as e:
            logging.error(f"❌ [{dataset_name} - {label}] 保存失败: {e}")

def main():
    # --- 参数解析 ---
    parser = argparse.ArgumentParser(description="Batch Parallel SQL Validator")
    parser.add_argument('--workers', type=int, default=min(32, (multiprocessing.cpu_count() or 1)))
    parser.add_argument('--myscale_password', type=str, default='', help="MyScale password")
    
    # 新增参数：允许通过命令行指定数据集列表，用逗号分隔
    parser.add_argument('--datasets', type=str, default='arxiv,spider,bird,wikipedia_multimodal', 
                        help="Comma-separated list of datasets (e.g., arxiv,spider,bird). If empty, uses default list.")
    
    args = parser.parse_args()
    
    if args.datasets:
        dataset_list = [d.strip() for d in args.datasets.split(',') if d.strip()]
    else:
        dataset_list = DEFAULT_DATASETS

    if tqdm is None:
        logging.warning("tqdm not found. Progress bars hidden.")

    logging.info("--- 步骤 1: 全局数据库连接检查 ---")
    # 配置数据库连接参数
    pg_args = argparse.Namespace(host='localhost', port=5432, user='postgres', password='postgres')
    ch_args = argparse.Namespace(host='localhost', port=9000, user='default', password='') 
    ms_args = argparse.Namespace(host='112.126.57.89', port=8123, user='default', password=args.myscale_password) 
    
    # 获取所有可用数据库（只做一次，避免每个数据集都去扫全库）
    common_dbs_pg_ch = set(get_pg_ch_common_databases(pg_args, ch_args))
    common_dbs_ms = set(get_myscale_databases(ms_args))
    all_available_dbs = common_dbs_pg_ch.union(common_dbs_ms)

    if not all_available_dbs:
        logging.error("未找到任何可用数据库，程序退出。")
        return

    logging.info(f"可用数据库总数: {len(all_available_dbs)}")
    logging.info(f"待处理数据集列表: {dataset_list}")

    # --- 循环处理每个数据集 ---
    for dataset_name in dataset_list:
        process_one_dataset(
            dataset_name, 
            args, 
            common_dbs_pg_ch, 
            common_dbs_ms, 
            all_available_dbs,
            pg_args, 
            ch_args, 
            ms_args
        )

    logging.info("\n--- 全部任务集处理完成 ---")

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s'
    )
    main()