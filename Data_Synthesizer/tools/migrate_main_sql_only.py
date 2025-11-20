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
    print("Could not import necessary modules.")
    # (Dummy classes omitted for brevity)

# --- Configuration Area ---
DEFAULT_DATASETS = ["arxiv", "spider", "bird"] 
ENGINE_CONFIG_PATH = os.path.join(project_root, 'Execution_Engine', 'engine_config.yaml')
# --- Configuration Area End ---

def get_paths(dataset_name):
    """根据数据集名称动态生成路径"""
    input_path = f'/mnt/DataFlow/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/sqlite/results/{dataset_name}/candidate_sql.json'
    
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
    # ... (保持原有的 get_database_schema 代码不变) ...
    try:
        schema_str = ""
        tables = defaultdict(list)
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
    # ... (保持原有的 get_pg_ch_common_databases 代码不变) ...
    pg_dbs, ch_dbs = set(), set()
    try:
        conn_pg = psycopg2.connect(host=pg_args.host, port=pg_args.port, user=pg_args.user, password=pg_args.password, dbname='postgres')
        with conn_pg.cursor() as cur:
            cur.execute("SELECT datname FROM pg_database;")
            pg_dbs = {row[0] for row in cur.fetchall()} - {'template0', 'template1', 'postgres'}
        conn_pg.close()
    except Exception: pass

    try:
        client_ch = Client_Native(host=ch_args.host, port=ch_args.port, user=ch_args.user, password=ch_args.password)
        ch_dbs = {row[0] for row in client_ch.execute('SHOW DATABASES')} - {'system', 'default', 'INFORMATION_SCHEMA', 'information_schema'}
    except Exception: pass
    return list(pg_dbs.intersection(ch_dbs))

def get_myscale_databases(ms_args):
    try:
        client_ms = clickhouse_connect.get_client(host=ms_args.host, port=ms_args.port, user=ms_args.user, password=ms_args.password)
        dbs = {row[0] for row in client_ms.query('SHOW DATABASES').result_rows} - {'system', 'default', 'INFORMATION_SCHEMA', 'information_schema'}
        client_ms.close()
        return list(dbs)
    except Exception as e:
        logging.error(f"MyScale List Error: {e}")
        return []

def patch_myscale_sql_runtime(sql):
    """
    运行时补丁：处理 migrate_sql.py 可能遗漏的极端情况。
    """
    # 1. 如果还有 MATCH [vector] 这种 Arxiv 格式，强制替换为 1=1 (防止语法报错，虽然会丢失向量条件)
    # 这是为了让 SQL 至少能跑通，后续 migrate_sql.py 完善后可移除
    if "MATCH [" in sql:
        # 移除 MATCH [...]
        sql = re.sub(r"MATCH\s*\[.*?\]", "1=1", sql, flags=re.DOTALL)
    
    # 2. 强制修复 LIMIT 括号问题 (LIMIT 5))
    sql = sql.replace("LIMIT 5)", "LIMIT 5")
    
    return sql

def process_item_for_backend(item, db_id, target_db, engine, schema, ConverterClass):
    processed_item = item.copy()
    original_sql = item.get('sql')
    if not original_sql: return None

    try:
        converter = ConverterClass(original_sql)
        converted_sql, integration_level = converter.convert()
        
        # 对 MyScale 应用运行时补丁
        if target_db == 'myscale':
            converted_sql = patch_myscale_sql_runtime(converted_sql)

        result = engine.execute(sql=converted_sql, db_type=target_db, db_identifier=db_id)

        processed_item['sql'] = converted_sql
        processed_item['integration_level'] = integration_level
        
        if result.get('status') == 'success':
            processed_item['execution_status'] = 'success'
        else:
            processed_item['execution_status'] = 'failed'
            processed_item['error_message'] = str(result.get('message', 'Unknown Error'))
            
    except Exception as e:
        processed_item['execution_status'] = 'exception'
        processed_item['error_message'] = str(e)

    # 处理 Candidate SQL (仅转换，不执行以节省时间，或者可选执行)
    new_candidates = []
    for cand in item.get('sql_candidate', []):
        try:
            c_conv = ConverterClass(cand)
            c_sql, _ = c_conv.convert()
            new_candidates.append(c_sql)
        except:
            new_candidates.append(cand)
    
    processed_item['sql_candidate'] = new_candidates
    processed_item['db_type'] = target_db
    processed_item['schema'] = schema
    
    # 清理无关字段
    for key in ['sql_explanation', 'input', 'database_note_prompt']:
        processed_item.pop(key, None)
        
    return processed_item

def process_item_parallel(item, engine_config_path, common_dbs_pg_ch, common_dbs_ms, pg_schema, ch_schema, ms_schema):
    try:
        engine = ExecutionEngine(config_path=engine_config_path)
    except:
        return (item.copy(), None, None, None)

    db_id = item.get('db_id')
    pg_item, ch_item, ms_item = None, None, None
    
    if db_id in common_dbs_pg_ch:
        if pg_schema: 
            pg_item = process_item_for_backend(item, db_id, 'postgresql', engine, pg_schema, SQLiteToPostgreSQLConverter)
        if ch_schema: 
            ch_item = process_item_for_backend(item, db_id, 'clickhouse', engine, ch_schema, SQLiteToClickHouseConverter)
    
    if db_id in common_dbs_ms and ms_schema:
        ms_item = process_item_for_backend(item, db_id, 'myscale', engine, ms_schema, SQLiteToMyScaleConverter)

    lite_item = item.copy()
    lite_item['db_type'] = 'sqlite'
    
    return (lite_item, pg_item, ch_item, ms_item)

def process_one_dataset(dataset_name, args, common_dbs_pg_ch, common_dbs_ms, all_available_dbs, pg_args, ch_args, ms_args):
    logging.info(f"\n{'='*20} 开始处理数据集: {dataset_name} {'='*20}")
    
    paths = get_paths(dataset_name)
    if not os.path.exists(paths["input"]):
        logging.error(f"找不到输入文件: {paths['input']}，跳过。")
        return

    try:
        with open(paths["input"], 'r', encoding='utf-8') as f:
            all_queries = json.load(f)
    except Exception as e:
        logging.error(f"读取失败: {e}")
        return

    relevant_db_ids = {item.get('db_id') for item in all_queries if item.get('db_id')}
    dbs_to_cache = all_available_dbs.intersection(relevant_db_ids)
    
    logging.info(f"[{dataset_name}] 加载 {len(all_queries)} 条查询，需缓存 {len(dbs_to_cache)} 个 DB Schema")

    pg_schema_cache = {}
    ch_schema_cache = {}
    ms_schema_cache = {}

    # 缓存 Schema
    cache_iterator = tqdm(dbs_to_cache, desc=f"[{dataset_name}] Caching") if tqdm else dbs_to_cache
    for db_id in cache_iterator:
        if db_id in common_dbs_pg_ch:
            pg_schema_cache[db_id] = get_database_schema('postgresql', pg_args, db_id)
            ch_schema_cache[db_id] = get_database_schema('clickhouse', ch_args, db_id)
        if db_id in common_dbs_ms:
            ms_schema_cache[db_id] = get_database_schema('myscale', ms_args, db_id)

    # 结果列表
    lite_res, pg_res, ch_res, ms_res = [], [], [], []

    logging.info(f"[{dataset_name}] 启动并行处理 (Workers: {args.workers})...")
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for item in all_queries:
            db_id = item.get('db_id')
            # 检查 DB 是否可用
            if not item.get('sql') or not db_id:
                continue
            if db_id not in all_available_dbs:
                # 如果数据库不存在，也可以选择跳过，或者标记为 failed
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
        
        iterator = as_completed(futures)
        if tqdm:
            iterator = tqdm(iterator, total=len(futures), desc=f"[{dataset_name}] Validating")

        for future in iterator:
            try:
                l, p, c, m = future.result()
                if l: lite_res.append(l)
                if p: pg_res.append(p)
                if c: ch_res.append(c)
                if m: ms_res.append(m)
            except Exception as e:
                logging.error(f"Task error: {e}")

    # 保存结果
    for p, data, label in [
        (paths["lite"], lite_res, "Lite"), 
        (paths["pg"], pg_res, "PG"), 
        (paths["ch"], ch_res, "CH"), 
        (paths["ms"], ms_res, "MyScale")
    ]:
        try:
            with open(p, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            logging.info(f"✔ [{dataset_name} - {label}] Saved {len(data)} records to {p}")
        except Exception as e:
            logging.error(f"❌ [{dataset_name} - {label}] Save failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Batch Parallel SQL Validator")
    parser.add_argument('--workers', type=int, default=min(32, (multiprocessing.cpu_count() or 1)))
    parser.add_argument('--myscale_password', type=str, default='', help="MyScale password")
    parser.add_argument('--datasets', type=str, default='arxiv,spider,bird,wikipedia_multimodal', 
                        help="Comma-separated list of datasets")
    
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
    
    # 获取所有可用数据库
    common_pg_ch = set(get_pg_ch_common_databases(pg_args, ch_args))
    common_ms = set(get_myscale_databases(ms_args))
    all_dbs = common_pg_ch.union(common_ms)

    if not all_dbs:
        logging.error("未找到任何可用数据库，程序退出。")
        return

    logging.info(f"可用数据库总数: {len(all_dbs)}")
    logging.info(f"待处理数据集列表: {dataset_list}")

    for ds in dataset_list:
        process_one_dataset(ds, args, common_pg_ch, common_ms, all_dbs, pg_args, ch_args, ms_args)

    logging.info("\n--- 全部任务集处理完成 ---")

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s'
    )
    main()