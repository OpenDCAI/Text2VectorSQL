import argparse
import os
import sqlite3
import re
import struct
import glob
import logging
from contextlib import contextmanager
from datetime import datetime

# 尝试导入依赖
try:
    from clickhouse_driver import Client
except ImportError:
    Client = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

BATCH_SIZE = 10000  # 你可以根据内存调整

def cleanup_myscale_db(args, db_name):
    """连接到 MyScale/ClickHouse 服务器并删除一个特定的数据库。"""
    client = None
    try:
        logging.info("  🧹 正在清理失败的 MyScale 数据库 '%s'...", db_name)
        # 连接时不需要指定数据库
        client = Client(
            host=args.host,
            port=args.port,
            user=args.user,
            password=args.password,
            # secure=True # MyScale 通常使用 SSL/TLS
        )
        client.execute(f'DROP DATABASE IF EXISTS `{db_name}`')
        logging.info("  ✔ 清理成功。")
    except Exception as e:
        logging.error("  🔴 清理过程中出错: %s", e)
        logging.error("     你可能需要手动删除数据库 '%s'。", db_name)
    finally:
        if client:
            client.disconnect()

def coerce_value(value, target_type_str):
    """
    根据目标数据库的列类型字符串，将 Python 值强制转换为更具体的类型。
    (从原脚本中重用)
    """
    if value is None:
        return None

    target_type_str = target_type_str.lower()
    if 'nullable' in target_type_str:
        target_type_str = re.sub(r'nullable\((.+?)\)', r'\1', target_type_str)

    if any(int_type in target_type_str for int_type in ['int', 'bigint', 'smallint', 'tinyint']):
        try:
            return int(value)
        except (ValueError, TypeError):
            return None

    if ('array' not in target_type_str) and ('vector' not in target_type_str) and any(float_type in target_type_str for float_type in ['float', 'double', 'real', 'numeric', 'decimal']):
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    is_date_type = 'date' in target_type_str and 'datetime' not in target_type_str
    is_datetime_type = 'datetime' in target_type_str or 'timestamp' in target_type_str
    if isinstance(value, str) and (is_date_type or is_datetime_type):
        datetime_formats = [
            '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%d %H:%M', '%Y-%m-%d',
        ]
        for fmt in datetime_formats:
            try:
                return datetime.strptime(value, fmt)
            except (ValueError, TypeError):
                continue
        return None

    if 'bool' in target_type_str:
        if isinstance(value, int): return value != 0
        if isinstance(value, str):
            val_lower = value.lower()
            if val_lower in ('true', 't', '1', 'yes', 'y'): return True
            if val_lower in ('false', 'f', '0', 'no', 'n'): return False
        return bool(value)

    if any(str_type in target_type_str for str_type in ['string', 'text', 'char', 'clob']):
        if not isinstance(value, str):
            return str(value)
    return value

# --- Schema Translation Module ---

def translate_type_for_myscale(sqlite_type):
    """将 SQLite 数据类型映射到 MyScale/ClickHouse 类型。(与原脚本相同)"""
    sqlite_type = sqlite_type.upper()
    if 'INT' in sqlite_type: return 'Int64'
    if 'CHAR' in sqlite_type or 'TEXT' in sqlite_type or 'CLOB' in sqlite_type: return 'String'
    if 'BLOB' in sqlite_type: return 'String' # BLOB 将被解包为 Array(Float32) 或保留为 String
    if 'REAL' in sqlite_type or 'FLOA' in sqlite_type or 'DOUB' in sqlite_type: return 'Float64'
    if 'NUMERIC' in sqlite_type or 'DECIMAL' in sqlite_type: return 'Decimal(38, 6)'
    if 'DATE' in sqlite_type: return 'Date'
    if 'DATETIME' in sqlite_type: return 'DateTime'
    return 'String'

def translate_schema_for_myscale(create_sql):
    """
    将 SQLite CREATE TABLE 语句 (包括 vec0 虚拟表) 转换为 MyScale 兼容的格式。
    *** 这是关键的修改点 ***
    """
    create_sql = re.sub(r'--.*', '', create_sql)
    table_name_match = re.search(
        r'CREATE\s+(?:VIRTUAL\s+)?TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?\s*([`"\[]\S+[`"\]]|[^\s(]+)',
        create_sql, flags=re.IGNORECASE
    )
    if not table_name_match:
        raise ValueError(f"无法从 SQL 中解析表名: {create_sql}")
    table_name = table_name_match.group(1).strip('`"[]\'')
    
    lines, constraints, indices = [], [], []
    
    columns_part_match = re.search(r'\((.*)\)', create_sql, re.DOTALL)
    if not columns_part_match:
        # 可能是 vec0，使用 USING 语法
        columns_part_match = re.search(r'USING\s+vec0\s*\((.*)\)', create_sql, re.IGNORECASE | re.DOTALL)
        if not columns_part_match:
            raise ValueError(f"无法从 SQL 中解析列定义: {create_sql}")
            
    columns_part = columns_part_match.group(1)
    columns_defs = re.split(r',(?![^\(]*\))', columns_part)
    
    for col_def in columns_defs:
        col_def = col_def.strip()
        if not col_def or col_def.upper().startswith(('PRIMARY KEY', 'FOREIGN KEY', 'UNIQUE', 'CONSTRAINT', 'CHECK')):
            continue
        
        # 匹配: `my_vector` float[384]
        vec_col_match = re.match(r'([`"\[]?\w+[`"\]]?)\s+float\s*\[\s*(\d+)\s*\]', col_def, re.IGNORECASE)
        if vec_col_match:
            col_name = vec_col_match.group(1).strip('`"')
            dimension = int(vec_col_match.group(2))
            
            # 1. 添加列定义
            lines.append(f'`{col_name}` Array(Float32)')
            # 2. 添加维度约束 (MyScale/ClickHouse 推荐)
            constraints.append(f'CONSTRAINT `{col_name}_dims` CHECK length(`{col_name}`) = {dimension}')
            # 3. 添加 MyScale 向量索引 (这里使用 MSTG 作为示例, 你可以换成 HNSW 等)
            indices.append(f'VECTOR INDEX `v_idx_{col_name}` `{col_name}` TYPE MSTG')
        else:
            parts = re.split(r'\s+', col_def, 2)
            col_name = parts[0].strip('`"')
            if len(parts) > 1:
                sqlite_type = parts[1].split('(')[0]
                ch_type = translate_type_for_myscale(sqlite_type)
                if 'NOT NULL' not in col_def.upper():
                    ch_type = f'Nullable({ch_type})'
                lines.append(f'`{col_name}` {ch_type}')
                
    all_definitions = lines + constraints + indices
    create_table_ch = f"CREATE TABLE `{table_name}` (\n    "
    create_table_ch += ',\n    '.join(all_definitions)
    # MyScale 推荐使用 MergeTree 或 ReplicatedMergeTree
    create_table_ch += f"\n) ENGINE = MergeTree() ORDER BY tuple();"
    return create_table_ch

# --- Database File Search ---

def find_database_files(source_path):
    """(从原脚本中重用)"""
    database_files = []
    if os.path.isfile(source_path):
        if source_path.lower().endswith(('.db', '.sqlite', '.sqlite3')):
            database_files.append(source_path)
    elif os.path.isdir(source_path):
        logging.info("🔍 正在搜索目录 '%s' 中的数据库文件...", source_path)
        patterns = ['**/*.db', '**/*.sqlite', '**/*.sqlite3']
        for pattern in patterns:
            files = glob.glob(os.path.join(source_path, pattern), recursive=True)
            database_files.extend(files)
        database_files = sorted(list(set(database_files)))
        logging.info("✔ 找到 %d 个数据库文件。", len(database_files))
    else:
        logging.error("✖ 路径 '%s' 不存在。", source_path)
    return database_files

# --- Database Connection & Operations ---

@contextmanager
def sqlite_conn(db_path):
    """(从原脚本中重用)"""
    conn = sqlite3.connect(db_path)
    try:
        conn.enable_load_extension(True)
        import sqlite_vec
        sqlite_vec.load(conn)
        logging.info("✔ sqlite-vec 扩展已加载。")
    except Exception as e:
        logging.warning("🟡 加载 sqlite-vec 扩展失败: %s", e)
    try:
        yield conn
    finally:
        conn.close()

def get_sqlite_schema(conn):
    """(从原脚本中重用)"""
    cursor = conn.cursor()
    cursor.execute("SELECT name, sql FROM sqlite_master WHERE (type='table' OR type='view') AND name NOT LIKE 'sqlite_%' AND sql IS NOT NULL;")
    schema = cursor.fetchall()
    if not schema:
        raise RuntimeError("源 SQLite 数据库中没有找到表。")
    return schema

def get_vec_info(conn):
    """(从原脚本中重用)"""
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

def migrate_to_myscale(args, db_name, source_db_path):
    """执行到 MyScale 的迁移 (基于 migrate_to_clickhouse)。"""
    if not Client:
        logging.error("clickhouse-driver 未安装。请运行 'pip install clickhouse-driver'。")
        return
    if not tqdm:
        logging.warning("tqdm 未安装。进度条将不会显示。")
    
    logging.info("\n🚀 开始迁移到 MyScale...")
    client = None
    try:
        logging.info("连接到 MyScale (Host: %s) 来创建数据库...", args.host)
        # 1. 连接到 admin 数据库 (通常是 'default' 或不指定) 以创建新数据库
        with Client(
            host=args.host,
            port=args.port,
            user=args.user,
            password=args.password,
            # secure=True  # MyScale 需要安全连接
        ) as admin_client:
            logging.info("正在创建数据库 (如果不存在): %s", db_name)
            admin_client.execute(f'CREATE DATABASE IF NOT EXISTS `{db_name}`')
            logging.info("✔ 数据库 '%s' 已准备好。", db_name)

        # 2. 连接到新创建的数据库
        logging.info("正在连接到新数据库 '%s'...", db_name)
        client = Client(
            host=args.host,
            port=args.port,
            user=args.user,
            password=args.password,
            database=db_name,
            # secure=True # 确保安全连接
        )

        with sqlite_conn(source_db_path) as source_conn:
            vec_tables_info = get_vec_info(source_conn)
            if vec_tables_info:
                logging.info("✔ 检测到 %d 个 sqlite-vec 表: %s", len(vec_tables_info), ', '.join(vec_tables_info.keys()))

            source_cur = source_conn.cursor()
            tables_schema = get_sqlite_schema(source_conn)

            for table_name, create_sql in tables_schema:
                # 过滤掉 sqlite-vec 的内部分片表
                if table_name == 'sqlite_sequence' or table_name.endswith(('_info', '_chunks', '_rowids')) or any(suffix in table_name for suffix in ('_metadatatext', '_metadatachunks', '_vector_chunks')):
                    continue
                
                logging.info("\n-- 正在处理表: %s --", table_name)
                logging.info("  - 转换 Schema...")
                myscale_create_sql = translate_schema_for_myscale(create_sql)
                
                logging.info("  - 正在 MyScale 中创建表 '%s'...", table_name)
                client.execute(f"DROP TABLE IF EXISTS `{table_name}`")
                client.execute(myscale_create_sql)
                
                logging.info("  - 获取 MyScale 目标表结构...")
                target_column_types = {name: type_str for name, type_str in client.execute(f"SELECT name, type FROM system.columns WHERE database = '{db_name}' AND table = '{table_name}'")}
                
                logging.info("  - 准备从 SQLite 提取数据...")
                source_cur.execute(f'SELECT COUNT(*) FROM `{table_name.strip("`[]")}`')
                total_rows = source_cur.fetchone()[0]
                if total_rows == 0:
                    logging.info("  - 表 '%s' 为空, 跳过数据插入。", table_name)
                    continue

                source_cur.execute(f'SELECT * FROM `{table_name.strip("`[]")}`')
                original_column_names = [desc[0] for desc in source_cur.description]
                column_names = list(original_column_names)

                try:
                    rowid_index = column_names.index('rowid')
                    del column_names[rowid_index]
                except ValueError:
                    rowid_index = -1
                
                vec_info_for_table = vec_tables_info.get(table_name)
                vector_column_indices = {}
                if vec_info_for_table:
                    logging.info("  - 正在解码 sqlite-vec 向量数据...")
                    vector_column_indices = {i: item['dim'] for i, name in enumerate(column_names) for item in vec_info_for_table['columns'] if item['name'] == name}

                logging.info("  - 开始向 MyScale 批量插入 %d 行数据...", total_rows)
                progress_bar = tqdm(total=total_rows, desc=f"  📤 迁移 {table_name}", unit="rows") if tqdm else None
                
                data_to_insert = []

                while True:
                    rows_batch = source_cur.fetchmany(BATCH_SIZE)
                    if not rows_batch: break
                    
                    processed_rows = [list(row) for row in rows_batch]
                    if rowid_index != -1:
                        original_rowid_index = original_column_names.index('rowid')
                        for row in processed_rows: del row[original_rowid_index]
                    
                    if vec_info_for_table:
                        for row in processed_rows:
                            for i, dim in vector_column_indices.items():
                                blob = row[i]
                                if isinstance(blob, bytes):
                                    num_floats = len(blob) // 4
                                    if num_floats == dim:
                                        row[i] = list(struct.unpack(f'<{dim}f', blob))
                                    else:
                                        # 填充或截断 (与原脚本逻辑相同)
                                        unpacked_data = list(struct.unpack(f'<{num_floats}f', blob))
                                        row[i] = (unpacked_data + [0.0] * dim)[:dim]

                    final_data_batch = []
                    for row in processed_rows:
                        coerced_row = [coerce_value(row[i], target_column_types.get(col_name, 'String')) for i, col_name in enumerate(column_names)]
                        final_data_batch.append(tuple(coerced_row))
                    
                    if not final_data_batch: continue
                    
                    data_to_insert.extend(final_data_batch)

                    # 为了提高性能，可以累积更多数据再插入，但这里我们保持原脚本的逻辑
                    insert_statement = f"INSERT INTO `{table_name}` ({', '.join([f'`{c}`' for c in column_names])}) VALUES"
                    client.execute(insert_statement, data_to_insert, types_check=True)
                    if progress_bar: progress_bar.update(len(data_to_insert))
                    data_to_insert = [] # 清空批次

                if progress_bar: progress_bar.close()
                logging.info("  ✔ 表 '%s' 数据迁移完成。", table_name)
        logging.info("\n🎉 所有表已成功迁移到 MyScale！")

    except Exception as e:
        logging.error("❌ 迁移到 MyScale 失败: %s", e)
        # 触发清理
        cleanup_myscale_db(args, db_name)
        raise e # 重新抛出异常以停止脚本
    finally:
        if client: client.disconnect()

def main():
    logging.basicConfig(
        level=logging.INFO, # 将日志级别设为 INFO 以查看更多详情
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    if Client is None:
        logging.error("clickhouse-driver 未找到。请运行 'pip install clickhouse-driver'。")
        logging.warning("MyScale 迁移将不可用。")
    if tqdm is None:
        logging.info("tqdm 未找到。进度条将不会显示。")

    parser = argparse.ArgumentParser(
        description="将 SQLite 数据库 (包括 sqlite-vec) 迁移到 MyScale。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--source', default='/mnt/DataFlow/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/sqlite/results/synthesis_data/vector_databases', help="源 SQLite 数据库文件或文件夹路径。")
    parser.add_argument('--host', default='8.140.37.123', help="[MyScale] 集群主机名 (例如 'your-cluster.db.myscale.com')。")
    parser.add_argument('--port', type=int, default=9000, help="[MyScale] 安全连接端口 (通常是 8443)。")
    parser.add_argument('--user', default='default', help="[MyScale] 用户名。")
    parser.add_argument('--password', default='', help="[MyScale] 密码。")
    
    args = parser.parse_args()

    if not os.path.exists(args.source):
        logging.error("✖ 源路径 '%s' 不存在。", args.source)
        return

    database_files = find_database_files(args.source)
    if not database_files:
        logging.warning("✖ 未找到可迁移的数据库文件。")
        return

    success_count, error_count = 0, 0

    logging.info("\n📊 迁移任务:")
    logging.info("  源路径: %s", args.source)
    logging.info("  目标类型: MyScale")
    logging.info("  目标主机: %s:%s", args.host, args.port)
    logging.info("  找到的数据库文件: %d", len(database_files))

    for i, db_file in enumerate(database_files, 1):
        logging.info("\n======================================================================")
        logging.info("🔄 正在处理数据库 %d/%d: %s", i, len(database_files), os.path.basename(db_file))
        logging.info("======================================================================")
        try:
            # 标准化数据库名称 (MyScale/ClickHouse 对名称有限制)
            db_name = os.path.splitext(os.path.basename(db_file))[0]
            db_name = re.sub(r'[^a-zA-Z0-9_]', '_', str(db_name)).lower()
            logging.info("目标数据库名称: %s", db_name)
            
            migrate_to_myscale(args, db_name, db_file)
            
            success_count += 1
            logging.info("✅ 数据库 %d/%d 迁移成功: %s", i, len(database_files), os.path.basename(db_file))
        except Exception:
            error_count += 1
            logging.exception("❌ 数据库 %d/%d 迁移失败: %s", i, len(database_files), os.path.basename(db_file))

    logging.info("\n======================================================================")
    logging.info("🎯 迁移完成 (模式: MyScale):")
    logging.info("   总文件数: %d", len(database_files))
    logging.info("   成功: %d", success_count)
    logging.info("   失败: %d", error_count)
    logging.info("======================================================================")

    if error_count > 0:
        logging.warning("⚠️  %d 个数据库迁移任务未成功完成。请检查上面的日志。", error_count)
    else:
        logging.info("🎉 所有数据库迁移任务均已成功完成！")


if __name__ == '__main__':
    main()
