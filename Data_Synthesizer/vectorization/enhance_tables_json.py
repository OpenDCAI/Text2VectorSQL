import os
import json
import sqlite3
import sqlite_vec
import csv
import re
from collections import defaultdict
import glob
from typing import List, Dict, Optional, Any
import numpy as np
import traceback
import base64
from dotenv import load_dotenv

def find_database_file(base_path: str, db_id: str) -> Optional[str]:
    """
    在指定路径下查找数据库文件，优先检查 .sqlite 后缀，其次检查 .db 后缀。
    
    Args:
        base_path (str): 数据库文件所在的目录。
        db_id (str): 数据库的ID（不含后缀）。
        
    Returns:
        Optional[str]: 如果找到文件，则返回完整的文件路径；否则返回 None。
    """
    path_sqlite = os.path.join(base_path, f"{db_id}.sqlite")
    if os.path.exists(path_sqlite):
        return path_sqlite
    
    path_db = os.path.join(base_path, f"{db_id}.db")
    if os.path.exists(path_db):
        return path_db
        
    return None

def truncate_value(value: Any, max_len: int = 300) -> Any:
    """
    如果值是字符串且长度超过 max_len，则进行截断。
    
    Args:
        value (Any): 要处理的值。
        max_len (int): 字符串的最大允许长度。
        
    Returns:
        Any: 处理后的值。
    """
    if isinstance(value, str) and len(value) > max_len:
        return value[:max_len] + '...'
    return value

def write_large_json(data: List[Dict], output_path: str, chunk_size: int = 500):
    """分块写入字典数组到 JSON 文件（避免嵌套数组）"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('[')
        
        if len(data) > 0:
            json.dump(data[0], f, ensure_ascii=False, indent=None)
        
        for i in range(1, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            f.write(',\n')
            for j, item in enumerate(chunk):
                if j > 0:
                    f.write(',')
                json.dump(item, f, ensure_ascii=False, indent=2)
        
        f.write(']')

def process_arxiv_dataset(base_dir="train"):
    """处理ArXiv数据集的表信息，为每个表添加描述和示例数据"""
    table_json_path = os.path.join(base_dir, "table.json")
    if not os.path.exists(table_json_path):
        raise ValueError(f"table.json not found for {table_json_path}")
        
    with open(table_json_path, "r", encoding="utf-8") as f:
        db_infos = json.load(f)
    
    if not isinstance(db_infos, list):
        db_infos = [db_infos]

    train_databases = base_dir
    for db_info in db_infos:
        db_id = db_info["db_id"]
        database_path = find_database_file(train_databases, db_id)

        db_info["table_samples"] = {}
        if database_path:
            try:
                conn = sqlite3.connect(database_path)
                cursor = conn.cursor()
                
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                db_tables = [row[0] for row in cursor.fetchall()]
                
                for table in db_tables:
                    try:
                        cursor.execute(f'SELECT * FROM "{table}" LIMIT 2')
                        rows = cursor.fetchall()
                        col_names = [description[0] for description in cursor.description]
                        
                        db_info["table_samples"][table] = []
                        for row in rows:
                            # 使用字典推导式和新的辅助函数进行转换和截断
                            row_dict = {col: truncate_value(val) for col, val in zip(col_names, row)}
                            db_info["table_samples"][table].append(row_dict)
                    except sqlite3.OperationalError as e:
                        print(f"  Error reading table {table}: {str(e)}")
                
                conn.close()
            except Exception as e:
                print(f"  Error connecting to SQLite database: {str(e)}")
                traceback.print_exc()
        else:
            print(f"警告: 数据库文件未找到，跳过处理: {os.path.join(train_databases, db_id)} (已检查 .sqlite 和 .db)")

    output_dir = "./results"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "enhanced_train_tables.json")
    write_large_json(db_infos,output_path,2000)
    print(f"Have generated enhanced_train_tables.json in {output_path}!")


def process_bird_dataset(base_dir="train"):
    """处理BIRD数据集的表信息，为每个表添加描述和示例数据"""
    table_json_path = os.path.join(base_dir, "train_tables.json")
    if not os.path.exists(table_json_path):
        raise ValueError(f"train_tables.json not found for {table_json_path}")
        
    with open(table_json_path, "r", encoding="utf-8") as f:
        db_infos = json.load(f)
    
    if not isinstance(db_infos, list):
        db_infos = [db_infos]

    train_databases = os.path.join(base_dir, "train_databases")
    for db_info in db_infos:
        db_id = db_info["db_id"]
        database_dir = os.path.join(train_databases, db_id)
        database_path = find_database_file(database_dir, db_id)
        database_description = os.path.join(database_dir, "database_description")
    
        if os.path.isdir(database_description):
            db_info["table_description"] = {}
            for desc in os.listdir(database_description):
                if desc.startswith('.'): continue
                desc_path = os.path.join(database_description, desc)
                if not os.path.exists(desc_path): continue
                table_name = os.path.splitext(desc)[0]
                if table_name in db_info.get("table_description", {}): continue
                try:
                    with open(desc_path, "r", encoding="utf-8", errors='ignore') as f:
                        db_info["table_description"][table_name] = f.read().strip()
                except Exception as e:
                    print(f"error in {desc}: {e}")

        db_info["table_samples"] = {}
        if database_path:
            try:
                conn = sqlite3.connect(database_path)
                cursor = conn.cursor()
                
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                db_tables = [row[0] for row in cursor.fetchall()]
                
                for table in db_tables:
                    try:
                        cursor.execute(f'SELECT * FROM "{table}" LIMIT 2')
                        rows = cursor.fetchall()
                        col_names = [description[0] for description in cursor.description]
                        
                        db_info["table_samples"][table] = []
                        for row in rows:
                            # 使用字典推导式和新的辅助函数进行转换和截断
                            row_dict = {col: truncate_value(val) for col, val in zip(col_names, row)}
                            db_info["table_samples"][table].append(row_dict)
                    except sqlite3.OperationalError as e:
                        print(f"  Error reading table {table}: {str(e)}")
                
                conn.close()
            except Exception as e:
                print(f"  Error connecting to SQLite database: {str(e)}")
                traceback.print_exc()
        else:
            print(f"警告: 数据库文件未找到，跳过处理: {os.path.join(database_dir, db_id)} (已检查 .sqlite 和 .db)")

    output_dir = "./results"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "enhanced_train_tables.json")
    write_large_json(db_infos,output_path,2000)
    print(f"Have generated enhanced_train_tables.json in {output_path}!")

def process_toy_dataset(base_dir="train/toy_spider",output_dir = "sqlite/results/toy_spider",output_json_name="enhanced_train_tables.json"):
    """处理数据集的表信息，为每个表添加描述和示例数据"""
    train_databases = base_dir
    table_json_path = os.path.join(base_dir, "tables.json")
    if not os.path.exists(table_json_path):
        raise ValueError(f"tables.json not found for {table_json_path}")
        
    with open(table_json_path, "r", encoding="utf-8") as f:
        db_infos = json.load(f)

    if not isinstance(db_infos, list):
        db_infos = [db_infos]

    for db_info in db_infos:
        db_id = db_info["db_id"]
        database_dir = os.path.join(train_databases, db_id)
        database_path = find_database_file(database_dir, db_id)

        db_info["table_samples"] = {}
        if database_path:
            try:
                conn = sqlite3.connect(database_path)
                cursor = conn.cursor()
                
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                db_tables = [row[0] for row in cursor.fetchall()]
                
                for table in db_tables:
                    try:
                        cursor.execute(f'SELECT * FROM "{table}" LIMIT 2')
                        rows = cursor.fetchall()
                        col_names = [description[0] for description in cursor.description]
                        
                        db_info["table_samples"][table] = []
                        for row in rows:
                            # 使用字典推导式和新的辅助函数进行转换和截断
                            row_dict = {col: truncate_value(val) for col, val in zip(col_names, row)}
                            db_info["table_samples"][table].append(row_dict)
                    except sqlite3.OperationalError as e:
                        print(f"  Error reading table {table}: {str(e)}")
                
                conn.close()
            except Exception as e:
                print(f"  Error connecting to SQLite database: {str(e)}")
                traceback.print_exc()
        else:
            print(f"警告: 数据库文件未找到，跳过处理: {os.path.join(database_dir, db_id)} (已检查 .sqlite 和 .db)")

    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, output_json_name)
    write_large_json(db_infos,output_path,2000)
    print(f"Have generated {output_json_name} in {output_path}!")
    
def process_spider_dataset(base_dir="spider_data"):
    """处理Spider数据集的表信息，为每个表添加描述和示例数据"""
    table_json_path = os.path.join(base_dir, "tables.json")
    if not os.path.exists(table_json_path):
        raise ValueError(f"tables.json not found for {table_json_path}")
        
    with open(table_json_path, "r", encoding="utf-8") as f:
        db_infos = json.load(f)
    
    if not isinstance(db_infos, list):
        db_infos = [db_infos]

    train_databases = os.path.join(base_dir, "database")
    for db_info in db_infos:
        db_id = db_info["db_id"]
        database_path = find_database_file(train_databases, db_id)

        db_info["table_samples"] = {}
        if database_path:
            try:
                conn = sqlite3.connect(database_path)
                cursor = conn.cursor()
                
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                db_tables = [row[0] for row in cursor.fetchall()]
                
                for table in db_tables:
                    try:
                        cursor.execute(f'SELECT * FROM "{table}" LIMIT 2')
                        rows = cursor.fetchall()
                        col_names = [description[0] for description in cursor.description]
                        
                        db_info["table_samples"][table] = []
                        for row in rows:
                            # 使用字典推导式和新的辅助函数进行转换和截断
                            row_dict = {col: truncate_value(val) for col, val in zip(col_names, row)}
                            db_info["table_samples"][table].append(row_dict)
                    except sqlite3.OperationalError as e:
                        print(f"  Error reading table {table}: {str(e)}")
                
                conn.close()
            except Exception as e:
                print(f"  Error connecting to SQLite database: {str(e)}")
                traceback.print_exc()
        else:
            print(f"警告: 数据库文件未找到，跳过处理: {os.path.join(train_databases, db_id)} (已检查 .sqlite 和 .db)")

    output_dir = "./results"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "enhanced_train_tables_spider.json")
    write_large_json(db_infos,output_path,2000)
    print(f"Have generated enhanced_train_tables_spider.json in {output_path}!")

def process_dataset_vector(base_dir, table_schema_path, databases_path, output_dir, output_schema_name):
    """处理向量数据库的表信息，为每个表添加描述和示例数据"""
    table_json_path = os.path.join(base_dir, table_schema_path)
    if not os.path.exists(table_json_path):
        raise ValueError(f"tables.json not found for {table_json_path}")
        
    with open(table_json_path, "r", encoding="utf-8") as f:
        db_infos = json.load(f)
    
    if not isinstance(db_infos, list):
        db_infos = [db_infos]

    train_databases = os.path.join(base_dir, databases_path)
    for db_info in db_infos:
        db_id = db_info["db_id"]
        database_dir = os.path.join(train_databases, db_id)
        database_path = find_database_file(database_dir, db_id)

        db_info["table_samples"] = {}
        if database_path:
            try:
                conn = sqlite3.connect(database_path)
                cursor = conn.cursor()
                conn.enable_load_extension(True)
                sqlite_vec.load(conn)
                
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                db_tables = [row[0] for row in cursor.fetchall()]
                
                for table in db_tables:
                    try:
                        cursor.execute(f'SELECT * FROM "{table}" LIMIT 2')
                        rows = cursor.fetchall()
                        col_names = [description[0] for description in cursor.description]
                        
                        db_info["table_samples"][table] = []
                        for row in rows:
                            processed_row = {}
                            for col_name, value in zip(col_names, row):
                                if isinstance(value, bytes):
                                    processed_row[col_name] = base64.b64encode(value).decode('ascii')
                                # 在这里直接整合截断逻辑
                                elif isinstance(value, str) and len(value) > 300:
                                    processed_row[col_name] = value[:300] + '...'
                                else:
                                    processed_row[col_name] = value
                            db_info["table_samples"][table].append(processed_row)
                    except sqlite3.OperationalError as e:
                        print(f"  Error reading table {table}: {str(e)}")
                
                conn.close()
            except Exception as e:
                print(f"  Error connecting to SQLite database: {str(e)}")
                traceback.print_exc()
        else:
            print(f"警告: 数据库文件未找到，跳过处理: {os.path.join(database_dir, db_id)} (已检查 .sqlite 和 .db)")
               
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, output_schema_name)
    write_large_json(db_infos, output_path, 2000)
    print(f"Have generated {output_schema_name} in {output_path}!")

def main_vector():
    load_dotenv()
    print("Attempting to load configuration from .env file...")

    base_dir = os.getenv("BASE_DIR_ENHANCE_VECTOR")
    table_schema_path = os.getenv("TABLE_SCHEMA_PATH_ENHANCE_VECTOR")
    databases_path = os.getenv("DATAPATH_PATH_ENHANCE_VECTOR")
    output_dir = os.getenv("OUTPUT_DIR_ENHANCE_VECTOR")
    output_schema_name = os.getenv("OUTPUT_SCHEMA_NAME_ENHANCE_VECTOR")

    required_vars = {
        "BASE_DIR_ENHANCE_VECTOR": base_dir,
        "TABLE_SCHEMA_PATH_ENHANCE_VECTOR": table_schema_path,
        "DATAPATH_PATH_ENHANCE_VECTOR": databases_path,
        "OUTPUT_DIR_ENHANCE_VECTOR": output_dir,
        "OUTPUT_SCHEMA_NAME_ENHANCE_VECTOR": output_schema_name
    }

    missing_vars = [key for key, value in required_vars.items() if value is None]
    if missing_vars:
        raise ValueError(f"错误：以下环境变量未在 .env 文件中设置，请检查: {', '.join(missing_vars)}")

    print("✅ 配置加载成功！")
    
    process_dataset_vector(
        base_dir=base_dir,
        table_schema_path=table_schema_path,
        databases_path=databases_path,
        output_dir=output_dir,
        output_schema_name=output_schema_name
    )

def main_bird(): 
    process_bird_dataset()

if __name__ == "__main__":
    FUNCTION_MAP = {
        "enhance_bird": main_bird,
        "enhance_vec_bird": main_vector,
        "enhance_vec": main_vector,
        "enhance_spider": process_spider_dataset,
        "spider_vector": main_vector,
        "enhance_arxiv": process_arxiv_dataset
    }

    print("正在加载 .env 文件...")
    load_dotenv()
    print("加载完成。")

    mode = os.getenv("ENHANCE_TABLE_MODE")
    print(f"当前配置的模式是: {mode}")

    selected_function = FUNCTION_MAP.get(mode, main_bird)
    
    selected_function()
