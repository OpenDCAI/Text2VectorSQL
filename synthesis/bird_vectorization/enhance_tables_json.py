import os
import json
import sqlite3
import sqlite_vec
import csv
import re
from collections import defaultdict
import glob
from typing import List, Dict
import numpy as np
import traceback
import base64
from dotenv import load_dotenv

def write_large_json(data: List[Dict], output_path: str, chunk_size: int = 500):
    """分块写入字典数组到 JSON 文件（避免嵌套数组）"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('[')  # 开始 JSON 数组
        
        # 写入第一个元素（避免开头多余逗号）
        if len(data) > 0:
            json.dump(data[0], f, ensure_ascii=False, indent=None)
        
        # 分块写入剩余元素
        for i in range(1, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            f.write(',\n')  # 添加分隔符
            # 逐元素写入（而非整个 chunk）
            for j, item in enumerate(chunk):
                if j > 0:
                    f.write(',')
                json.dump(item, f, ensure_ascii=False, indent=2)
        
        f.write(']')  # 结束 JSON 数组

def process_bird_dataset(base_dir="train"):
    """处理BIRD数据集的表信息，为每个表添加描述和示例数据"""
    # 1. 加载table.json
    table_json_path = os.path.join(base_dir, "train_tables.json")
    if not os.path.exists(table_json_path):
        raise ValueError(f"train_tables.json not found for {table_json_path}")
        
    with open(table_json_path, "r", encoding="utf-8") as f:
        db_infos = json.load(f)
    
    # 检查数据结构（数组还是对象）
    if not isinstance(db_infos, list):
        db_infos = [db_infos]

    train_databases = os.path.join(base_dir, "train_databases")
    for db_info in db_infos:
        db_id = db_info["db_id"]
        database_dir = os.path.join(train_databases, db_id)
        database_path = os.path.join(database_dir, f"{db_id}.sqlite")
        database_description = os.path.join(database_dir, "database_description")
    
        # 第一步：写入描述
        if not os.path.isdir(database_description):
                continue
        
        db_info["table_description"] = {}

        for desc in os.listdir(database_description):
            desc_path = os.path.join(database_description, desc)

            # 1. 跳过macOS的系统文件和其他隐藏文件
            if desc.startswith('.'):
                continue

            if not os.path.exists(desc_path):
                continue           

            # 表格名
            table_name = os.path.splitext(desc)[0]

            # 如果已经处理过则跳过
            if table_name in db_info["table_description"]:
                 continue
            
            try:
                # print(desc_path)
                with open(desc_path, "r", encoding="utf-8", errors='ignore') as f:
                    db_info["table_description"][table_name] = f.read().strip()
                    # print(db_info["table_description"][table_name])
            except Exception as e:
                print(f"error in {desc}: {e}")
                pass

        db_info["table_samples"] = {}
        # 第二步 写入示例数据  
        if os.path.exists(database_path):
            try:
                conn = sqlite3.connect(database_path)
                cursor = conn.cursor()
                
                # 获取所有表名
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                db_tables = [row[0] for row in cursor.fetchall()]
                
                # 为每个表获取示例行
                for table in db_tables:
                    try:
                        cursor.execute(f'SELECT * FROM "{table}" LIMIT 2')
                        rows = cursor.fetchall()
                        
                        # 获取列名
                        col_names = [description[0] for description in cursor.description]
                        
                        db_info["table_samples"][table] = []
                        # 转换为字典格式
                        for row in rows:
                            db_info["table_samples"][table].append(
                                dict(zip(col_names, row)))
                            # db_info["table_samples"][table][col_names] = row
                    except sqlite3.OperationalError as e:
                        print(f"  Error reading table {table}: {str(e)}")
                
                conn.close()
            except Exception as e:
                print(f"  Error connecting to SQLite database: {str(e)}")
                traceback.print_exc()
   
    output_dir = "./results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"目录 {output_dir} 已创建")
    else:
        print(f"目录 {output_dir} 已存在")
    
    output_path = os.path.join(output_dir, "enhanced_train_tables.json")
    # print("Processing completed!")
    write_large_json(db_infos,output_path,2000)
        
    print(f"Have generated enhanced_train_tables.json in {output_path}!")

def process_spider_dataset(base_dir="spider_data"):
    """处理BIRD数据集的表信息，为每个表添加描述和示例数据"""
    # 1. 加载table.json
    table_json_path = os.path.join(base_dir, "tables.json")
    if not os.path.exists(table_json_path):
        raise ValueError(f"tables.json not found for {table_json_path}")
        
    with open(table_json_path, "r", encoding="utf-8") as f:
        db_infos = json.load(f)
    
    # 检查数据结构（数组还是对象）
    if not isinstance(db_infos, list):
        db_infos = [db_infos]

    train_databases = os.path.join(base_dir, "database")
    for db_info in db_infos:
        db_id = db_info["db_id"]
        database_dir = os.path.join(train_databases, db_id)
        database_path = os.path.join(database_dir, f"{db_id}.sqlite")
        database_description = os.path.join(database_dir, "database_description")

        db_info["table_samples"] = {}
        # 写入示例数据  
        if os.path.exists(database_path):
            try:
                conn = sqlite3.connect(database_path)
                cursor = conn.cursor()
                
                # 获取所有表名
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                db_tables = [row[0] for row in cursor.fetchall()]
                
                # 为每个表获取示例行
                for table in db_tables:
                    try:
                        cursor.execute(f'SELECT * FROM "{table}" LIMIT 2')
                        rows = cursor.fetchall()
                        
                        # 获取列名
                        col_names = [description[0] for description in cursor.description]
                        
                        db_info["table_samples"][table] = []
                        # 转换为字典格式
                        for row in rows:
                            db_info["table_samples"][table].append(
                                dict(zip(col_names, row)))
                            # db_info["table_samples"][table][col_names] = row
                    except sqlite3.OperationalError as e:
                        print(f"  Error reading table {table}: {str(e)}")
                
                conn.close()
            except Exception as e:
                print(f"  Error connecting to SQLite database: {str(e)}")
                traceback.print_exc()
   
    output_dir = "./results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"目录 {output_dir} 已创建")
    else:
        print(f"目录 {output_dir} 已存在")
    
    output_path = os.path.join(output_dir, "enhanced_train_tables_spider.json")
    # print("Processing completed!")
    write_large_json(db_infos,output_path,2000)
        
    print(f"Have generated enhanced_train_tables.json in {output_path}!")

def process_dataset_vector(base_dir, table_schema_path, databases_path, output_dir, output_schema_name):
    """处理BIRD数据集的表信息，为每个表添加描述和示例数据"""
    # 1. 加载table.json
    table_json_path = os.path.join(base_dir, table_schema_path)
    if not os.path.exists(table_json_path):
        raise ValueError(f"tables.json not found for {table_json_path}")
        
    with open(table_json_path, "r", encoding="utf-8") as f:
        db_infos = json.load(f)
    
    # 检查数据结构（数组还是对象）
    if not isinstance(db_infos, list):
        db_infos = [db_infos]

    train_databases = os.path.join(base_dir, databases_path)
    for db_info in db_infos:
        db_id = db_info["db_id"]
        database_dir = os.path.join(train_databases, db_id)
        database_path = os.path.join(database_dir, f"{db_id}.sqlite")

        db_info["table_samples"] = {}
        # 写入示例数据  
        if os.path.exists(database_path):
            try:
                conn = sqlite3.connect(database_path)
                cursor = conn.cursor()
                conn.enable_load_extension(True)
                sqlite_vec.load(conn)
                
                # 获取所有表名
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                db_tables = [row[0] for row in cursor.fetchall()]
                
                # 为每个表获取示例行
                for table in db_tables:
                    try:
                        cursor.execute(f'SELECT * FROM "{table}" LIMIT 2')
                        rows = cursor.fetchall()
                        
                        # 获取列名
                        col_names = [description[0] for description in cursor.description]
                        
                        db_info["table_samples"][table] = []
                        # 转换为字典格式
                        for row in rows:
                            # --- 这是修改的核心部分 ---
                            processed_row = {}
                            for col_name, value in zip(col_names, row):
                                if isinstance(value, bytes):
                                    # 如果是 bytes 类型，进行 Base64 编码
                                    processed_row[col_name] = base64.b64encode(value).decode('ascii')
                                else:
                                    processed_row[col_name] = value
                            db_info["table_samples"][table].append(processed_row)
                            # --- 修改结束 ---

                    except sqlite3.OperationalError as e:
                        print(f"  Error reading table {table}: {str(e)}")
                
                conn.close()
            except Exception as e:
                print(f"  Error connecting to SQLite database: {str(e)}")
                traceback.print_exc()
   
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"目录 {output_dir} 已创建")
    else:
        print(f"目录 {output_dir} 已存在")
    
    output_path = os.path.join(output_dir, output_schema_name)
    write_large_json(db_infos, output_path, 2000)
        
    print(f"Have generated enhanced_train_tables.json in {output_path}!")

def main_vector():
    # 1. 加载 .env 文件中的环境变量
    load_dotenv()
    print("Attempting to load configuration from .env file...")

    # 2. 从环境变量中读取配置项
    base_dir = os.getenv("BASE_DIR_ENHANCE_VECTOR")
    table_schema_path = os.getenv("TABLE_SCHEMA_PATH_ENHANCE_VECTOR")
    databases_path = os.getenv("DATAPATH_PATH_ENHANCE_VECTOR")
    output_dir = os.getenv("OUTPUT_DIR_ENHANCE_VECTOR")
    output_schema_name = os.getenv("OUTPUT_SCHEMA_NAME_ENHANCE_VECTOR")

    # 3. (推荐) 检查所有变量是否都已成功加载
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
    
    # 4. 使用加载的配置作为参数来调用您的主函数
    process_dataset_vector(
        base_dir=base_dir,
        table_schema_path=table_schema_path,
        databases_path=databases_path,
        output_dir=output_dir,
        output_schema_name=output_schema_name
    )

def main_brid(): 
    # 为原始BRID数据集添加每个表的示例数据，顺便把表格描述一起存入schema
    process_bird_dataset()

if __name__ == "__main__":
    # --- 2. 创建从字符串到函数对象的映射字典 ---
    # 键是 .env 文件中 APP_MODE 的值
    # 值是上面定义的函数对象（注意：没有括号！）
    FUNCTION_MAP = {
        "enhance_bird": main_brid,
        "enhance_vec_bird": main_vector,
        "enhance_spider": process_spider_dataset,
        "spider_vector": main_vector,
    }

    # --- 3. 加载 .env 文件中的环境变量 ---
    print("正在加载 .env 文件...")
    load_dotenv()
    print("加载完成。")

    # --- 4. 从环境变量中读取模式配置 ---
    # os.getenv() 可以安全地获取环境变量，如果不存在则返回 None
    mode = os.getenv("ENHANCE_TABLE_MODE")
    print(f"当前配置的模式是: {mode}")

    # --- 5. 从字典中查找并执行对应的函数 ---
    # 使用 .get() 方法来安全地获取函数
    # 如果 mode 对应的键在字典中不存在，.get() 会返回第二个参数（这里是 default_function）
    selected_function = FUNCTION_MAP.get(mode, main_brid)
    
    # 执行找到的函数
    selected_function()
