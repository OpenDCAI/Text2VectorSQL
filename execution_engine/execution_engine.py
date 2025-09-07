# execution_engine.py

import argparse
import json
import logging
import re
import sys
from typing import Any, Dict, List, Tuple

import psycopg2
import requests
import sqlite3
import yaml
import clickhouse_connect

# --- 日志设置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ExecutionEngine")


class ExecutionEngine:
    """
    一个用于翻译和执行VectorSQL的引擎。
    
    该引擎负责:
    1. 解析SQL中的自定义 `lembed(model, text)` 函数。
    2. 调用外部的Embedding Service将文本转换为向量。
    3. 将 `lembed` 函数和向量替换为目标数据库的原生语法。
    4. 连接到指定的数据库并执行翻译后的SQL。
    5. 返回执行结果或错误信息。
    """

    def __init__(self, config_path: str = "engine_config.yaml"):
        """
        初始化引擎并加载配置。
        
        :param config_path: YAML配置文件的路径。
        """
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            self.embedding_url = self.config['embedding_service']['url']
            logger.info(f"引擎配置已从 '{config_path}' 加载。")
        except Exception as e:
            logger.error(f"加载配置文件 '{config_path}' 失败: {e}")
            raise

    def _get_embeddings(self, lembed_calls: List[Tuple[str, str]]) -> Dict[str, List[float]]:
        """
        批量调用Embedding Service获取向量。
        
        :param lembed_calls: 从SQL中解析出的 `(model, text)` 元组列表。
        :return: 一个字典，将原始 `lembed` 字符串映射到其向量。
        """
        embeddings_map = {}
        # 按模型分组，以进行高效的批量调用
        calls_by_model: Dict[str, List[str]] = {}
        for model, text in lembed_calls:
            if model not in calls_by_model:
                calls_by_model[model] = []
            calls_by_model[model].append(text)

        for model, texts in calls_by_model.items():
            try:
                payload = {"model": model, "texts": texts}
                logger.info(f"正在为模型 '{model}' 请求 {len(texts)} 个文本的向量...")
                response = requests.post(self.embedding_url, json=payload)
                response.raise_for_status()
                results = response.json()['embeddings']
                
                # 将返回的向量映射回原始的lembed调用
                for i, text in enumerate(texts):
                    # 使用元组 (model, text)作为键，比字符串更可靠
                    embeddings_map[(model, text)] = results[i]
                logger.info(f"成功获取模型 '{model}' 的向量。")
            except requests.RequestException as e:
                logger.error(f"调用Embedding Service时出错 (模型: {model}): {e}")
                raise
        
        return embeddings_map

    def _translate_sql(self, sql: str, db_type: str) -> str:
        """
        【已修正】将包含 `lembed` 函数的SQL翻译成特定数据库的方言。
        
        :param sql: 原始VectorSQL查询。
        :param db_type: 数据库类型 ('postgresql', 'clickhouse', 'sqlite')。
        :return: 翻译后的原生SQL查询。
        """
        lembed_pattern = re.compile(r"lembed\('([^']*)',\s*'([^']*)'\)")
        matches = lembed_pattern.findall(sql)
        
        if not matches:
            logger.info("SQL中未找到 'lembed' 函数，无需翻译。")
            return sql

        unique_calls = sorted(list(set(matches)))
        try:
            # embeddings_map 的键现在是元组 (model, text)
            embeddings_map = self._get_embeddings(unique_calls)
        except Exception:
            raise ValueError("无法从Embedding Service获取向量，翻译中止。")

        def replacer(match):
            """re.sub的回调函数，用于替换每个匹配项。"""
            model, text = match.groups()
            vector = embeddings_map.get((model, text))
            if vector is None:
                raise ValueError(f"未找到lembed('{model}', '{text}')对应的向量。")
            
            if db_type == 'postgresql':
                return f"'{str(vector)}'"
            elif db_type == 'clickhouse':
                return str(vector)
            else:
                # 其他数据库方言可以在此添加
                raise ValueError(f"不支持的数据库类型: {db_type}")

        # 使用 re.sub 和回调函数进行可靠的替换
        translated_sql = lembed_pattern.sub(replacer, sql)
        return translated_sql
    
    def execute(self, sql: str, db_type: str, db_identifier: str) -> Dict[str, Any]:
        """
        执行SQL查询的核心方法。
        
        :param sql: 待执行的VectorSQL。
        :param db_type: 数据库类型 ('postgresql', 'clickhouse', 'sqlite')。
        :param db_identifier: 数据库标识符 (对于PG/CH是数据库名, 对于SQLite是文件路径)。
        :return: 包含执行结果的字典。
        """
        conn = None
        try:
            # --- 对于SQLite，由于需要传递BLOB，处理方式特殊 ---
            if db_type == 'sqlite':
                lembed_pattern = re.compile(r"lembed\('([^']*)',\s*'([^']*)'\)")
                matches = lembed_pattern.findall(sql)
                params = []
                if matches:
                    unique_calls = sorted(list(set(matches)))
                    embeddings_map = self._get_embeddings(unique_calls)
                    translated_sql = lembed_pattern.sub('?', sql)
                    all_calls_in_order = lembed_pattern.findall(sql)
                    for model, text in all_calls_in_order:
                        vector = embeddings_map[(model, text)]
                        params.append(json.dumps(vector))
                else:
                    translated_sql = sql

                conn = sqlite3.connect(db_identifier)
                cursor = conn.cursor()
                cursor.execute(translated_sql, params)

            # --- 对于PostgreSQL和ClickHouse，可以先翻译再执行 ---
            else:
                translated_sql = self._translate_sql(sql, db_type)
                logger.info(f"翻译后的SQL ({db_type}): {translated_sql}")
                
                if db_type == 'postgresql':
                    db_config = self.config['database_connections']['postgresql'].copy()
                    db_config['dbname'] = db_identifier
                    conn = psycopg2.connect(**db_config)
                    cursor = conn.cursor()
                elif db_type == 'clickhouse':
                    db_config = self.config['database_connections']['clickhouse'].copy()
                    db_config['database'] = db_identifier
                    client = clickhouse_connect.get_client(**db_config)
                    result = client.query(translated_sql)
                    return {
                        "status": "success",
                        "columns": result.column_names,
                        "data": result.result_rows,
                        "row_count": len(result.result_rows)
                    }
                else:
                    raise ValueError(f"不支持的数据库类型: {db_type}")
                
                cursor.execute(translated_sql)

            # --- 获取并格式化结果 ---
            if cursor.description:
                columns = [desc[0] for desc in cursor.description]
                data = cursor.fetchall()
                row_count = len(data)
            else:
                columns = []
                data = []
                row_count = cursor.rowcount
            
            if conn and db_type != 'sqlite': conn.commit()
            
            return {
                "status": "success",
                "columns": columns,
                "data": data,
                "row_count": row_count
            }
        
        except Exception as e:
            logger.error(f"在数据库 '{db_identifier}' ({db_type}) 上执行查询失败: {e}")
            if conn: conn.rollback()
            return {"status": "error", "message": str(e)}
        
        finally:
            if conn:
                conn.close()

# --- Main execution block (no changes) ---
def main():
    parser = argparse.ArgumentParser(description="Text2VectorSQL 执行与验证引擎")
    parser.add_argument("--sql", required=True, type=str, help="要执行的VectorSQL查询语句。")
    parser.add_argument("--db-type", required=True, choices=['postgresql', 'clickhouse', 'sqlite'], help="目标数据库的类型。")
    parser.add_argument("--db-identifier", required=True, type=str, help="数据库标识符 (数据库名或SQLite文件路径)。")
    parser.add_argument("--config", default="engine_config.yaml", type=str, help="引擎配置文件的路径。")
    
    args = parser.parse_args()
    
    try:
        engine = ExecutionEngine(config_path=args.config)
        result = engine.execute(sql=args.sql, db_type=args.db_type, db_identifier=args.db_identifier)
        
        sys.stdout.write(json.dumps(result, indent=2, ensure_ascii=False))
        sys.stdout.write('\n')

    except Exception as e:
        error_result = {"status": "error", "message": f"引擎初始化或执行过程中发生致命错误: {e}"}
        sys.stderr.write(json.dumps(error_result, indent=2, ensure_ascii=False))
        sys.stderr.write('\n')
        sys.exit(1)

if __name__ == "__main__":
    main()