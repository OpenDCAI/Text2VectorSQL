#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
过滤 LLM 生成的 SQL：
1. 仅保留 SELECT/CTE 查询
2. 剔除语法错误
3. 剔除执行超时
4. 进行去重
5. 统计信息并保存结果
"""

import os
import sys
import re
import time
import json
import multiprocessing as mp

from tqdm import tqdm
from func_timeout import func_timeout, FunctionTimedOut
import ijson

# ----------------------------------------------------------
# 1. 选择合适的 sqlite3 模块（优先 pysqlite3，自带最新版 SQLite）
# ----------------------------------------------------------
try:
    import pysqlite3 as sqlite3
    print(f"使用 pysqlite3，SQLite 版本: {sqlite3.sqlite_version}")
except ImportError:
    import sqlite3
    print(f"使用系统自带 sqlite3，SQLite 版本: {sqlite3.sqlite_version}")
    print("⚠️  如果后续仍出现 SQL logic error，可先安装 pysqlite3-binary：\n"
          "   pip install pysqlite3-binary\n")

# ----------------------------------------------------------
# 2. 加载向量/嵌入扩展所需的动态库
# ----------------------------------------------------------
import sqlite_vec               # pip install sqlite_vec
import sqlite_lembed            # pip install sqlite_lembed


# ----------------------------------------------------------
# 3. SQL 执行工具函数
# ----------------------------------------------------------
MODEL_NAME  = "all-MiniLM-L6-v2"
MODEL_PATH  = "../all-MiniLM-L6-v2.e4ce9877.q8_0.gguf"      # 请根据实际路径调整
REGISTER_SQL = """
INSERT OR IGNORE INTO main.lembed_models (name, model)
VALUES (?, lembed_model_from_file(?))
"""

def execute_sql(sql: str, db_path: str):
    """
    在指定 sqlite 数据库上执行 SQL，返回 (结果, 列数)。
    若出错则返回 (None, None) 并打印错误信息。
    """
    if not sql.strip():
        return None, None

    conn = None
    try:
        conn = sqlite3.connect(db_path, timeout=30)  # 最长等待 30 秒
        conn.execute("PRAGMA journal_mode=WAL")      # 切到 WAL
        # conn = sqlite3.connect(db_path)
        conn.enable_load_extension(True)

        # 加载扩展
        sqlite_vec.load(conn)
        sqlite_lembed.load(conn)

        cursor = conn.cursor()

        # 注册模型（仅首次插入时生效）
        if not os.path.exists(MODEL_PATH):
            print(f"❌ 模型文件不存在: {MODEL_PATH}")
            return None, None

        cursor.execute(REGISTER_SQL, (MODEL_NAME, MODEL_PATH))
        conn.commit()

        # 真正查询
        cursor.execute("BEGIN")
        cursor.execute(sql)
        result = cursor.fetchall()
        col_cnt = len(cursor.description)
        cursor.execute("ROLLBACK")

        return result, col_cnt

    except Exception as e:
        print(f"SQL 执行错误: {e}")
        print(f"出错 SQL: {sql}")
        return None, None

    finally:
        if conn is not None:
            conn.close()


# ----------------------------------------------------------
# 4. 供多进程使用的包装函数
# ----------------------------------------------------------
def _execute_wrapper(idx, db_id, sql, complexity, timeout, db_dir):
    try:
        res, col_cnt = func_timeout(
            timeout,
            execute_sql,
            args=(sql, os.path.join(db_dir, db_id, db_id + ".sqlite"))
        )

        success = int(res is not None and col_cnt is not None)
        row_cnt = len(res) if success else 0
        col_cnt = col_cnt if success else 0

        return [idx, db_id, sql, complexity, success, col_cnt, row_cnt]

    except FunctionTimedOut:
        return [idx, db_id, sql, complexity, 0, 0, 0]
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception:
        return [idx, db_id, sql, complexity, 0, 0, 0]


def _execute_callback(result):
    idx, db_id, sql, complexity, success, col_cnt, rows = result
    if success:
        shared_results.append({
            "db_id": db_id,
            "sql": sql,
            "complexity": complexity,
            "column_count": col_cnt,
            "rows": rows
        })


def remove_timeout_sqls_parallel(sql_infos, db_dir, num_cpus=10, timeout=10):
    """
    使用多进程并发执行 SQL，删除超时 / 报错 SQL。
    """
    batch_size = 1024
    batches = [sql_infos[i:i + batch_size] for i in range(0, len(sql_infos), batch_size)]

    for b_idx, batch in enumerate(batches):
        print(f"并行执行进度: {b_idx + 1}/{len(batches)}")
        with mp.Pool(processes=num_cpus) as pool:
            for idx, info in enumerate(batch):
                pool.apply_async(
                    _execute_wrapper,
                    args=(idx, info["db_id"], info["sql"], info["complexity"], timeout, db_dir),
                    callback=_execute_callback
                )
            pool.close()
            pool.join()
        # 防止一次性开太多进程导致系统负载过高
        time.sleep(8)


# ----------------------------------------------------------
# 5. 一系列工具函数（与原脚本相同，稍作整理）
# ----------------------------------------------------------
def parse_response(text: str) -> str:
    """从 LLM 响应中截取最后一段 ```sql ... ``` 块"""
    blocks = re.findall(r"```sql\s*(.*?)\s*```", text, re.S | re.I)
    return blocks[-1].strip() if blocks else ""


def filter_select_sqls(sql_infos):
    """仅保留 SELECT/CTE 类型查询"""
    out = []
    for info in sql_infos:
        sql = re.sub(r'/\*.*?\*/', '', info["sql"], flags=re.S)      # /* ... */
        sql = re.sub(r'--.*',        '', sql)
        sql = sql.strip()
        if sql.lower().startswith(("select", "with")):
            out.append(info)
    return out


def execute_and_attach_plan(sql_infos, db_dir):
    """利用 EXPLAIN QUERY PLAN 剔除语法错误"""
    valid = []
    for info in tqdm(sql_infos, desc="EXPLAIN"):
        res, _ = execute_sql("EXPLAIN QUERY PLAN " + info["sql"],
                             os.path.join(db_dir, info["db_id"], info["db_id"] + ".sqlite"))
        if res is not None:
            info["query_plan"] = str(res)
            valid.append(info)
    return valid


def dedup_by_template(sql_infos):
    """按照模板去重（把常量替换为 <value>）"""
    def to_template(sql):
        pat = r"""
            (?<!\w)'(?:\\.|[^'])*'   |   # 单引号字符串
            (?<!\w)"(?:\\.|[^"])*"   |   # 双引号字符串
            -?\b\d+(\.\d+)?([eE][-+]?\d+)?\b |  # 数字
            \bNULL\b | \bTRUE\b | \bFALSE\b
        """
        tpl = re.sub(pat, "<value>", sql, flags=re.I | re.X)
        tpl = re.sub(r'\s+', ' ', tpl).lower().strip()
        return tpl

    seen, uniq = set(), []
    for info in sql_infos:
        tpl = to_template(info["sql"])
        if tpl not in seen:
            seen.add(tpl)
            uniq.append(info)
    return uniq


def analyze_column_count(sql_infos):
    cnt = {}
    for info in sql_infos:
        cnt[info["column_count"]] = cnt.get(info["column_count"], 0) + 1
    print("列数分布:", cnt)


def load_ndjson(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for obj in tqdm(ijson.items(f, 'item')):
            data.append(obj)
    return data


# ----------------------------------------------------------
# 6. 主流程
# ----------------------------------------------------------
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    DB_DIR         = "../brid_vectorization/results/vector_databases_brid"
    LLM_JSON_FILE  = "./results/sql_synthesis.json"
    OUTPUT_FILE    = "./results/synthetic_sqls.json"

    # 读取 LLM 响应
    responses = load_ndjson(LLM_JSON_FILE)

    synthesized = []
    for r in responses:
        sql = parse_response(r["response"])
        if not sql:
            continue
        synthesized.append({
            "db_id": r["db_id"][:-3] if r["db_id"].endswith(".db") else r["db_id"],
            "sql": sql,
            "complexity": r["prompt"].split("Ensure the SQL query matches the ")[1]
                            .split(" level, defined as follows:")[0]
        })

    print("原始 SQL 数量:", len(synthesized))

    # 1) 去掉非 SELECT
    synthesized = filter_select_sqls(synthesized)
    print("仅保留 SELECT 后:", len(synthesized))

    # 2) 去掉语法错误
    synthesized = execute_and_attach_plan(synthesized, DB_DIR)
    print("去掉语法错误后:", len(synthesized))

    # 3) 并发执行，去掉超时
    manager = mp.Manager()
    shared_results = manager.list()
    remove_timeout_sqls_parallel(synthesized, DB_DIR, num_cpus=10, timeout=60)
    synthesized = list(shared_results)
    print("去掉超时后:", len(synthesized))
    analyze_column_count(synthesized)

    # 4) 按模板去重
    synthesized = dedup_by_template(synthesized)
    print("模板级去重后:", len(synthesized))
    analyze_column_count(synthesized)

    # 5) 保存结果
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(synthesized, f, indent=2, ensure_ascii=False)

    print(f"✅ 处理完成，结果写入 {OUTPUT_FILE}")
