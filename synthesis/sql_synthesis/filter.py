#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
功能：
1. 仅保留 SELECT / CTE 查询
2. EXPLAIN 检查，剔除语法错误
3. 并发真实执行，剔除运行错误 / 超时
4. 按模板去重
5. 统计并输出结果

参数：
你需要修改MODEL_PATH变量。
MODEL_PATH：embedding模型的位置


--------------------------------------------------------------------
输入文件示例（数组形式）：
[
  {
    "db_id": "sakila",
    "sql": "SELECT * FROM actor LIMIT 3;",
    "sql_complexity": "easy"          # 可省略
  },
  ...
]

使用示例：
    python filter.py \
        --db_dir  ../brid_vectorization/results/vector_databases_brid   \
        --input   ../question_synthesis/results/question_and_sql_pairs.json      \
        --output  ./results/test_sqls.json     \
        --cpus    10 \
        --timeout 60
"""

import os, re, json, time, argparse, multiprocessing as mp
from typing import List, Dict, Tuple, Any

from tqdm import tqdm
from func_timeout import func_timeout, FunctionTimedOut

# ----------------------------------------------------------
# 1. 选择 sqlite3 实现（优先 pysqlite3，自带新版 SQLite）
# ----------------------------------------------------------
try:
    import pysqlite3 as sqlite3
    print(f"✅ 使用 pysqlite3，SQLite 版本: {sqlite3.sqlite_version}")
except ImportError:
    import sqlite3
    print(f"⚠️  使用系统自带 sqlite3，SQLite 版本: {sqlite3.sqlite_version}")
    print("   如遇到 SQL logic error，可尝试: pip install pysqlite3-binary\n")

# ----------------------------------------------------------
# 2. 加载向量/嵌入扩展（如不需要可注释掉）
# ----------------------------------------------------------
import sqlite_vec               # pip install sqlite-vec>=0.5.0
import sqlite_lembed            # pip install sqlite-lembed>=0.2.3

MODEL_NAME  = "all-MiniLM-L6-v2"
MODEL_PATH  = "../all-MiniLM-L6-v2.e4ce9877.q8_0.gguf"   # 修改成你的模型文件路径

REGISTER_MODEL_SQL = """
INSERT OR IGNORE INTO main.lembed_models (name, model)
VALUES (?, lembed_model_from_file(?))
"""

SKIP_EXPLAIN_PATTERN = re.compile(r'MATCH\s+lembed\(', re.I)

# ----------------------------------------------------------
# 3. SQLite 帮手函数
# ----------------------------------------------------------
def _connect_sqlite(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    sqlite_lembed.load(conn)
    # 注册嵌入模型（第一次插入，之后被 IGNORE）
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ 嵌入模型文件不存在: {MODEL_PATH}")
    conn.execute(REGISTER_MODEL_SQL, (MODEL_NAME, MODEL_PATH))
    conn.commit()
    return conn

def execute_sql(sql: str, db_path: str) -> Tuple[Any, int]:
    """在 db_path 数据库上执行 sql，成功返回 (rows, 列数)，失败抛异常"""
    if not sql.strip():
        raise ValueError("空 SQL")
    conn = None
    try:
        conn = _connect_sqlite(db_path)
        cur  = conn.cursor()
        cur.execute("BEGIN")
        cur.execute(sql)
        rows = cur.fetchall()
        col_cnt = len(cur.description)
        cur.execute("ROLLBACK")
        return rows, col_cnt
    finally:
        if conn: conn.close()

def explain_ok(sql: str, db_path: str) -> bool:
    """EXPLAIN 通过返回 True，否则 False"""
    if SKIP_EXPLAIN_PATTERN.search(sql):
        return True
    try:
        _ = execute_sql("EXPLAIN QUERY PLAN " + sql, db_path)
        return True
    except Exception:
        return False

# ----------------------------------------------------------
# 4. 多进程执行包装
# ----------------------------------------------------------
def _worker(idx: int, db_id: str, sql: str, sql_complexity: str,
            timeout: int, db_dir: str):
    db_path = os.path.join(db_dir, db_id, f"{db_id}.sqlite")
    try:
        rows, col_cnt = func_timeout(timeout, execute_sql, args=(sql, db_path))
        return [idx, db_id, sql, sql_complexity, True, col_cnt, len(rows)]
    except (FunctionTimedOut, Exception):
        return [idx, db_id, sql, sql_complexity, False, 0, 0]

def _callback(res):
    idx, db_id, sql, sql_complexity, ok, col_cnt, row_cnt = res
    if ok:
        shared_results.append(dict(
            db_id=db_id, sql=sql, sql_complexity=sql_complexity,
            column_count=col_cnt, rows=row_cnt
        ))

def parallel_execute(sql_infos: List[Dict], db_dir: str,
                     num_cpus: int = 8, timeout: int = 30):
    """并发执行 SQL，过滤运行错误 / 超时"""
    batch = 1024
    chunks = [sql_infos[i:i+batch] for i in range(0, len(sql_infos), batch)]
    for i, part in enumerate(chunks, 1):
        print(f"并行执行进度 {i}/{len(chunks)}")
        with mp.Pool(num_cpus) as pool:
            for idx, info in enumerate(part):
                pool.apply_async(_worker,
                                 args=(idx, info["db_id"], info["sql"],
                                       info["sql_complexity"], timeout, db_dir),
                                 callback=_callback)
            pool.close(); pool.join()
        time.sleep(4)               # 给系统降温

# ----------------------------------------------------------
# 5. 过滤与去重辅助
# ----------------------------------------------------------
def filter_select(sql_infos: List[Dict]) -> List[Dict]:
    """仅保留 SELECT / WITH 开头的查询"""
    out = []
    for info in sql_infos:
        sql = re.sub(r'/\*.*?\*/', '', info["sql"], flags=re.S)   # /* ... */
        sql = re.sub(r'--.*', '', sql)                            # -- ...
        if sql.lower().lstrip().startswith(("select", "with")):
            info["sql"] = sql.strip()
            out.append(info)
    return out

def dedup_by_template(sql_infos: List[Dict]) -> List[Dict]:
    """将常量、字符串、NULL/TRUE/FALSE 抽象掉后去重"""
    def to_tpl(sql: str) -> str:
        pat = r"""
            (?<!\w)'(?:\\.|[^'])*' |       # 'str'
            (?<!\w)"(?:\\.|[^"])*" |       # "str"
            -?\b\d+(\.\d+)?([eE][-+]?\d+)?\b |  # number
            \bNULL\b | \bTRUE\b | \bFALSE\b
        """
        tpl = re.sub(pat, "<v>", sql, flags=re.I | re.X)
        return re.sub(r'\s+', ' ', tpl).lower().strip()
    seen, uniq = set(), []
    for info in sql_infos:
        k = to_tpl(info["sql"])
        if k not in seen:
            seen.add(k)
            uniq.append(info)
    return uniq

def analyze_col_cnt(sql_infos: List[Dict]):
    cnt = {}
    for x in sql_infos:
        cnt[x["column_count"]] = cnt.get(x["column_count"], 0) + 1
    print("列数分布:", cnt)

# ----------------------------------------------------------
# 6. 主流程
# ----------------------------------------------------------
def main(args):
    mp.set_start_method("spawn", force=True)

    # 读取 JSON（数组）
    with open(args.input, "r", encoding="utf-8") as f:
        sql_infos = json.load(f)
    print("原始 SQL 数量:", len(sql_infos))

    # 1. 仅保留 SELECT / CTE
    sql_infos = filter_select(sql_infos)
    print("仅保留 SELECT 后:", len(sql_infos))

    # 2. EXPLAIN 过滤语法错误
    ok = []
    for info in tqdm(sql_infos, desc="EXPLAIN"):
        db_path = os.path.join(args.db_dir, info["db_id"],
                               f"{info['db_id']}.sqlite")
        if explain_ok(info["sql"], db_path):
            ok.append(info)
    sql_infos = ok
    print("去掉语法错误后:", len(sql_infos))

    # 3. 并发真实执行
    global shared_results
    shared_results = mp.Manager().list()
    parallel_execute(sql_infos, args.db_dir,
                     num_cpus=args.cpus, timeout=args.timeout)
    sql_infos = list(shared_results)
    print("去掉运行错误/超时后:", len(sql_infos))
    analyze_col_cnt(sql_infos)

    # 4. 模板级去重
    sql_infos = dedup_by_template(sql_infos)
    print("模板级去重后:", len(sql_infos))
    analyze_col_cnt(sql_infos)

    # 5. 保存结果
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(sql_infos, f, ensure_ascii=False, indent=2)
    print(f"✅ 处理完成，结果已写入 {args.output}")

# ----------------------------------------------------------
# 7. CLI
# ----------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_dir", required=True,
                        help="包含多个 *.sqlite 子目录的根路径")
    parser.add_argument("--input",   required=True,
                        help="输入 JSON 文件（数组，每项含 db_id/sql/complexity）")
    parser.add_argument("--output",  required=True,
                        help="输出 JSON 文件路径")
    parser.add_argument("--cpus",    type=int, default=8,
                        help="并发进程数")
    parser.add_argument("--timeout", type=int, default=30,
                        help="单条 SQL 执行超时时间 (秒)")
    args = parser.parse_args()
    main(args)
