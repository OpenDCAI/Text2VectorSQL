#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
过滤 LLM 生成的 SQL：
1. 仅保留 SELECT/CTE 查询
2. 剔除语法错误 / 执行错误 / 超时
3. 去重
4. 统计并写入结果

用法示例：
    python post_process_sqls_new.py \
        --db_dir ../bird_vectorization/results/vector_databases_bird \
        --llm_json ./results/sql_synthesis.json \
        --output   ./results/synthetic_sqls.json \
        --cpus 10 \
        --timeout 60
"""
import os, re, sys, time, json, argparse, multiprocessing as mp, traceback
from typing import List, Dict, Tuple, Any

from tqdm import tqdm
from func_timeout import func_timeout, FunctionTimedOut
import ijson

# ----------------------------------------------------------
# 1. 选择 sqlite3 实现（优先 pysqlite3，自带最新版 SQLite）
# ----------------------------------------------------------
try:
    import pysqlite3 as sqlite3
    print(f"✅ 使用 pysqlite3，SQLite 版本: {sqlite3.sqlite_version}")
except ImportError:
    import sqlite3
    print(f"⚠️  使用系统自带 sqlite3，SQLite 版本: {sqlite3.sqlite_version}")
    print("   如果后续仍出现 SQL logic error，可执行: pip install pysqlite3-binary\n")

# ----------------------------------------------------------
# 2. 加载向量 / 嵌入扩展
# ----------------------------------------------------------
import sqlite_vec               # pip install sqlite-vec>=0.5.0
import sqlite_lembed            # pip install sqlite-lembed>=0.2.3

# ----------------------------------------------------------
# 3. 全局常量与工具
# ----------------------------------------------------------
MODEL_NAME  = "all-MiniLM-L6-v2"
MODEL_PATH  = "../all-MiniLM-L6-v2.e4ce9877.q8_0.gguf"      # 请改成你的本地路径

REGISTER_MODEL_SQL = """
INSERT OR IGNORE INTO main.lembed_models (name, model)
VALUES (?, lembed_model_from_file(?))
"""

# 旧版扩展在 EXPLAIN 阶段会对含有 MATCH lembed(...) 的查询报错。
SKIP_EXPLAIN_PATTERN = re.compile(r'MATCH\s+lembed\(', re.I)

def _connect_sqlite(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    sqlite_lembed.load(conn)
    # 注册嵌入模型（第一次会真正插入，之后 IGNORE）
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ 嵌入模型文件不存在: {MODEL_PATH}")
    conn.execute(REGISTER_MODEL_SQL, (MODEL_NAME, MODEL_PATH))
    conn.commit()
    return conn

# ----------------------------------------------------------
# 4. 基础执行函数
# ----------------------------------------------------------
def execute_sql(sql: str, db_path: str) -> Tuple[Any, int]:
    """
    在 db_path 上执行 sql；成功返回 (rows, 列数)，失败抛异常
    """
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
    """
    若 sql 通过 EXPLAIN 检查返回 True。
    遇到扩展限制可通过 SKIP_EXPLAIN_PATTERN 跳过。
    """
    if SKIP_EXPLAIN_PATTERN.search(sql):
        return True
    try:
        _ = execute_sql("EXPLAIN QUERY PLAN " + sql, db_path)
        return True
    except Exception as ex:
        # 打开注释可调试
        # print("EXPLAIN 失败:", ex, "\nSQL:", sql)
        return False

# ----------------------------------------------------------
# 5. 多进程包装
# ----------------------------------------------------------
def _worker(idx: int, db_id: str, sql: str, complexity: str,
            timeout: int, db_dir: str):
    db_path = os.path.join(db_dir, db_id, db_id + ".sqlite")
    try:
        rows, col_cnt = func_timeout(timeout, execute_sql, args=(sql, db_path))
        return [idx, db_id, sql, complexity, True, col_cnt, len(rows)]
    except FunctionTimedOut:
        return [idx, db_id, sql, complexity, False, 0, 0]
    except Exception:
        return [idx, db_id, sql, complexity, False, 0, 0]

def _callback(res):
    idx, db_id, sql, complexity, ok, col_cnt, row_cnt = res
    if ok:
        shared_results.append(dict(
            db_id=db_id, sql=sql, complexity=complexity,
            column_count=col_cnt, rows=row_cnt
        ))

def parallel_execute(sql_infos: List[Dict], db_dir: str,
                     num_cpus: int = 8, timeout: int = 30):
    """
    多进程并发执行 SQL，剔除超时 / 执行错误
    """
    batch = 1024
    chunks = [sql_infos[i:i+batch] for i in range(0, len(sql_infos), batch)]
    for i, part in enumerate(chunks, 1):
        print(f"并行执行进度 {i}/{len(chunks)}")
        with mp.Pool(num_cpus) as pool:
            for idx, info in enumerate(part):
                pool.apply_async(_worker,
                                 args=(idx, info["db_id"], info["sql"],
                                       info["complexity"], timeout, db_dir),
                                 callback=_callback)
            pool.close(); pool.join()
        time.sleep(6)        # 给系统降温

# ----------------------------------------------------------
# 6. 若干帮助函数
# ----------------------------------------------------------
def parse_response(text: str) -> str:
    """提取最后一段 ```sql ... ```"""
    blocks = re.findall(r"```sql\s*(.*?)\s*```", text, re.S | re.I)
    return blocks[-1].strip() if blocks else ""

def filter_select(sql_infos: List[Dict]) -> List[Dict]:
    out = []
    for info in sql_infos:
        sql = re.sub(r'/\*.*?\*/', '', info["sql"], flags=re.S)   # /* … */
        sql = re.sub(r'--.*', '', sql)
        if sql.lower().lstrip().startswith(("select", "with")):
            info["sql"] = sql.strip()
            out.append(info)
    return out

def dedup_by_template(sql_infos: List[Dict]) -> List[Dict]:
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
            seen.add(k); uniq.append(info)
    return uniq

def analyze_col_cnt(sql_infos):
    cnt = {}
    for x in sql_infos:
        cnt[x["column_count"]] = cnt.get(x["column_count"], 0) + 1
    print("列数分布:", cnt)

def load_ndjson(path: str) -> List[Dict]:
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for obj in tqdm(ijson.items(f, 'item'), desc="加载 LLM 输出"):
            data.append(obj)
    return data

# ----------------------------------------------------------
# 7. 主流程
# ----------------------------------------------------------
def main(args):
    mp.set_start_method("spawn", force=True)

    # 读取 LLM 输出
    llm_resps = load_ndjson(args.llm_json)
    sql_infos = []
    for r in llm_resps:
        sql = parse_response(r["response"])
        if not sql:
            continue
        db_id = r["db_id"][:-3] if r["db_id"].endswith(".db") else r["db_id"]
        # 这里的复杂度解析可按自己的 prompt 来改
        complexity = r.get("complexity") or \
                     r["prompt"].split("Ensure the SQL query matches the ")[1] \
                                 .split(" level")[0]
        sql_infos.append(dict(db_id=db_id, sql=sql, complexity=complexity))

    print("原始 SQL 数量:", len(sql_infos))

    # 1. 仅保留 SELECT / CTE
    sql_infos = filter_select(sql_infos)
    print("仅保留 SELECT 后:", len(sql_infos))

    # 2. EXPLAIN 过滤语法错误
    ok = []
    for info in tqdm(sql_infos, desc="EXPLAIN"):
        if explain_ok(info["sql"],
                      os.path.join(args.db_dir, info["db_id"],
                                   info["db_id"] + ".sqlite")):
            ok.append(info)
    sql_infos = ok
    print("去掉语法错误后:", len(sql_infos))

    # 3. 多进程执行，过滤运行错误 / 超时
    global shared_results
    shared_results = mp.Manager().list()
    parallel_execute(sql_infos, args.db_dir,
                     num_cpus=args.cpus, timeout=args.timeout)
    sql_infos = list(shared_results)
    print("去掉超时后:", len(sql_infos))
    analyze_col_cnt(sql_infos)

    # 4. 去重
    sql_infos = dedup_by_template(sql_infos)
    print("模板级去重后:", len(sql_infos))
    analyze_col_cnt(sql_infos)

    # 5. 保存
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(sql_infos, f, indent=2, ensure_ascii=False)
    print(f"✅ 处理完成，结果写入 {args.output}")

# ----------------------------------------------------------
# 8. CLI
# ----------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_dir",   required=True,
                        help="包含多个 *.sqlite 子目录的根路径")
    parser.add_argument("--llm_json", required=True,
                        help="LLM 生成结果 (ndjson)")
    parser.add_argument("--output",   required=True,
                        help="输出 JSON 文件")
    parser.add_argument("--cpus",     type=int, default=8,
                        help="并发进程数")
    parser.add_argument("--timeout",  type=int, default=30,
                        help="单条 SQL 执行超时时间 (秒)")
    args = parser.parse_args()
    main(args)
