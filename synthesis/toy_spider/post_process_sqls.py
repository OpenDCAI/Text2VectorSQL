#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
过滤 LLM 生成的 SQL：
... (docstring a s before) ...
"""
import os, re, sys, time, json, multiprocessing as mp, traceback
from typing import List, Dict, Tuple, Any
# from dotenv import load_dotenv

from tqdm import tqdm
from func_timeout import func_timeout, FunctionTimedOut
import ijson

# --- Load environment variables at the start ---
# load_dotenv()

# ----------------------------------------------------------
# 1. 选择 sqlite3 实现
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
import sqlite_vec
import sqlite_lembed

# ----------------------------------------------------------
# 3. 全局常量与工具
# ----------------------------------------------------------
MODEL_NAME  = "all-MiniLM-L6-v2"
MODEL_PATH = "../../model/all-MiniLM-L6-v2.e4ce9877.q8_0.gguf"
print(f"模型绝对路径: {MODEL_PATH}")


REGISTER_MODEL_SQL = """
INSERT OR IGNORE INTO main.lembed_models (name, model)
VALUES (?, lembed_model_from_file(?))
"""

SKIP_EXPLAIN_PATTERN = re.compile(r'MATCH\s+lembed\(', re.I)

def _connect_sqlite(MODEL_PATH,db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    sqlite_lembed.load(conn)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ 嵌入模型文件不存在: {MODEL_PATH}")
    conn.execute(REGISTER_MODEL_SQL, (MODEL_NAME, MODEL_PATH))
    conn.commit()
    return conn

# ----------------------------------------------------------
# 4. 基础执行函数 (保持不变)
# ----------------------------------------------------------
def execute_sql(MODEL_PATH, sql: str, db_path: str) -> Tuple[Any, int]:
    if not sql.strip():
        raise ValueError("空 SQL")
    conn = None
    try:
        conn = _connect_sqlite(MODEL_PATH,db_path)
        cur  = conn.cursor()
        cur.execute("BEGIN")
        cur.execute(sql)
        rows = cur.fetchall()
        col_cnt = len(cur.description)
        cur.execute("ROLLBACK")
        return rows, col_cnt
    finally:
        if conn: conn.close()

### MODIFIED ###
# 修正 explain_ok 函数签名，使其接收 MODEL_PATH 参数
def explain_ok(MODEL_PATH: str, sql: str, db_path: str) -> bool:
    if SKIP_EXPLAIN_PATTERN.search(sql):
        return True
    try:
        # 现在这里调用的 execute_sql 使用的是传入的 MODEL_PATH 参数
        _ = execute_sql(MODEL_PATH, "EXPLAIN QUERY PLAN " + sql, db_path)
        return True
    except Exception as ex:
        return False

# ----------------------------------------------------------
# 5. 多进程包装 (使用增强诊断的版本)
# ----------------------------------------------------------
def _worker(idx: int, db_id: str, MODEL_PATH:str, sql: str, complexity: str,
            timeout: int, db_dir: str):
    db_path = os.path.join(db_dir, db_id, db_id + ".sqlite")
    try:
        rows, col_cnt = func_timeout(timeout, execute_sql, args=(MODEL_PATH, sql, db_path))
        return [idx, db_id, sql, complexity, True, col_cnt, len(rows), None]
    except FunctionTimedOut:
        return [idx, db_id, sql, complexity, False, 0, 0, f"DB: {db_id} - TIMEOUT"]
    except Exception as e:
        error_info = f"DB: {db_id}\nError: {traceback.format_exc()}"
        return [idx, db_id, sql, complexity, False, 0, 0, error_info]

def _callback(res):
    idx, db_id, sql, complexity, ok, col_cnt, row_cnt, error_info = res
    if ok:
        shared_results.append(dict(
            db_id=db_id, sql=sql, complexity=complexity,
            column_count=col_cnt, rows=row_cnt
        ))
    elif error_info: # Only print if there's an error message
        print("-------------------- WORKER FAILED --------------------", file=sys.stderr)
        print(error_info, file=sys.stderr)
        print("-------------------------------------------------------", file=sys.stderr)

# ... (其他帮助函数保持不变) ...
### MODIFIED ###
# 修正 parallel_execute 函数签名，使其接收并传递 MODEL_PATH
def parallel_execute(MODEL_PATH: str, sql_infos: List[Dict], db_dir: str,
                     num_cpus: int = 8, timeout: int = 30):
    batch = 1024
    chunks = [sql_infos[i:i+batch] for i in range(0, len(sql_infos), batch)]
    for i, part in enumerate(chunks, 1):
        print(f"并行执行进度 {i}/{len(chunks)}")
        with mp.Pool(num_cpus) as pool:
            for idx, info in enumerate(part):
                ### MODIFIED ###
                # 在 args 中按 _worker 的参数顺序添加 MODEL_PATH
                pool.apply_async(_worker,
                                 args=(idx, info["db_id"], MODEL_PATH, info["sql"],
                                       info["complexity"], timeout, db_dir),
                                 callback=_callback)
            pool.close(); pool.join()
        time.sleep(6)

def parse_response(text: str) -> str:
    blocks = re.findall(r"```sql\s*(.*?)\s*```", text, re.S | re.I)
    return blocks[-1].strip() if blocks else ""

def filter_select(sql_infos: List[Dict]) -> List[Dict]:
    out = []
    for info in sql_infos:
        sql = re.sub(r'/\*.*?\*/', '', info["sql"], flags=re.S)
        sql = re.sub(r'--.*', '', sql)
        if sql.lower().lstrip().startswith(("select", "with")):
            info["sql"] = sql.strip()
            out.append(info)
    return out

def dedup_by_template(sql_infos: List[Dict]) -> List[Dict]:
    def to_tpl(sql: str) -> str:
        pat = r"""
            (?<!\w)'(?:\\.|[^'])*' | (?<!\w)"(?:\\.|[^"])*" |
            -?\b\d+(\.\d+)?([eE][-+]?\d+)?\b |
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
def post_process_sqls(MODEL_PATH, db_dir,output_path,llm_json_path,num_cpus=8,sql_timeout=30):
    required_vars = {"DB_DIR": db_dir, "LLM_JSON_PATH": llm_json_path, "OUTPUT_JSON_PATH": output_path}
    missing_vars = [key for key, value in required_vars.items() if not value or not os.path.exists(value.rsplit('/', 1)[0])]
    if missing_vars:
        raise ValueError(f"❌ 错误: 缺少或不正确的配置: {', '.join(missing_vars)}")

    print("--- 配置加载成功 ---")
    print(f"数据库绝对路径: {db_dir}")
    print(f"LLM 输入文件: {llm_json_path}")
    print(f"输出文件: {output_path}")
    print(f"CPU 核心数: {num_cpus}")
    print(f"SQL 超时 (秒): {sql_timeout}")
    print("--------------------")

    mp.set_start_method("spawn", force=True)

    llm_resps = load_ndjson(llm_json_path)
    sql_infos = []
    for r in llm_resps:
        sql = parse_response(r["response"])
        if not sql:
            continue
        db_id = r["db_id"][:-3] if r["db_id"].endswith(".db") else r["db_id"]
        complexity = r.get("complexity") or \
                     r["prompt"].split("Ensure the SQL query matches the ")[1] \
                                 .split(" level")[0]
        sql_infos.append(dict(db_id=db_id, sql=sql, complexity=complexity))

    print("原始 SQL 数量:", len(sql_infos))
    sql_infos = filter_select(sql_infos)
    print("仅保留 SELECT 后:", len(sql_infos))

    ok = []
    for info in tqdm(sql_infos, desc="EXPLAIN"):
        db_path_for_explain = os.path.join(db_dir, info["db_id"], info["db_id"] + ".sqlite")
        ### MODIFIED ###
        # 在调用 explain_ok 时传递 MODEL_PATH
        if explain_ok(MODEL_PATH, info["sql"], db_path_for_explain):
            ok.append(info)
    sql_infos = ok
    print("去掉语法错误后:", len(sql_infos))

    global shared_results
    shared_results = mp.Manager().list()
    ### MODIFIED ###
    # 在调用 parallel_execute 时传递 MODEL_PATH
    parallel_execute(MODEL_PATH, sql_infos, db_dir, num_cpus=num_cpus, timeout=sql_timeout)
    sql_infos = list(shared_results)
    print("去掉运行错误/超时后:", len(sql_infos))
    analyze_col_cnt(sql_infos)

    sql_infos = dedup_by_template(sql_infos)
    print("模板级去重后:", len(sql_infos))
    analyze_col_cnt(sql_infos)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sql_infos, f, indent=2, ensure_ascii=False)
    print(f"✅ 处理完成，结果写入 {output_path}")

# ----------------------------------------------------------
# 8. CLI
# ----------------------------------------------------------
# if __name__ == "__main__":
#     main()
