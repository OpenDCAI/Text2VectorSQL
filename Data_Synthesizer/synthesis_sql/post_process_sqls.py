#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
过滤 LLM 生成的 SQL：
- 移除 sqlite-lembed 依赖
- 通过HTTP请求从本地VLLM服务获取嵌入向量
- 采用“先串行预处理，后并行执行”的策略解决多进程数据库锁问题
"""
import os, re, sys, time, json, multiprocessing as mp, traceback
from typing import List, Dict, Tuple, Any

# ### 新增 ### - 引入 requests 库用于HTTP通信
import requests

from tqdm import tqdm
from func_timeout import func_timeout, FunctionTimedOut
import ijson

# ----------------------------------------------------------
# 1. 选择 sqlite3 实现
# ----------------------------------------------------------
try:
    import pysqlite3 as sqlite3
    print(f"✅ 使用 pysqlite3，SQLite 版本: {sqlite3.sqlite_version}")
except ImportError:
    import sqlite3
    print(f"⚠️  使用系统自带 sqlite3，SQLite 版本: {sqlite3.sqlite_version}")

# ----------------------------------------------------------
# 2. 加载向量扩展
# ----------------------------------------------------------
import sqlite_vec

# ----------------------------------------------------------
# 3. 与VLLM服务交互的工具函数
# ----------------------------------------------------------

def get_embedding_from_server(text: str, server_url: str, model_name: str) -> List[float]:
    """
    向你本地的 embedding_server.py 服务发送请求并返回向量。
    """
    # 构建符合 embedding_server.py API 的请求体
    payload = {
        "model": model_name,
        "texts": [text]  # 服务端接收一个文本列表
    }
    try:
        response = requests.post(server_url, json=payload, timeout=60)
        # 如果请求失败 (如 4xx, 5xx 错误), 抛出异常
        response.raise_for_status()
        result = response.json()
        # 从返回的列表中获取第一个（也是唯一一个）嵌入向量
        embedding = result["embeddings"][0]
        return embedding
    except requests.exceptions.RequestException as e:
        print(f"❌ 请求VLLM服务失败: {e}", file=sys.stderr)
        raise
    except (KeyError, IndexError) as e:
        print(f"❌ 解析VLLM响应失败，检查返回的JSON结构: {result}", file=sys.stderr)
        raise

def preprocess_sql(sql: str, server_url: str, model_name: str) -> str:
    """
    使用正则表达式查找所有的lembed()调用，并将其替换为从VLLM获取的真实向量。
    """
    # 正则表达式，用于匹配 lembed('model', "text") 或 lembed("model", 'text')
    # re.DOTALL 使得 . 可以匹配换行符
    lembed_pattern = re.compile(r"lembed\s*\(\s*[^,]+?,\s*(['\"])(.*?)\1\s*\)", re.IGNORECASE | re.DOTALL)

    def replacer(match):
        text_to_embed = match.group(2)
        vector = get_embedding_from_server(text_to_embed, server_url, model_name)
        # 将向量转换为JSON字符串，这是 sqlite-vec 所需的格式
        return "'"+json.dumps(vector)+"'"

    return lembed_pattern.sub(replacer, sql)

# ----------------------------------------------------------
# 4. 数据库连接与执行函数
# ----------------------------------------------------------

def _connect_sqlite(db_path: str) -> sqlite3.Connection:
    """简化的数据库连接函数，只加载sqlite-vec。"""
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    return conn

def execute_sql_simple(sql: str, db_path: str) -> Tuple[Any, int]:
    """一个简化的执行函数，只负责执行已处理好的、不含网络请求的纯SQL。"""
    if not sql.strip():
        raise ValueError("空 SQL")
    conn = None
    try:
        conn = _connect_sqlite(db_path)
        cur  = conn.cursor()
        cur.execute("BEGIN")
        cur.execute(sql)
        rows = cur.fetchall()
        col_cnt = len(cur.description) if cur.description else 0
        cur.execute("ROLLBACK")
        return rows, col_cnt
    except Exception as e:
        print(f"error : {e}")
        print(f"error sql: {sql}")
    finally:
        if conn: conn.close()

# ----------------------------------------------------------
# 5. 为“后处理”阶段设计的多进程包装
# ----------------------------------------------------------

shared_results = None # 全局变量，用于在进程间共享结果

def _worker_processed(idx: int, db_id: str, original_sql: str, processed_sql: str, complexity: str,
                      timeout: int, db_dir: str):
    """为已处理SQL设计的worker，不执行任何网络请求。"""
    db_path = os.path.join(db_dir, db_id, db_id + ".sqlite")
    try:
        args = (processed_sql, db_path)
        rows, col_cnt = func_timeout(timeout, execute_sql_simple, args=args)
        # 返回结果时，我们仍然使用 original_sql，因为它更具可读性
        return [idx, db_id, original_sql, complexity, True, col_cnt, len(rows), None]
    except FunctionTimedOut:
        return [idx, db_id, original_sql, complexity, False, 0, 0, f"DB: {db_id} - TIMEOUT"]
    except Exception:
        error_info = f"DB: {db_id}\nOriginal SQL: {original_sql}\nError: {traceback.format_exc()}"
        return [idx, db_id, original_sql, complexity, False, 0, 0, error_info]

def _callback(res):
    """多进程回调函数，用于收集结果。"""
    _, db_id, original_sql, complexity, ok, col_cnt, row_cnt, error_info = res
    if ok:
        shared_results.append(dict(
            db_id=db_id, sql=original_sql, complexity=complexity,
            column_count=col_cnt, rows=row_cnt
        ))
    elif error_info:
        print("-------------------- WORKER FAILED --------------------", file=sys.stderr)
        print(error_info, file=sys.stderr)
        print("-------------------------------------------------------", file=sys.stderr)

def parallel_execute_processed(sql_infos: List[Dict], db_dir: str, num_cpus: int = 8, timeout: int = 30):
    """为已处理SQL设计的并行执行器。"""
    print(f"开始并行执行 {len(sql_infos)} 条已处理的SQL...")
    with mp.Pool(num_cpus) as pool:
        for idx, info in enumerate(sql_infos):
            pool.apply_async(_worker_processed,
                             args=(idx, info["db_id"], info["sql"], info["processed_sql"],
                                   info["complexity"], timeout, db_dir),
                             callback=_callback)
        pool.close()
        pool.join()

# ----------------------------------------------------------
# 6. 辅助函数 (保持不变)
# ----------------------------------------------------------
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
def post_process_sqls(db_dir: str, output_path: str, llm_json_path: str,
                        server_url: str, model_name: str,
                        num_cpus: int = 8, sql_timeout: int = 30):

    print("--- 配置加载成功 ---")
    print(f"VLLM 服务地址: {server_url}")
    print(f"使用的模型名: {model_name}")
    print(f"数据库目录: {db_dir}")
    print(f"LLM 输入文件: {llm_json_path}")
    print(f"最终输出文件: {output_path}")
    print(f"CPU 核心数: {num_cpus}")
    print(f"SQL 超时 (秒): {sql_timeout}")
    print("--------------------")

    # 安全地设置多进程启动方法
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")

    # --- 阶段一: 加载并进行串行预处理 ---
    llm_resps = load_ndjson(llm_json_path)
    sql_infos = []
    for r in llm_resps:
        sql = parse_response(r["response"])
        if not sql: continue
        db_id = r["db_id"][:-3] if r["db_id"].endswith(".db") else r["db_id"]
        complexity = r.get("complexity") or r["prompt"].split("Ensure the SQL query matches the ")[1].split(" level")[0]
        sql_infos.append(dict(db_id=db_id, sql=sql, complexity=complexity))

    print(f"原始 SQL 数量: {len(sql_infos)}")
    sql_infos = filter_select(sql_infos)
    print(f"仅保留 SELECT 后: {len(sql_infos)}")

    # 在主进程中串行调用VLLM服务，替换lembed()
    processed_sql_infos = []
    for info in tqdm(sql_infos, desc="串行预处理SQL (调用VLLM)"):
        try:
            processed_sql = preprocess_sql(info['sql'], server_url, model_name)
            new_info = info.copy()
            new_info['processed_sql'] = processed_sql  # 存储处理好的SQL
            processed_sql_infos.append(new_info)
        except Exception as e:
            print(f"预处理失败，跳过SQL: {info['sql'][:100]}... Error: {e}", file=sys.stderr)
            
    sql_infos = processed_sql_infos
    print(f"预处理完成，成功处理 {len(sql_infos)} 条SQL")

    if not sql_infos:
        print("没有可执行的SQL，程序退出。")
        # 创建一个空文件并退出
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([], f)
        return

    # --- 阶段二: 并行执行已处理好的SQL ---
    global shared_results
    shared_results = mp.Manager().list()
    
    parallel_execute_processed(sql_infos, db_dir, num_cpus=num_cpus, timeout=sql_timeout)
    
    sql_infos_final = list(shared_results)
    print(f"去掉运行错误/超时后: {len(sql_infos_final)}")
    analyze_col_cnt(sql_infos_final)

    sql_infos_final = dedup_by_template(sql_infos_final)
    print(f"模板级去重后: {len(sql_infos_final)}")
    analyze_col_cnt(sql_infos_final)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sql_infos_final, f, indent=2, ensure_ascii=False)
    print(f"✅ 处理完成，结果写入 {output_path}")

# ----------------------------------------------------------
# 8. CLI - 如何调用
# ----------------------------------------------------------
if __name__ == "__main__":
    # !!!重要!!!: 运行此脚本前，请确保你的 embedding_server.py 服务正在运行!
    
    # --- 请在这里配置你的参数 ---
    DB_DIRECTORY = "sqlite/results/wikipedia_multimodal/vector_databases_wiki"
    LLM_OUTPUT_FILE = "sqlite/results/wikipedia_multimodal/sql_synthesis.json"
    FINAL_OUTPUT_FILE = "sqlite/results/wikipedia_multimodal/synthetic_sqls.json"
    
    # 你的VLLM服务地址和在服务中加载的模型名称
    VLLM_SERVER_URL = "http://127.0.0.1:8000/embed"  # 确保这与你 embedding_server.py 的地址和端口匹配
    MODEL_IN_VLLM = "CLIP-ViT-B-32-laion2B-s34B-b79K" # 必须与你服务 config.yaml 中的 'name' 字段完全匹配

    # 调用主函数
    post_process_sqls(
        db_dir=DB_DIRECTORY,
        output_path=FINAL_OUTPUT_FILE,
        llm_json_path=LLM_OUTPUT_FILE,
        server_url=VLLM_SERVER_URL,
        model_name=MODEL_IN_VLLM,
        num_cpus=8,        # 根据你的机器CPU核心数调整
        sql_timeout=60     # 如果单个SQL查询复杂，可以适当增加超时时间
    )
