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

def analyze_col_cnt(sql, db_path):
    """
    分析 SQL 查询中的列数，包括别名处理和复杂查询解析
    
    Args:
        sql (str): SQL 查询语句
        db_path (str): 数据库路径
        
    Returns:
        int: 列数，如果出错返回 -1
    """
    try:
        # 移除注释和多余空白
        cleaned_sql = re.sub(r'--.*?$', '', sql, flags=re.MULTILINE)
        cleaned_sql = re.sub(r'/\*.*?\*/', '', cleaned_sql, flags=re.DOTALL)
        cleaned_sql = ' '.join(cleaned_sql.split())
        
        # 连接数据库
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        try:
            # 使用 EXPLAIN QUERY PLAN 获取查询计划而不执行
            cursor.execute(f"EXPLAIN QUERY PLAN {cleaned_sql}")
            
            # 尝试获取列信息 - 使用 LIMIT 0 避免实际执行
            limited_sql = f"SELECT * FROM ({cleaned_sql}) LIMIT 0"
            cursor.execute(limited_sql)
            
            # 获取列描述
            columns = cursor.description
            if columns:
                col_count = len(columns)
                logger.debug(f"通过 LIMIT 0 查询检测到 {col_count} 列")
                return col_count
            
        except sqlite3.Error as e:
            logger.warning(f"数据库查询失败: {e}")
            # 如果数据库查询失败，使用静态分析
            return _static_analyze_columns(cleaned_sql)
            
        finally:
            conn.close()
            
    except Exception as e:
        logger.error(f"列数分析失败: {e}")
        return _static_analyze_columns(sql)

def _static_analyze_columns(sql):
    """
    静态分析 SQL 中的列数（备用方法）
    
    Args:
        sql (str): SQL 查询语句
        
    Returns:
        int: 估计的列数
    """
    try:
        # 移除子查询中的 SELECT 来避免误计算
        main_select = _extract_main_select(sql)
        
        # 查找 SELECT 和 FROM 之间的内容
        select_match = re.search(r'SELECT\s+(.*?)\s+FROM', main_select, re.IGNORECASE | re.DOTALL)
        if not select_match:
            return -1
            
        select_clause = select_match.group(1).strip()
        
        # 处理 SELECT * 的情况
        if select_clause.strip() == '*':
            # 尝试从 FROM 子句推断表结构（简化处理）
            logger.debug("检测到 SELECT *，返回估计列数")
            return 5  # 默认估计值
            
        # 分割列，处理括号内的内容
        columns = _split_columns_smart(select_clause)
        
        # 过滤掉空的列
        columns = [col.strip() for col in columns if col.strip()]
        
        logger.debug(f"静态分析检测到 {len(columns)} 列: {columns}")
        return len(columns)
        
    except Exception as e:
        logger.error(f"静态列数分析失败: {e}")
        return -1

def _extract_main_select(sql):
    """
    提取主 SELECT 语句，排除子查询
    
    Args:
        sql (str): 完整的 SQL 语句
        
    Returns:
        str: 主 SELECT 语句
    """
    # 简化处理：找到第一个 SELECT 到第一个分号或字符串结尾
    sql = sql.strip()
    if sql.endswith(';'):
        sql = sql[:-1]
    
    # 找到主要的 SELECT 语句结构
    # 这里可以进一步改进以处理更复杂的嵌套查询
    return sql

def _split_columns_smart(select_clause):
    """
    智能分割 SELECT 子句中的列，处理函数调用和表达式
    
    Args:
        select_clause (str): SELECT 子句内容
        
    Returns:
        list: 分割后的列列表
    """
    columns = []
    current_col = ""
    paren_depth = 0
    in_quotes = False
    quote_char = None
    
    i = 0
    while i < len(select_clause):
        char = select_clause[i]
        
        # 处理引号
        if char in ('"', "'", '`') and not in_quotes:
            in_quotes = True
            quote_char = char
        elif char == quote_char and in_quotes:
            in_quotes = False
            quote_char = None
            
        # 如果在引号内，直接添加字符
        if in_quotes:
            current_col += char
        elif char == '(':
            paren_depth += 1
            current_col += char
        elif char == ')':
            paren_depth -= 1
            current_col += char
        elif char == ',' and paren_depth == 0:
            # 找到列分隔符
            columns.append(current_col.strip())
            current_col = ""
        else:
            current_col += char
            
        i += 1
    
    # 添加最后一列
    if current_col.strip():
        columns.append(current_col.strip())
    
    return columns

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

def fix_sql_syntax_errors(sql, error_msg):
    """
    根据错误信息修复 SQL 语法错误
    
    Args:
        sql (str): 原始 SQL 语句
        error_msg (str): 错误信息
        
    Returns:
        str: 修复后的 SQL 语句
    """
    fixed_sql = sql.strip()
    
    try:
        # 确保 SQL 以分号结尾
        if not fixed_sql.endswith(';'):
            fixed_sql += ';'
            
        # 修复常见的语法错误
        error_lower = error_msg.lower()
        
        # 处理表名或列名错误
        if 'no such table' in error_lower:
            table_match = re.search(r"no such table:\s*(['\"]?)(\w+)\1", error_lower)
            if table_match:
                missing_table = table_match.group(2)
                logger.info(f"检测到缺失表: {missing_table}")
                # 这里可以添加表名映射逻辑
                
        elif 'no such column' in error_lower:
            col_match = re.search(r"no such column:\s*(['\"]?)(\w+\.?\w*)\1", error_lower)
            if col_match:
                missing_col = col_match.group(2)
                logger.info(f"检测到缺失列: {missing_col}")
                # 尝试修复列名
                fixed_sql = _fix_column_names(fixed_sql, missing_col)
                
        # 修复 GROUP BY 错误
        elif 'group by' in error_lower and 'aggregate' in error_lower:
            fixed_sql = _fix_group_by_issues(fixed_sql)
            
        # 修复引号问题
        elif 'unrecognized token' in error_lower:
            fixed_sql = _fix_quote_issues(fixed_sql)
            
        # 修复 JOIN 语法
        elif 'join' in error_lower:
            fixed_sql = _fix_join_syntax(fixed_sql)
            
        # 修复括号不匹配
        elif 'syntax error' in error_lower:
            fixed_sql = _fix_parentheses(fixed_sql)
            
        logger.debug(f"SQL 修复尝试: {sql[:100]}... -> {fixed_sql[:100]}...")
        return fixed_sql
        
    except Exception as e:
        logger.error(f"SQL 修复过程出错: {e}")
        return sql

def _fix_column_names(sql, missing_col):
    """修复列名错误"""
    try:
        # 移除表前缀尝试
        if '.' in missing_col:
            simple_col = missing_col.split('.')[-1]
            sql = sql.replace(missing_col, simple_col)
            
        # 常见列名映射
        column_mappings = {
            'id': ['ID', 'Id', 'identifier', 'key'],
            'name': ['Name', 'title', 'label'],
            'date': ['Date', 'created_at', 'timestamp'],
            'count': ['Count', 'total', 'number']
        }
        
        for standard, variants in column_mappings.items():
            if missing_col.lower() == standard:
                # 尝试替换为可能的变体
                for variant in variants:
                    if variant in sql:
                        sql = sql.replace(missing_col, variant)
                        break
                        
        return sql
    except:
        return sql

def _fix_group_by_issues(sql):
    """修复 GROUP BY 相关问题"""
    try:
        # 查找 SELECT 子句中的聚合函数
        select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql, re.IGNORECASE | re.DOTALL)
        if not select_match:
            return sql
            
        select_clause = select_match.group(1)
        
        # 查找非聚合列
        non_agg_cols = []
        agg_functions = ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN', 'GROUP_CONCAT']
        
        columns = _split_columns_smart(select_clause)
        for col in columns:
            col_upper = col.upper()
            is_aggregate = any(func in col_upper for func in agg_functions)
            if not is_aggregate and not col.strip() == '*':
                # 提取列名（去除别名）
                col_name = col.split(' AS ')[0].strip()
                col_name = re.sub(r'\s+(AS\s+)?\w+$', '', col_name, flags=re.IGNORECASE).strip()
                if col_name:
                    non_agg_cols.append(col_name)
        
        # 如果有非聚合列但没有 GROUP BY，添加 GROUP BY
        if non_agg_cols and 'GROUP BY' not in sql.upper():
            group_by_clause = f" GROUP BY {', '.join(non_agg_cols)}"
            # 在 ORDER BY 之前或 LIMIT 之前或语句结尾添加
            if 'ORDER BY' in sql.upper():
                sql = re.sub(r'\s+ORDER\s+BY', group_by_clause + ' ORDER BY', sql, flags=re.IGNORECASE)
            elif 'LIMIT' in sql.upper():
                sql = re.sub(r'\s+LIMIT', group_by_clause + ' LIMIT', sql, flags=re.IGNORECASE)
            else:
                sql = sql.rstrip(';') + group_by_clause + ';'
                
        return sql
    except:
        return sql

def _fix_quote_issues(sql):
    """修复引号问题"""
    try:
        # 修复单引号和双引号混用
        # 将所有字符串字面量统一使用单引号
        sql = re.sub(r'"([^"]*)"', r"'\1'", sql)
        
        # 修复反引号（用于标识符）
        sql = re.sub(r'`([^`]*)`', r'\1', sql)
        
        return sql
    except:
        return sql

def _fix_join_syntax(sql):
    """修复 JOIN 语法问题"""
    try:
        # 确保 JOIN 前有适当的关键字
        sql = re.sub(r'\bJOIN\b', ' JOIN ', sql, flags=re.IGNORECASE)
        sql = re.sub(r'\s+JOIN\s+', ' JOIN ', sql, flags=re.IGNORECASE)
        
        # 修复 ON 子句
        sql = re.sub(r'\bON\b', ' ON ', sql, flags=re.IGNORECASE)
        sql = re.sub(r'\s+ON\s+', ' ON ', sql, flags=re.IGNORECASE)
        
        return sql
    except:
        return sql

def _fix_parentheses(sql):
    """修复括号不匹配问题"""
    try:
        # 统计括号
        open_count = sql.count('(')
        close_count = sql.count(')')
        
        if open_count > close_count:
            # 缺少右括号
            sql = sql.rstrip(';') + ')' * (open_count - close_count) + ';'
        elif close_count > open_count:
            # 多余的右括号，移除末尾的
            diff = close_count - open_count
            for _ in range(diff):
                last_paren = sql.rfind(')')
                if last_paren != -1:
                    sql = sql[:last_paren] + sql[last_paren+1:]
                    
        return sql
    except:
        return sql

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
