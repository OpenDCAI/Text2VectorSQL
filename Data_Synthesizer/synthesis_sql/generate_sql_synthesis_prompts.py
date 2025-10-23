#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate SQL-synthesis prompts for each database in `db_path`.

与旧版相比的主要修正：
1. compute_embedding_column / obtain_insert_statements 传入正确的
   *.sqlite 文件绝对路径，保证能统计到 *_embedding 列。
2. 没有 *_embedding 列时也会至少生成 1 条 prompt，避免输出文件为空。
3. 抽样列值字符串拼接方式修正。

新功能：
- 增加数据库判空逻辑，只为非空数据库生成 prompts。
- 统计并报告空数据库的数量。
"""
import json, os, random, sqlite3, traceback, re
from typing import List, Dict, Any
from pathlib import Path

import sqlite_vec, sqlite_lembed
import numpy as np
from tqdm import tqdm


# --------------------------- 模板 & 常量 --------------------------- #
sql_func_template = '''
### SQL Functions
You may consider one or more of the following SQL functions while generating the query:
{sql_funcs}
Important tips:
Except for the functions listed above, you may use any other functions as long as they conform to the syntax of the database engine.
'''

rich_semantic_column_sample_template = '''
### rich semantic column sample
You are provided with sample data for all semantically rich columns across all tables in the database. When generating a predicate (i.e., a WHERE clause) for a vector matching query on a specific column, you must refer to the corresponding examples for that particular column. Do not directly copy this reference data into your answer. Your task is to generate a new predicate that is similar in format and meaning to the provided samples for that column.

sample:
{rich_semantic_column_sample}
'''

# ---- 复杂度描述 ---- #
simple_criterion = '''**Criteria:**
Simple SQL queries may satisfy one or more of the following criteria:
- Simple queries should select data from a single table only.
- Basic aggregate functions are permitted, such as `COUNT`, `SUM`, `AVG`, `MIN`, `MAX`.
- No joins are allowed; the query must operate on a single table.
'''
moderate_criterion = '''**Criteria:**
Moderate SQL queries may satisfy one or more of the following criteria:
- Involves table joins, such as `JOIN`, `INNER JOIN`, `LEFT JOIN`, `CROSS JOIN`, etc.
- Includes subqueries within the `SELECT` or `WHERE` clauses.
- Utilizes aggregate functions alongside a `GROUP BY` clause.
- Contains complex `WHERE` conditions, including `IN`, `BETWEEN`, `LIKE`.
- Incorporate a `HAVING` clause to filter aggregated results.
'''
complex_criterion = '''**Criteria:**
Complex SQL queries may satisfy one or more of the following criteria:
- Contains complex nested subqueries.
- Utilizes multiple types of joins, including self-joins.
- Includes window functions, such as `ROW_NUMBER`, `RANK`, etc.
- Uses Common Table Expressions (CTEs) for improved readability.
'''
highly_complex_criterion = '''**Criteria:**
Highly complex SQL queries may satisfy one or more of the following criteria:
- Includes multiple Common Table Expressions (CTEs) for readability.
- Utilizes recursive CTEs, advanced window functions, UNION/UNION ALL 等。
'''

# ---- vec 查询复杂度描述 ---- #
simple_vec_criterion   = '''**Criteria:** Basic vector similarity search on a single table.'''
moderate_vec_criterion = '''**Criteria:** Vector search + simple joins / filters.'''
complex_vec_criterion  = '''**Criteria:** Vector search combined with CTEs / hybird search.'''
highly_complex_vec_criterion = '''**Criteria:** Multi-stage or recursive vector search with advanced analytics.'''


# =========================== 数据库判空函数 (新增) =========================== #
def is_db_empty(db_path: Path) -> bool:
    """
    根据新标准检查 SQLite 数据库是否为空：
    一个数据库被视为空，当且仅当它的所有用户表都包含 0 行数据。

    参数:
        db_path (Path): SQLite 数据库文件的路径。

    返回:
        bool: 如果数据库为空则返回 True，否则返回 False。
              对于 0 字节文件或损坏的文件也返回 True。
    """
    # 如果文件大小为0，直接视为空数据库
    if not db_path.exists() or db_path.stat().st_size == 0:
        return True

    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) # 以只读模式打开
        con.enable_load_extension(True)
        sqlite_vec.load(con)
        cursor = con.cursor()
        
        # 1. 获取所有用户表的列表 (排除 sqlite 系统表)
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        tables = [row[0] for row in cursor.fetchall()]

        # 2. 如果数据库中没有任何用户表，也视为空
        if not tables:
            con.close()
            return True

        # 3. 遍历每个表，检查其行数
        for table_name in tables:
            # 使用双引号确保表名中的特殊字符被正确处理
            query = f'SELECT COUNT(*) FROM "{table_name}"'
            cursor.execute(query)
            row_count = cursor.fetchone()[0]
            
            # 4. 只要发现任何一个表有数据，就立刻判断该数据库不为空
            if row_count > 0:
                con.close()
                return False
        
        # 5. 如果循环正常结束，说明所有表都是空的
        con.close()
        return True
        
    except sqlite3.Error as e:
        # 如果文件无法作为数据库打开或查询（例如已损坏），则报告警告并视为空
        print(f"警告: 无法查询 '{db_path}'。错误: {e}。将此文件视为空。")
        return True


# =========================== DB 辅助函数 =========================== #
def _open_connection(db_file: str) -> sqlite3.Connection:
    try:
        conn = sqlite3.connect(db_file)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        sqlite_lembed.load(conn)
        return conn
    except Exception as e:
        print(f"db_file:{db_file}")
        print(e)
        traceback.print_exc()


def contains_virtual_table(statements):
    """检查 schema 里是否包含 vec0 虚拟表等"""
    if isinstance(statements, str):
        statements = [statements]
    patterns = [
        r'\bvirtual\b', r'\bvec0\b',
        r'_embedding\b', r'\bfloat\['
    ]
    for stmt in statements:
        if not stmt:
            continue
        for pat in patterns:
            if re.search(pat, stmt, re.I):
                return True
    return False


def obtain_db_schema(db_file: str):
    conn = _open_connection(db_file)
    cur  = conn.cursor()
    cur.execute("SELECT name, sql FROM sqlite_master WHERE type='table';")
    rows = cur.fetchall()
    cur.close(); conn.close()
    table_names, create_sqls = zip(*rows) if rows else ([], [])
    return list(table_names), list(create_sqls)


# -------- 1) 统计 *_embedding 列 -------- #
def compute_embedding_column(db_file: str) -> Dict[str, Any]:
    conn = _open_connection(db_file)
    cur  = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tbls = [r[0] for r in cur.fetchall()]

    summary = {"total_num": 0, "tables": {}}

    for tbl in tbls:
        try:
            cur.execute(f'PRAGMA table_info("{tbl}")')
            cols = [r[1] for r in cur.fetchall()]     # column name
            bases = []
            for col in cols:
                if col.endswith("_embedding"):
                    base = col[:-10]
                    if base in cols:
                        bases.append(base)
            if bases:
                summary["tables"][tbl] = bases
                summary["total_num"] += len(bases)
        except Exception:
            traceback.print_exc()

    cur.close(); conn.close()
    return summary

# -------- 1) 统计 *image_embedding 列 -------- #
def compute_image_column(db_file: str) -> Dict[str, Any]:
    conn = _open_connection(db_file)
    cur  = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tbls = [r[0] for r in cur.fetchall()]

    summary = {"total_num": 0, "tables": {}}

    for tbl in tbls:
        try:
            cur.execute(f'PRAGMA table_info("{tbl}")')
            cols = [r[1] for r in cur.fetchall()]     # column name
            bases = []
            for col in cols:
                if col.endswith("_embedding"):
                    base = col[:-10]
                    if 'image' in base.lower():
                        bases.append(base)
            if bases:
                summary["tables"][tbl] = bases
                summary["total_num"] += len(bases)
        except Exception:
            traceback.print_exc()

    cur.close(); conn.close()
    return summary

# -------- 2) 抽样列值 -------- #
def obtain_insert_statements(db_file: str,
                             embedding_summary: Dict[str, Any],
                             row_num: int = 2) -> Dict[str, Dict[str, List[Any]]]:
    if not embedding_summary or "tables" not in embedding_summary:
        return {}

    conn = _open_connection(db_file)
    cur  = conn.cursor()
    result: Dict[str, Dict[str, List[Any]]] = {}

    for tbl, bases in embedding_summary["tables"].items():
        tbl_dict = {}
        for col in bases:
            try:
                cur.execute(f'SELECT "{col}" FROM "{tbl}" LIMIT ?', (row_num,))
                tbl_dict[col] = [r[0] for r in cur.fetchall()]
            except Exception:
                print(f"error in col:{col}, table:{tbl}")
                traceback.print_exc()
        if tbl_dict:
            result[tbl] = tbl_dict

    cur.close(); conn.close()
    return result


# ---------------------- 辅助：写大 JSON --------------------- #
def write_large_json(data: List[Dict], output_path: str, chunk_size: int = 500):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('[')
        if data:
            # 为了避免首行逗号，单独处理第一个元素
            json.dump(data[0], f, ensure_ascii=False, indent=2)
            # 从第二个元素开始循环
            for i in range(1, len(data)):
                f.write(',\n')
                json.dump(data[i], f, ensure_ascii=False, indent=2)
        f.write(']')

def main_generate_sql_synthesis_prompts(db_path="./results/vector_databases_toy",prompt_tpl_path="./prompt_templates/sql_synthesis_prompt.txt",functions_path="./prompt_templates/sqlite_funcs.json",output_dir="./prompts",output_name="sql_synthesis_prompts.json",embedding_model="all-MiniLM-L6-v2",sql_num=6):
    # random.seed(42)

    # 目录配置 --------------------------------------------------
    prompt_tpl    = open(prompt_tpl_path,encoding='utf-8').read()
    functions     = json.load(open(functions_path,encoding='utf-8'))

    # 输出目录 --------------------------------------------------
    prompts_dir   = output_dir
    output_path   = os.path.join(prompts_dir, output_name)
    os.makedirs(prompts_dir, exist_ok=True)

    # ----------------------------------------------------------
    prompts: List[Dict] = []
    db_names = os.listdir(db_path)
    empty_db_count = 0 # <--- 新增：空数据库计数器

    for db_name in tqdm(db_names, desc="Scanning databases"):
        if not os.path.isdir(os.path.join(db_path, db_name)):
            continue
        try:
            db_file_path_str = os.path.join(db_path, db_name, f"{db_name}.sqlite")
            db_file_final_path_str = os.path.join(db_path, db_name, f"{db_name}_final.sqlite")

            db_file_to_check = db_file_path_str
            if os.path.exists(db_file_final_path_str):
                db_file_to_check = db_file_final_path_str
            
            # --- 新增逻辑：检查数据库是否为空 ---
            if is_db_empty(Path(db_file_to_check)):
                print(f"数据库 '{db_name}' 为空，已跳过。")
                empty_db_count += 1
                continue # 跳过当前循环，不生成 prompt
            # ---------------------------------

            table_names, create_sqls = obtain_db_schema(db_file_to_check)

            # 根据是否包含 vec0 / embedding 列切换 criterion
            if contains_virtual_table(create_sqls):
                complexity2criterion = {
                    "Simple":   simple_vec_criterion,
                    "Moderate": moderate_vec_criterion,
                    "Complex":  complex_vec_criterion,
                    "Highly Complex": highly_complex_vec_criterion
                }
            else:
                complexity2criterion = {
                    "Simple":   simple_criterion,
                    "Moderate": moderate_criterion,
                    "Complex":  complex_criterion,
                    "Highly Complex": highly_complex_criterion
                }

            # 统计 embedding 列
            summary = compute_embedding_column(db_file_to_check)
            # 抽样列值
            tbl2values = obtain_insert_statements(db_file_to_check, summary, row_num=2)
            
            # support image_embedding
            image_summary = compute_image_column(db_file_to_check)

            total_rich_cols = summary.get("total_num", 0) + image_summary.get("total_num", 0)

            loop_times = max(1, sql_num * total_rich_cols) # 确保即使没有 rich_cols 也至少生成一个 prompt

            for _ in range(loop_times):
                complexity = random.choice(
                    ["Simple", "Moderate", "Complex", "Highly Complex"])

                # ---- 组装 rich_semantic_column_sample ---- #
                rich_samples: List[str] = []
                for tbl in table_names:
                    col2vals = tbl2values.get(tbl, {})
                    for col, vals in col2vals.items():
                        for v in vals:
                            # 确保值中的双引号被正确转义
                            v_escaped = str(v).replace('"', '""')
                            rich_samples.append(
                                f'INSERT INTO {tbl}({col}) VALUES ("{v_escaped}");')

                if rich_samples:
                    if len(rich_samples) > 4:
                        rich_samples = random.sample(rich_samples, 4)
                    db_value_prompt = rich_semantic_column_sample_template.format(
                        rich_semantic_column_sample="\n\n".join(rich_samples))
                else:
                    db_value_prompt = ""

                # ---- SQL function 片段 ---- #
                func_num = random.randint(0, 2)
                if func_num == 0:
                    sql_func_prompt = ("### SQL Functions\n"
                                       "You can use any function supported "
                                       "by the database engine.")
                else:
                    sampled = random.sample(functions, func_num)
                    sql_funcs = ""
                    for idx, func in enumerate(sampled):
                        sql_funcs += f"Function {idx+1}:\n{func.strip()}\n"
                    sql_func_prompt = sql_func_template.format(sql_funcs=sql_funcs)

                column_count = np.random.geometric(0.6)

                prompt = prompt_tpl.format(
                    schema_str="\n\n".join(create_sqls),
                    sql_function_prompt=sql_func_prompt.strip(),
                    db_value_prompt=db_value_prompt.strip(),
                    complexity=complexity,
                    criterion=complexity2criterion[complexity].strip(),
                    db_engine="SQLite",
                    column_count=int(column_count),
                    db_extension="SQLite-vec and sqlite-lembed",
                    embedding_model=embedding_model
                )

                prompts.append({"prompt": prompt, "db_id": db_name})

        except Exception as e:
            print(f"[Error] {db_name}: {e}")
            traceback.print_exc()

    # ---------------------- 写文件 ---------------------- #
    if prompts:
        write_large_json(prompts, output_path, chunk_size=500)
        print(f"✅ 生成完成，共 {len(prompts)} 条 prompts，写入 {output_path}")
    else:
        print("未生成任何 prompts，可能所有数据库都为空或处理失败。")

    # --- 新增逻辑：打印最终统计结果 ---
    print(f"\n📊 统计报告:")
    print(f" - 在 '{db_path}' 中共扫描到 {len(db_names)} 个条目。")
    print(f" - 发现 {empty_db_count} 个空数据库。")
    print(f" - 为 {len(db_names) - empty_db_count} 个非空数据库生成了 prompts。")
    # ---------------------------------

# =========================== 主流程 =========================== #
if __name__ == "__main__":
    main_generate_sql_synthesis_prompts()
