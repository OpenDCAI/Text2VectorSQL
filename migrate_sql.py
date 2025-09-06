import re
import argparse

def translate_sqlite_vec_to_clickhouse(sqlite_sql: str, distance_func: str = 'cosineDistance') -> str:
    """
    将包含 sqlite-vec 向量搜索语法的 SQL 查询自动转换为 ClickHouse 兼容的语法。

    Args:
        sqlite_sql (str): 原始的 SQLite 查询语句。
        distance_func (str, optional): 用于 ClickHouse 的距离计算函数。
                                       默认为 'cosineDistance'，可以是 'L2Distance' 等。

    Returns:
        str: 转换后的 ClickHouse 查询语句。
    """
    # 移除 SQL 语句中的换行符和多余空格，便于正则处理
    sql = re.sub(r'\s+', ' ', sqlite_sql.strip())
    # 移除多余 ;
    sql = sql.rstrip(';')

    # 1. 判断是否为向量搜索查询
    if "MATCH lembed" not in sql:
        print("INFO: 未检测到 'MATCH lembed' 关键字，返回原始 SQL。")
        return sqlite_sql

    # 2. 提取向量搜索的关键信息 (MATCH lembed 子句)
    # 正则表达式: 匹配 "table.column_embedding MATCH lembed('model', 'text')"
    match_pattern = re.compile(
        r'(\w+\.)?(\w+_embedding)\s+MATCH\s+lembed\s*\(\s*\'(.*?)\'\s*,\s*\'(.*?)\'\s*\)',
        re.IGNORECASE
    )
    match_info = match_pattern.search(sql)

    if not match_info:
        raise ValueError("无法解析 'MATCH lembed' 子句。请检查语法。")

    table_prefix = match_info.group(1) or ''
    embedding_col = table_prefix + match_info.group(2)
    model_name = match_info.group(3)
    embedding_text = match_info.group(4)

    # 3. 提取 Top-K 限制 (k=N 或 LIMIT N)
    limit = None
    # 优先匹配 "AND k = N"
    k_pattern = re.compile(r'AND\s+(?:\w+\.)?k\s*=\s*(\d+)', re.IGNORECASE)
    k_match = k_pattern.search(sql)
    if k_match:
        limit = k_match.group(1)
        # 从原始 SQL 中移除 k=N 条件，避免干扰后续的 WHERE 子句提取
        sql = k_pattern.sub('', sql)
    else:
        # 如果没有 k=N，则查找 LIMIT N
        limit_pattern = re.compile(r'LIMIT\s+(\d+)', re.IGNORECASE)
        limit_match = limit_pattern.search(sql)
        if limit_match:
            limit = limit_match.group(1)
            sql = limit_pattern.sub('', sql)

    if not limit:
        raise ValueError("向量搜索查询必须包含 'k=N' 或 'LIMIT N' 约束。")

    # 4. 从 SQL 中移除已处理的 MATCH 子句，得到一个 "干净" 的基础查询
    sql = match_pattern.sub('', sql)

    # 5. 提取 SQL 的主要部分 (SELECT, FROM, WHERE)
    select_pattern = re.compile(r'SELECT\s+(.*?)\s+FROM', re.IGNORECASE | re.DOTALL)
    select_match = select_pattern.search(sql)
    if not select_match:
        raise ValueError("无法解析 SELECT ... FROM 结构。")
    
    select_cols_str = select_match.group(1).strip()
    
    # 提取 FROM 和 WHERE 之间的部分
    # 我们找到 FROM 的起始位置和 WHERE 的起始位置
    from_start_index = sql.lower().find(' from ') + len(' from ')
    where_start_index = sql.lower().find(' where ')
    
    if where_start_index != -1:
        from_clause = sql[from_start_index:where_start_index].strip()
        # 提取 WHERE 之后的所有剩余条件
        # 注意：需要清理掉可能由移除 MATCH 和 k=N 留下的多余 AND
        remaining_where_clause = sql[where_start_index + len(' where '):].strip()
        remaining_where_clause = re.sub(r'^\s*AND\s+', '', remaining_where_clause).strip()
        remaining_where_clause = re.sub(r'\s+AND\s+AND\s+', ' AND ', remaining_where_clause).strip()
        remaining_where_clause = re.sub(r'\s+AND\s*$', '', remaining_where_clause).strip()
    else:
        from_clause = sql[from_start_index:].strip()
        remaining_where_clause = ""
        
    # 如果原始 SELECT 中包含了隐式的 distance 列，我们将其移除，因为 ClickHouse 会显式计算
    select_cols_list = [col.strip() for col in select_cols_str.split(',')]
    select_cols_list = [col for col in select_cols_list if 'distance' not in col.lower()]
    cleaned_select_cols = ', '.join(select_cols_list)


    # 6. 组装 ClickHouse 查询
    
    # WITH 子句：使用一个假设的 embed 函数来表示向量化过程
    with_clause = f"WITH embed('{model_name}', '{embedding_text}') AS reference_vector"

    # SELECT 子句：包含原始列和新的距离计算列
    new_select_clause = f"SELECT {cleaned_select_cols}, {distance_func}({embedding_col}, reference_vector) AS distance"

    # WHERE 子句
    where_clause = ""
    if remaining_where_clause:
        where_clause = f"WHERE {remaining_where_clause}"

    # ORDER BY 和 LIMIT 子句
    order_by_clause = "ORDER BY distance ASC"
    limit_clause = f"LIMIT {limit}"

    # 最终组合
    clickhouse_sql = (
        f"{with_clause}\n"
        f"{new_select_clause}\n"
        f"FROM {from_clause}\n"
        f"{where_clause}\n"
        f"{order_by_clause}\n"
        f"{limit_clause};"
    ).replace('\n\n', '\n') # 清理空行

    return clickhouse_sql

# --- 主程序入口 ---
def main():
    parser = argparse.ArgumentParser(description="将 SQLite-vec SQL 查询转换为 ClickHouse 语法。")
    parser.add_argument(
        "sqlite_query",
        type=str,
        help="要转换的 SQLite 查询语句字符串。"
    )
    parser.add_argument(
        "--dist",
        type=str,
        default="cosineDistance",
        choices=['cosineDistance', 'L2Distance'],
        help="在 ClickHouse 中使用的距离函数。"
    )
    args = parser.parse_args()

    print("--- 原始 SQLite 查询 ---")
    print(args.sqlite_query)
    print("\n" + "="*30 + "\n")

    try:
        clickhouse_query = translate_sqlite_vec_to_clickhouse(args.sqlite_query, distance_func=args.dist)
        print("--- 转换后的 ClickHouse 查询 ---")
        print(clickhouse_query)
    except ValueError as e:
        print(f"转换错误: {e}")


if __name__ == '__main__':
    # 为了直接在脚本中运行示例，可以取消下面的注释
    example_sql = "SELECT COUNT(m.Musical_ID) FROM musical m JOIN actor a ON m.Musical_ID = a.Musical_ID WHERE m.Category_embedding MATCH lembed('all-MiniLM-L6-v2', 'Outstanding performance in musical theater') AND m.k = 3 AND a.age BETWEEN 20 AND 40 AND m.Year >= 2000 AND m.Result = 'Won';"
    another_example = "SELECT Name, Year, distance FROM musical WHERE Category_embedding MATCH lembed('all-MiniLM-L6-v2', 'Best Choreography') AND Year > 1990 and k = 5"

    try:
        print("--- 示例 1 ---")
        print(f"原始SQL: {example_sql}\n")
        ch_sql = translate_sqlite_vec_to_clickhouse(example_sql)
        print(f"转换后SQL:\n{ch_sql}")
        print("\n" + "="*30 + "\n")
        
        print("--- 示例 2 ---")
        print(f"原始SQL: {another_example}\n")
        ch_sql_2 = translate_sqlite_vec_to_clickhouse(another_example)
        print(f"转换后SQL:\n{ch_sql_2}")
        
    except ValueError as e:
        print(f"转换错误: {e}")

    # 命令行调用逻辑
    # 如果希望通过命令行参数运行，请保持 main() 的原始结构并从命令行调用。
    # main()