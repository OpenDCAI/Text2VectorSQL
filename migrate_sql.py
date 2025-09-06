import re
import uuid

# --- 正则表达式常量 ---

# 匹配 "table.column_embedding MATCH lembed('model', 'text')"
# 增加了对表别名的捕获
MATCH_PATTERN = re.compile(
    r'\b(\w+)\.(\w+_embedding)\s+MATCH\s+lembed\s*\(\s*\'(.*?)\'\s*,\s*\'(.*?)\'\s*\)',
    re.IGNORECASE | re.DOTALL
)

# 匹配 "AND table.k = N" 或 "AND k = N"
K_PATTERN = re.compile(r'AND\s+(?:(\w+)\.)?k\s*=\s*(\d+)', re.IGNORECASE)

# 匹配 CTE: WITH alias AS (...)
CTE_PATTERN = re.compile(r'\bWITH\b\s+(.+?)\s+AS\s+\((.*)\)\s*(SELECT.*)', re.IGNORECASE | re.DOTALL)

def process_select_block(sql_block: str, distance_func: str = 'cosineDistance') -> str:
    """
    处理一个独立的 SELECT...FROM...WHERE 块，核心转换逻辑在此。
    """
    
    # 1. 查找此块中所有的向量搜索请求
    vector_searches = {}
    
    # 使用 finditer 查找所有 MATCH 子句
    for match in MATCH_PATTERN.finditer(sql_block):
        alias = match.group(1).lower()
        if alias not in vector_searches:
            vector_searches[alias] = []
        
        search_info = {
            "full_match": match.group(0),
            "alias": alias,
            "column": match.group(2),
            "model": match.group(3),
            "text": match.group(4),
            "k": None,
            "distance_alias": f"distance_{alias}_{uuid.uuid4().hex[:4]}" # 保证唯一
        }
        vector_searches[alias].append(search_info)

    # 如果没有向量搜索，直接返回原始块
    if not vector_searches:
        return sql_block

    # 2. 为每个找到的向量搜索匹配其 'k' 值
    cleaned_sql = sql_block
    for k_match in K_PATTERN.finditer(sql_block):
        k_alias = k_match.group(1)
        k_value = k_match.group(2)
        
        # 找到这个 k 对应的 search_info 并赋值
        target_alias = k_alias.lower() if k_alias else None
        if target_alias and target_alias in vector_searches:
            # 假定每个表只有一个MATCH
            vector_searches[target_alias][0]['k'] = k_value
        # 如果k没有别名，尝试匹配到唯一的那个search
        elif not target_alias and len(vector_searches) == 1:
            key = list(vector_searches.keys())[0]
            vector_searches[key][0]['k'] = k_value
            
    # 清理原始SQL中的 k=N 子句
    cleaned_sql = K_PATTERN.sub('', cleaned_sql)

    # 3. 解析 FROM/JOIN 子句，找到原始表名和别名
    from_clause_match = re.search(r'\bFROM\b(.*?)(?:\bWHERE\b|\bGROUP BY\b|\bORDER BY\b|\bLIMIT\b|\)$|$)', cleaned_sql, re.IGNORECASE | re.DOTALL)
    if not from_clause_match:
        raise ValueError("无法解析 FROM 子句。")
    
    from_clause = from_clause_match.group(1).strip()
    
    # 4. 重写 FROM 子句：将原始表替换为向量搜索子查询
    rewritten_from_clause = from_clause
    with_clauses = []

    table_pattern = re.compile(r'\b(city|farm_competition|competition_record|farm)\s+(?:AS\s+)?(\w+)\b', re.IGNORECASE)
    
    # 存储原始表名和别名的映射
    table_alias_map = {m.group(2).lower(): m.group(1) for m in table_pattern.finditer(from_clause)}

    for alias, searches in vector_searches.items():
        if alias not in table_alias_map:
            continue # 如果找不到别名对应的表，暂时跳过
        
        original_table_name = table_alias_map[alias]
        
        for search in searches:
            if not search['k']:
                raise ValueError(f"未找到与别名 '{alias}' 匹配的 'k=N' 约束。")
            
            # 创建 WITH 子句
            ref_vector_name = f"ref_vector_{alias}"
            with_clauses.append(f"WITH embed('{search['model']}', '{search['text']}') AS {ref_vector_name}")
            
            # 创建向量搜索子查询
            subquery = (
                f"(SELECT *, {distance_func}({search['column']}, {ref_vector_name}) AS {search['distance_alias']} "
                f"FROM {original_table_name} "
                f"ORDER BY {search['distance_alias']} ASC "
                f"LIMIT {search['k']})"
            )
            
            # 在 FROM 子句中用子查询替换原表
            # 使用更精确的正则替换，避免错误替换
            table_ref_pattern = re.compile(fr'\b{original_table_name}\s+(?:AS\s+)?{alias}\b', re.IGNORECASE)
            rewritten_from_clause = table_ref_pattern.sub(f"{subquery} AS {alias}", rewritten_from_clause)
            
            # 清理原始SQL中的 MATCH 子句
            cleaned_sql = cleaned_sql.replace(search['full_match'], '')
            
            # 替换 SELECT 和 ORDER BY 中的 'distance' 引用
            select_dist_pattern = re.compile(fr'\b{alias}\.distance\b(\s+AS\s+\w+)?', re.IGNORECASE)
            cleaned_sql = select_dist_pattern.sub(f"{search['distance_alias']}\\1", cleaned_sql)

    # 5. 清理 WHERE 子句中可能残留的 'AND'
    where_match = re.search(r'(\bWHERE\b)(.*)', cleaned_sql, re.IGNORECASE | re.DOTALL)
    if where_match:
        where_keyword = where_match.group(1)
        where_content = where_match.group(2)
        
        # 移除开头、结尾和连续的 AND
        where_content = re.sub(r'^\s*AND\s+', '', where_content.strip()).strip()
        where_content = re.sub(r'\s+AND\s+AND\s+', ' AND ', where_content).strip()
        where_content = re.sub(r'\s+AND\s*$', '', where_content).strip()

        if where_content:
            cleaned_sql = cleaned_sql[:where_match.start(2)] + " " + where_content
        else: # 如果 WHERE 子句变空，则移除整个 WHERE
            cleaned_sql = cleaned_sql[:where_match.start(1)] + " " + cleaned_sql[where_match.end(2):]

    # 6. 重新组装整个查询
    # 将 WITH 子句放在最前面
    final_with_clause = "\n".join(with_clauses)
    
    # 找到原始 SELECT 关键字，将重写的 FROM 子句插入
    final_sql_parts = re.split(r'\bFROM\b', cleaned_sql, 1, re.IGNORECASE)
    final_sql = final_sql_parts[0] + "FROM " + rewritten_from_clause
    
    # 找到原始 FROM 子句结束的位置，拼接剩余部分
    original_from_end = from_clause_match.end(0)
    remaining_sql = sql_block[original_from_end:]
    
    # 对剩余部分进行清理
    remaining_sql = K_PATTERN.sub('', remaining_sql)
    for alias, searches in vector_searches.items():
        for search in searches:
            remaining_sql = remaining_sql.replace(search['full_match'], '')
            select_dist_pattern = re.compile(fr'\b{alias}\.distance\b(\s+AS\s+\w+)?', re.IGNORECASE)
            remaining_sql = select_dist_pattern.sub(f"{search['distance_alias']}\\1", remaining_sql)
            
    # 再次清理 WHERE 子句
    if ' WHERE ' in remaining_sql.upper():
        where_parts = re.split(r'\bWHERE\b', remaining_sql, 1, re.IGNORECASE)
        where_content = where_parts[1]
        where_content = re.sub(r'^\s*AND\s+', '', where_content.strip()).strip()
        where_content = re.sub(r'\s+AND\s+AND\s+', ' AND ', where_content).strip()
        where_content = re.sub(r'\s+AND\s*$', '', where_content).strip()
        if where_content:
            remaining_sql = where_parts[0] + " WHERE " + where_content
        else:
            remaining_sql = where_parts[0]

    # 重新拼接
    final_sql = final_sql_parts[0] + " FROM " + rewritten_from_clause + " " + remaining_sql
    
    return f"{final_with_clause}\n{final_sql}".strip()

def translate_sql_recursively(sql: str) -> str:
    """
    递归地解析和转换 SQL 语句，支持 CTE 和子查询。
    """
    sql = sql.strip()
    
    # 检查并处理 CTE
    cte_match = CTE_PATTERN.match(sql)
    if cte_match:
        cte_name = cte_match.group(1)
        cte_body = cte_match.group(2)
        main_query = cte_match.group(3)
        
        # 递归转换 CTE 的主体
        translated_cte_body = translate_sql_recursively(cte_body)
        
        # 递归转换主查询
        translated_main_query = translate_sql_recursively(main_query)
        
        return f"WITH {cte_name} AS (\n{translated_cte_body}\n)\n{translated_main_query}"
    
    # 检查并处理嵌套子查询 (简化版：处理 FROM/JOIN 中的子查询)
    # 这是一个复杂的问题，这里做一个简化的示例逻辑
    # 一个完整的方案需要一个真正的SQL解析器
    # for subquery_match in re.finditer(r'\(\s*(SELECT .*?)\s*\)', sql, re.IGNORECASE | re.DOTALL):
    #     subquery_sql = subquery_match.group(1)
    #     translated_subquery = translate_sql_recursively(subquery_sql)
    #     sql = sql.replace(subquery_sql, translated_subquery)

    # 如果不是 CTE 或已知子查询模式，则作为独立的 SELECT 块处理
    return process_select_block(sql)

# --- 主程序 ---
if __name__ == '__main__':
    complex_sqlite_sql = """
    WITH RankedCities AS (
      SELECT
        c.City_ID,
        c.Official_Name,
        c.Status,
        c.Population,
        f.Farm_ID,
        f.Year,
        cr.Rank,
        fc.Theme,
        fc.Hosts,
        c.distance as city_distance,
        fc.distance as theme_distance
      FROM
        city c
      JOIN
        farm_competition fc ON c.City_ID = fc.Host_city_ID
      JOIN
        competition_record cr ON fc.Competition_ID = cr.Competition_ID
      JOIN
        farm f ON cr.Farm_ID = f.Farm_ID
      WHERE
        c.Status_embedding MATCH lembed('all-MiniLM-L6-v2', 'Major city status with high population density') AND c.k = 5
        AND fc.Theme_embedding MATCH lembed('all-MiniLM-L6-v2', 'Annual agricultural showcase with international participants') AND fc.k = 3
        AND f.Year > 2010
      ORDER BY
        city_distance, theme_distance
      LIMIT 10
    )
    SELECT
      Official_Name,
      MAX(Population) AS Max_Population
    FROM
      RankedCities
    GROUP BY
      Official_Name
    ORDER BY
      Max_Population DESC;
    """

    print("--- 原始复杂 SQLite 查询 ---")
    print(complex_sqlite_sql)
    print("\n" + "="*40 + "\n")

    try:
        clickhouse_query = translate_sql_recursively(complex_sqlite_sql)
        print("--- 转换后的 ClickHouse 查询 ---")
        print(clickhouse_query)
    except ValueError as e:
        print(f"\n转换时发生错误: {e}")