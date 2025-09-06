import re
import uuid
from textwrap import dedent

class SQLiteToClickHouseConverter:
    """
    自动将使用 sqlite-vec 和 sqlite-lembed 的复杂 SQLite 查询
    转换为 ClickHouse 的等效向量搜索查询。

    该转换器支持：
    - 多个 Common Table Expressions (CTE)。
    - 对多个连接（JOIN）的表同时进行向量搜索。
    - 遵循“先过滤，后连接”的原则，以最高效地执行查询。
    """

    def __init__(self, sqlite_query, distance_function='L2Distance'):
        """
        初始化转换器。

        Args:
            sqlite_query (str): 原始的 SQLite 查询语句。
            distance_function (str): 在 ClickHouse 中使用的距离函数，
                                     例如 'L2Distance', 'cosineDistance'。
        """
        self.sqlite_query = sqlite_query.strip()
        self.distance_function = distance_function
        # 存储所有向量搜索的信息
        self.vector_searches = {}  # {search_id: info}
        # 存储原始CTE的信息
        self.original_ctes = {}  # {cte_name: body}
        # 存储表名和别名的映射
        self.table_alias_map = {} # {alias: table_name}


    def _get_placeholder_embedding(self, text: str) -> str:
        """
        为给定的文本生成一个向量占位符。
        在实际应用中，您需要在此处调用嵌入模型服务并将结果填充进去。
        """
        clean_text = text.replace('*/', '* /') #避免破坏注释
        return f"/* embedding_of('{clean_text}') */ [ ... ] -- 请在此处填充真实向量"

    def _parse_query_structure(self):
        """
        解析整个查询，分离出顶层的 WITH 子句和主查询。
        注意：这个基于正则表达式的解析器主要针对结构良好的SQL，
              对于极端复杂的嵌套或格式可能不够健壮。
        """
        # 提取表名和别名
        from_join_pattern = re.compile(
            r'\b(FROM|JOIN)\s+([\w\.]+)\s+(?:AS\s+)?(\w+)\b', re.IGNORECASE)
        for match in from_join_pattern.finditer(self.sqlite_query):
            _, table_name, alias = match.groups()
            self.table_alias_map[alias] = table_name

        # 分离 CTEs
        with_clause_match = re.match(r'\s*WITH\s+', self.sqlite_query, re.IGNORECASE)
        if not with_clause_match:
            return self.sqlite_query # 没有 CTE

        # 寻找 WITH 子句的结束位置
        query_after_with = self.sqlite_query[with_clause_match.end():]
        open_parens = 0
        last_cte_end_index = 0
        in_string = False

        for i, char in enumerate(query_after_with):
            if char == "'":
                in_string = not in_string
            if in_string:
                continue
            
            if char == '(':
                open_parens += 1
            elif char == ')':
                open_parens -= 1
                if open_parens == 0:
                    # 检查此括号后是否跟着另一个CTE定义或主查询
                    next_segment = query_after_with[i+1:].lstrip()
                    if not next_segment.startswith(','):
                        last_cte_end_index = i + 1
                        break
        
        if last_cte_end_index == 0: # 解析失败
             return self.sqlite_query

        cte_definitions_str = query_after_with[:last_cte_end_index]
        main_query = query_after_with[last_cte_end_index:].strip()

        # 使用正则表达式分割 CTE 定义
        cte_defs = re.split(r',\s*(?=[a-zA-Z0-9_]+\s+AS\s*\()', cte_definitions_str)

        for cte_def in cte_defs:
            match = re.match(r'(\w+)\s+AS\s+\((.*)\)', cte_def.strip(), re.IGNORECASE | re.DOTALL)
            if match:
                name, content = match.groups()
                self.original_ctes[name.strip()] = content.strip()

        return main_query

    def _find_and_process_vector_searches(self, query_block: str) -> tuple[str, dict]:
        """
        在给定的查询块中查找、处理并替换所有向量搜索相关的子句。
        """
        modified_block = query_block
        local_searches = {}

        # 1. 查找所有 MATCH 和 k=N 子句
        match_pattern = re.compile(
            r"(?P<table_alias>\w+)\.(?P<column_name>\w+)\s+MATCH\s+"
            r"lembed\('(?P<model>.*?)',\s*'(?P<text>.*?)'\)",
            re.IGNORECASE | re.DOTALL
        )
        k_pattern = re.compile(
            r"(?:(?P<table_alias>\w+)\.)?k\s*=\s*(?P<k_value>\d+)",
            re.IGNORECASE
        )

        found_matches = {m.group('table_alias'): m for m in match_pattern.finditer(modified_block)}
        found_ks = {m.group('table_alias') or list(found_matches.keys())[0]: m for m in k_pattern.finditer(modified_block) if len(found_matches) > 0}

        # 2. 整合信息并从块中移除子句
        for alias, match in found_matches.items():
            if alias not in found_ks:
                raise ValueError(f"在表 '{alias}' 上的向量搜索缺少 'k=N' 约束。")

            k_match = found_ks[alias]
            search_id = str(uuid.uuid4())
            search_info = {
                "match": match.groupdict(),
                "k": k_match.groupdict(),
            }
            self.vector_searches[search_id] = search_info
            local_searches[alias] = {"id": search_id}

            # 从 WHERE 子句中移除 MATCH 和 k
            for clause in [match.group(0), k_match.group(0)]:
                # 尝试移除 'AND clause', 'clause AND', 'clause'
                modified_block = re.sub(r'\bAND\s+' + re.escape(clause) + r'\b', '', modified_block, flags=re.IGNORECASE | re.DOTALL)
                modified_block = re.sub(r'\b' + re.escape(clause) + r'\s+AND\b', '', modified_block, flags=re.IGNORECASE | re.DOTALL)
                modified_block = re.sub(r'\b' + re.escape(clause) + r'\b', '', modified_block, flags=re.IGNORECASE | re.DOTALL)

        # 3. 替换 FROM/JOIN 中的表名为过滤后的 CTE
        def replace_table_with_cte(m):
            keyword, table_name, alias = m.groups()
            if alias in local_searches:
                return f"{keyword} {alias}_filtered AS {alias}"
            return m.group(0)
        
        from_join_pattern = re.compile(r'\b(FROM|JOIN)\s+([\w\.]+)\s+(?:AS\s+)?(\w+)\b', re.IGNORECASE)
        modified_block = from_join_pattern.sub(replace_table_with_cte, modified_block)

        # 4. 替换 'distance' 列的引用
        for alias in local_searches:
            dist_pattern = re.compile(r'\b' + re.escape(alias) + r'\.distance\b')
            modified_block = dist_pattern.sub(f'{alias}.distance_{alias}', modified_block)

        # 5. 清理空的 WHERE 子句
        modified_block = re.sub(r'\bWHERE\s*(\bAND\b\s*)+', 'WHERE ', modified_block, flags=re.IGNORECASE)
        modified_block = re.sub(r'\bWHERE\s*$', '', modified_block, flags=re.IGNORECASE).strip()

        return modified_block, local_searches

    def convert(self) -> str:
        """
        执行完整的转换过程。
        """
        # 1. 解析原始查询结构
        main_query = self._parse_query_structure()

        # 2. 分别处理每个查询块（CTE 和主查询）
        modified_blocks = {}
        all_local_searches = {}

        for name, body in self.original_ctes.items():
            modified_body, local_searches = self._find_and_process_vector_searches(body)
            modified_blocks[name] = modified_body
            all_local_searches.update(local_searches)

        modified_main, local_searches = self._find_and_process_vector_searches(main_query)
        modified_blocks['main'] = modified_main
        all_local_searches.update(local_searches)

        if not self.vector_searches:
            return "-- 无需转换：未检测到向量搜索语法。\n" + self.sqlite_query

        # 3. 构建最终的 ClickHouse 查询
        output_parts = ["WITH"]
        
        # 3a. 创建引用向量的 WITH 子句
        ref_vec_clauses = []
        vec_alias_map = {}
        for i, (search_id, search_info) in enumerate(self.vector_searches.items()):
            vec_alias = f"ref_vec_{i}"
            vec_alias_map[search_id] = vec_alias
            embedding_text = search_info['match']['text']
            clause = f"    {self._get_placeholder_embedding(embedding_text)} AS {vec_alias}"
            ref_vec_clauses.append(clause)
        
        output_parts.append(",\n".join(ref_vec_clauses))

        # 3b. 创建过滤用的 CTE
        filtering_ctes = []
        for alias, local_search in all_local_searches.items():
            search_id = local_search['id']
            info = self.vector_searches[search_id]
            table_name = self.table_alias_map.get(alias, alias) # 如果找不到映射，则回退到别名
            
            cte = dedent(f"""\
                {alias}_filtered AS (
                    SELECT
                        *,
                        {self.distance_function}({info['match']['column_name']}, {vec_alias_map[search_id]}) AS distance_{alias}
                    FROM {table_name}
                    ORDER BY distance_{alias}
                    LIMIT {info['k']['k_value']}
                )""")
            filtering_ctes.append(cte)
        
        if filtering_ctes:
            output_parts.append(",\n\n".join(filtering_ctes))

        # 3c. 添加修改后的原始 CTE
        modified_original_ctes = []
        for name, body in self.original_ctes.items():
            modified_body = "    " + modified_blocks[name].replace("\n", "\n    ")
            cte = f"{name} AS (\n{modified_body}\n)"
            modified_original_ctes.append(cte)
        
        if modified_original_ctes:
            output_parts.append(",\n\n".join(modified_original_ctes))

        # 4. 组装最终查询
        final_with_clause = ",\n\n".join(filter(None, output_parts))
        final_query = f"{final_with_clause}\n\n{modified_blocks['main']};"
        
        return final_query

# --- 使用示例 ---
if __name__ == '__main__':
    # 复杂的 SQLite 查询示例 (来自用户问题)
    complex_sqlite_query = """
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
         c.distance as city_distance
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
       ORDER BY
         city_distance
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

    # 简单的 SQLite 查询示例
    simple_sqlite_query = """
    SELECT
      m.Musical_ID,
      m.Name,
      a.Actor_ID,
      m.distance
    FROM musical m
    JOIN actor a ON m.Musical_ID = a.Musical_ID
    WHERE m.Category_embedding MATCH lembed('all-MiniLM-L6-v2', 'Outstanding performance in musical theater')
      AND m.k = 3
      AND a.age BETWEEN 20 AND 40
      AND m.Musical_ID > 1
    ORDER BY m.distance;
    """
    
    print("="*40)
    print(">>> 正在转换复杂查询...")
    print("="*40)
    
    # 创建转换器实例并执行转换
    # converter_complex = SQLiteToClickHouseConverter(complex_sqlite_query)
    # clickhouse_query_complex = converter_complex.convert()
    
    # print("--- 原始 SQLite 查询 ---\n")
    # print(complex_sqlite_query.strip())
    # print("\n\n--- 转换后的 ClickHouse 查询 ---\n")
    # print(clickhouse_query_complex)

    print("\n" + "="*40)
    print(">>> 正在转换简单查询...")
    print("="*40)

    converter_simple = SQLiteToClickHouseConverter(simple_sqlite_query)
    clickhouse_query_simple = converter_simple.convert()

    print("--- 原始 SQLite 查询 ---\n")
    print(simple_sqlite_query.strip())
    print("\n\n--- 转换后的 ClickHouse 查询 ---\n")
    print(clickhouse_query_simple)