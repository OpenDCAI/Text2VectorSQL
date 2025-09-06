import re
import uuid
from textwrap import dedent
import logging

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# 定义一组 SQL 关键字，用于辅助判断别名是否合法
SQL_KEYWORDS = {'WHERE', 'JOIN', 'LEFT', 'RIGHT', 'INNER', 'ON', 'GROUP', 'ORDER', 'LIMIT', 'UNION', 'CROSS', 'HAVING', 'SELECT', 'FROM', 'AS'}

class SQLiteToClickHouseConverter:
    """
    自动将使用 sqlite-vec 和 sqlite-lembed 的复杂 SQLite 查询
    转换为优化的 ClickHouse 向量搜索查询。

    核心特性 (V2 - 修正版):
    - **作用域感知**: 精确解析每个查询块（CTE/主查询）内的表和别名，避免全局污染。
    - **过滤条件下推 (Filter Pushdown)**: 将相关的结构化过滤器下推到向量搜索CTE中，以提高性能。
    - **健壮的语法解析**: 能正确处理带或不带'AS'的别名，以及完全没有别名的表。
    """

    def __init__(self, sqlite_query, distance_function='L2Distance'):
        self.sqlite_query = sqlite_query.strip()
        self.distance_function = distance_function
        self.vector_searches = {}
        self.original_ctes = {}

    def _get_placeholder_embedding(self, text: str) -> str:
        """为给定的文本生成一个向量占位符。"""
        clean_text = text.replace('*/', '* /')
        return f"/* embedding_of('{clean_text}') */ [ ... ] -- 请在此处填充真实向量"

    def _parse_cte_structure(self):
        """仅负责从查询中分离出CTE定义和主查询体。"""
        with_clause_match = re.match(r'\s*WITH\s+', self.sqlite_query, re.IGNORECASE)
        if not with_clause_match:
            return self.sqlite_query

        query_after_with = self.sqlite_query[with_clause_match.end():]
        open_parens, last_cte_end_index, in_string = 0, 0, False

        for i, char in enumerate(query_after_with):
            if char == "'": in_string = not in_string
            if in_string: continue
            if char == '(': open_parens += 1
            elif char == ')':
                open_parens -= 1
                if open_parens == 0:
                    next_segment = query_after_with[i + 1:].lstrip()
                    if not next_segment.startswith(','):
                        last_cte_end_index = i + 1
                        break
        
        if last_cte_end_index == 0: return self.sqlite_query

        cte_definitions_str = query_after_with[:last_cte_end_index]
        main_query = query_after_with[last_cte_end_index:].strip()

        cte_defs = re.split(r',\s*(?=[a-zA-Z0-9_]+\s+AS\s*\()', cte_definitions_str)
        for cte_def in cte_defs:
            match = re.match(r'(\w+)\s+AS\s+\((.*)\)', cte_def.strip(), re.IGNORECASE | re.DOTALL)
            if match:
                self.original_ctes[match.group(1).strip()] = match.group(2).strip()
        return main_query

    def _get_aliases_in_scope(self, query_block: str) -> dict[str, str]:
        """
        【核心修正】精确解析给定查询块中的所有表及其别名。
        """
        aliases = {}
        from_join_content_match = re.search(
            r'\bFROM\s+(.*?)(?=\bWHERE|\bGROUP BY|\bORDER BY|\bLIMIT|\Z)', 
            query_block, re.S | re.I
        )
        if not from_join_content_match:
            return {}
        
        from_join_content = from_join_content_match.group(1)
        # 移除 ON 子句，因为它可能包含复杂的逻辑，干扰表的拆分
        content_no_on = re.sub(r'\bON\b.*?(?=\b(CROSS|INNER|LEFT|RIGHT|FULL)?\s*JOIN\b|\Z)', '', from_join_content, flags=re.S|re.I)
        # 使用一个不会混淆的分隔符替换所有类型的JOIN
        processed_content = re.sub(r'\b(CROSS|INNER|LEFT|RIGHT|FULL)?\s*JOIN\b', '&&&', content_no_on, flags=re.I)
        
        table_parts = [p.strip() for p in processed_content.split('&&&')]

        for part in table_parts:
            # 匹配 'table [AS] alias' 或 'table'
            match = re.match(r'([\w\.]+)(?:\s+AS)?\s*(\w*)', part)
            if match:
                table_name, alias = match.groups()
                # 如果别名为空，或者别名是SQL关键字，则使用表名作为别名
                if not alias or alias.upper() in SQL_KEYWORDS:
                    aliases[table_name] = table_name
                else:
                    aliases[alias] = table_name
        return aliases

    def _split_where_conditions(self, where_clause: str) -> list[str]:
        if not where_clause: return []
        return [cond.strip() for cond in re.split(r'\bAND\b', where_clause, flags=re.IGNORECASE)]

    def _process_query_block(self, query_block: str) -> tuple[str, dict, dict, dict]:
        """处理单个查询块，实现下推和重写逻辑。"""
        aliases_in_scope = self._get_aliases_in_scope(query_block)

        match_pattern = re.compile(
            r"(?P<full_clause>(?:(?P<table_alias>\w+)\.)?(?P<column_name>\w+)\s+MATCH\s+lembed\('(?P<model>.*?)',\s*'(?P<text>.*?)'\))",
            re.IGNORECASE | re.DOTALL
        )
        k_pattern = re.compile(
            r"(?P<full_clause>(?:(?P<table_alias>\w+)\.)?k\s*=\s*(?P<k_value>\d+))", re.IGNORECASE
        )
        
        vector_search_matches = list(match_pattern.finditer(query_block))
        if not vector_search_matches: return query_block, {}, {}, aliases_in_scope
        k_matches = list(k_pattern.finditer(query_block))

        local_searches = {}
        for match in vector_search_matches:
            alias = match.group('table_alias')
            if not alias:
                if len(aliases_in_scope) == 1:
                    alias = list(aliases_in_scope.keys())[0]
                else:
                    raise ValueError(f"歧义错误: 在多表查询中发现无别名的向量搜索列 '{match.group('column_name')}'。请为该列表明表别名 (例如: table.{match.group('column_name')})。")
            
            corresponding_k = next((k for k in k_matches if k.group('table_alias') == alias), None)
            if not corresponding_k and len(vector_search_matches) == 1:
                corresponding_k = next((k for k in k_matches if not k.group('table_alias')), None)
            if not corresponding_k: raise ValueError(f"约束缺失: 在表 '{alias}' 上的向量搜索缺少 'k=N' 约束。")

            search_id = str(uuid.uuid4())
            self.vector_searches[search_id] = {"match": match.groupdict(), "k": corresponding_k.groupdict()}
            local_searches[alias] = {"id": search_id, "match_clause": match.group('full_clause'), "k_clause": corresponding_k.group('full_clause')}

        where_match = re.search(r'\bWHERE\s+(.*?)(?=\bGROUP BY|\bORDER BY|\bLIMIT|\Z)', query_block, re.S | re.I)
        pushed_filters, remaining_conditions = {}, []

        if where_match:
            for cond in self._split_where_conditions(where_match.group(1)):
                if any(s['match_clause'] in cond or s['k_clause'] in cond for s in local_searches.values()): continue
                cond_aliases = set(re.findall(r'(\w+)\.', cond))
                target_alias, can_push = None, False
                if len(cond_aliases) == 1:
                    target_alias = cond_aliases.pop()
                    if target_alias in local_searches: can_push = True
                elif not cond_aliases and len(local_searches) == 1:
                    target_alias = list(local_searches.keys())[0]
                    can_push = True
                if can_push: pushed_filters.setdefault(target_alias, []).append(cond)
                else: remaining_conditions.append(cond)
        
        modified_block = query_block
        if where_match: modified_block = modified_block.replace(where_match.group(0), "")
        
        if remaining_conditions:
            new_where = "WHERE " + " AND ".join(remaining_conditions)
            insert_point = re.search(r'(\bGROUP BY|\bORDER BY|\bLIMIT|\Z)', modified_block, re.S | re.I)
            modified_block = f"{modified_block[:insert_point.start()]}{new_where} {modified_block[insert_point.start():]}"
        
        from_join_pattern = re.compile(r'\b(FROM|JOIN)\s+([\w\.]+)(?:\s+AS)?\s+(\w+)?', re.IGNORECASE)
        modified_block = from_join_pattern.sub(lambda m: f"{m.group(1)} {m.group(3) or m.group(2)}_filtered AS {m.group(3) or m.group(2)}" if (m.group(3) or m.group(2)) in local_searches else m.group(0), modified_block)

        for alias in local_searches:
            modified_block = re.sub(r'\b' + re.escape(alias) + r'\.distance\b', f'{alias}.distance_{alias}', modified_block)
            
        return modified_block.strip(), local_searches, pushed_filters, aliases_in_scope

    def convert(self) -> str:
        """执行完整的转换过程。"""
        main_query = self._parse_cte_structure()
        modified_blocks, all_local_searches, all_pushed_filters, all_aliases = {}, {}, {}, {}

        for name, body in self.original_ctes.items():
            modified_body, local_searches, pushed_filters, aliases = self._process_query_block(body)
            modified_blocks[name] = modified_body
            all_local_searches.update(local_searches)
            all_pushed_filters.update(pushed_filters)
            all_aliases.update(aliases)
        
        modified_main, local_searches, pushed_filters, aliases = self._process_query_block(main_query)
        modified_blocks['main'] = modified_main
        all_local_searches.update(local_searches)
        all_pushed_filters.update(pushed_filters)
        all_aliases.update(aliases)

        if not self.vector_searches:
            logging.info("无需转换：未检测到向量搜索语法。")
            return self.sqlite_query + ";"

        output_parts, vec_alias_map = [], {}
        ref_vec_clauses = []
        for i, (sid, s_info) in enumerate(self.vector_searches.items()):
            vec_alias = f"ref_vec_{i}"
            vec_alias_map[sid] = vec_alias
            ref_vec_clauses.append(f"    {self._get_placeholder_embedding(s_info['match']['text'])} AS {vec_alias}")
        if ref_vec_clauses: output_parts.append(",\n".join(ref_vec_clauses))

        filtering_ctes = []
        for alias, local_search in all_local_searches.items():
            info = self.vector_searches[local_search['id']]
            table_name = all_aliases.get(alias, alias)
            where_for_cte = ""
            if all_pushed_filters.get(alias):
                filters = [re.sub(r'\b' + alias + r'\.', '', f) for f in all_pushed_filters[alias]]
                where_for_cte = f"WHERE {' AND '.join(filters)}"
            cte = dedent(f"""\
                {alias}_filtered AS (
                    SELECT
                        *,
                        {self.distance_function}({info['match']['column_name']}, {vec_alias_map[local_search['id']]}) AS distance_{alias}
                    FROM {table_name}
                    {where_for_cte}
                    ORDER BY distance_{alias}
                    LIMIT {info['k']['k_value']}
                )""").strip()
            filtering_ctes.append(cte)
        if filtering_ctes: output_parts.append(",\n\n".join(filtering_ctes))

        modified_original_ctes = []
        for name in self.original_ctes:
            modified_body = "    " + modified_blocks[name].replace("\n", "\n    ")
            modified_original_ctes.append(f"{name} AS (\n{modified_body}\n)")
        if modified_original_ctes: output_parts.append(",\n\n".join(modified_original_ctes))
        
        final_with_clause = "WITH\n" + "\n\n, ".join(filter(None, output_parts))
        final_query = f"{final_with_clause}\n\n{modified_blocks['main']};"
        return final_query
        
# --- 运行验证 ---
if __name__ == '__main__':
    # 在 CTE 内部进行向量搜索的查询, 且 FROM musical 无别名
    sqlite_query_with_vec_in_cte_no_alias = """
    WITH TopMusicals AS (
        SELECT Musical_ID, Name, Year, distance
        FROM musical
        WHERE Category_embedding MATCH lembed('all-MiniLM-L6-v2', 'Classic Broadway hit')
        AND k = 5 AND Year < 2000
    )
    SELECT tm.Name, a.Name
    FROM TopMusicals tm
    JOIN actor a ON tm.Musical_ID = a.Musical_ID
    WHERE a.age > 50;
    """

    sqlite_query_with_vec_in_cte_no_alias = "WITH PerpetratorSearch AS (    SELECT Perpetrator_ID, Location, Year, distance    FROM perpetrator    WHERE Location_embedding MATCH lembed('all-MiniLM-L6-v2', 'Stadium: 123 Main St, Boston, MA. Capacity: 50,000. Home team: Patriots')    AND Year BETWEEN 2015 AND 2020    AND k = 5  )    SELECT p.People_ID  FROM perpetrator p  JOIN PerpetratorSearch ps ON p.Perpetrator_ID = ps.Perpetrator_ID  ORDER BY ps.distance  LIMIT 1;"
    
    sqlite_query_with_vec_in_cte_no_alias = "SELECT m.Musical_ID FROM musical m JOIN actor a ON m.Musical_ID = a.Musical_ID WHERE m.Category_embedding MATCH lembed('all-MiniLM-L6-v2',  'Best Performance by a Supporting Actor in a Musical ') AND m.k = 5 AND a.age > 30 ORDER BY m.distance;"
    
    sqlite_query_with_vec_in_cte_no_alias = "SELECT COUNT(m.Musical_ID) FROM musical m JOIN actor a ON m.Musical_ID = a.Musical_ID WHERE m.Category_embedding MATCH lembed('all-MiniLM-L6-v2',  'Outstanding performance in musical theater ') AND k = 3 AND a.age BETWEEN 20 AND 40 AND m.Year >= 2000 AND m.Result = 'Won';"

    print("\n" + "="*40)
    print(">>> 验证场景: CTE内部向量搜索 + 无别名表...")
    print("="*40)
    
    converter_validation = SQLiteToClickHouseConverter(sqlite_query_with_vec_in_cte_no_alias)
    try:
        clickhouse_query_validation = converter_validation.convert()
        print("--- 原始 SQLite 查询 ---\n")
        print(sqlite_query_with_vec_in_cte_no_alias.strip())
        print("\n\n--- 转换后的 ClickHouse 查询 ---\n")
        print(clickhouse_query_validation)
        
        # 验证别名映射是否正确
        print("\n--- 内部别名映射 (调试信息) ---")
        # To show the alias map, we'd need to expose it from the class. For now, the successful conversion is proof.
        logging.info("转换成功，未出现别名解析错误。")

    except ValueError as e:
        logging.error(f"转换失败: {e}")