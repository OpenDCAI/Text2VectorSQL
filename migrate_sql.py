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

    核心特性 (V10 - CTE拼接逻辑修正):
    - **修正主查询丢失**: 修复了当查询包含CTE且向量搜索在主查询中时，主查询丢失的严重Bug。
    - **精准定位SELECT**: 修正了ORDER BY子句中的distance干扰SELECT列生成的Bug。
    - **智能替换/注入**: 智能处理（替换或注入）别名或裸distance列。
    - **LIMIT约束支持**: 能正确识别并处理 `LIMIT k` 作为向量搜索的约束条件。
    - **动态嵌入占位符**: 生成 `lembed('model', 'text')` 格式的占位符。
    """

    def __init__(self, sqlite_query, distance_function='L2Distance'):
        self.sqlite_query = sqlite_query.strip()
        self.distance_function = distance_function
        self.vector_searches = {}
        self.original_ctes = {}

    def _get_placeholder_embedding(self, text: str, model: str) -> str:
        """
        为给定的文本和模型生成一个 lembed 函数调用字符串。
        对文本中的单引号进行转义以防止SQL注入或语法错误。
        """
        escaped_text = text.replace("'", "''")
        return f"lembed('{model}', '{escaped_text}')"

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
        
        balance, last_split = 0, 0
        in_string = False
        for i, char in enumerate(cte_definitions_str):
            if char == "'":
                in_string = not in_string
            if not in_string:
                if char == '(':
                    balance += 1
                elif char == ')':
                    balance -= 1
                elif char == ',' and balance == 0:
                    cte_def = cte_definitions_str[last_split:i].strip()
                    match = re.match(r'(\w+)\s+AS\s+\((.*)\)', cte_def, re.IGNORECASE | re.DOTALL)
                    if match: self.original_ctes[match.group(1).strip()] = match.group(2).strip()
                    last_split = i + 1
        
        last_cte_def = cte_definitions_str[last_split:].strip()
        match = re.match(r'(\w+)\s+AS\s+\((.*)\)', last_cte_def, re.IGNORECASE | re.DOTALL)
        if match: self.original_ctes[match.group(1).strip()] = match.group(2).strip()

        return main_query

    def _get_aliases_in_scope(self, query_block: str) -> dict[str, str]:
        """精确解析给定查询块中的所有表及其别名。"""
        aliases = {}
        from_join_content_match = re.search(
            r'\bFROM\s+(.*?)(?=\bWHERE|\bGROUP BY|\bORDER BY|\bLIMIT|\Z)', 
            query_block, re.S | re.I
        )
        if not from_join_content_match: return {}
        
        from_join_content = from_join_content_match.group(1)
        content_no_on = re.sub(r'\bON\b.*?(?=\b(CROSS|INNER|LEFT|RIGHT|FULL)?\s*JOIN\b|\Z)', '', from_join_content, flags=re.S|re.I)
        processed_content = re.sub(r'\b(CROSS|INNER|LEFT|RIGHT|FULL)?\s*JOIN\b', '&&&', content_no_on, flags=re.I)
        
        table_parts = [p.strip() for p in processed_content.split('&&&')]

        for part in table_parts:
            match = re.match(r'([\w\.\(\)]+)(?:\s+AS)?\s*(\w*)', part.strip())
            if match:
                table_name, alias = match.groups()
                if not alias or alias.upper() in SQL_KEYWORDS:
                    aliases[table_name] = table_name
                else:
                    aliases[alias] = table_name
        return aliases

    def _split_where_conditions(self, where_clause: str) -> list[str]:
        """智能地将 WHERE 子句按 AND 分割，能正确处理 BETWEEN ... AND ... 语法。"""
        if not where_clause:
            return []
        placeholder = "##_AND_##"
        masked_clause = re.sub(r'(\bBETWEEN\b\s+.*?)\s+\bAND\b', rf'\1 {placeholder}', where_clause, flags=re.IGNORECASE)
        conditions = [cond.strip() for cond in re.split(r'\bAND\b', masked_clause, flags=re.IGNORECASE)]
        unmasked_conditions = [cond.replace(placeholder, 'AND') for cond in conditions]
        return unmasked_conditions

    def _process_query_block(self, query_block: str) -> tuple[str, dict, dict, dict]:
        """处理单个查询块，能识别 k=N 和 LIMIT N 作为约束。"""
        aliases_in_scope = self._get_aliases_in_scope(query_block)

        match_pattern = re.compile(
            r"(?P<full_clause>(?:(?P<table_alias>\w+)\.)?(?P<column_name>\w+)\s+MATCH\s+lembed\('(?P<model>.*?)',\s*'(?P<text>.*?)'\))",
            re.IGNORECASE | re.DOTALL
        )
        k_pattern = re.compile(
            r"(?P<full_clause>(?:(?P<table_alias>\w+)\.)?k\s*=\s*(?P<k_value>\d+))", re.IGNORECASE
        )
        limit_pattern = re.compile(r"\bLIMIT\s+(?P<k_value>\d+)", re.IGNORECASE | re.DOTALL)
        
        vector_search_matches = list(match_pattern.finditer(query_block))
        if not vector_search_matches: return query_block, {}, {}, aliases_in_scope
        
        k_matches = list(k_pattern.finditer(query_block))
        limit_match = limit_pattern.search(query_block)

        local_searches = {}
        limit_was_used_as_k = False

        for match in vector_search_matches:
            alias = match.group('table_alias')
            if not alias:
                if len(aliases_in_scope) == 1:
                    alias = list(aliases_in_scope.keys())[0]
                else:
                    real_name_alias = next((k for k,v in aliases_in_scope.items() if k==v), None)
                    if real_name_alias: alias = real_name_alias
                    else: raise ValueError(f"歧义错误: 在多表查询中发现无别名的向量搜索列 '{match.group('column_name')}'。请为该列表明表别名。")

            corresponding_k_match = next((k for k in k_matches if k.group('table_alias') == alias), None)
            if not corresponding_k_match and len(k_matches) == 1 and not k_matches[0].group('table_alias'):
                corresponding_k_match = k_matches[0]

            k_info = None
            if corresponding_k_match:
                k_info = corresponding_k_match.groupdict()
            elif len(vector_search_matches) == 1 and limit_match:
                k_info = { "full_clause": limit_match.group(0), "table_alias": None, "k_value": limit_match.group('k_value') }
                limit_was_used_as_k = True
            
            if not k_info:
                raise ValueError(f"约束缺失: 在表 '{alias}' 上的向量搜索缺少 'k=N' 或 'LIMIT N' 约束。")

            search_id = str(uuid.uuid4())
            self.vector_searches[search_id] = {"match": match.groupdict(), "k": k_info}
            local_searches[alias] = { "id": search_id, "match_clause": match.group('full_clause'), "k_clause": k_info['full_clause'] }

        where_match = re.search(r'\bWHERE\s+(.*?)(?=\bGROUP BY|\bORDER BY|\bLIMIT|\Z)', query_block, re.S | re.I)
        pushed_filters, remaining_conditions = {}, []

        if where_match:
            for cond in self._split_where_conditions(where_match.group(1)):
                if any(s['match_clause'] in cond or s['k_clause'] in cond for s in local_searches.values()):
                    continue
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
        if limit_was_used_as_k:
            modified_block = limit_pattern.sub("", modified_block)

        if where_match: modified_block = modified_block.replace(where_match.group(0), "")
        
        if remaining_conditions:
            new_where = "WHERE " + " AND ".join(remaining_conditions)
            insert_point = re.search(r'(\bGROUP BY|\bORDER BY|\bLIMIT|\Z)', modified_block, re.S | re.I)
            modified_block = f"{modified_block[:insert_point.start()]} {new_where} {modified_block[insert_point.start():]}"
        
        from_join_pattern = re.compile(r'\b(FROM|JOIN)\s+([\w\.]+)(?:\s+AS)?\s*(\w*)', re.IGNORECASE)
        def replace_from_join(m):
            keyword, table, alias = m.groups()
            effective_alias = alias if alias and alias.upper() not in SQL_KEYWORDS else table
            if effective_alias in local_searches:
                return f"{keyword} {effective_alias}_filtered AS {effective_alias}"
            return m.group(0)
        modified_block = from_join_pattern.sub(replace_from_join, modified_block)

        for alias in local_searches:
            modified_block = re.sub(r'\b' + re.escape(alias) + r'\.distance\b', f'{alias}.distance_{alias}', modified_block, flags=re.I)
            
        return modified_block.strip(), local_searches, pushed_filters, aliases_in_scope

    def convert(self) -> str:
        """执行完整的转换过程，根据向量搜索数量选择最优策略。"""
        main_query = self._parse_cte_structure()
        
        modified_blocks, all_local_searches, all_pushed_filters, all_aliases = {}, {}, {}, {}

        for name, body in self.original_ctes.items():
            modified_body, local_searches, pushed_filters, aliases = self._process_query_block(body)
            modified_blocks[name] = modified_body
            if local_searches: all_local_searches[name] = local_searches
            if pushed_filters: all_pushed_filters[name] = pushed_filters
            all_aliases.update(aliases)
        
        modified_main, local_searches, pushed_filters, aliases = self._process_query_block(main_query)
        modified_blocks['main'] = modified_main
        if local_searches: all_local_searches['main'] = local_searches
        if pushed_filters: all_pushed_filters['main'] = pushed_filters
        all_aliases.update(aliases)

        if not self.vector_searches:
            logging.info("无需转换：未检测到向量搜索语法。")
            return self.sqlite_query + ";"

        if len(self.vector_searches) == 1:
            logging.info("检测到单个向量搜索，采用就地转换优化策略。")
            
            sid, s_info = next(iter(self.vector_searches.items()))
            vec_alias = "ref_vec_0"
            
            search_block_key = next(iter(all_local_searches))
            search_alias = next(iter(all_local_searches[search_block_key]))

            original_block = self.original_ctes.get(search_block_key, main_query)
            clean_block = original_block.replace(s_info['match']['full_clause'], '1=1')
            
            if "limit" in s_info['k']['full_clause'].lower():
                clean_block = re.sub(r'\bLIMIT\s+\d+', '', clean_block, flags=re.I|re.S)
            else:
                clean_block = clean_block.replace(s_info['k']['full_clause'], '1=1')

            clean_block = re.sub(r'\bAND\s+1=1\b', '', clean_block, flags=re.I)
            clean_block = re.sub(r'\b1=1\s+AND\b', '', clean_block, flags=re.I)
            clean_block = re.sub(r'\bWHERE\s+1=1\b', '', clean_block, flags=re.I)

            from_match = re.search(r'\s+\bFROM\b', clean_block, flags=re.I | re.S)
            if not from_match:
                raise ValueError("转换失败：无法在查询块中定位 FROM 关键字。")

            select_clause_part = clean_block[:from_match.start()]
            rest_of_block = clean_block[from_match.start():]
            
            distance_func_call = f"{self.distance_function}({s_info['match']['table_alias'] or search_alias}.{s_info['match']['column_name']}, {vec_alias})"
            aliased_distance_pattern = r'\b' + re.escape(search_alias) + r'\.distance\b'
            bare_distance_pattern = r'(?<!\.)\bdistance\b'
            
            if re.search(aliased_distance_pattern, select_clause_part, flags=re.I):
                select_clause_part = re.sub(aliased_distance_pattern, f"{distance_func_call} AS distance", select_clause_part, flags=re.I, count=1)
            elif re.search(bare_distance_pattern, select_clause_part, flags=re.I):
                select_clause_part = re.sub(bare_distance_pattern, f"{distance_func_call} AS distance", select_clause_part, flags=re.I, count=1)
            else:
                select_clause_part = re.sub(
                    r'(\bSELECT(?:\s+DISTINCT)?\s+)(.*)',
                    rf'\1\2, {distance_func_call} AS distance',
                    select_clause_part, count=1, flags=re.I | re.S).strip()

            clean_block = select_clause_part + rest_of_block
            clean_block = re.sub(r'\bORDER BY.*', '', clean_block, flags=re.I|re.S).strip()
            clean_block = re.sub(r'\bLIMIT\s+\d+\s*$', '', clean_block, flags=re.I).strip()
            clean_block += f"\nORDER BY distance\nLIMIT {s_info['k']['k_value']}"
            
            # --- **核心修正逻辑 V10** ---
            placeholder = self._get_placeholder_embedding(s_info['match']['text'], s_info['match']['model'])
            with_clause = f"WITH\n    {placeholder} AS {vec_alias}"
            
            final_cte_strings = []
            main_part = ""

            # 遍历原始 CTE 并构建最终的 CTE 字符串列表
            for name, body in self.original_ctes.items():
                body_to_add = clean_block if name == search_block_key else body
                indented_body = "    " + body_to_add.replace("\n", "\n    ")
                final_cte_strings.append(f"{name} AS (\n{indented_body}\n)")
            
            # 确定主查询部分
            if search_block_key == 'main':
                main_part = clean_block
            else:
                main_part = main_query
            
            # 组装最终查询
            if final_cte_strings:
                with_clause += ",\n\n" + ",\n\n".join(final_cte_strings)
            
            return f"{with_clause}\n\n{main_part};"
            # --- **修正结束** ---

        else: # 多向量搜索逻辑 (保持不变)
            logging.info(f"检测到 {len(self.vector_searches)} 个向量搜索，采用CTE下推过滤策略。")
            output_parts, vec_alias_map = [], {}
            ref_vec_clauses = []
            for i, (sid, s_info) in enumerate(self.vector_searches.items()):
                vec_alias = f"ref_vec_{i}"
                vec_alias_map[sid] = vec_alias
                placeholder = self._get_placeholder_embedding(s_info['match']['text'], s_info['match']['model'])
                ref_vec_clauses.append(f"    {placeholder} AS {vec_alias}")
            if ref_vec_clauses: output_parts.append(",\n".join(ref_vec_clauses))

            filtering_ctes = []
            for block_name, local_searches in all_local_searches.items():
                for alias, local_search in local_searches.items():
                    info = self.vector_searches[local_search['id']]
                    table_name = all_aliases.get(alias, alias)
                    where_for_cte = ""
                    pushed_filters = all_pushed_filters.get(block_name, {}).get(alias, [])
                    if pushed_filters:
                        filters = [re.sub(r'\b' + alias + r'\.', '', f, flags=re.I) for f in pushed_filters]
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
            
            final_with_clause = "WITH\n" + ",\n\n".join(filter(None, output_parts))
            final_query = f"{final_with_clause}\n\n{modified_blocks['main']};"
            return final_query

# --- 运行验证 ---
if __name__ == '__main__':
    import json
    path = '/mnt/b_public/data/wangzr/Text2VectorSQL/synthesis/toy_spider/results/question_and_sql_pairs.json'
    with open(path, 'r') as f:
        data = json.load(f)
        for item in data:
            print(item['question'])
            sql = item['sql']
            sql = sql.strip().rstrip(';').replace('"',"'")
            print('='*40)
            print(sql)
            converter = SQLiteToClickHouseConverter(sql)
            print('-'*20)
            ch_sql = converter.convert()
            print(ch_sql)
            print('='*40)
            