import re
import uuid
from textwrap import dedent
import logging
import json

# 假设 execution_engine.py 在同一个目录下
from execution_engine.execution_engine import ExecutionEngine, TimeoutError

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# 定义一组 SQL 关键字，用于辅助判断别名是否合法
SQL_KEYWORDS = {'WHERE', 'JOIN', 'LEFT', 'RIGHT', 'INNER', 'ON', 'GROUP', 'ORDER', 'LIMIT', 'UNION', 'CROSS', 'HAVING', 'SELECT', 'FROM', 'AS'}

class SQLiteToClickHouseConverter:
    """
    自动将使用 sqlite-vec 和 sqlite-lembed 的复杂 SQLite 查询
    转换为优化的 ClickHouse 向量搜索查询。

    核心特性 (V11 - 简化CTE列名):
    - **简化CTE列名**: 在多向量搜索的CTE下推策略中，将生成的距离列名（如`distance_c`）统一简化为`distance`。
    - **修正主查询丢失**: 修复了当查询包含CTE且向量搜索在主查询中时，主查询丢失的严重Bug。
    - **精准定位SELECT**: 修正了ORDER BY子句中的distance干扰SELECT列生成的Bug。
    - **智能替换/注入**: 智能处理（替换或注入）别名或裸distance列。
    - **LIMIT约束支持**: 能正确识别并处理 `LIMIT k` 作为向量搜索的约束条件。
    """

    def __init__(self, sqlite_query, distance_function='L2Distance'):
        self.sqlite_query = sqlite_query.strip().rstrip(';').replace('"',"'")
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
            
            placeholder = self._get_placeholder_embedding(s_info['match']['text'], s_info['match']['model'])
            with_clause = f"WITH\n    {placeholder} AS {vec_alias}"
            
            final_cte_strings = []
            main_part = ""

            for name, body in self.original_ctes.items():
                body_to_add = clean_block if name == search_block_key else body
                indented_body = "    " + body_to_add.replace("\n", "\n    ")
                final_cte_strings.append(f"{name} AS (\n{indented_body}\n)")
            
            if search_block_key == 'main':
                main_part = clean_block
            else:
                main_part = main_query
            
            if final_cte_strings:
                with_clause += ",\n\n" + ",\n\n".join(final_cte_strings)
            
            return f"{with_clause}\n\n{main_part};"

        else: # 多向量搜索逻辑
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
                                {self.distance_function}({info['match']['column_name']}, {vec_alias_map[local_search['id']]}) AS distance
                            FROM {table_name}
                            {where_for_cte}
                            ORDER BY distance
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
            

# 定义一组 SQL 关键字和已知函数，用于辅助判断标识符是否需要加引号
SQL_KEYWORDS_FUNCTIONS = {
    'ABS', 'ACOS', 'ALL', 'ALTER', 'AND', 'ANY', 'AS', 'ASC', 'ASIN', 'ATAN', 'AVG',
    'BETWEEN', 'BY', 'CASE', 'CAST', 'CEIL', 'CEILING', 'CHAR', 'CHARACTER', 'CHECK',
    'COALESCE', 'CONCAT', 'CONSTRAINT', 'COS', 'COT', 'COUNT', 'CREATE', 'CROSS',
    'CURRENT_DATE', 'CURRENT_TIME', 'CURRENT_TIMESTAMP', 'CURRENT_USER', 'DATE',
    'DECIMAL', 'DECLARE', 'DEFAULT', 'DELETE', 'DESC', 'DISTINCT', 'DROP', 'ELSE',
    'END', 'ESCAPE', 'EXCEPT', 'EXISTS', 'EXP', 'EXTRACT', 'FALSE', 'FETCH', 'FLOOR',
    'FOR', 'FOREIGN', 'FROM', 'FULL', 'FUNCTION', 'GROUP', 'HAVING', 'IN', 'INDEX',
    'INNER', 'INSERT', 'INT', 'INTEGER', 'INTERSECT', 'INTERVAL', 'INTO', 'IS',
    'JOIN', 'KEY', 'LEFT', 'LENGTH', 'LIKE', 'LIMIT', 'LN', 'LOG', 'LOG10', 'LOWER',
    'LTRIM', 'MATCH', 'MAX', 'MIN', 'MOD', 'NATURAL', 'NOT', 'NULL', 'NULLIF',
    'NUMERIC', 'OF', 'ON', 'OR', 'ORDER', 'OUTER', 'PI', 'POWER', 'PRIMARY',
    'PROCEDURE', 'REAL', 'REFERENCES', 'RIGHT', 'ROUND', 'ROW', 'ROWNUM', 'RTRIM',
    'SELECT', 'SESSION_USER', 'SET', 'SIGN', 'SIN', 'SMALLINT', 'SOME', 'SQRT',
    'SUBSTR', 'SUBSTRING', 'SUM', 'SYSTEM_USER', 'TABLE', 'TAN', 'THEN', 'TIME',
    'TIMESTAMP', 'TO', 'TRANSLATE', 'TRIGGER', 'TRIM', 'TRUE', 'TRUNC', 'TRUNCATE',
    'UNION', 'UNIQUE', 'UPDATE', 'UPPER', 'USER', 'USING', 'VALUES', 'VARCHAR',
    'VIEW', 'WHEN', 'WHERE', 'WITH',
    'LEMBED'
}

class SQLiteToPostgreSQLConverter:
    """
    自动将使用 sqlite-vec 和 sqlite-lembed 的复杂 SQLite 查询
    转换为优化的、使用 pgvector 扩展的 PostgreSQL 查询。

    核心特性 (V3 - 优化列顺序):
    - **优化列顺序**: 在单向量搜索注入中，将生成的 `distance` 列移动到 SELECT 字段的末尾。
    - **修正引号处理**: 重写了标识符加引号的逻辑，采用更稳定的方式，完美处理SQL语句中的字符串字面量。
    - **PostgreSQL 语法适配**: 生成使用 '<->' 操作符的 pgvector 标准语法。
    - **智能标识符加引号**: 自动为表名和列名添加双引号，以符合PostgreSQL的大小写敏感规则。
    - **增强的lembed解析**: 支持解析由单引号或双引号包裹的字符串。
    - **保留复杂查询转换**: 完整保留了对 CTE、多向量搜索、预过滤下推等复杂场景的支持。
    """

    def __init__(self, sqlite_query):
        self.sqlite_query = sqlite_query.strip().rstrip(';').replace('"',"'")
        self.vector_searches = {}
        self.original_ctes = {}

    def _get_placeholder_embedding(self, text: str, model: str) -> str:
        escaped_text = text.replace("'", "''")
        return f"lembed('{model}', '{escaped_text}')"

    def _quote_all_identifiers(self, sql: str) -> str:
        identifier_pattern = re.compile(r'\b([a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)*)\b')
        string_literal_pattern = re.compile(r"('(?:''|[^'])*'|\"(?:\"\"|[^\"])*\")")

        def replacer(match):
            full_identifier = match.group(1)
            parts = full_identifier.split('.')
            if len(parts) == 1 and parts[0].upper() in SQL_KEYWORDS_FUNCTIONS:
                return parts[0]
            
            quoted_parts = [f'"{p}"' for p in parts if not p.isnumeric()]
            return '.'.join(quoted_parts)

        def process_code_part(part):
            return identifier_pattern.sub(replacer, part)

        last_end = 0
        result_parts = []
        for match in string_literal_pattern.finditer(sql):
            start, end = match.span()
            code_part = sql[last_end:start]
            result_parts.append(process_code_part(code_part))
            result_parts.append(match.group(0))
            last_end = end
        
        final_code_part = sql[last_end:]
        result_parts.append(process_code_part(final_code_part))
        
        return "".join(result_parts)

    def _parse_cte_structure(self):
        with_clause_match = re.match(r'\s*WITH\s+', self.sqlite_query, re.IGNORECASE)
        if not with_clause_match:
            return self.sqlite_query

        query_after_with = self.sqlite_query[with_clause_match.end():]
        open_parens, last_cte_end_index, in_string = 0, 0, False
        string_char = ''

        for i, char in enumerate(query_after_with):
            if not in_string and char in ("'", '"'):
                in_string = True
                string_char = char
            elif in_string and char == string_char:
                if i + 1 < len(query_after_with) and query_after_with[i+1] == string_char:
                    continue
                in_string = False
                string_char = ''
            
            if not in_string:
                if char == '(':
                    open_parens += 1
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
        string_char = ''
        for i, char in enumerate(cte_definitions_str):
            if not in_string and char in ("'", '"'):
                in_string = True
                string_char = char
            elif in_string and char == string_char:
                if i + 1 < len(cte_definitions_str) and cte_definitions_str[i+1] == string_char:
                    continue
                in_string = False
                string_char = ''

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
            match = re.match(r'([\w\.\(\)]+|\(.*\))(?:\s+AS)?\s*(\w*)', part.strip(), re.S)
            if match:
                table_name, alias = match.groups()
                table_name = table_name.strip()
                alias = alias.strip()
                if not alias or alias.upper() in SQL_KEYWORDS_FUNCTIONS:
                    aliases[table_name] = table_name
                else:
                    aliases[alias] = table_name
        return aliases

    def _split_where_conditions(self, where_clause: str) -> list[str]:
        if not where_clause:
            return []
        placeholder = "##_AND_##"
        masked_clause = re.sub(r'(\bBETWEEN\b\s+.*?)\s+\bAND\b', rf'\1 {placeholder}', where_clause, flags=re.IGNORECASE)
        conditions = [cond.strip() for cond in re.split(r'\bAND\b', masked_clause, flags=re.IGNORECASE)]
        unmasked_conditions = [cond.replace(placeholder, 'AND') for cond in conditions]
        return unmasked_conditions

    def _process_query_block(self, query_block: str) -> tuple[str, dict, dict, dict]:
        aliases_in_scope = self._get_aliases_in_scope(query_block)

        match_pattern = re.compile(
            r"""(?P<full_clause>
                (?:(?P<table_alias>\w+)\.)?(?P<column_name>\w+)\s+MATCH\s+
                lembed\('(?P<model>[^']*)',\s*
                (?:'(?P<text_single>(?:''|[^'])*)'|"(?P<text_double>(?:""|[^"])*)")
                \)
            )""",
            re.IGNORECASE | re.DOTALL | re.VERBOSE
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
            match_dict = match.groupdict()
            match_dict['text'] = (match_dict.pop('text_single') or match_dict.pop('text_double') or "").replace("''", "'")

            alias = match_dict['table_alias']
            if not alias:
                if len(aliases_in_scope) == 1:
                    alias = list(aliases_in_scope.keys())[0]
                else:
                    real_name_alias = next((k for k,v in aliases_in_scope.items() if k == v), None)
                    if real_name_alias:
                        alias = real_name_alias
                    else:
                        raise ValueError(f"歧义错误: 在多表查询中发现无别名的向量搜索列 '{match_dict['column_name']}'。请为该列表明表别名。")

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
            self.vector_searches[search_id] = {"match": match_dict, "k": k_info}
            local_searches[alias] = { "id": search_id, "match_clause": match_dict['full_clause'], "k_clause": k_info['full_clause'] }

        where_match = re.search(r'\bWHERE\s+(.*?)(?=\bGROUP BY|\bORDER BY|\bLIMIT|\Z)', query_block, re.S | re.I)
        pushed_filters, remaining_conditions = {}, []

        if where_match:
            for cond in self._split_where_conditions(where_match.group(1)):
                is_vec_clause = False
                for s in local_searches.values():
                    if s['match_clause'] in cond or s['k_clause'] in cond:
                        is_vec_clause = True
                        break
                if is_vec_clause:
                    continue

                cond_aliases = set(re.findall(r'(\w+)\.', cond))
                target_alias, can_push = None, False
                if len(cond_aliases) == 1:
                    target_alias = cond_aliases.pop()
                    if target_alias in local_searches: can_push = True
                elif not cond_aliases and len(local_searches) == 1:
                    target_alias = list(local_searches.keys())[0]
                    can_push = True
                
                if can_push:
                    pushed_filters.setdefault(target_alias, []).append(cond)
                else:
                    remaining_conditions.append(cond)
        
        modified_block = query_block
        if limit_was_used_as_k:
            modified_block = limit_pattern.sub("", modified_block)

        if where_match:
            clean_where_parts = []
            original_conditions = self._split_where_conditions(where_match.group(1))
            all_vec_clauses = {s['match_clause'] for s in local_searches.values()} | \
                              {s['k_clause'] for s in local_searches.values()}
            all_pushed_filters_flat = {pf for pf_list in pushed_filters.values() for pf in pf_list}
            
            for cond in original_conditions:
                if cond not in all_vec_clauses and cond not in all_pushed_filters_flat:
                    clean_where_parts.append(cond)

            if clean_where_parts:
                new_where_clause = "WHERE " + " AND ".join(clean_where_parts)
                modified_block = modified_block.replace(where_match.group(0), new_where_clause)
            else:
                modified_block = modified_block.replace(where_match.group(0), '')
        
        from_join_pattern = re.compile(r'\b(FROM|JOIN)\s+([\w\.]+)(?:\s+AS)?\s*(\w*)', re.IGNORECASE)
        def replace_from_join(m):
            keyword, table, alias = m.groups()
            effective_alias = alias if alias and alias.upper() not in SQL_KEYWORDS_FUNCTIONS else table
            if effective_alias in local_searches:
                return f"{keyword} {effective_alias}_filtered AS {effective_alias}"
            return m.group(0)
        
        if len(self.vector_searches) > 1:
            modified_block = from_join_pattern.sub(replace_from_join, modified_block)
            
        return modified_block.strip(), local_searches, pushed_filters, aliases_in_scope

    def convert(self) -> str:
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
            final_sql = self.sqlite_query if self.sqlite_query.endswith(';') else self.sqlite_query + ";"
            return self._quote_all_identifiers(final_sql)

        final_query = ""
        if len(self.vector_searches) == 1:
            logging.info("检测到单个向量搜索，采用就地转换优化策略。")
            
            sid, s_info = next(iter(self.vector_searches.items()))
            
            search_block_key = next(iter(all_local_searches))
            search_alias = next(iter(all_local_searches[search_block_key]))

            original_block = self.original_ctes.get(search_block_key, main_query)
            clean_block = original_block.replace(s_info['match']['full_clause'], '1=1')
            if "limit" in s_info['k']['full_clause'].lower():
                clean_block = re.sub(r'\bLIMIT\s+\d+', '', clean_block, flags=re.I|re.S)
            else:
                clean_block = clean_block.replace(s_info['k']['full_clause'], '1=1')

            clean_block = re.sub(r'\bAND\s+1=1\b|\b1=1\s+AND\b', '', clean_block, flags=re.I)
            clean_block = re.sub(r'\bWHERE\s+1=1\b', '', clean_block, flags=re.I)

            from_match = re.search(r'\sFROM\b', clean_block, re.I | re.S)
            if not from_match:
                raise ValueError("转换失败：无法在查询块中定位 FROM 关键字。")

            select_clause_part = clean_block[:from_match.start()]
            rest_of_block = clean_block[from_match.start():]
            
            placeholder = self._get_placeholder_embedding(s_info['match']['text'], s_info['match']['model'])
            table_ref = s_info['match']['table_alias'] or search_alias
            distance_calc = f"{table_ref}.{s_info['match']['column_name']} <-> {placeholder}"
            
            aliased_distance_pattern = r'\b' + re.escape(search_alias) + r'\.distance\b'
            bare_distance_pattern = r'(?<!\.)\bdistance\b'
            
            if re.search(aliased_distance_pattern, select_clause_part, flags=re.I):
                select_clause_part = re.sub(aliased_distance_pattern, f"{distance_calc} AS distance", select_clause_part, flags=re.I, count=1)
            elif re.search(bare_distance_pattern, select_clause_part, flags=re.I):
                select_clause_part = re.sub(bare_distance_pattern, f"{distance_calc} AS distance", select_clause_part, flags=re.I, count=1)
            else:
                select_clause_part = select_clause_part.strip().rstrip(',') + f", {distance_calc} AS distance"

            clean_block = select_clause_part + rest_of_block
            clean_block = re.sub(r'\bORDER BY.*', '', clean_block, flags=re.I|re.S).strip()
            clean_block = re.sub(r'\bLIMIT\s+\d+\s*$', '', clean_block, flags=re.I).strip()
            clean_block += f"\nORDER BY distance\nLIMIT {s_info['k']['k_value']}"
            
            final_cte_strings = []
            main_part = ""
            if self.original_ctes:
                with_prefix = "WITH "
                for name, body in self.original_ctes.items():
                    body_to_add = clean_block if name == search_block_key else body
                    indented_body = "    " + body_to_add.replace("\n", "\n    ")
                    final_cte_strings.append(f"{name} AS (\n{indented_body}\n)")
                main_part = main_query if search_block_key != 'main' else clean_block
                final_query = with_prefix + ",\n\n".join(final_cte_strings) + "\n\n" + main_part
            else:
                final_query = clean_block

        else: # 多向量搜索逻辑
            logging.info(f"检测到 {len(self.vector_searches)} 个向量搜索，采用CTE下推过滤策略。")
            output_parts = []

            filtering_ctes = []
            for block_name, local_searches_in_block in all_local_searches.items():
                for alias, local_search in local_searches_in_block.items():
                    info = self.vector_searches[local_search['id']]
                    table_name = all_aliases.get(alias, alias)
                    where_for_cte = ""
                    pushed_filters_list = all_pushed_filters.get(block_name, {}).get(alias, [])
                    if pushed_filters_list:
                        filters = [re.sub(r'\b' + re.escape(alias) + r'\.', '', f, flags=re.I) for f in pushed_filters_list]
                        where_for_cte = f"WHERE {' AND '.join(filters)}"
                    
                    placeholder = self._get_placeholder_embedding(info['match']['text'], info['match']['model'])
                    distance_calc = f"{info['match']['column_name']} <-> {placeholder}"

                    cte = dedent(f"""\
                        {alias}_filtered AS (
                            SELECT
                                *,
                                {distance_calc} AS distance
                            FROM {table_name}
                            {where_for_cte}
                            ORDER BY distance
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
            final_query = f"{final_with_clause}\n\n{modified_blocks['main']}"

        final_sql = final_query if final_query.endswith(';') else final_query + ";"
        return self._quote_all_identifiers(final_sql)

if __name__ == '__main__':
    # --- 新增功能：集成执行引擎 ---
    
    # --- 配置区 ---
    # 请根据您的环境修改这些配置
    INPUT_FILE_PATH = '/mnt/b_public/data/wangzr/Text2VectorSQL/synthesis/toy_spider/results/question_and_sql_pairs.json'
    TARGET_DB_TYPE = 'postgresql'  # 或 'clickhouse'
    ENGINE_CONFIG_PATH = 'execution_engine/engine_config.yaml' # 执行引擎的配置文件路径
    # --- 配置区结束 ---

    try:
        engine = ExecutionEngine(config_path=ENGINE_CONFIG_PATH)
    except Exception as e:
        logging.error(f"无法初始化 ExecutionEngine: {e}")
        logging.error("请确保 'engine_config.yaml' 配置文件存在且格式正确。")
        exit(1)

    try:
        with open(INPUT_FILE_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        logging.error(f"输入文件未找到: {INPUT_FILE_PATH}")
        exit(1)
    except json.JSONDecodeError:
        logging.error(f"无法解析JSON文件: {INPUT_FILE_PATH}")
        exit(1)

    for i, item in enumerate(data):
        original_sql = item.get('sql')
        db_id = item.get('db_id')
        
        if not original_sql or not db_id:
            logging.warning(f"跳过第 {i+1} 条记录，缺少 'sql' 或 'db_id' 字段。")
            continue

        # print(f"\n{'='*40}")
        # print(f"处理第 {i+1}/{len(data)} 条记录 | 数据库: {db_id}")
        # print(f"原始 SQLite SQL:\n{original_sql}")
        # print(f"{'-'*20}")
        
        try:
            # 1. 选择转换器并转换SQL
            if TARGET_DB_TYPE == 'postgresql':
                converter = SQLiteToPostgreSQLConverter(original_sql)
            elif TARGET_DB_TYPE == 'clickhouse':
                converter = SQLiteToClickHouseConverter(original_sql)
            else:
                logging.error(f"不支持的目标数据库类型: {TARGET_DB_TYPE}")
                continue
                
            converted_sql = converter.convert()
            # print(f"转换后的 {TARGET_DB_TYPE.capitalize()} SQL:\n{converted_sql}")
            # print(f"{'-'*20}")
            
            # 2. 执行转换后的SQL
            logging.info(f"正在目标数据库 '{db_id}' ({TARGET_DB_TYPE}) 上执行...")
            result = engine.execute(
                sql=converted_sql, 
                db_type=TARGET_DB_TYPE, 
                db_identifier=db_id
            )
            
            # 3. 处理执行结果
            if result.get('status') == 'success':
                logging.info(f"执行成功！返回 {result.get('row_count', 'N/A')} 行。")
                # print("执行结果:")
                # print(json.dumps(result, indent=2, ensure_ascii=False))
            else:
                # 捕获由引擎处理的错误（例如，SQL语法错误）
                print(db_id)
                print(f"原始 SQLite SQL:\n{original_sql}")
                print(f"转换后的 {TARGET_DB_TYPE.capitalize()} SQL:\n{converted_sql}")
                print(f"{'-'*20}")
                logging.warning(f"执行失败: {result.get('message', '未知错误')}")
                logging.warning("跳过此SQL。")

        except (TimeoutError, ValueError) as e:
            # 捕获转换或执行过程中的超时和值错误
            logging.error(f"处理过程中发生错误: {e}")
            logging.error("跳过此SQL。")
        except Exception as e:
            # 捕获其他意外错误
            logging.error(f"发生意外错误: {e}", exc_info=True)
            logging.error("跳过此SQL。")
            
        print(f"{'='*40}")