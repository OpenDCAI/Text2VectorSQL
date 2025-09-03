# schema_utils.py

import json
import re
import hashlib
import os
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import Dict, Any, Set, List, Tuple

from dotenv import load_dotenv

# (get_detailed_semantic_type, load_and_process_schema, get_table_aliases, 
#  get_where_conditions, find_semantic_candidates, LLMCache, call_llm 函数与上一版本相同，为简洁省略)
# ...
def get_detailed_semantic_type(column_name: str) -> str:
    col = column_name.lower()
    person_keywords = ['fname', 'lname', 'fullname', 'firstname', 'lastname', 'customer', 'employee', 'student', 'pilot', 'driver', 'captain', 'director', 'founder', 'manager', 'reviewer', 'owner', 'scientist', 'architect', 'supervisor']
    location_keywords = ['country', 'city', 'state', 'location', 'address', 'headquarter', 'nationality', 'building', 'airport', 'district']
    organization_keywords = ['department', 'company', 'school', 'team', 'brand', 'publisher', 'university', 'organization', 'club']
    thing_keywords = ['product', 'title', 'album', 'song', 'movie', 'track', 'appellation', 'service']
    if any(keyword in col for keyword in person_keywords): return "person_name"
    if any(keyword in col for keyword in location_keywords): return "location_name"
    if any(keyword in col for keyword in organization_keywords): return "organization_name"
    if any(keyword in col for keyword in thing_keywords): return "product_name"
    if 'name' in col: return "generic_name"
    return "other"

def load_and_process_schema(filepath: str) -> Dict[str, Dict[str, Any]]:
    try:
        with open(filepath, 'r', encoding='utf-8') as f: schemas_raw = json.load(f)
    except FileNotFoundError:
        print(f"Error: Schema file not found at {filepath}"); return {}
    processed_schemas = {}
    for db_schema in schemas_raw:
        db_id = db_schema['db_id']
        table_map = {i: name for i, name in enumerate(db_schema['table_names_original'])}
        vector_columns: Set[str] = set(); table_columns: Dict[str, Set[str]] = {name: set() for name in db_schema['table_names_original']}
        semantic_info: Dict[str, Dict[str, str]] = {name: {} for name in db_schema['table_names_original']}
        for col_idx, (table_idx, col_name) in enumerate(db_schema['column_names_original']):
            if table_idx >= 0 and table_idx in table_map:
                table_name = table_map[table_idx]; table_columns[table_name].add(col_name.lower())
                if col_name.lower().endswith('_embedding'): vector_columns.add(f"{table_name}.{col_name.lower()}")
        if 'column_alter' in db_schema:
            for table_name, alterations in db_schema['column_alter'].items():
                if table_name in semantic_info:
                    for alt in alterations:
                        col_name = alt.get('column_name'); sem_type = alt.get('semantic_type')
                        if col_name and sem_type: semantic_info[table_name][col_name.lower()] = sem_type
        processed_schemas[db_id] = {"table_names": db_schema['table_names_original'], "table_columns": table_columns, "vector_columns": vector_columns, "semantic_info": semantic_info, "table_samples": db_schema.get('table_samples', {})}
    return processed_schemas

def get_table_aliases(sql: str) -> Dict[str, str]:
    aliases = {}; pattern = re.compile(r"\b(?:FROM|JOIN)\s+([`'\"\w\.]+)\s+(?:AS\s+)?([`'\"\w]+)", re.IGNORECASE)
    matches = pattern.findall(sql)
    for table, alias in matches:
        table_name = table.split('.')[-1].strip("`'\""); aliases[alias.upper()] = table_name
    return aliases

def get_where_conditions(sql: str) -> List[Tuple[str, str, str]]:
    conditions = []; where_clause_match = re.search(r"WHERE\s+(.*?)(?:\sGROUP BY|\sORDER BY|\sLIMIT|\sHAVING|\sUNION|\sINTERSECT|\sEXCEPT|$)", sql, re.IGNORECASE | re.DOTALL)
    if not where_clause_match: return []
    where_clause = where_clause_match.group(1); pattern = re.compile(r"([`'\"\w\.]*[\w`'\"]+)\s*(=|LIKE)\s*['\"]([^'\"]+)['\"]", re.IGNORECASE)
    matches = pattern.findall(where_clause)
    for col, op, val in matches: conditions.append((col.strip(), op.strip(), val.strip()))
    return conditions

def find_semantic_candidates(sql: str, schema: Dict) -> List[Dict]:
    aliases = get_table_aliases(sql); conditions = get_where_conditions(sql); candidates = []
    semantic_info = schema.get("semantic_info", {}); vector_columns = schema.get("vector_columns", set())
    column_to_table_map = {}
    for table, columns in schema.get("table_columns", {}).items():
        for col in columns: column_to_table_map[col.lower()] = table
    for full_col, op, value in conditions:
        table_name = None; col_name = None
        if '.' in full_col:
            table_alias, col_name_raw = full_col.split('.', 1); table_name = aliases.get(table_alias.upper()); col_name = col_name_raw.lower()
        else:
            col_name = full_col.lower(); table_name = column_to_table_map.get(col_name)
        if not table_name: continue
        table_semantic_info = semantic_info.get(table_name, {})
        col_semantic_type_from_schema = table_semantic_info.get(col_name)
        if col_semantic_type_from_schema == 'name':
            detailed_type = get_detailed_semantic_type(col_name)
            if detailed_type not in ['location_name', 'other']:
                description_col = next((c for c, s_type in table_semantic_info.items() if s_type == 'description'), None)
                if description_col:
                    vector_col_name = f"{description_col}_embedding"; full_vector_col_name = f"{table_name}.{vector_col_name}"
                    if full_vector_col_name in vector_columns:
                        candidates.append({"conversion_type": "name_to_description_vector", "table_name": table_name, "name_column": col_name, "description_column": description_col, "vector_column_name": vector_col_name, "value_to_embed": str(value), "original_condition": f"{full_col} {op} '{value}'"}); continue
        potential_direct_vector_col = f"{col_name}_embedding"; full_potential_direct_vector_col = f"{table_name}.{potential_direct_vector_col}"
        if full_potential_direct_vector_col in vector_columns:
            candidates.append({"conversion_type": "direct_filter", "table_name": table_name, "column_name": col_name, "vector_column_name": potential_direct_vector_col, "operator": op, "value": value, "original_condition": f"{full_col} {op} '{value}'"}); continue
        table_description_col = next((c for c, s_type in table_semantic_info.items() if s_type == 'description'), None)
        if table_description_col:
            vector_col_name = f"{table_description_col}_embedding"; full_vector_col_name = f"{table_name}.{vector_col_name}"
            if full_vector_col_name in vector_columns:
                candidates.append({"conversion_type": "generic_to_description_vector", "table_name": table_name, "original_column": col_name, "description_column": table_description_col, "vector_column_name": vector_col_name, "value": value, "original_condition": f"{full_col} {op} '{value}'"}); continue
        candidates.append({"conversion_type": "unconvertible", "table_name": table_name, "column_name": col_name, "value": value, "original_condition": f"{full_col} {op} '{value}'"})
    return candidates

class LLMCache:
    def __init__(self, cache_path: str):
        self.cache_path = cache_path; self._cache = self._load_cache()
    def _load_cache(self) -> Dict[str, Any]:
        if not self.cache_path: return {}
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'r', encoding='utf-8') as f: return json.load(f)
        return {}
    def get(self, key: str) -> Any: return self._cache.get(key)
    def set(self, key: str, value: Any): self._cache[key] = value; self._save_cache()
    def _save_cache(self):
        if self.cache_path:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, 'w', encoding='utf-8') as f: json.dump(self._cache, f, indent=2, ensure_ascii=False)

def call_llm(client: OpenAI, model_name: str, prompt: str) -> Dict[str, str]:
    try:
        response = client.chat.completions.create(model=model_name, messages=[{"role": "user", "content": prompt}], temperature=0.2, response_format={"type": "json_object"})
        content = response.choices[0].message.content; return json.loads(content)
    except Exception as e:
        print(f"Error calling LLM: {e}"); return {}

def process_batch_with_llm(
    items: List[Dict],
    api_key: str,
    base_url: str,
    llm_model: str,
    embedding_model: str,
    max_workers: int
) -> List[Dict]:
    """
    Processes a batch of items using the LLM with caching and parallel execution.
    This function now dynamically loads the correct prompt template based on the candidate type.
    """
    load_dotenv()
    client = OpenAI(api_key=api_key, base_url=base_url)
    cache = LLMCache(os.getenv("CACHE_FILE_PATH"))
    results = [None] * len(items)

    # --- ▼▼▼ 核心改动：加载所有提示词模板 ▼▼▼ ---
    prompt_templates = {}
    template_files = {
        "name_to_description_vector": "prompt_name_to_desc.txt",
        "direct_filter": "prompt_direct_vector.txt",
        "generic_to_description_vector": "prompt_generic_to_desc.txt"
    }
    try:
        for key, filename in template_files.items():
            with open(f"./prompt_templates/{filename}", 'r', encoding='utf-8') as f:
                prompt_templates[key] = f.read()
    except FileNotFoundError as e:
        print(f"Error: Prompt template file not found: {e.filename}")
        return items
    # --- ▲▲▲ 核心改动结束 ▲▲▲ ---

    schemas = load_and_process_schema(os.getenv("SCHEMA_FILE_PATH"))
    futures_map = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i, item in enumerate(items):
            candidate = item.get("_candidate")
            db_id = item.get("db_id")
            if not candidate or not db_id:
                results[i] = item
                continue
            
            schema_for_db = schemas.get(db_id)
            if not schema_for_db:
                results[i] = item
                continue

            candidate_type = candidate.get("conversion_type")
            table_name = candidate.get("table_name")

            # --- ▼▼▼ 核心改动：准备模板变量并选择模板 ▼▼▼ ---
            PROMPT_TEMPLATE = prompt_templates.get(candidate_type)
            if not PROMPT_TEMPLATE:
                results[i] = item
                continue # 如果候选类型没有对应的模板，则跳过

            # 准备填充模板所需的变量
            template_vars = {
                "simplified_schema": "", # Placeholder, will be built next
                "original_question": item['question'],
                "original_sql": item['query'],
                "original_condition": candidate['original_condition'],
                "value_from_condition": candidate.get("value_to_embed") or candidate.get("value"),
                "embedding_model_name": embedding_model,
                "target_vector_column": candidate.get("vector_column_name"),
                "original_column": candidate.get("column_name") or candidate.get("original_column"),
                "name_column": candidate.get("name_column")
            }
            # --- ▲▲▲ 核心改动结束 ▲▲▲ ---

            # (准备 Schema 字符串的逻辑保持不变)
            table_cols = schema_for_db["table_columns"].get(table_name, set())
            table_sem_info = schema_for_db["semantic_info"].get(table_name, {})
            table_samples = schema_for_db["table_samples"].get(table_name, [])
            sem_cols_str_list = [f"`{col}` ({stype})" for col, stype in table_sem_info.items()]
            simplified_schema_str = f"- **Table: `{table_name}`**\n"
            simplified_schema_str += f"  - Columns: {', '.join(sorted(list(table_cols)))}\n"
            if sem_cols_str_list: simplified_schema_str += f"  - Semantic Columns: {', '.join(sem_cols_str_list)}\n"
            if table_samples:
                headers = [h for h in table_samples[0].keys() if not h.endswith('_embedding')]
                simplified_schema_str += "  - Sample Rows:\n"; simplified_schema_str += f"    | {' | '.join(headers)} |\n"; simplified_schema_str += f"    |{'---|' * len(headers)}\n"
                for sample_row in table_samples:
                    row_values = [str(sample_row.get(h, '')).replace('\n', ' ').replace('|', ' ') for h in headers]; simplified_schema_str += f"    | {' | '.join(row_values)} |\n"
            
            template_vars["simplified_schema"] = simplified_schema_str
            
            prompt = PROMPT_TEMPLATE.format(**template_vars)
            
            prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
            cached_result = cache.get(prompt_hash)
            
            if cached_result:
                processed_item = {**item}; processed_item["vector_sql"] = cached_result.get("new_sql"); processed_item["vector_question"] = cached_result.get("new_question")
                results[i] = processed_item
            else:
                future = executor.submit(call_llm, client, llm_model, prompt)
                futures_map[future] = (i, prompt_hash)

        if not futures_map:
             print("No new items to process with LLM. All candidates were either invalid or found in cache.")
             return [res for res in results if res is not None]

        # (处理 futures 的逻辑保持不变)
        desc = "Generating Semantic Pairs"
        for future in tqdm(as_completed(futures_map), total=len(futures_map), desc=desc):
            index, prompt_hash = futures_map[future]
            original_item = items[index]
            try:
                llm_result = future.result()
                if llm_result and "new_sql" in llm_result:
                    cache.set(prompt_hash, llm_result)
                    processed_item = {**original_item}; processed_item["vector_sql"] = llm_result.get("new_sql"); processed_item["vector_question"] = llm_result.get("new_question")
                    results[index] = processed_item
                else:
                    results[index] = original_item 
            except Exception as e:
                print(f"Future resulted in an exception: {e}")
                results[index] = original_item

    return [res for res in results if res is not None]
