# sql_vectorize.py

import json
import os
from typing import List, Dict, Any

# Import configurations and utility functions
from dotenv import load_dotenv
from schema_utils import load_and_process_schema, find_semantic_candidates, process_batch_with_llm

def main():
    """Main orchestration function."""
    print("--- Starting TextSQL to VectorSQL Expansion Process ---")

    # 1. Load and process schemas
    load_dotenv()
    print(f'''Loading schemas from {os.getenv("SCHEMA_FILE_PATH")}...''')
    schemas = load_and_process_schema(os.getenv("SCHEMA_FILE_PATH"))
    if not schemas:
        return
    print(f"Successfully loaded and processed {len(schemas)} database schemas.")

    # 2. Load input data
    print(f'''Loading input data from {os.getenv("INPUT_FILE_PATH")}...''')
    try:
        with open(os.getenv("INPUT_FILE_PATH"), 'r', encoding='utf-8') as f:
            input_data = json.load(f)
    except FileNotFoundError:
        print(f'''Error: Input file not found at {os.getenv("INPUT_FILE_PATH")}''')
        return
    print(f"Loaded {len(input_data)} question/SQL pairs.")
    
    # --- ▼▼▼ 新增：初始化提交给 LLM 的任务计数器 ▼▼▼ ---
    llm_submission_counts = {
        "name_to_description_vector": 0,
        "direct_filter": 0,
        "generic_to_description_vector": 0
    }
    # --- ▲▲▲ 新增结束 ▲▲▲ ---

    # 3. Identify convertible items
    items_to_process = []
    unconvertible_items = []

    for item in input_data:
        db_id = item.get("db_id")
        sql = item.get("query")
        schema = schemas.get(db_id)

        if not db_id or not sql or not schema:
            item["CONVERT"] = "false"
            unconvertible_items.append(item)
            continue

        all_candidates = find_semantic_candidates(sql, schema)
        
        valid_llm_candidates = []
        unconvertible_candidates_for_debug = []

        for candidate in all_candidates:
            if candidate.get("conversion_type") in [
                "name_to_description_vector", 
                "direct_filter", 
                "generic_to_description_vector"
            ]:
                valid_llm_candidates.append(candidate)
            else:
                unconvertible_candidates_for_debug.append(candidate)

        if valid_llm_candidates:
            # --- ▼▼▼ 核心改动：为有效候选者设置优先级并选择最优的一个 ▼▼▼ ---
            
            # 1. 定义优先级 (数字越小，优先级越高)
            priority_map = {
                "name_to_description_vector": 1,
                "direct_filter": 2,
                "generic_to_description_vector": 3
            }

            # 2. 根据优先级对候选者列表进行排序
            valid_llm_candidates.sort(key=lambda c: priority_map.get(c.get("conversion_type"), 99))
            
            # 3. 选择排序后的第一个，即为最优候选者
            best_candidate = valid_llm_candidates[0]
            
            item["CONVERT"] = "true"
            item["_candidate"] = best_candidate
            items_to_process.append(item)
            
            # 4. 更新计数器
            best_candidate_type = best_candidate.get("conversion_type")
            if best_candidate_type in llm_submission_counts:
                llm_submission_counts[best_candidate_type] += 1

            # --- ▲▲▲ 核心改动结束 ▲▲▲ ---
        else:
            item["CONVERT"] = "false"
            unconvertible_items.append(item)
            if unconvertible_candidates_for_debug:
                # (调试打印逻辑保持不变)
                pass # You can re-enable the debug prints here if needed

    print(f"Found {len(items_to_process)} items that are candidates for semantic conversion.")
    
    # 4. Process convertible items with LLM
    if items_to_process:
        processed_items = process_batch_with_llm(
            items=items_to_process,
            api_key=os.getenv("API_KEY"),
            base_url=os.getenv("BASE_URL"),
            llm_model=os.getenv("LLM_MODEL_NAME"),
            embedding_model=os.getenv("EMBEDDING_MODEL_NAME"),
            max_workers=int(os.getenv("MAX_WORKERS"))
        )
    else:
        processed_items = []

    # 5. Combine results and save
    final_data = unconvertible_items + processed_items
    
    for item in final_data:
        item.pop("_candidate", None)

    output_path = os.getenv("OUTPUT_FILE_PATH")
    print(f"Saving {len(final_data)} processed items to {output_path}...")
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=4, ensure_ascii=False)
    
    # --- ▼▼▼ 新增：打印最终的分类计数结果 ▼▼▼ ---
    print("\n--- LLM Submission Summary ---")
    print(f"Total items submitted to LLM: {len(items_to_process)}")
    for type_name, count in llm_submission_counts.items():
        print(f"  - {type_name}: {count}")
    print("------------------------------\n")
    # --- ▲▲▲ 新增结束 ▲▲▲ ---

    print("--- Process Completed Successfully! ---")


if __name__ == "__main__":
    main()
