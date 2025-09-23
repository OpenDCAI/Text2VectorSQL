import json
import re
import os

from tqdm import tqdm

def remove_sql_comments(sql):
    # Remove single-line comments
    sql = re.sub(r'--.*', '', sql)
    # Remove multi-line comments
    sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)
    return sql.strip()

def generate_cot_prompts(dataset_json_path="./results/question_and_sql_pairs.json", tables_json_path="sqlite/results/enhanced_embedding_table_vector.json", prompt_tamplate_path="sqlite/prompt_templates/cot_synthesis_prompt_template.txt", output_prompt_path="sqlite/prompts/cot_synthesis_prompts.json"):
    dataset_json = json.load(open(dataset_json_path))
    tables_json = json.load(open(tables_json_path))
    print("len(tables):", len(tables_json))
    
    prompts = []
    db_id2ddls = dict()
    for table in tables_json:
        db_id2ddls[table["db_id"]] = table["ddls"]
    print("len(db_id2ddls):", len(db_id2ddls))

    prompt_tamplate = open(prompt_tamplate_path).read()
    for data in tqdm(dataset_json):
        if data["external_knowledge"] != "":
            question = data["external_knowledge"] + "\n" + data["question"]
        else:
            question = data["question"]

        data["cot_synthesis_prompt"] = prompt_tamplate.format(
            schema = "\n\n".join(db_id2ddls[data["db_id"]]),
            question = question,
            sql = remove_sql_comments(data["sql"])
        )
    # 创建输出目录
    os.makedirs("sqlite/prompts", exist_ok=True)
    with open(output_prompt_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(dataset_json, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    generate_cot_prompts()
    