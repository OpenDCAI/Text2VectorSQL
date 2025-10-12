import json
import re
import os

from tqdm import tqdm

def generate_sql_prompts(dataset_json_path="../pipeline/sqlite/results/toy_spider/candidate_sql.json", tables_json_path="../pipeline/sqlite/results/toy_spider/embedding_table_vector.json", prompt_tamplate_path="../pipeline/sqlite/prompt_templates/sql_generate_prompt_template.txt", output_prompt_path="../pipeline/sqlite/prompts/sql_generate_prompts.json",dataset_backend="sqlite",database_note_prompt_path="../prompt_templates/sqlite_vec_note_prompt.txt",embedding_model_name="all-MiniLM-L6-v2"):
    dataset_json = json.load(open(dataset_json_path))
    tables_json = json.load(open(tables_json_path))
    print("len(question-vecsql):", len(dataset_json))
    
    prompts = []
    db_id2ddls = dict()
    for table in tables_json:
        db_id2ddls[table["db_id"]] = table["ddls"]
    print("len(db_id2ddls):", len(db_id2ddls))

    database_note_prompt = open(database_note_prompt_path).read().format(embedding_model = embedding_model_name)
    prompt_tamplate = open(prompt_tamplate_path).read()
    for data in tqdm(dataset_json):
        if data["external_knowledge"] != "":
            question = data["external_knowledge"] + "\n" + data["question"]
        else:
            question = data["question"]

        data["sql_synthesis_prompt"] = prompt_tamplate.format(
            schema = "\n\n".join(db_id2ddls[data["db_id"]]),
            question = question,
            dataset_backend =dataset_backend,
            database_note_prompt = database_note_prompt
        )

        
    # 创建输出目录
    os.makedirs("../pipeline/sqlite/prompts", exist_ok=True)
    with open(output_prompt_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(dataset_json, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    generate_sql_prompts()
    