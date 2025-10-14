import json
import re
import os

from tqdm import tqdm

def generate_input_llm(dataset_json_path="../pipeline/sqlite/results/toy_spider/candidate_sql.json", tables_json_path="../pipeline/sqlite/results/toy_spider/embedding_table_vector.json", prompt_tamplate_path="../pipeline/sqlite/prompt_templates/sql_generate_prompt_template.txt", output_input_path="../pipeline/sqlite/results/toy_spider/input_llm.json",dataset_backend="sqlite",database_note_prompt_path="../pipeline/sqlite/prompt_templates/sqlite_vec_note_prompt.txt",embedding_model_name="all-MiniLM-L6-v2"):
    dataset_json = json.load(open(dataset_json_path))
    print("len(question-vecsql):", len(dataset_json))
    
    if os.path.exists(tables_json_path):
        db_id2ddls = dict()
        tables_json = json.load(open(tables_json_path))
        for table in tables_json:
            db_id2ddls[table["db_id"]] = table["ddls"]
        print("len(db_id2ddls):", len(db_id2ddls))
    else:
        assert "schema" in dataset_json[0], "When tables_json_path not exists, the schema should be in dataset_json"

    database_note_prompt = open(database_note_prompt_path).read().format(embedding_model = embedding_model_name)
    prompt_tamplate = open(prompt_tamplate_path).read()
    for data in tqdm(dataset_json):
        if data["external_knowledge"] != "":
            question = data["external_knowledge"] + "\n" + data["question"]
        else:
            question = data["question"]

        if os.path.exists(tables_json_path):
            schema = "\n\n".join(db_id2ddls[data["db_id"]])
        else:
            schema = data["schema"]

        data["db_type"] = dataset_backend
        data["embedding_model_name"] = embedding_model_name
        data["database_note_prompt"] = database_note_prompt
        data["input"] = prompt_tamplate.format(
            dataset_backend =dataset_backend,
            schema = schema,
            database_note_prompt = database_note_prompt,
            embedding_model_name = embedding_model_name,
            question = question,
        )

        
    # 创建输出目录
    # os.makedirs("../pipeline/sqlite/results", exist_ok=True)
    with open(output_input_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(dataset_json, indent=2, ensure_ascii=False))

def generate_output_llm(dataset_json_path="../pipeline/sqlite/results/toy_spider/input_llm.json", output_path_input="../pipeline/sqlite/results/toy_spider/output_llm.json",dataset_backend="sqlite"):
    dataset_json = json.load(open(dataset_json_path))
    print("len(question-vecsql):", len(dataset_json))
    
    for data in tqdm(dataset_json):

        data['output'] = {
            "sql": data["sql"],
            "cot": data["cot"],
        }

        
    # 创建输出目录
    os.makedirs("../pipeline/sqlite/results", exist_ok=True)
    with open(output_path_input, "w", encoding="utf-8") as f:
        f.write(json.dumps(dataset_json, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    generate_input_llm()
    