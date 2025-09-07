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

if __name__ == "__main__":
    dataset = json.load(open("./results/question_and_sql_pairs.json"))
    tables = json.load(open("./results/enhanced_embedding_table_vector.json"))
    print("len(tables):", len(tables))
    
    prompts = []
    db_id2ddls = dict()
    for table in tables:
        db_id2ddls[table["db_id"]] = table["ddls"]
    print("len(db_id2ddls):", len(db_id2ddls))

    prompt_tamplate = open("./prompt_templates/cot_synthesis_prompt_template.txt").read()
    for data in tqdm(dataset):
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
    os.makedirs("./prompts", exist_ok=True)
    with open("./prompts/cot_synthesis_prompts.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(dataset, indent=2, ensure_ascii=False))
    