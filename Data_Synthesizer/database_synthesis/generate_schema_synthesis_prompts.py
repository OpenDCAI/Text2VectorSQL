import json
import random
import numpy as np
import os

def generate_a_normal_integer(mean = 10, std_dev = 4, lower_bound = 1, upper_bound = 20):
    sample = np.random.normal(mean, std_dev)
    sample = np.clip(sample, lower_bound, upper_bound)
    
    return int(sample)

if __name__ == '__main__':
    random.seed(42)
    tables = json.load(open("web_tables.json", "r", encoding = "utf-8"))
    prompt_template = open("./prompt_templates/schema_prompt.txt", "r", encoding = "utf-8").read()

    prompts = []
    for table in tables:
        random_table_num = generate_a_normal_integer()
        print(random_table_num)
        prompt = prompt_template.format(
            table_num = random_table_num, 
            table = table
        )
        prompts.append(prompt.strip())

    random.shuffle(prompts)

    # Check whether output dir exists
    output_dir = "./prompts"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"目录 {output_dir} 已创建")
    else:
        print(f"目录 {output_dir} 已存在")
    with open("./prompts/prompts_schema_synthesis.json", "w", encoding = "utf-8") as file:
        file.write(json.dumps(prompts, ensure_ascii = False, indent = 2))