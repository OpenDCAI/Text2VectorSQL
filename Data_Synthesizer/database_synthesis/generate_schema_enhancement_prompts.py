import json
import random

random.seed(42)

if __name__ == '__main__':
    prompts = []
    prompt_template = open("./prompt_templates/enhance_prompt.txt", "r", encoding = "utf-8").read()
    schema_synthesis_results = json.load(open("./results/schema_synthesis.json"))

    no_res_num = 0
    for data in schema_synthesis_results:
        try:
            if data["generated_content"] == {}:
                no_res_num += 1
                continue

            domain = data["generated_content"]["domain"]
            scenario = data["generated_content"]["scenario"]
            schema_str = data["generated_content"]["schema"]
            
            prompts.append(
                prompt_template.format(domain = domain, scenario = scenario, schema = schema_str).strip()
            )

        except Exception as e:
            print(e)

    print("no_res_num:", no_res_num)
    print("len(prompts):", len(prompts))
    random.shuffle(prompts)

    with open("./prompts/prompts_schema_enhancement.json", "w", encoding="utf-8") as file:
        file.write(json.dumps(prompts, ensure_ascii=False, indent=2))