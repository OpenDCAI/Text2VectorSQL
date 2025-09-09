import json
import re
import time
import random
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np
import math
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def visualize_embeddings(embeddings, min_index):
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))

    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], color='red', label='Other Points')
    plt.scatter(embeddings_2d[min_index, 0], embeddings_2d[min_index, 1], color='blue', label='Central Point', s=100)

    plt.legend()

    plt.title('2D PCA of Embeddings')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')

    plt.savefig(f"embeddings/figure-{random.randint(0,10000000000)}")

def parse_llm_response(response, style):
    explanation_pattern = re.compile(r'\[EXPLANATION-START\](.*?)\[EXPLANATION-END\]', re.DOTALL)
    question_pattern = re.compile(r'\[QUESTION-START\](.*?)\[QUESTION-END\]', re.DOTALL)
    external_knowledge_pattern = re.compile(r'\[EXTERNAL-KNOWLEDGE-START\](.*?)\[EXTERNAL-KNOWLEDGE-END\]', re.DOTALL)

    # 获取所有匹配项并选择最后一个
    explanation_matches = list(explanation_pattern.finditer(response))
    question_matches = list(question_pattern.finditer(response))
    external_knowledge_matches = list(external_knowledge_pattern.finditer(response))

    # 提取最后一个匹配项的内容
    explanation_content = explanation_matches[-1].group(1).strip() if explanation_matches else ""
    question_content = question_matches[-1].group(1).strip() if question_matches else ""
    external_knowledge_content = external_knowledge_matches[-1].group(1).strip() if external_knowledge_matches else ""
    
    if style == "Multi-turn Dialogue":
        # parse dialogue
        try:
            dialog = ""
            for turn in json.loads(question_content):
                dialog += "**" + list(turn.keys())[0] + "**: " + list(turn.values())[0] + "\n"
            question_content = dialog
        except Exception as e:
            print(f"Error parsing dialogue: {e}")
            return None

    if explanation_content == "" or question_content == "":
        return None
    else:
        return {
            "question": question_content.strip(),
            "explanation": explanation_content.strip(),
            "external_knowledge": external_knowledge_content.strip()
        }
    
def integrate_info(sql2question_prompt_info, question_info):
    if sql2question_prompt_info["db_id"].endswith(".db"):
        db_id = sql2question_prompt_info["db_id"][:-3]
    else:
        db_id = sql2question_prompt_info["db_id"]
    return {
        "db_id": db_id,
        "sql": sql2question_prompt_info["sql"],
        "sql_result_column_count": sql2question_prompt_info["column_count"],
        "sql_result_rows_count": sql2question_prompt_info["rows"],
        "sql_complexity": sql2question_prompt_info["complexity"],
        "question_style": sql2question_prompt_info["style"],
        "sql_explanation": question_info["explanation"],
        "question": question_info["question"],
        "external_knowledge": question_info["external_knowledge"]
    }

def edu_distance(vector1, vector2):
    distance = 0
    for num1, num2 in zip(vector1, vector2):
        distance += (num1-num2) ** 2
    return math.sqrt(distance)

def post_process_questions(input_dataset_path="./results/question_synthesis.json",output_file = "./results/question_and_sql_pairs.json",model_name_or_path = "sentence-transformers/all-mpnet-base-v2"):
    input_dataset = json.load(open(input_dataset_path))

    print("loading SentenceTransformer....")
    embedding_model = SentenceTransformer(model_name_or_path)

    valid_questions_num = []
    result_dataset = []
    for data in tqdm(input_dataset):
        question_infos = []
        for response in data["responses"]:
            question_info = parse_llm_response(response, data["style"])
            if question_info is not None:
                question_infos.append(question_info)
        
        valid_questions_num.append(len(question_infos))

        if len(question_infos) == 0: # no valid question
            continue
        elif len(question_infos) == 1: # only one valid question
            result_dataset.append(integrate_info(data, question_infos[0]))
        elif len(question_infos) == 2: # two valid questions
            # we randomly select one of them
            result_dataset.append(integrate_info(data, random.sample(question_infos, 1)[0]))
        else: # more than two valid questions
            # we vote the final question according to the EK+question embeddings
            texts = [question_info["external_knowledge"] + " " + question_info["question"] for question_info in question_infos]
            texts = [text.strip() for text in texts]

            # we vote the final question according to the question embeddings
            # texts = [question_info["question"] for question_info in question_infos]
            embeddings = embedding_model.encode(texts)
            
            # find the index of the question at the central point
            distance_matrix = cdist(embeddings, embeddings, metric = 'cosine') # metric='cityblock' or metric='euclidean'
            distance_sums = distance_matrix.sum(axis = 1)
            min_index = np.argmin(distance_sums)
            
            result_dataset.append(integrate_info(data, question_infos[min_index]))

            # print("EK:\n", integrate_info(data, question_infos[min_index])["external_knowledge"])
            # print("Question:\n", integrate_info(data, question_infos[min_index])["question"])
            # print("SQL:\n", integrate_info(data, question_infos[min_index])["sql"])
            # print("---------------------------------------")
            # visualize_embeddings(embeddings, min_index)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(result_dataset, indent=2, ensure_ascii=False))

    question_num2count = dict()
    for num in valid_questions_num:
        if num in question_num2count:
            question_num2count[num] += 1
        else:
            question_num2count[num] = 1
    print(question_num2count)
