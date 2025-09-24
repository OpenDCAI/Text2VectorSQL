#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
import time
import random
import numpy as np
import math
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
import os
from typing import List, Dict, Any

# --- 新增 --- - 引入 requests 库用于HTTP通信
import requests

# --- 配置日志 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ----------------------------------------------------------
# 1. (新增) 与VLLM服务交互的工具函数
#    用于替换本地的 SentenceTransformer 模型
# ----------------------------------------------------------

def get_embeddings_from_server_batch(texts: List[str], server_url: str, model_name: str) -> np.ndarray:
    """
    向本地的 embedding_server.py 服务发送批量文本请求并返回嵌入向量。
    
    :param texts: 需要编码的文本列表
    :param server_url: VLLM 服务的 URL
    :param model_name: 在 VLLM 服务中加载的模型名称
    :return: 嵌入向量的 Numpy 数组
    """
    if not texts:
        return np.array([])

    # 构建符合 embedding_server.py API 的请求体
    payload = {
        "model": model_name,
        "texts": texts
    }
    try:
        logging.info(f"向 {server_url} 发送 {len(texts)} 条文本进行编码...")
        response = requests.post(server_url, json=payload, timeout=180) # 增加超时以应对大批量
        # 如果请求失败 (如 4xx, 5xx 错误), 抛出异常
        response.raise_for_status()
        result = response.json()
        # 从响应中获取嵌入向量列表
        embeddings = result["embeddings"]
        logging.info("成功从VLLM服务获取嵌入向量。")
        return np.array(embeddings)
    except requests.exceptions.RequestException as e:
        logging.error(f"❌ 请求VLLM服务失败: {e}")
        raise
    except (KeyError, IndexError) as e:
        logging.error(f"❌ 解析VLLM响应失败，检查返回的JSON结构: {result}")
        raise

# ----------------------------------------------------------
# 2. (已移除) 不再需要本地模型加载函数
#    `load_embedding_model` 函数已被删除
# ----------------------------------------------------------

# def visualize_embeddings(embeddings, min_index):
#     pca = PCA(n_components=2)
#     embeddings_2d = pca.fit_transform(embeddings)

#     plt.figure(figsize=(8, 6))

#     plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], color='red', label='Other Points')
#     plt.scatter(embeddings_2d[min_index, 0], embeddings_2d[min_index, 1], color='blue', label='Central Point', s=100)

#     plt.legend()

#     plt.title('2D PCA of Embeddings')
#     plt.xlabel('PCA Component 1')
#     plt.ylabel('PCA Component 2')

#     plt.savefig(f"embeddings/figure-{random.randint(0,10000000000)}")

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

# ----------------------------------------------------------
# 3. (已修改) 主流程函数
# ----------------------------------------------------------
def post_process_questions(input_dataset_path: str, output_file: str, server_url: str, model_name: str):
    """
    主处理函数，现在调用外部服务获取嵌入。
    
    :param input_dataset_path: 输入的JSON文件路径
    :param output_file: 输出的JSON文件路径
    :param server_url: VLLM 服务的 URL
    :param model_name: 在 VLLM 服务中加载的模型名称
    """
    with open(input_dataset_path, 'r', encoding='utf-8') as f:
        input_dataset = json.load(f)

    # --- (修改点) 不再加载本地模型 ---
    # print("loading SentenceTransformer....")
    # embedding_model = load_embedding_model(model_name=model_name_or_path,cache_folder=model_embedding_cache)
    logging.info(f"配置完成，将使用VLLM服务 at {server_url} with model {model_name}")

    valid_questions_num = []
    result_dataset = []
    for data in tqdm(input_dataset, desc="处理问题数据"):
        question_infos = []
        # 假设 'responses' 键存在并且是一个列表
        for response in data.get("responses", []):
            question_info = parse_llm_response(response, data.get("style", ""))
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

            # --- (修改点) 调用外部服务获取嵌入 ---
            try:
                embeddings = get_embeddings_from_server_batch(texts, server_url, model_name)
            except Exception as e:
                logging.error(f"获取嵌入失败，跳过此数据点 (SQL: {data['sql'][:50]}...): {e}")
                # 出现异常时，可以选择随机选一个作为备用方案
                result_dataset.append(integrate_info(data, random.sample(question_infos, 1)[0]))
                continue
            
            # find the index of the question at the central point
            distance_matrix = cdist(embeddings, embeddings, metric = 'cosine') # metric='cityblock' or metric='euclidean'
            distance_sums = distance_matrix.sum(axis = 1)
            min_index = np.argmin(distance_sums)
            
            result_dataset.append(integrate_info(data, question_infos[min_index]))

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result_dataset, f, indent=2, ensure_ascii=False)
    logging.info(f"✅ 处理完成，结果已写入 {output_file}")

    question_num2count = dict()
    for num in valid_questions_num:
        if num in question_num2count:
            question_num2count[num] += 1
        else:
            question_num2count[num] = 1
    print("有效问题数量分布:")
    print(question_num2count)

# ----------------------------------------------------------
# 4. (已修改) CLI - 如何调用
# ----------------------------------------------------------
if __name__ == "__main__":
    # !!!重要!!!: 运行此脚本前，请确保你的 embedding_server.py 服务正在运行!

    # --- 请在这里配置你的参数 ---
    INPUT_FILE = "./results/question_synthesis.json"
    OUTPUT_FILE = "./results/question_and_sql_pairs.json"
    
    # 你的VLLM服务地址和在服务中加载的模型名称
    VLLM_SERVER_URL = "http://127.0.0.1:8000/embed"  # 确保这与你 embedding_server.py 的地址和端口匹配
    MODEL_IN_VLLM = "CLIP-ViT-B-32-laion2B-s34B-b79K" # 必须与你服务 config.yaml 中的 'name' 字段完全匹配

    # 创建输出目录（如果不存在）
    output_dir = os.path.dirname(OUTPUT_FILE)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 调用主函数
    post_process_questions(
        input_dataset_path=INPUT_FILE,
        output_file=OUTPUT_FILE,
        server_url=VLLM_SERVER_URL,
        model_name=MODEL_IN_VLLM
    )
