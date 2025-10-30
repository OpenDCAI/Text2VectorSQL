import ijson
import json
import random
import argparse
from tqdm import tqdm
import os

def create_model_list():
    """
    创建包含带前缀和不带前缀的40个模型名称的列表。
    """
    base_models = [
        # 商业闭源模型
        "OpenAI/text-embedding-3-large",
        "OpenAI/text-embedding-3-small",
        "OpenAI/text-embedding-ada-02",
        "Voyage-AI/voyage-large-2",
        "Voyage-AI/voyage-code-2",
        "Voyage-AI/voyage-2",
        "Google/text-embedding-004", 
        "Google/text-embedding-gecko@003",
        "Cohere/embed-english-v3.0",
        "Cohere/embed-multilingual-v3.0",
        
        # 开源顶级性能模型
        "BAAI/bge-large-en-v1.5",
        "NVIDIA/NV-Embed-v2",
        "Alibaba-NLP/gte-Qwen2-7B-instruct",
        "intfloat/E5-Mistral-7B-Instruct",
        "Salesforce/SFR-Embedding-2_R",
        "nomic-ai/nomic-embed-text-v1.5",
        "intfloat/e5-large-v2",
        "Alibaba-NLP/gte-large",
        "hkunlp/instructor-xl",
        
        # 开源高效与经典模型
        "sentence-transformers/all-mpnet-base-v2",
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-MiniLM-L12-v2", # 新增：L6 的稍大（12层）版本，性能更强
        "sentence-transformers/msmarco-distilbert-base-v4", # 新增：专为语义搜索（MS MARCO）优化的经典模型
        "princeton-nlp/sup-simcse-bert-base-uncased", # 新增：SimCSE，对比学习领域的经典之作
        "intfloat/e5-base-v2", # 新增：E5 系列的 base 版本
        "BAAI/bge-base-en-v1.5",
        "BAAI/bge-small-en-v1.5",
        "Alibaba-NLP/gte-base",
        "jina-ai/jina-embeddings-v2-base-en",
        "Grit-AI/g-gt-large",
        
        # 开源多语言模型
        "BAAI/bge-m3",
        "intfloat/multilingual-e5-large",
        "intfloat/multilingual-e5-base", # 新增：多语言 E5 的 base 版本
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", # 新增：高效的多语言 MiniLM
        "sentence-transformers/distiluse-base-multilingual-cased-v1", # 新增：SBERT 经典的多语言 DistilUSE
        "sentence-transformers/LaBSE",
        "google-bert/bert-base-multilingual-cased"
    ]
    
    full_list = []
    for model in base_models:
        # 添加带前缀的完整名称
        full_list.append(model)
        # 如果有前缀，则添加不带前缀的名称
        if "/" in model:
            unprefixed_name = model.split('/')[-1]
            full_list.append(unprefixed_name)
            
    # 去重后返回，确保唯一性
    return list(set(full_list))

def process_large_json(input_path, output_path):
    """
    流式处理大型JSON文件，替换指定字符串。
    
    :param input_path: 输入的JSON文件路径。
    :param output_path: 输出的JSON文件路径。
    """
    
    target_string = "all-MiniLM-L6-v2"
    model_replacements = create_model_list()
    
    print(f"输入文件: {input_path}")
    print(f"输出文件: {output_path}")
    print(f"将从 {len(model_replacements)} 个候选项中随机选择模型进行替换...")

    try:
        # 使用 ijson 流式读取顶层数组的每个元素
        with open(input_path, 'rb') as f_in, open(output_path, 'w', encoding='utf-8') as f_out:
            # 开始写入JSON数组
            f_out.write('[\n')
            
            is_first_item = True
            # 使用 ijson.items 解析文件，'item' 表示我们期望一个数组作为根元素
            parser = ijson.items(f_in, 'item')
            
            # 使用 tqdm 显示进度条
            for item in tqdm(parser, desc="正在处理JSON对象"):
                # 确保 item 是一个字典
                if not isinstance(item, dict):
                    continue

                # 为当前这一个字典元素，只选择一个随机的替换值
                chosen_replacement = random.choice(model_replacements)
                
                updated_item = {}
                for key, value in item.items():
                    # 检查字段值是否为字符串
                    if isinstance(value, str):
                        # 替换所有出现的子字符串
                        updated_item[key] = value.replace(target_string, chosen_replacement)
                    else:
                        # 如果不是字符串，保持原样
                        updated_item[key] = value
                
                # 写入处理后的对象
                if not is_first_item:
                    f_out.write(',\n')
                
                json.dump(updated_item, f_out, ensure_ascii=False, indent=2)
                is_first_item = False

            # 结束JSON数组
            f_out.write('\n]')
            
        print("\n处理完成！")

    except Exception as e:
        print(f"\n处理过程中发生错误: {e}")
        # 如果出错，可能需要清理不完整的输出文件
        if os.path.exists(output_path):
            # os.remove(output_path) # 可选：如果希望出错时删除不完整的文件
            print(f"注意: 输出文件 '{output_path}' 可能不完整。")


if __name__ == "__main__":
    # 直接在代码中定义文件路径
    input_file_path = "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/tools/results/mixed_datasets/input_llm_4_1.json"
    output_file_path = "/mnt/b_public/data/ydw/Text2VectorSQL/LLaMA-Factory/data/input_llm_4_1.json"
    
    # 确保文件存在，给出更友好的提示
    if not os.path.exists(input_file_path):
        print(f"错误: 输入文件未找到 -> {input_file_path}")
    else:
        # 直接调用处理函数
        process_large_json(input_file_path, output_file_path)
