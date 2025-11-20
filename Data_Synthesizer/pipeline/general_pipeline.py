# 这个文件是通用pipeline，可以用来生成训练数据（你需要先准备好向量数据库）
# main文件中放了所有算子，你可以选择性地使用它们。
# 如果是多模态的wiki数据集的处理，它比其他数据集要多一个build_final_db_with_images算子
# 如果是训练数据，可以把cot相关的算子也加进来
import yaml
import os
from pprint import pprint
from pathlib import Path
import sys

# 配置hugging face代理
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 只需要修改这里，就可以加载不同的数据集配置！
DATASET_BACKEND = "postgresql" # "myscale" "clickhouse" "postgresql" 
DATASET_TO_LOAD = "arxiv" # "arxiv" "wikipedia_multimodal" "bird" "spider"
# "synthesis_data"用于训练集
# DATASET_TO_LOAD = "bird" # 例如，切换到bird数据集

# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)

# 获取 project 目录的路径 (当前文件的父目录的父目录)
project_root_path = os.path.dirname(os.path.dirname(current_file_path))

# 将 project 目录添加到 sys.path
if project_root_path not in sys.path:
    sys.path.append(project_root_path)


from vectorization.batch_vectorize_databases import main_batch_vectorize_databases
from vectorization.enhance_tables_json import process_toy_dataset
from vectorization.find_semantic_rich_column import main_find_rich_semantic_column
from vectorization.generate_schema import generate_schema
from vectorization.generate_vector_schema import generate_vector_schema

from synthesis_nl.generate_question_synthesis_prompts import sqlite_generate_question_synthesis_prompts
from synthesis_nl.synthesize_question import synthesize_questions
from synthesis_nl.post_process_questions import post_process_questions
from synthesis_nl.synthesize_candidate import synthesize_candidate

from synthesis_cot.generate_cot_synthesis_prompts import generate_cot_prompts
from synthesis_cot.synthesize_cot import synthesize_cot
from synthesis_cot.post_process_cot import post_process_cot

from synthesis_sql.generate_sql_synthesis_prompts import main_generate_sql_synthesis_prompts
from synthesis_sql.synthesize_sql import run_sql_synthesis
from synthesis_sql.post_process_sqls import post_process_sqls

from vectorization.image_embedding_adder import build_final_db_with_images

from synthesis_eval.generate_input import generate_input_llm
from synthesis_eval.generate_input import generate_output_llm

# --------------------------------------------------------------------
# 安装提示 (Installation Tip)
# --------------------------------------------------------------------
# 如果您的环境中没有 PyYAML 库，请先安装它。
# You need to install the PyYAML library if you don't have it.
# pip install PyYAML
# --------------------------------------------------------------------
from typing import Any

class DynamicConfig:
    def __init__(self, config_dict: dict):
        if config_dict:
            for key, value in config_dict.items():
                # 如果值是字典，也将其转换为DynamicConfig实例，以支持链式调用
                if isinstance(value, dict):
                    setattr(self, key, DynamicConfig(value))
                else:
                    setattr(self, key, value)
    
    # 增加一个get方法以安全地获取属性，类似字典
    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

class ServicesConfig(DynamicConfig):
    pass

class PathsConfig(DynamicConfig):
    pass

class ParametersConfig(DynamicConfig):
    pass

class AppConfig:
    def __init__(self, base_dir: str, services_dict: dict, paths_dict: dict, params_dict: dict):
        self.base_dir = base_dir
        self.services = ServicesConfig(services_dict)
        self.paths = PathsConfig(paths_dict)
        self.parameters = ParametersConfig(params_dict)

# -------------------------------------------------------------------
#  在这里添加辅助函数并修改 load_config
# -------------------------------------------------------------------
def _format_config_paths(config_node: Any, dataset_name: str) -> Any:
    """
    【新增的辅助函数】
    递归地遍历配置节点（字典或列表），格式化所有字符串。
    """
    if isinstance(config_node, dict):
        return {key: _format_config_paths(value, dataset_name) for key, value in config_node.items()}
    elif isinstance(config_node, list):
        return [_format_config_paths(item, dataset_name) for item in config_node]
    elif isinstance(config_node, str):
        # 核心逻辑：用实际的数据集名称替换占位符 {dataset}
        return config_node.format(dataset=dataset_name)
    else:
        return config_node

def load_config(database: str, dataset: str, config_path: str = 'config.yaml') -> AppConfig:
    """
    【修改后的函数】
    从 YAML 文件中加载、解析、格式化并封装配置。
    """
    config_file = Path(config_path)
    if not config_file.is_file():
        raise FileNotFoundError(f"配置文件未找到: {config_file.resolve()}")

    with open(config_file, 'r', encoding='utf-8') as f:
        full_config = yaml.safe_load(f)

    try:
        db_config = full_config[database]
        dataset_config = db_config[dataset]
    except KeyError as e:
        raise KeyError(f"在 '{config_path}' 中找不到配置路径: {database}.{dataset} - {e}")
        
    base_dir = db_config.get('base_dir')
    # 1. 先像原来一样获取原始的字典
    services_dict = dataset_config.get('services', {})
    paths_dict = dataset_config.get('paths', {})
    params_dict = dataset_config.get('parameters', {})
    
    # 2. 【关键新增步骤】调用辅助函数，用 `dataset` 变量格式化路径字典
    formatted_paths_dict = _format_config_paths(paths_dict, dataset)
    
    # 3. 将格式化后的字典传入 AppConfig
    return AppConfig(base_dir, services_dict, formatted_paths_dict, params_dict)

def create_directory_with_os(directory_name: str):
    """
    使用 os 模块创建目录。
    如果路径已作为文件存在，则发出警告；如果目录已存在，则静默处理。
    """
    # 检查目标路径是否已经存在，并且是一个文件
    if os.path.exists(directory_name) and not os.path.isdir(directory_name):
        # 如果是文件，打印警告并直接返回，不做任何事
        print(f"warning: 路径 '{directory_name}' 已作为文件存在, 无法创建同名目录。")
        return

    # 尝试创建目录，exist_ok=True 会处理目录已存在的情况
    try:
        os.makedirs(directory_name, exist_ok=True)
    except OSError as e:
        # 捕获其他可能的OS错误
        print(f"创建目录 '{directory_name}' 时发生未知错误: {e}")

def main():
    try:
        config = load_config(database=DATASET_BACKEND, dataset=DATASET_TO_LOAD)

        print(f"--- 成功加载 '{DATASET_TO_LOAD}' 数据集的配置! ---")
        
        # 访问路径时，它们已经是被正确替换后的最终路径了
        print("\n--- 动态生成的路径 ---")
        # 使用 hasattr() 安全地检查并打印每个路径
        if hasattr(config.paths, 'source_db_root'):
            print(f"  源数据库根目录: {config.paths.source_db_root}")

        if hasattr(config.paths, 'result_path'):
            print(f"  结果存放路径:   {config.paths.result_path}")

        if hasattr(config.paths, 'cache_file'):
            print(f"  缓存文件路径:   {config.paths.cache_file}")

        # --- 【关键修改】更准确、更稳健的目录创建逻辑 ---
        print("\n正在创建所有必需的目录...")
        
        for path_name, path_value in vars(config.paths).items():
            if not isinstance(path_value, str) or not path_value:
                continue # 如果值不是字符串或为空，则跳过

            # 使用 os.path.splitext() 来判断一个路径是否指向文件
            # 如果路径有扩展名（如 .json, .txt），我们认为它是一个文件路径。
            _, file_extension = os.path.splitext(path_value)
            
            if file_extension:
                # 如果有文件扩展名, 说明这是一个文件路径。
                # 我们需要创建的是它的【父目录】。
                directory_to_create = os.path.dirname(path_value)
            # else:
            #     # 如果没有文件扩展名, 我们假定它本身就是一个【目录路径】。
            #     directory_to_create = path_value
            
            # 只有当 `directory_to_create` 非空时才创建
            if directory_to_create:
                create_directory_with_os(directory_to_create)

        print("所有相关目录已创建完毕。")


    except (FileNotFoundError, KeyError, yaml.YAMLError) as e:
        print(f"错误: 配置加载失败 - {e}")

    # #开始执行pipeline

    # #如果数据库没有tables.json文件，那么你就必须运行这个代码为数据库生成tables.json文件
    # print("################################################")
    # print("为数据库生成schema")
    # generate_schema(config.paths.source_db_root, config.paths.generate_tables_json_path)




    # print("################################################")
    # print("将每张表格的两行样本数据填入schema")
    # process_toy_dataset(config.paths.source_db_root,config.paths.result_path,config.paths.enhance_json_name)

    # print("################################################")
    # print("查找所有数据库中语义丰富的列，并计入schema")
    # enhance_json_path = os.path.join(config.paths.result_path,config.paths.enhance_json_name)
    # main_find_rich_semantic_column(config.services.openai.get('llm_model_name'),config.services.openai.get('api_key'),config.services.openai.get('base_url'),enhance_json_path,config.paths.find_semantic_table_json,config.parameters.no_parallel_find_semantic_rich,config.paths.find_semantic_prompt_template_path)

    # print("################################################")
    # print("为语义丰富的列生成embedding，并构建向量数据库")
    # main_batch_vectorize_databases(config.paths.source_db_root,config.paths.sql_script_dir,config.paths.vector_db_root,config.paths.find_semantic_table_json,config.services.openai.get('embedding_model_name'),config.paths.EMBED_MODEL_PATH_CACHE)



    # # 这个算子只有wiki数据库需要
    # print("################################################")
    # print("为Image表格中的图片添加embedding")
    # build_final_db_with_images(config.paths.intermediate_sql_path,config.paths.final_image_vec_db_path,config.paths.target_vec_db_path,config.paths.table_to_modify,config.paths.model_name_or_path,config.paths.model_cache_directory,config.paths.article_table,config.paths.image_root_directory)




    # print("################################################")
    # print("为向量数据库更新schema，并且添加ddls字段，便于生成cot")
    # generate_vector_schema(config.paths.vector_db_root,config.paths.find_semantic_table_json,config.paths.schema_output_dir,config.paths.schema_output_json)

    # print("################################################")
    # print("生成合成sql提示词")
    # main_generate_sql_synthesis_prompts(config.paths.vector_db_root,config.paths.prompt_tpl_path,config.paths.functions_path,config.paths.sql_prompts_output_dir,config.paths.sql_prompts_output_name,config.services.openai.get('embedding_model_name'),config.parameters.sql_number)

    # print("################################################")
    # print("合成sql")
    # run_sql_synthesis(config.paths.synthesize_sql_input_file,config.paths.synthesize_sql_output_file,config.services.openai.get('llm_model_name'),config.services.openai.get('api_key'),config.services.openai.get('base_url'),config.paths.cache_file_path_sql,config.parameters.max_workers)

    # print("################################################")   
    # print("过滤可以成功运行的sql")
    # post_process_sqls(config.paths.vector_db_root,config.paths.post_sql_output_path,config.paths.post_sql_llm_json_path,config.services.embed.get('api_url'),config.services.openai.get('embedding_model_name'),config.parameters.num_cpus,config.parameters.sql_exec_timeout)

    # print("################################################")
    # print("生成合成question提示词")
    # sqlite_generate_question_synthesis_prompts(config.paths.vector_db_root,config.paths.sql_infos_path,config.paths.question_synthesis_template_path,config.paths.question_prompts_output_json_path)

    # print("################################################")
    # print("合成question")
    # synthesize_questions(config.paths.synthesize_question_input_file,config.paths.synthesize_question_output_file,config.services.openai.get('llm_model_name'),config.services.openai.get('api_key'),config.services.openai.get('base_url'),config.parameters.max_workers,config.paths.cache_file_path_question)

    # print("################################################")
    # print("多数投票表决选取合适的question")
    # post_process_questions(config.paths.post_process_questions_input_dataset_path,config.paths.post_process_questions_output_file,config.services.embed.get('api_url'),config.services.openai.get('embedding_model_name'))

    # print("################################################")
    # print("为每个问题生成多个sql候选")
    # synthesize_candidate(config.services.openai.get('llm_model_name'),config.services.openai.get('api_key'),config.services.openai.get('base_url'),config.parameters.num_candidates,config.parameters.max_workers,config.paths.synthesiaze_candidate_input_file,config.paths.synthesiaze_candidate_output_file)
    # print("如果发现结果少了,可能是执行过快,这里要重新执行一遍")



    # #下面的算子只有训练数据集需要
    # print("################################################")
    # print("生成合成cot提示词")
    # generate_cot_prompts(config.paths.gene_cot_prompts_dataset_json_path,config.paths.gene_cot_prompts_tables_json_path,config.paths.gene_cot_prompts_prompt_tamplate_path,config.paths.database_note_prompt_path,config.paths.gene_cot_prompts_output_prompt_path, config.parameters.dataset_backend, config.services.openai.get('embedding_model_name'))

    # print("################################################")
    # print("合成cot")
    # synthesize_cot(config.paths.gene_cot_prompts_output_prompt_path,config.paths.synthesize_cot_output_file,config.services.openai.get('llm_model_name'),config.services.openai.get('api_key'),config.services.openai.get('base_url'),config.parameters.max_workers,config.paths.cache_file_path_cot,5,0.8, config.parameters.dataset_backend)

    # print("################################################")
    # print("选择合适的cot，并且过滤可以成功运行的sql")
    # post_process_cot(config.paths.post_process_cot_results_path,config.paths.post_process_cot_output_dir, config.parameters.dataset_backend)




    # 下面的算子是生成llm的输入输出时, 给我llm做微调用的
    print("################################################")
    print("生成llm输入")
    generate_input_llm(config.paths.dataset_json_path_input, config.paths.tables_json_path_input, config.paths.prompt_tamplate_path_infer_input, config.paths.output_path_input,config.parameters.dataset_backend, config.paths.database_note_prompt_path,config.services.openai.get('embedding_model_name'))

    # # 下面的算子是生成llm的输入输出时, 给我llm做微调用的（只有训练数据需要）（弃用）
    # print("################################################")
    # print("生成llm输出")
    # generate_output_llm(config.paths.output_path_input, config.paths.output_llm_train_path)

if __name__ == "__main__":
    main()
