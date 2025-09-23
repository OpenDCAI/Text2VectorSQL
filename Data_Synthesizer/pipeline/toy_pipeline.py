import yaml
import os
from pprint import pprint
from pathlib import Path
import sys

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

# --------------------------------------------------------------------
# 安装提示 (Installation Tip)
# --------------------------------------------------------------------
# 如果您的环境中没有 PyYAML 库，请先安装它。
# You need to install the PyYAML library if you don't have it.
# pip install PyYAML
# --------------------------------------------------------------------
class ServicesConfig:
    """封装服务相关的配置"""
    def __init__(self, services_dict: dict):
        self.vllm = services_dict.get('vllm', {})
        self.openai = services_dict.get('openai', {})

class PathsConfig:
    """封装所有路径相关的配置"""
    def __init__(self, paths_dict: dict):
        # 使用 .get() 为每个属性赋值，以增加灵活性
        self.generate_tables_json_path = paths_dict.get('generate_tables_json_path')
        self.result_path = paths_dict.get('result_path')
        self.enhance_json_name = paths_dict.get('enhance_json_name')
        self.source_db_root = paths_dict.get('source_db_root')
        self.sql_script_dir = paths_dict.get('sql_script_dir')
        self.vector_db_root = paths_dict.get('vector_db_root')
        self.find_semantic_table_json = paths_dict.get('find_semantic_table_json')
        self.EMBED_MODEL_PATH_CACHE = paths_dict.get('EMBED_MODEL_PATH_CACHE')
        self.find_semantic_prompt_template_path = paths_dict.get('find_semantic_prompt_template_path')
        self.model_download = paths_dict.get('model_download')
        self.cache_file = paths_dict.get('cache_file')
        self.original_schema = paths_dict.get('original_schema')
        self.schema_output_dir = paths_dict.get('schema_output_dir')
        self.schema_output_json = paths_dict.get('schema_output_json')
        self.prompt_tpl_path = paths_dict.get('prompt_tpl_path')
        self.functions_path = paths_dict.get('functions_path')
        self.sql_prompts_output_dir = paths_dict.get('sql_prompts_output_dir')
        self.sql_prompts_output_name = paths_dict.get('sql_prompts_output_name')
        self.synthesize_sql_input_file = paths_dict.get('synthesize_sql_input_file')
        self.synthesize_sql_output_file = paths_dict.get('synthesize_sql_output_file')
        self.EMBED_MODEL_PATH = paths_dict.get('EMBED_MODEL_PATH')
        self.post_sql_output_path = paths_dict.get('post_sql_output_path')
        self.post_sql_llm_json_path = paths_dict.get('post_sql_llm_json_path')
        self.sql_infos_path = paths_dict.get('sql_infos_path')
        self.question_synthesis_template_path = paths_dict.get('question_synthesis_template_path')
        self.question_prompts_output_json_path = paths_dict.get('question_prompts_output_json_path')
        self.synthesize_question_input_file = paths_dict.get('synthesize_question_input_file')
        self.synthesize_question_output_file = paths_dict.get('synthesize_question_output_file')
        self.post_process_questions_input_dataset_path = paths_dict.get('post_process_questions_input_dataset_path')
        self.post_process_questions_output_file = paths_dict.get('post_process_questions_output_file')
        self.model_name_or_path = paths_dict.get('model_name_or_path')
        self.synthesiaze_candidate_input_file = paths_dict.get('synthesiaze_candidate_input_file')
        self.synthesiaze_candidate_output_file = paths_dict.get('synthesiaze_candidate_output_file')
        self.gene_cot_prompts_dataset_json_path = paths_dict.get('gene_cot_prompts_dataset_json_path')
        self.gene_cot_prompts_tables_json_path = paths_dict.get('gene_cot_prompts_tables_json_path')
        self.gene_cot_prompts_prompt_tamplate_path = paths_dict.get('gene_cot_prompts_prompt_tamplate_path')
        self.gene_cot_prompts_output_prompt_path = paths_dict.get('gene_cot_prompts_output_prompt_path')
        self.synthesize_cot_output_file = paths_dict.get('synthesize_cot_output_file')
        self.cache_file_path_cot = paths_dict.get('cache_file_path_cot')
        self.post_process_cot_results_path = paths_dict.get('post_process_cot_results_path')
        self.post_process_cot_db_dir = paths_dict.get('post_process_cot_db_dir')
        self.post_process_cot_output_dir = paths_dict.get('post_process_cot_output_dir')
        # 嵌套的字典结构可以直接赋值
        # self.enhance_vector = paths_dict.get('enhance_vector', {})
        # self.sql_vectorize = paths_dict.get('sql_vectorize', {})
        # self.post_process = paths_dict.get('post_process', {})

class ParametersConfig:
    """封装所有参数相关的配置"""
    def __init__(self, params_dict: dict):
        self.max_workers = params_dict.get('max_workers')
        self.enhance_table_mode = params_dict.get('enhance_table_mode')
        self.no_parallel_find_semantic_rich = params_dict.get('no_parallel_find_semantic_rich')
        self.num_cpus = params_dict.get('num_cpus')
        self.sql_exec_timeout = params_dict.get('sql_exec_timeout')
        self.num_candidates = params_dict.get('num_candidates')

class AppConfig:
    """主配置类，整合所有配置部分"""
    def __init__(self, base_dir: str, services_dict: dict, paths_dict: dict, params_dict: dict):
        self.base_dir = base_dir
        self.services = ServicesConfig(services_dict)
        self.paths = PathsConfig(paths_dict)
        self.parameters = ParametersConfig(params_dict)

def load_config(database: str, dataset: str, config_path: str = 'config.yaml') -> AppConfig:
    """
    从 YAML 文件中加载、解析并封装配置。

    Args:
        database (str): 数据库类型 (例如 'sqlite').
        dataset (str): 数据集名称 (例如 'toy_spider').
        config_path (str): YAML 配置文件的路径.

    Returns:
        AppConfig: 一个包含所有配置的 AppConfig 对象实例。
        
    Raises:
        FileNotFoundError: 如果配置文件不存在。
        KeyError: 如果在配置文件中找不到指定的 database 或 dataset。
    """
    config_file = Path(config_path)
    if not config_file.is_file():
        raise FileNotFoundError(f"配置文件未找到: {config_file.resolve()}")

    with open(config_file, 'r', encoding='utf-8') as f:
        full_config = yaml.safe_load(f)

    # --- 逐级解析配置 ---
    try:
        db_config = full_config[database]
    except KeyError:
        raise KeyError(f"在 '{config_path}' 中找不到数据库配置: '{database}'")
    
    try:
        dataset_config = db_config[dataset]
    except KeyError:
        raise KeyError(f"在 '{database}' 配置下找不到数据集: '{dataset}'")
        
    # 提取顶层和各部分的配置字典
    base_dir = db_config.get('base_dir')
    services_dict = dataset_config.get('services', {})
    paths_dict = dataset_config.get('paths', {})
    params_dict = dataset_config.get('parameters', {})
    
    # 创建并返回总配置对象
    return AppConfig(base_dir, services_dict, paths_dict, params_dict)

def create_directory_with_os(directory_name: str):
    """
    使用 os 模块在当前工作目录下创建一个新目录。

    Args:
        directory_name (str): 要创建的目录的名称。
    """
    try:
        # os.makedirs 可以一次性创建所有必需的中间目录
        # exist_ok=True 的作用和 pathlib 中一样
        os.makedirs(directory_name, exist_ok=True)
        
        # os.path.abspath 可以获取绝对路径
        abs_path = os.path.abspath(directory_name)
        print(f"目录 '{abs_path}' 创建成功或已存在。")
        
    except OSError as e:
        print(f"创建目录时出错: {e}")

def main():
    # 配置hugging face代理
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

    # 加载配置
    try:
        # 1. 调用函数加载 'sqlite' 的 'toy_spider' 配置
        config = load_config(database="sqlite", dataset="toy_spider")

        # 2. 演示如何通过点符号访问所有变量
        print("--- 配置加载成功! ---")
        print(f"\n基础目录 (Base Directory): {config.base_dir}")
        create_directory_with_os(config.base_dir)

        print("\n--- 服务配置 (Services) ---")
        print(f"  OpenAI API Key: ...{config.services.openai.get('api_key')[:6]}")
        print(f"  OpenAI base_url: ...{config.services.openai.get('base_url')}")
        print(f"  OpenAI llm_model_name: ...{config.services.openai.get('llm_model_name')}")
        print(f"  embedding_model_name: ...{config.services.openai.get('embedding_model_name')}")
        

        print("\n--- 路径配置 (Paths) ---")
        print(f"  源数据库根目录: {config.paths.source_db_root}")
        print(f"  模型下载路径: {config.paths.model_download}")

        print("\n--- 参数配置 (Parameters) ---")
        print(f"  最大工作线程数: {config.parameters.max_workers}")
        print(f"  SQL 执行超时: {config.parameters.sql_exec_timeout}s")
        
        # 也可以使用 pprint 打印某个配置部分的完整内容
        # print("\n--- 完整的路径配置字典 ---")
        # vars() 可以将对象的属性转为字典以便打印
        # pprint(vars(config.paths))
    except (FileNotFoundError, KeyError, yaml.YAMLError) as e:
        print(f"错误: 配置加载失败 - {e}")

    # 开始执行pipeline
    # 如果数据库没有tables.json文件，那么你就必须运行这个代码为数据库生成tables.json文件
    print("################################################")
    print("为数据库生成schema")
    generate_schema(config.paths.source_db_root, config.paths.generate_tables_json_path)

    print("################################################")
    print("将每张表格的两行样本数据填入schema")
    process_toy_dataset(config.paths.source_db_root,config.paths.result_path,config.paths.enhance_json_name)
    enhance_json_path = os.path.join(config.paths.result_path,config.paths.enhance_json_name)

    print("################################################")
    print("查找所有数据库中语义丰富的列，并计入schema")
    main_find_rich_semantic_column(config.services.openai.get('llm_model_name'),config.services.openai.get('api_key'),config.services.openai.get('base_url'),enhance_json_path,config.paths.find_semantic_table_json,config.parameters.no_parallel_find_semantic_rich,config.paths.find_semantic_prompt_template_path)

    print("################################################")
    print("为语义丰富的列生成embedding，并构建向量数据库")
    main_batch_vectorize_databases(config.paths.source_db_root,config.paths.sql_script_dir,config.paths.vector_db_root,config.paths.find_semantic_table_json,config.services.openai.get('embedding_model_name'),config.paths.EMBED_MODEL_PATH_CACHE)

    print("################################################")
    print("为向量数据库更新schema，并且添加ddls字段，便于生成cot")
    generate_vector_schema(config.paths.vector_db_root,config.paths.find_semantic_table_json,config.paths.schema_output_dir,config.paths.schema_output_json)

    print("################################################")
    print("生成合成sql提示词")
    main_generate_sql_synthesis_prompts(config.paths.vector_db_root,config.paths.prompt_tpl_path,config.paths.functions_path,config.paths.sql_prompts_output_dir,config.paths.sql_prompts_output_name,config.services.openai.get('embedding_model_name'))

    print("################################################")
    print("合成sql")
    run_sql_synthesis(config.paths.synthesize_sql_input_file,config.paths.synthesize_sql_output_file,config.services.openai.get('llm_model_name'),config.services.openai.get('api_key'),config.services.openai.get('base_url'),True)

    print("################################################")   
    print("过滤可以成功运行的sql")
    post_process_sqls(config.paths.EMBED_MODEL_PATH,config.services.openai.get('embedding_model_name'),config.paths.vector_db_root,config.paths.post_sql_output_path,config.paths.post_sql_llm_json_path,8,30)

    print("################################################")
    print("生成合成question提示词")
    sqlite_generate_question_synthesis_prompts(config.paths.vector_db_root,config.paths.sql_infos_path,config.paths.question_synthesis_template_path,config.paths.question_prompts_output_json_path)

    print("################################################")
    print("合成question")
    synthesize_questions(config.paths.synthesize_question_input_file,config.paths.synthesize_question_output_file,config.services.openai.get('llm_model_name'),config.services.openai.get('api_key'),config.services.openai.get('base_url'),config.parameters.max_workers)

    print("################################################")
    print("多数投票表决选取合适的question")
    post_process_questions(config.paths.post_process_questions_input_dataset_path,config.paths.post_process_questions_output_file,config.paths.model_name_or_path)

    print("################################################")
    print("为每个问题生成多个sql候选")
    synthesize_candidate(config.services.openai.get('llm_model_name'),config.services.openai.get('api_key'),config.services.openai.get('base_url'),config.parameters.num_candidates,config.parameters.max_workers,config.paths.synthesiaze_candidate_input_file,config.paths.synthesiaze_candidate_output_file)

    print("################################################")
    print("生成合成cot提示词")
    generate_cot_prompts(config.paths.gene_cot_prompts_dataset_json_path,config.paths.gene_cot_prompts_tables_json_path,config.paths.gene_cot_prompts_prompt_tamplate_path,config.paths.gene_cot_prompts_output_prompt_path)

    print("################################################")
    print("合成cot")
    synthesize_cot(config.paths.gene_cot_prompts_output_prompt_path,config.paths.synthesize_cot_output_file,config.services.openai.get('llm_model_name'),config.services.openai.get('api_key'),config.services.openai.get('base_url'),config.parameters.max_workers,config.paths.cache_file_path_cot,5,0.8)

    print("################################################")
    print("选择合适的cot，并且过滤可以成功运行的sql")
    post_process_cot(config.paths.post_process_cot_results_path,config.paths.post_process_cot_db_dir,config.paths.post_process_cot_output_dir,config.paths.EMBED_MODEL_PATH)

    

if __name__ == "__main__":
    main()
