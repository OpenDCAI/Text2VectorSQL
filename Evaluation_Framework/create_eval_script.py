import os

def generate_eval_script(datasets, eval_report_files, db_types, api_models, vllm_models, exec_results_files):
    """
    根据输入的参数数组，生成一个用于运行评估流程的bash脚本。
    """
    
    # --- 常量定义 ---
    # Use paths relative to project root (auto-detected from this script's location)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_dir_template = os.path.join(project_root, "Data_Synthesizer/pipeline/{db_type}/results/{dataset}/vector_databases")
    eval_data_file_template = os.path.join(project_root, "Evaluation_Framework/results/{db_type}/{dataset}/out_llm_{model_type}_{model_name_safe}.json")
    
    # --- 修改点: 在输出路径模板中加入了 {dataset} ---
    output_dir_template = "./results/{db_type}/{dataset}"
    output_script_name = "run_evaluation.sh"

    all_commands = []
    
    # --- 循环所有组合来生成命令 ---
    for db_type in db_types:
        all_commands.append(f"\n#################################################")
        all_commands.append(f"#    Evaluation for DB Backend: {db_type.upper()}")
        all_commands.append(f"#################################################\n")

        for dataset in datasets:
            # --- 修改点: 将目录创建和定义移到数据集循环内部 ---
            # 这样可以为每个数据集创建独立的输出文件夹
            output_dir = output_dir_template.format(db_type=db_type, dataset=dataset)
            all_commands.append(f"# Ensure output directory for {db_type}/{dataset} exists")
            all_commands.append(f"mkdir -p {output_dir}\n")

            # --- API 模型循环 ---
            if api_models:
                all_commands.append(f"# --- API Models on Dataset: {dataset} for DB: {db_type} ---")
            for model_name in api_models:
                eval_report_base = eval_report_files[0].rsplit('.', 1)[0]
                exec_results_base = exec_results_files[0].rsplit('.', 1)[0]

                base_dir = base_dir_template.format(db_type=db_type, dataset=dataset)
                eval_data_file = eval_data_file_template.format(db_type=db_type, dataset=dataset, model_type='api', model_name_safe=model_name)
                
                evaluation_report_file = f"{output_dir}/{eval_report_base}_api_{model_name}.json"
                execution_results_file = f"{output_dir}/{exec_results_base}_api_{model_name}.json"

                all_commands.append(f'echo "==> Evaluating API Model [{model_name}] on Dataset [{dataset}] for DB [{db_type}]"')
                command = (
                    f'python run_eval_pipeline.py --all \\\n'
                    f'    --base_dir "{base_dir}" \\\n'
                    f'    --evaluation_report_file "{evaluation_report_file}" \\\n'
                    f'    --db_type "{db_type}" \\\n'
                    f'    --eval_data_file "{eval_data_file}" \\\n'
                    f'    --execution_results_file "{execution_results_file}"'
                )
                all_commands.append(command + "\n")

            # --- VLLM 模型循环 ---
            if vllm_models:
                all_commands.append(f"# --- VLLM Models on Dataset: {dataset} for DB: {db_type} ---")
            for model_path in vllm_models:
                eval_report_base = eval_report_files[0].rsplit('.', 1)[0]
                exec_results_base = exec_results_files[0].rsplit('.', 1)[0]
                
                model_name_safe = os.path.basename(model_path.strip('/')).replace('/', '-')

                base_dir = base_dir_template.format(db_type=db_type, dataset=dataset)
                eval_data_file = eval_data_file_template.format(db_type=db_type, dataset=dataset, model_type='vllm', model_name_safe=model_name_safe)
                
                evaluation_report_file = f"{output_dir}/{eval_report_base}_vllm_{model_name_safe}.json"
                execution_results_file = f"{output_dir}/{exec_results_base}_vllm_{model_name_safe}.json"

                all_commands.append(f'echo "==> Evaluating VLLM Model [{model_name_safe}] on Dataset [{dataset}] for DB [{db_type}]"')
                command = (
                    f'python run_eval_pipeline.py --all \\\n'
                    f'    --base_dir "{base_dir}" \\\n'
                    f'    --evaluation_report_file "{evaluation_report_file}" \\\n'
                    f'    --db_type "{db_type}" \\\n'
                    f'    --eval_data_file "{eval_data_file}" \\\n'
                    f'    --execution_results_file "{execution_results_file}"'
                )
                all_commands.append(command + "\n")

    # --- 将所有命令写入bash脚本文件 ---
    with open(output_script_name, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# This script is auto-generated to run the evaluation pipeline.\n")
        f.write("set -e  # Exit immediately if a command exits with a non-zero status.\n")
        f.write("\n".join(all_commands))
            
    try:
        os.chmod(output_script_name, 0o755)
        print(f"✅ Bash 脚本 '{output_script_name}' 已成功生成并被设置为可执行文件。")
    except OSError:
        print(f"✅ Bash 脚本 '{output_script_name}' 已成功生成。")


if __name__ == '__main__':
    # --- 在这里配置你的参数数组 ---
    
    DATASETS = [
        'arxiv',
        'bird',
        'wikipedia_multimodal',
        'spider'
    ]

    EVALUATION_REPORT_FILES = [
        'evaluation_report.json'
    ]

    DB_TYPES = [
        # 'sqlite',
        # 'clickhouse',
        # 'postgresql',
        'myscale'
    ]
    
    API_MODELS = [
        # 'gpt-4o', 
        # 'gpt-4o-mini',
        # 'gpt-4-turbo',
        # 'claude-3-5-haiku-20241022',
        # 'claude-3-7-sonnet-20250219',
        # 'claude-4-sonnet',
        # 'gemini-2.5-flash',
        # 'gemini-2.5-pro',
        # 'qwen2.5-72b-instruct',
        # 'qwen2.5-32b-instruct',
        # 'deepseek-v3.1-250821',
        # 'grok-3',
        # 'grok-4',
        # 'command-r-plus-08-2024'
    ]
    
    # VLLM model paths - replace with your local model paths.
    # Download models via: python Data_Synthesizer/tools/download_model.py
    # Or specify HuggingFace model IDs / local paths.
    VLLM_MODELS = [
        # './models/UniVectorSQL-7B-LoRA-Step600',
        # './models/UniVectorSQL-7B-LoRA-Step800',
        # './models/UniVectorSQL-7B-LoRA-Step1000',
    ]
    
    EXECUTION_RESULTS_FILES = [
        'sql_execution_results.json'
    ]

    # --- 运行主函数 ---
    generate_eval_script(
        DATASETS,
        EVALUATION_REPORT_FILES,
        DB_TYPES,
        API_MODELS,
        VLLM_MODELS,
        EXECUTION_RESULTS_FILES
    )
