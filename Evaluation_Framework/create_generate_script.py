import os

def generate_bash_script(backends, modes, datasets, model_names, model_paths):
    """
    根据输入的后端、模式、数据集、模型名称和模型路径，生成一个bash脚本。

    Args:
        backends (list): 数据库后端列表, e.g., ['sqlite', 'clickhouse']
        modes (list): 模式列表, e.g., ['api', 'vllm']
        datasets (list): 数据集名称列表, e.g., ['test', 'bird']
        model_names (list): API模型名称列表, e.g., ['gpt-4o']
        model_paths (list): VLLM模型路径列表.
    """
    
    # --- 常量定义 ---
    base_path_template = "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/{backend}/results"
    output_path_template = "./results/{backend}/{dataset}"
    config_path = "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/generate_config.yaml"
    output_script_name = "generate.sh"

    all_commands = []
    
    # --- 循环生成命令 ---
    for backend in backends:
        # 为每个后端添加一个大的标题块
        all_commands.append(f"\n#################################################")
        all_commands.append(f"#    Commands for Backend: {backend.upper()}")
        all_commands.append(f"#################################################\n")

        api_commands = []
        vllm_commands = []

        for mode in modes:
            if mode == 'api':
                for dataset in datasets:
                    for model_name in model_names:
                        dataset_path = f"{base_path_template.format(backend=backend)}/{dataset}/input_llm.json"
                        output_dir = output_path_template.format(backend=backend, dataset=dataset)
                        output_filename = f"out_llm_api_{model_name}.json"
                        output_path = f"{output_dir}/{output_filename}"
                        
                        # 添加创建目录的命令
                        api_commands.append(f"# Ensure output directory exists for {dataset} on {backend}")
                        api_commands.append(f"mkdir -p {output_dir}\n")
                        
                        api_commands.append(f'echo "--- Running API [{model_name}] on DATASET [{dataset}] for BACKEND [{backend}] ---"')
                        command = (
                            f'python generate.py \\\n'
                            f'    --mode "{mode}" \\\n'
                            f'    --dataset "{dataset_path}" \\\n'
                            f'    --output "{output_path}" \\\n'
                            f'    --model_name "{model_name}" \\\n'
                            f'    --config "{config_path}"'
                        )
                        api_commands.append(command)
                        api_commands.append("")

            elif mode == 'vllm':
                for dataset in datasets:
                    for model_path in model_paths:
                        dataset_path = f"{base_path_template.format(backend=backend)}/{dataset}/input_llm.json"
                        model_folder_name = os.path.basename(model_path.strip('/'))
                        output_dir = output_path_template.format(backend=backend, dataset=dataset)
                        output_filename = f"out_llm_vllm_{model_folder_name}.json"
                        output_path = f"{output_dir}/{output_filename}"

                        # 添加创建目录的命令
                        vllm_commands.append(f"# Ensure output directory exists for {dataset} on {backend}")
                        vllm_commands.append(f"mkdir -p {output_dir}\n")

                        vllm_commands.append(f'echo "--- Running VLLM [{model_folder_name}] on DATASET [{dataset}] for BACKEND [{backend}] ---"')
                        command = (
                            f'python generate.py \\\n'
                            f'    --mode "{mode}" \\\n'
                            f'    --dataset "{dataset_path}" \\\n'
                            f'    --model_path "{model_path}" \\\n'
                            f'    --output "{output_path}" \\\n'
                            f'    --config "{config_path}"'
                        )
                        vllm_commands.append(command)
                        vllm_commands.append("")

        # 按API -> VLLM的顺序将命令添加到总列表中
        if api_commands:
            all_commands.append("### API Mode Commands ###\n")
            all_commands.extend(api_commands)
        if vllm_commands:
            all_commands.append("### VLLM Mode Commands ###\n")
            all_commands.extend(vllm_commands)

    # --- 将所有命令写入bash脚本文件 ---
    with open(output_script_name, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# This script is auto-generated to run generation tasks across different backends, modes, and models.\n")
        f.write("set -e  # Exit immediately if a command exits with a non-zero status.\n")
        f.write("\n".join(all_commands))
            
    try:
        os.chmod(output_script_name, 0o755)
        print(f"✅ Bash 脚本 '{output_script_name}' 已成功生成并被设置为可执行文件。")
    except OSError:
        print(f"✅ Bash 脚本 '{output_script_name}' 已成功生成。")


if __name__ == '__main__':
    # --- 在这里输入你的参数 ---
    
    DATABASE_BACKENDS = [
        'sqlite',
        # 'clickhouse',
        # 'postgre'
    ]

    MODES = [
        'api', 
        # 'vllm'
    ]
    
    DATASETS = [
        # 'arxiv', 
        # 'bird', 
        # 'spider',
        'wikipedia_multimodal'
    ]
    
    MODEL_NAMES = [
        'gpt-4o', 
        'gpt-4o-mini',
        'gpt-4-turbo'
    ]
    
    MODEL_PATHS = [
        # '/mnt/b_public/data/ydw/model/Qwen/Qwen2.5-72B-Instruct',
        # '/mnt/b_public/data/ydw/model/Llama/Llama-3-70b-chat-hf'
    ]

    # --- 运行主函数 ---
    generate_bash_script(DATABASE_BACKENDS, MODES, DATASETS, MODEL_NAMES, MODEL_PATHS)
