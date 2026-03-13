import os
from huggingface_hub import snapshot_download, hf_hub_download

# 配置
REPO_ID = "dongwenyao/Text2vecsql_data"
REPO_TYPE = "dataset"

# 定义可选的下载模块
# 格式：{ "显示名称": "远程与本地对应的路径" }
DATA_MAP = {
    "1": ("database_synthesis/results", "Database Synthesis Results"),
    "2": ("database_synthesis/synthesis_data", "Database Synthesis Data"),
    "3": ("pipeline/sqlite/train", "SQLite Train Data"),
    "4": ("pipeline/sqlite/results", "SQLite Results"),
    "5": ("pipeline/sqlite/prompts", "SQLite Prompts"),
    "6": ("pipeline/postgresql/prompts", "PostgreSQL Prompts"),
    "7": ("pipeline/postgresql/results", "PostgreSQL Results"),
    "8": ("pipeline/myscale/prompts", "MyScale Prompts"),
    "9": ("pipeline/myscale/results", "MyScale Results"),
    "10": ("pipeline/clickhouse/prompts", "ClickHouse Prompts"),
    "11": ("pipeline/clickhouse/results", "ClickHouse Results"),
    "12": ("tools/results/collected_input_llm.json", "Collected Input LLM (Single File)"),
}

def main():
    # 1. 确定项目根目录 (tools 的上一级)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    
    print(f"🚀 目标项目根目录: {project_root}")
    print("请选择要下载的内容 (输入数字，多个请用逗号隔开，输入 'all' 下载全部):")
    
    for key, value in DATA_MAP.items():
        print(f"[{key}] {value[1]} ({value[0]})")

    user_input = input("\n请输入编号: ").strip().lower()
    
    selected_keys = []
    if user_input == 'all':
        selected_keys = list(DATA_MAP.keys())
    else:
        selected_keys = [k.strip() for k in user_input.split(',')]

    for key in selected_keys:
        if key not in DATA_MAP:
            print(f"⚠️ 跳过无效编号: {key}")
            continue
        
        rel_path, description = DATA_MAP[key]
        print(f"\n正在下载: {description}...")

        # 判断是文件还是文件夹
        # 你的脚本中，除了最后一个是 .json，其余都是文件夹
        if rel_path.endswith('.json'):
            # 下载单个文件
            hf_hub_download(
                repo_id=REPO_ID,
                filename=f"Data_Synthesizer/{rel_path}",
                repo_type=REPO_TYPE,
                local_dir=project_root,
                local_dir_use_symlinks=False
            )
        else:
            # 下载文件夹
            snapshot_download(
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                allow_patterns=f"Data_Synthesizer/{rel_path}/*",
                local_dir=project_root,
                local_dir_use_symlinks=False
            )
            
    print("\n✅ 下载任务完成！")

if __name__ == "__main__":
    main()