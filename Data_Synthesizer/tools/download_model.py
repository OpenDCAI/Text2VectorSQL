import os
from huggingface_hub import snapshot_download

# 配置
REPO_ID = "dongwenyao/Text2Vecsql"
MODEL_PATH = "model/UniVectorSQL-7B-LoRA"

def main():
    # 1. 自动获取项目根目录 (tools 的上一级是 Data_Synthesizer)
    # 如果你的项目结构是 Root/Data_Synthesizer/tools
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    
    # 2. 构造本地目标路径
    # 这样模型会放在 Data_Synthesizer/model/UniVectorSQL-7B-LoRA
    local_target_path = project_root
    
    print(f"🔍 正在从仓库 {REPO_ID} 下载模型...")
    print(f"📂 目标目录: {os.path.join(local_target_path, MODEL_PATH)}")

    try:
        # 3. 执行下载
        # allow_patterns 确保只下载指定的文件夹内容
        snapshot_download(
            repo_id=REPO_ID,
            allow_patterns=f"{MODEL_PATH}/*",
            local_dir=local_target_path,
            local_dir_use_symlinks=False,
            resume_download=True, # 如果中断了可以继续
            token=True # 如果是私有仓库，会自动读取你登录的 HF Token
        )
        
        print("\n✅ 模型下载完成！")
        print(f"🚀 位置: {os.path.join(local_target_path, MODEL_PATH)}")
        
    except Exception as e:
        print(f"\n❌ 下载过程中出错: {e}")

if __name__ == "__main__":
    main()