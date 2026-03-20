import os
import tarfile
from huggingface_hub import snapshot_download

# ================= 配置区 =================
REPO_ID = "dongwenyao/text2vecsql_zip"
REPO_TYPE = "dataset"
# 脚本在 Data_Synthesizer/tools 下，两级向上到达项目总根目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# ==========================================

def decompress_tar_gz(file_path, extract_path):
    print(f"📦 正在解压还原: {file_path}")
    try:
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(path=extract_path)
        os.remove(file_path)
        return True
    except Exception as e:
        print(f"❌ 解压失败: {e}")
        return False

def main():
    print(f"🚀 目标项目根目录: {PROJECT_ROOT}")
    
    # 1. 下载仓库所有内容
    try:
        snapshot_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            local_dir=PROJECT_ROOT,
            local_dir_use_symlinks=False,
            # 允许下载所有文件夹：Data_Synthesizer, Embedding_Service, Evaluation_Framework
            allow_patterns=["*"] 
        )
        print(f"✅ 下载/同步完成！")
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return

    # 2. 全局扫描并解压
    print("\n🔍 正在全局扫描压缩包并还原结构...")
    for root, dirs, files in os.walk(PROJECT_ROOT):
        # 跳过 .git 等隐藏目录
        if '.git' in root: continue
        
        for file in files:
            if file.endswith(".tar.gz"):
                full_file_path = os.path.join(root, file)
                # 解压到当前压缩包所在的父目录
                extract_target_dir = os.path.dirname(full_file_path)
                decompress_tar_gz(full_file_path, extract_target_dir)

    print("\n🎊 所有项目组件（Embedding/Data/Evaluation）已还原到位！")

if __name__ == "__main__":
    main()