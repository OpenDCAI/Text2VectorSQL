import os
import tarfile
from huggingface_hub import snapshot_download

# ================= 配置区 =================
REPO_ID = "dongwenyao/text2vecsql_zip"
REPO_TYPE = "dataset"
# 假设脚本在 Data_Synthesizer/tools 下，项目根目录就是脚本的上一级
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# ==========================================

def decompress_tar_gz(file_path, extract_path):
    """解压 tar.gz 文件到指定目录并删除原文件"""
    print(f"📦 正在解压: {file_path}")
    try:
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(path=extract_path)
        print(f"✅ 解压成功: {extract_path}")
        
        # 解压成功后删除压缩包
        os.remove(file_path)
        print(f"🗑️ 已清理压缩包: {os.path.basename(file_path)}")
        return True
    except Exception as e:
        print(f"❌ 解压失败 {file_path}: {e}")
        return False

def main():
    print(f"🚀 开始从 Hugging Face 下载仓库: {REPO_ID}")
    
    # 1. 下载整个 Data_Synthesizer 目录
    # local_dir 指定为项目根目录，snapshot_download 会自动处理子目录结构
    try:
        download_path = snapshot_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            local_dir=PROJECT_ROOT,
            allow_patterns="Data_Synthesizer/*",
            local_dir_use_symlinks=False
        )
        print(f"✅ 下载完成，存放在: {download_path}")
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return

    # 2. 遍历下载后的目录，寻找 .tar.gz 文件进行还原
    print("\n🔍 正在扫描压缩文件并还原目录结构...")
    
    for root, dirs, files in os.walk(PROJECT_ROOT):
        for file in files:
            if file.endswith(".tar.gz"):
                full_file_path = os.path.join(root, file)
                
                # 你的上传逻辑中，压缩包内包含了目标文件夹本身
                # 例如：pipeline/sqlite/train.tar.gz 解压后会自动生成 train/ 文件夹
                # 所以解压路径应该是该压缩包所在的父目录
                extract_target_dir = root 
                
                decompress_tar_gz(full_file_path, extract_target_dir)

    print("\n🎊 恭喜！所有数据已还原，环境清理完毕。")

if __name__ == "__main__":
    main()