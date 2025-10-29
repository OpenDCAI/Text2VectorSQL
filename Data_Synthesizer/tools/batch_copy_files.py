import os
import shutil

# --- 配置区域 ---

# 1. 定义需要处理的数据库列表
DATABASES = [
    "arxiv",
    "bird",
    "spider",
    "synthesis_data",
    "wikipedia_multimodal"
]

# 2. 定义后缀到目标数据库类型的映射关系
#    'pg' -> 'postgresql'
#    'ch' -> 'clickhouse'
SUFFIX_MAP = {
    "pg": "postgresql",
    "ch": "clickhouse"
}

# 3. 定义基础的源目录和目标目录
BASE_SOURCE_DIR = "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/sqlite/results"
BASE_DEST_DIR = "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline"

# --- 脚本主逻辑 ---

def run_batch_copy():
    """
    根据配置，批量复制文件到指定位置。
    """
    print("🚀 开始执行批量文件复制任务...")
    
    copied_count = 0
    skipped_count = 0
    
    # 遍历每一个数据库
    for db_name in DATABASES:
        # 遍历每一种后缀 ('pg' 和 'ch')
        for suffix, db_type in SUFFIX_MAP.items():
            
            # --- 步骤 1: 构建源文件路径 ---
            source_filename = f"input_llm_{suffix}.json"
            source_path = os.path.join(BASE_SOURCE_DIR, db_name, source_filename)
            
            # --- 步骤 2: 构建目标文件路径 ---
            # 目标目录，例如：.../pipeline/postgresql/results/arxiv/
            dest_dir = os.path.join(BASE_DEST_DIR, db_type, "results", db_name)
            # 目标文件，统一命名为 input_llm.json
            dest_path = os.path.join(dest_dir, "input_llm.json")
            
            # --- 步骤 3: 检查源文件是否存在 ---
            if os.path.exists(source_path):
                try:
                    # --- 步骤 4: 确保目标目录存在，如果不存在则创建 ---
                    os.makedirs(dest_dir, exist_ok=True)
                    
                    # --- 步骤 5: 执行文件复制操作 ---
                    # 使用 shutil.copy2 可以同时复制文件内容和元数据（如修改时间）
                    shutil.copy2(source_path, dest_path)
                    print(f"✅ 复制成功: \n   - 源: {source_path}\n   - 至: {dest_path}\n")
                    copied_count += 1
                    
                except (OSError, shutil.Error) as e:
                    print(f"❌ 复制失败: 从 {source_path} 到 {dest_path}\n   - 错误: {e}\n")
                    skipped_count += 1
            else:
                # 如果源文件不存在，则打印提示并跳过
                print(f"⚠️ 源文件不存在，已跳过: {source_path}\n")
                skipped_count += 1
                
    print("--- 任务摘要 ---")
    print(f"总计成功复制: {copied_count} 个文件")
    print(f"总计跳过(或失败): {skipped_count} 个文件")
    print("✨ 批量任务执行完毕。")


if __name__ == "__main__":
    run_batch_copy()