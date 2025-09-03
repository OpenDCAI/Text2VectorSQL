import os
import sys

# --- 配置 ---
# 请确保这个路径指向你存放 .sql 文件的目录
SQL_SCRIPT_DIR = "results/vector_sql_brid" # 或者 "results/vector_sql_spider1"
# --- 配置结束 ---

# 这是需要追加到文件末尾的文本
TEXT_TO_APPEND = "\nCOMMIT;\nPRAGMA foreign_keys = ON;\n"

# 定义从文件末尾读取的字节数，1024字节（1KB）足以检查COMMIT是否存在
CHUNK_SIZE = 1024

def fix_sql_files_fast():
    """
    高效地遍历指定目录下的 .sql 文件，并在末尾追加 COMMIT 命令。
    只检查文件末尾的一小部分，避免读取整个大文件。
    """
    if not os.path.isdir(SQL_SCRIPT_DIR):
        print(f"错误：目录 '{SQL_SCRIPT_DIR}' 不存在。请检查脚本中的路径配置。")
        sys.exit(1)

    print(f"开始高效扫描目录: {SQL_SCRIPT_DIR}")
    
    file_count = 0
    fixed_count = 0

    for filename in os.listdir(SQL_SCRIPT_DIR):
        if filename.endswith(".sql"):
            file_count += 1
            file_path = os.path.join(SQL_SCRIPT_DIR, filename)
            
            try:
                # --- 高效检查逻辑 ---
                is_fixed = False
                # 以二进制读取模式打开文件，用于seek操作
                with open(file_path, 'rb') as f:
                    # 移动到文件末尾附近
                    f.seek(0, os.SEEK_END)
                    file_size = f.tell()
                    
                    # 如果文件本身很小，就从头开始读
                    read_start = max(0, file_size - CHUNK_SIZE)
                    f.seek(read_start)
                    
                    # 读取文件末尾的数据块
                    last_chunk = f.read()
                
                # 在解码后的文本块中检查 "COMMIT;"
                if b"COMMIT;" in last_chunk:
                    is_fixed = True
                # --- 检查结束 ---

                if is_fixed:
                    print(f"跳过 (已修复): {filename}")
                    continue

                # 以追加模式 ('a') 打开文件，在末尾添加内容
                with open(file_path, 'a', encoding='utf-8') as f:
                    f.write(TEXT_TO_APPEND)
                
                print(f"成功修复: {filename}")
                fixed_count += 1

            except Exception as e:
                print(f"修复失败: {filename} - 错误: {e}")

    print("\n--- 修复完成 ---")
    print(f"总共扫描 .sql 文件数: {file_count}")
    print(f"本次成功修复文件数: {fixed_count}")

if __name__ == '__main__':
    fix_sql_files_fast()
