import os
from collections import defaultdict

def find_duplicate_prefix_sqlite_files(root_dir):
    """
    递归遍历指定目录，查找具有相同前缀的 .sqlite 文件。

    Args:
        root_dir (str): 要开始搜索的根目录的路径。

    Returns:
        dict: 一个字典，其中键是 .sqlite 文件的前缀，
              值是具有该前缀的文件路径列表。
              仅包含具有多个文件的条目。
    """
    if not os.path.isdir(root_dir):
        print(f"错误：提供的路径 '{root_dir}' 不是一个有效的目录。")
        return {}

    sqlite_files_by_prefix = defaultdict(list)

    # 递归遍历目录树
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.sqlite'):
                # 获取文件名（不包括.sqlite扩展名）作为前缀
                prefix = filename[:-7]  # ".sqlite" 的长度是 7
                full_path = os.path.join(dirpath, filename)
                sqlite_files_by_prefix[prefix].append(full_path)

    # 筛选出具有多个文件（即重复前缀）的条目
    duplicate_files = {
        prefix: paths for prefix, paths in sqlite_files_by_prefix.items() if len(paths) > 1
    }

    return duplicate_files

if __name__ == '__main__':
    # --- 使用说明 ---
    # 1. 将下面的 'your_target_directory' 替换为您要搜索的实际目录路径。
    #    例如: '/home/user/documents' 或 'C:\\Users\\User\\Documents'
    target_directory = '/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/sqlite/results' 

    # 检查并查找具有相同前缀的.sqlite文件
    duplicates = find_duplicate_prefix_sqlite_files(target_directory)

    if not duplicates:
        print(f"在目录 '{target_directory}' 及其子目录中没有找到具有相同前缀的 .sqlite 文件。")
    else:
        print(f"在 '{target_directory}' 中找到了以下具有相同前缀的 .sqlite 文件：\n")
        for prefix, paths in duplicates.items():
            print(f"前缀: '{prefix}.sqlite'")
            for path in paths:
                print(f"  - {path}")
            print("-" * 20)