import os
import sys

def add_prefix_to_dirs_and_files(target_directory, prefix="deverse_2_"):
    """
    遍历指定目录，为符合条件的子目录和其中的同名 .sqlite 文件添加前缀。

    条件:
    1. 目标必须是 `target_directory` 下的一个子目录。
    2. 该子目录中必须包含一个与子目录同名的 .sqlite 文件。
    
    例如:
    - /your/path/folder1/folder1.sqlite
    将被重命名为:
    - /your/path/deverse_2_folder1/deverse_2_folder1.sqlite

    Args:
        target_directory (str): 需要处理的根目录路径。
        prefix (str): 要添加的前缀。
    """
    # 检查目标目录是否存在
    if not os.path.isdir(target_directory):
        print(f"错误：目录 '{target_directory}' 不存在或不是一个有效的目录。")
        sys.exit(1) # 退出脚本

    print(f"开始扫描目录: {target_directory}\n")
    
    # 获取目录下所有的文件和文件夹名
    try:
        entries = os.listdir(target_directory)
    except OSError as e:
        print(f"错误：无法访问目录 '{target_directory}'。请检查权限。")
        print(f"详细信息: {e}")
        sys.exit(1)

    renamed_count = 0
    # 遍历所有条目
    for dir_name in entries:
        old_dir_path = os.path.join(target_directory, dir_name)

        # 检查当前条目是否是一个目录
        if os.path.isdir(old_dir_path):
            # 构造原始 sqlite 文件的路径
            sqlite_file_name = dir_name + ".sqlite"
            old_sqlite_file_path = os.path.join(old_dir_path, sqlite_file_name)

            # 检查同名的 sqlite 文件是否存在
            if os.path.isfile(old_sqlite_file_path):
                print(f"找到匹配项: 目录 '{dir_name}' 和文件 '{sqlite_file_name}'")
                
                # 1. 重命名目录
                new_dir_name = prefix + dir_name
                new_dir_path = os.path.join(target_directory, new_dir_name)
                
                try:
                    print(f"  -> 正在重命名目录 '{dir_name}' 为 '{new_dir_name}'")
                    os.rename(old_dir_path, new_dir_path)

                    # 2. 重命名目录内的 sqlite 文件
                    #    注意：此时目录已经改名，所以要用 new_dir_path
                    file_to_rename_path = os.path.join(new_dir_path, sqlite_file_name)
                    new_sqlite_file_name = new_dir_name + ".sqlite"
                    new_sqlite_file_path = os.path.join(new_dir_path, new_sqlite_file_name)
                    
                    print(f"  -> 正在重命名文件 '{sqlite_file_name}' 为 '{new_sqlite_file_name}'")
                    os.rename(file_to_rename_path, new_sqlite_file_path)

                    print("  -> 操作成功！\n")
                    renamed_count += 1
                
                except OSError as e:
                    print(f"  -> 操作失败！错误: {e}\n")
                    # 如果目录重命名成功但文件失败，尝试恢复目录名
                    if not os.path.exists(old_dir_path):
                        os.rename(new_dir_path, old_dir_path)
                        print(f"  -> 已将目录恢复为 '{dir_name}'")


    if renamed_count > 0:
        print(f"处理完成！总共重命名了 {renamed_count} 个目录和文件对。")
    else:
        print("处理完成！没有找到符合条件的目录和文件对。")


if __name__ == "__main__":
    # --- 请修改这里的路径 ---
    # Windows 示例: "C:\\Users\\YourUser\\Desktop\\my_databases"
    # macOS/Linux 示例: "/Users/youruser/Documents/my_databases"
    target_directory = "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/sqlite/results/synthesis_data_deversity/vector_databases"  # <--- 修改这里为你需要处理的目录路径！

    # 运行主函数
    add_prefix_to_dirs_and_files(target_directory)
