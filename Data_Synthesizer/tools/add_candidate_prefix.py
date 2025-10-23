import json
import sys

def add_prefix_to_db_id(input_file, output_file, prefix="deverse_2_"):
    """
    读取一个JSON文件，为文件中字典数组的每个'db_id'字段添加前缀，
    并保存到新的文件中。

    Args:
        input_file (str): 输入的JSON文件名。
        output_file (str): 输出的JSON文件名。
        prefix (str): 要添加的前缀。
    """
    # --- 1. 读取并解析JSON文件 ---
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"✅ 成功读取文件: '{input_file}'")
    except FileNotFoundError:
        print(f"❌ 错误: 输入文件 '{input_file}' 未找到。请检查文件名和路径。")
        sys.exit(1) # 退出脚本
    except json.JSONDecodeError:
        print(f"❌ 错误: 文件 '{input_file}' 不是有效的JSON格式。")
        sys.exit(1)

    # 检查数据是否为列表
    if not isinstance(data, list):
        print(f"❌ 错误: JSON文件的顶层结构不是一个数组（列表）。")
        sys.exit(1)

    # --- 2. 遍历并修改数据 ---
    modified_count = 0
    for item in data:
        # 确保元素是字典并且包含 'db_id' 键
        if isinstance(item, dict) and 'db_id' in item:
            original_id = item['db_id']
            item['db_id'] = prefix + original_id
            modified_count += 1
            # print(f"  - 已修改: '{original_id}' -> '{item['db_id']}'") # 如果需要详细日志可以取消此行注释

    print(f"🔄 已处理 {len(data)} 个元素，其中 {modified_count} 个元素的 'db_id' 被修改。")

    # --- 3. 将修改后的数据写入新文件 ---
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            # indent=2 使输出的JSON文件格式化，更易读
            # ensure_ascii=False 确保中文字符等能被正确写入
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"✅ 操作完成！结果已保存到: '{output_file}'")
    except IOError as e:
        print(f"❌ 错误: 无法写入到文件 '{output_file}'。")
        print(f"详细信息: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # --- 请在这里配置你的文件名 ---
    input_filename = "cot_synthesis_old.json"  # <--- 你的原始JSON文件名
    output_filename = "cot_synthesis.json" # <--- 你希望保存的新文件名

    # 运行主函数
    add_prefix_to_db_id(input_filename, output_filename)

