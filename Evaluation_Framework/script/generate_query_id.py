# -*- coding: utf-8 -*-
import json
import os

def add_query_ids_to_json(input_path: str, output_path: str):
    """
    读取一个包含字典列表的 JSON 文件，为每个字典添加一个唯一的 'query_id'，
    然后将结果写入新的 JSON 文件。

    Args:
        input_path (str): 输入的 JSON 文件路径。
        output_path (str): 输出的 JSON 文件路径。
    """
    try:
        # 1. 读取输入的 JSON 文件
        # 使用 'r' 模式和 utf-8 编码确保能正确处理中文字符
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 2. 检查数据格式是否为列表
        if not isinstance(data, list):
            print(f"错误: JSON 文件的顶层结构不是一个列表/数组。文件路径: {input_path}")
            return

        # 3. 遍历列表中的每个元素（字典）并添加字段
        # 使用 enumerate(..., start=1) 可以同时获得索引和元素，索引从 1 开始
        for index, item in enumerate(data, start=1):
            # 确保列表中的元素是字典，以防数据格式混淆
            if isinstance(item, dict):
                # 构造 query_id, 例如 "q1", "q2", ...
                query_id = f"q{index}"
                # 为字典添加新的键值对
                item['query_id'] = query_id
            else:
                print(f"警告: 在索引 {index-1} 处找到一个非字典类型的元素，已跳过。")

        # 4. 将修改后的数据写入新的 JSON 文件
        # 使用 'w' 模式写入
        # indent=4 让输出的 JSON 文件格式化，更易于阅读
        # ensure_ascii=False 确保中文字符能正常显示，而不是被转义成 \uXXXX
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        
        print(f"处理完成！已成功为 {len(data)} 个元素添加 query_id。")
        print(f"结果已保存至: {output_path}")

    except FileNotFoundError:
        print(f"错误: 找不到输入文件。请检查路径是否正确: {input_path}")
    except json.JSONDecodeError:
        print(f"错误: 文件内容不是有效的 JSON 格式。文件路径: {input_path}")
    except Exception as e:
        print(f"发生未知错误: {e}")

# --- 使用示例 ---
if __name__ == "__main__":
    # 定义输入和输出文件名
    input_file = "../data/candidate_sql.json"
    output_file = "../data/candidate_sql_query_id.json"

    # 调用函数进行处理
    add_query_ids_to_json(input_file, output_file)
