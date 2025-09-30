# -*- coding: utf-8 -*-
import json
import os

def transform_json_data(input_path: str, output_path: str):
    """
    读取一个包含特定结构字典数组的 JSON 文件，并将其转换为新的字典格式。

    输入格式:
    [
      {
        "query_id": "q1", "db_id": "...", "sql": "...", "sql_candidate": ["..."]
      }, ...
    ]

    输出格式:
    {
      "q1": { "db_name": "...", "sqls": ["...", "..."] }, ...
    }

    Args:
        input_path (str): 输入的 JSON 文件路径。
        output_path (str): 输出的 JSON 文件路径。
    """
    # 定义输入字典必须包含的字段
    required_keys = {"query_id", "db_id", "sql", "sql_candidate"}
    
    try:
        # 1. 读取并解析输入的 JSON 文件
        with open(input_path, 'r', encoding='utf-8') as f:
            source_data = json.load(f)

        # 2. 检查输入数据是否为列表
        if not isinstance(source_data, list):
            print(f"错误: JSON 文件的顶层结构应为一个列表 (array)。文件: {input_path}")
            return

        # 3. 初始化一个新的空字典用于存放结果
        transformed_data = {}

        # 4. 循环处理源列表中的每一个元素
        for index, item in enumerate(source_data):
            # 确保元素是字典并且包含所有必需的字段
            if not isinstance(item, dict):
                print(f"警告: 在索引 {index} 处找到一个非字典元素，已跳过。")
                continue
            
            if not required_keys.issubset(item.keys()):
                print(f"警告: 在索引 {index} 处的字典缺少必要字段，已跳过。必需字段: {required_keys}")
                continue
            
            # 提取所需数据
            query_id = item["query_id"]
            db_name = item["db_id"]
            main_sql = item["sql"]
            candidate_sqls = item["sql_candidate"]

            # 检查 sql_candidate 是否为列表
            if not isinstance(candidate_sqls, list):
                print(f"警告: query_id '{query_id}' 的 'sql_candidate' 字段不是列表，已跳过。")
                continue

            # 构造新的 sqls 列表
            # 将主 sql 字符串放在列表开头，然后拼接上候选 sql 列表
            all_sqls = [main_sql] + candidate_sqls
            
            # 按照目标格式，构建新的字典条目
            transformed_data[query_id] = {
                "db_name": db_name,
                "sqls": all_sqls
            }

        # 5. 将转换后的字典写入新的 JSON 文件
        # indent=4 使输出文件格式优美，易于阅读
        # ensure_ascii=False 保证中文字符的正确显示
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(transformed_data, f, indent=2, ensure_ascii=False) # 使用 indent=2 更接近示例

        print(f"处理成功！共转换 {len(transformed_data)} 条数据。")
        print(f"结果已保存至: {output_path}")

    except FileNotFoundError:
        print(f"错误: 找不到输入文件 '{input_path}'。")
    except json.JSONDecodeError:
        print(f"错误: 文件 '{input_path}' 的内容不是有效的 JSON 格式。")
    except Exception as e:
        print(f"发生了未知错误: {e}")

# --- 使用示例 ---
if __name__ == "__main__":
    input_file = "../data/candidate_sql_query_id.json"
    output_file = "../data/ground_truth.json"

    # 调用核心函数进行转换
    transform_json_data(input_file, output_file)

