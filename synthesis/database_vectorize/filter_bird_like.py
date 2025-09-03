import json
import os

def filter_sql_queries(input_path='results/vector_train.json', output_dir='results'):
    """
    筛选BIRD数据集中在WHERE子句里进行字符串匹配的SQL查询。

    Args:
        input_path (str): 输入的train.json文件路径。
        output_dir (str): 输出目录的路径。
    """
    # 定义输出文件路径
    vector_output_path = os.path.join(output_dir, 'vector_train_like.json')
    remain_output_path = os.path.join(output_dir, 'vector_remain_train.json')

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 初始化两个列表，用于存放筛选结果
    string_match_items = []
    remaining_items = []

    # 读取并处理JSON文件
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误：找不到文件 {input_path}。请确保文件路径正确。")
        return
    except json.JSONDecodeError:
        print(f"错误：文件 {input_path} 不是有效的JSON格式。")
        return

    # 遍历数据集中的每一个元素
    for item in data:
        # 获取SQL查询语句，如果不存在则视为空字符串
        sql_query = item.get("SQL", "")
        
        # 将SQL语句转为大写以便于不区分大小写地查找 'WHERE'
        sql_upper = sql_query.upper()

        is_string_match = False
        if ' WHERE ' in sql_upper:
            # 分割字符串，获取WHERE子句之后的部分
            where_clause = sql_query[sql_upper.find(' WHERE ') + len(' WHERE '):]
            
            # 核心判断：如果WHERE子句中包含LIKE，我们假定它正在进行模糊匹配
            # 这是一个简单但对大多数SQL方言都有效的判断方法
            if "LIKE" in where_clause:
                is_string_match = True

        import random
 
        # 创建一个包含四个True和四个False的列表
        values = [True] + [False] * 4
        
        # 随机选择一个元素
        selected_value = random.choice(values)
        
        # 根据判断结果将元素放入对应的列表
        if is_string_match or selected_value:
            string_match_items.append(item)
        else:
            remaining_items.append(item)

    # 将结果写入对应的JSON文件
    try:
        with open(vector_output_path, 'w', encoding='utf-8') as f:
            # indent=4 使JSON文件格式化，更易读
            # ensure_ascii=False 确保中文字符能正常写入
            json.dump(string_match_items, f, indent=4, ensure_ascii=False)

        with open(remain_output_path, 'w', encoding='utf-8') as f:
            json.dump(remaining_items, f, indent=4, ensure_ascii=False)

        # 打印成功信息
        print("✅ 筛选完成！")
        print(f"筛选出 {len(string_match_items)} 个符合条件的项目。")
        print(f"剩下 {len(remaining_items)} 个项目。")
        print("\n结果已成功写入：")
        print(f"  - {vector_output_path}")
        print(f"  - {remain_output_path}")

    except IOError as e:
        print(f"错误：写入文件时发生错误。 {e}")


if __name__ == '__main__':
    # 假设你的vector_train.json在 'vector_train/vector_train.json'下
    filter_sql_queries(input_path='results/vector_train.json')
