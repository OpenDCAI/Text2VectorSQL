import json
import random

# 1. 定义您提供的三个JSON文件的路径列表
file_paths = [
    '/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/sqlite/results/synthesis_data/input_llm.json',
    '/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/postgresql/results/synthesis_data/input_llm.json',
    '/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/clickhouse/results/synthesis_data/input_llm.json'
]

output_file_path = 'collected_input_llm.json'


# 2. 初始化一个空列表，用于存放所有合并后的数据
combined_list = []

print("开始处理文件...")

# 3. 遍历文件路径列表，读取并合并数据
for file_path in file_paths:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # 从文件中加载JSON数据（每个文件都是一个列表）
            data = json.load(f)
            # 使用 extend 方法将当前文件中的列表元素添加到主列表中
            combined_list.extend(data)
            print(f"成功从 {file_path} 加载了 {len(data)} 条数据。")
    except FileNotFoundError:
        print(f"错误：文件未找到 {file_path}")
    except json.JSONDecodeError:
        print(f"错误：无法解析文件中的JSON内容 {file_path}")
    except Exception as e:
        print(f"处理文件 {file_path} 时发生未知错误: {e}")

# 打印合并后的总数据量
print(f"\n数据合并完成。总共聚合了 {len(combined_list)} 条数据。")

# 4. 对合并后的列表进行随机排序 (in-place shuffle)
print("正在对聚合列表进行随机排序...")
random.shuffle(combined_list)
print("列表已成功打乱顺序。")

# (可选) 验证一下，打印前5个元素看看效果
# print("\n打乱顺序后列表的前5个元素示例:")
# for item in combined_list[:5]:
    # print(item)

# (可选) 5. 将最终的列表写入一个新的JSON文件
print(f"\n准备将结果保存到文件: {output_file_path}")
try:
    with open(output_file_path, 'w', encoding='utf-8') as f:
        # 使用 json.dump 将列表写入文件
        # ensure_ascii=False 确保中文字符等能正确显示
        # indent=4 让JSON文件格式化，更易读
        json.dump(combined_list, f, ensure_ascii=False, indent=4)
    print(f"数据已成功保存到 {output_file_path}")
except Exception as e:
    print(f"保存文件时发生错误: {e}")