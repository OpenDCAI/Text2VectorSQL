import ijson
import json
from tqdm import tqdm

def process_large_json(input_path, output_path):
    """
    流式读取一个大型JSON文件，修改每个对象的 'db_id' 字段，
    然后将修改后的对象流式写入一个新的JSON文件。

    Args:
        input_path (str): 输入的大型JSON文件路径。
        output_path (str): 输出的JSON文件路径。
    """
    # 路径的前缀
    prefix = "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/sqlite/results/synthesis_data_deversity/vector_databases/"
    
    # 以二进制模式读取输入文件，以文本模式写入输出文件
    with open(input_path, 'rb') as f_in, open(output_path, 'w', encoding='utf-8') as f_out:
        
        # ijson.items 会返回一个迭代器，逐个产出文件根数组（'item'）中的对象
        # 这样可以避免将整个文件加载到内存中
        json_objects = ijson.items(f_in, 'item')
        
        is_first_item = True
        
        # 手动开始写入JSON数组
        f_out.write('[')
        
        print(f"开始处理文件 {input_path}...")
        
        # 使用tqdm来显示处理进度条
        for item in tqdm(json_objects, desc="处理进度"):
            # 如果不是第一个元素，就在前面加上逗号，以符合JSON数组的格式
            if not is_first_item:
                f_out.write(',')
            
            # 获取原始的 "db_id"
            db_id_name = item.get("db_id")
            
            # 如果存在 "db_id" 字段
            if db_id_name and isinstance(db_id_name, str):
                # 拼接新的文件路径
                # 示例: prefix + db_id_name + / + db_id_name + .sqlite
                new_db_id_path = f"{prefix}{db_id_name}/{db_id_name}.sqlite"
                
                # 更新字典中的值
                item["db_id"] = new_db_id_path
            
            # 将处理过的单个Python字典转换成JSON字符串并写入输出文件
            # ensure_ascii=False 保证中文字符能被正确写入
            json.dump(item, f_out, ensure_ascii=False)
            
            is_first_item = False
        
        # 手动结束JSON数组
        f_out.write(']')
        
    print(f"\n处理完成！结果已保存至 {output_path}")

# ==============================================================================
# =================== 在这里修改您的输入和输出文件名 ===================
# ==============================================================================

# 1. 设置您的原始大JSON文件的名字或路径
input_filename = "embedding_table_vector_old.json" 

# 2. 设置您希望保存结果的新文件名
output_filename = "embedding_table_vector.json"

# ==============================================================================

# 执行主函数
if __name__ == "__main__":
    process_large_json(input_filename, output_filename)
