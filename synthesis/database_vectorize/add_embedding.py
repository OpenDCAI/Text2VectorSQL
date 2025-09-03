import json
import argparse
from collections import defaultdict

# 配置参数
STRING_TYPES = {'varchar', 'text', 'string', 'char'}
SEMANTIC_KEYWORDS = {
    'name', 'title', 'description', 'summary', 'content', 
    'comment', 'note', 'detail', 'message', 'bio',
    'label', 'value', 'status', 'type', 'category'
}

def is_semantic_column(column_name, column_type):
    """判断列是否包含语义信息"""
    name_lower = column_name.lower()
    type_lower = column_type.lower()
    
    # 排除ID和标志性列
    if '_id' in name_lower or 'id_' in name_lower:
        return False
    if 'flag' in name_lower or 'is_' in name_lower:
        return False
    
    # 检查类型和名称中的关键字
    return (any(kw in name_lower for kw in SEMANTIC_KEYWORDS) and
            any(st in type_lower for st in STRING_TYPES))

def generate_embeddings_json(input_file, output_file):
    """生成带有嵌入列的新JSON结构"""
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # 记录每个表需要添加嵌入列的位置和名称
    table_modifications = defaultdict(dict)
    
    for table in data["tables"]:
        new_column_names = []
        new_column_types = []
        insert_positions = []  # 记录在哪些列后插入嵌入列
        
        # 第一轮：收集原始列和插入位置
        for i, (col_name, col_type) in enumerate(zip(table["column_names"], table["column_types"])):
            new_column_names.append(col_name)
            new_column_types.append(col_type)
            
            # 如果是语义列且没有对应的嵌入列
            if is_semantic_column(col_name, col_type):
                embedding_col_name = f"{col_name}_embedding"
                # 检查是否已有嵌入列
                if embedding_col_name not in new_column_names:
                    insert_positions.append(i + len(insert_positions) + 1)
        
        # 第二轮：在收集的位置插入嵌入列
        for pos in insert_positions:
            orig_col_name = new_column_names[pos-1]
            embedding_col_name = f"{orig_col_name}_embedding"
            
            new_column_names.insert(pos, embedding_col_name)
            new_column_types.insert(pos, "BLOB")
            
            # 记录需要修改的位置
            table_modifications[table["table_name"]][embedding_col_name] = pos
    
    # 更新示例数据
    for table in data["tables"]:
        modifications = table_modifications[table["table_name"]]
        if not modifications:
            continue
        
        new_sample_rows = []
        for row in table["sample_rows"]:
            new_row = list(row)
            # 在指定位置插入嵌入列的null值
            for pos in sorted(modifications.values(), reverse=True):
                new_row.insert(pos, None)
            new_sample_rows.append(new_row)
        
        # 应用新列名和类型
        table["column_names"] = new_column_names
        table["column_types"] = new_column_types
        table["sample_rows"] = new_sample_rows
    
    # 保存结果
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Successfully generated {output_file} with embedding columns")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='生成带有嵌入列的JSON文件')
    parser.add_argument('input', help='输入的JSON文件路径')
    parser.add_argument('output', help='输出的JSON文件路径')
    args = parser.parse_args()
    
    generate_embeddings_json(args.input, args.output)
