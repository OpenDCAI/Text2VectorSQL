import os
import json
import sqlite3
import argparse
from tqdm import tqdm

def get_schema_for_db(db_path):
    """
    连接到单个 SQLite 数据库，检查其 schema，并返回一个字典。

    Args:
        db_path (str): 数据库文件的路径。

    Returns:
        dict: 包含数据库 schema 信息的字典，如果出错则返回 None。
    """
    try:
        # 从文件路径中提取 db_id
        db_id = os.path.splitext(os.path.basename(db_path))[0]
        
        # 初始化 schema 字典结构
        schema = {
            "db_id": db_id,
            "table_names_original": [],
            "table_names": [],
            "column_names_original": [[-1, "*"]], # 包含通配符 '*'
            "column_names": [[-1, "*"]],
            "column_types": ["text"],
            "primary_keys": [],
            "foreign_keys": []
        }

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 1. 获取所有表名
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        table_names = [row[0] for row in cursor.fetchall()]
        schema["table_names_original"] = table_names
        schema["table_names"] = table_names
        
        # 用于快速查找表名对应的索引
        table_name_to_idx = {name: i for i, name in enumerate(table_names)}
        
        # 用于后续查找外键时，快速定位列的全局索引
        column_to_global_idx = {}
        current_col_idx = 1 # 从 1 开始，因为 0 被 '*' 占用

        # 2. 遍历每个表，获取列信息和主键
        for table_idx, table_name in enumerate(table_names):
            cursor.execute(f'PRAGMA table_info("{table_name}");')
            columns_info = cursor.fetchall()

            for col in columns_info:
                # col 格式: (cid, name, type, notnull, dflt_value, pk)
                col_name = col[1]
                col_type = col[2].lower()
                is_primary_key = col[5] == 1

                # 添加列信息
                schema["column_names_original"].append([table_idx, col_name])
                schema["column_names"].append([table_idx, col_name])
                schema["column_types"].append(col_type)

                # 记录主键
                if is_primary_key:
                    schema["primary_keys"].append(current_col_idx)
                
                # 建立 (表名, 列名) -> 全局索引 的映射
                column_to_global_idx[(table_name, col_name)] = current_col_idx
                current_col_idx += 1

        # 3. 遍历每个表，获取外键信息
        for table_name in table_names:
            cursor.execute(f'PRAGMA foreign_key_list("{table_name}");')
            foreign_keys_info = cursor.fetchall()

            for fk in foreign_keys_info:
                # fk 格式: (id, seq, table, from, to, on_update, on_delete, match)
                from_table = table_name
                from_column = fk[3]
                to_table = fk[2]
                to_column = fk[4]

                # 查找源列和目标列的全局索引
                from_col_idx = column_to_global_idx.get((from_table, from_column))
                to_col_idx = column_to_global_idx.get((to_table, to_column))

                if from_col_idx is not None and to_col_idx is not None:
                    schema["foreign_keys"].append([from_col_idx, to_col_idx])

        conn.close()
        return schema

    except sqlite3.Error as e:
        print(f"  [错误] 处理数据库 {db_id} 失败: {e}")
        return None


def main():
    """
    主函数，用于解析命令行参数，并为指定目录中的所有数据库生成 schema。
    """
    parser = argparse.ArgumentParser(description="为目录中的 SQLite 数据库生成 Schema JSON 文件。")
    parser.add_argument(
        "--db-dir", 
        required=True, 
        help="包含 .sqlite 或 .db 数据库文件的目录路径。"
    )
    parser.add_argument(
        "--output-file", 
        required=True, 
        help="生成的 schema JSON 文件的输出路径。"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.db_dir):
        print(f"✖ 错误：目录 '{args.db_dir}' 不存在。")
        return

    print(f"🚀 开始从目录 '{args.db_dir}' 生成 schema...")

    db_files = [
        os.path.join(args.db_dir, f) 
        for f in os.listdir(args.db_dir) 
        if f.endswith(('.sqlite', '.db'))
    ]

    if not db_files:
        print("🟡 警告：在指定目录中未找到 .sqlite 或 .db 文件。")
        return

    all_schemas = []
    for db_path in tqdm(db_files, desc="处理数据库中"):
        schema_data = get_schema_for_db(db_path)
        if schema_data:
            all_schemas.append(schema_data)

    # 确保输出目录存在
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    try:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(all_schemas, f, indent=2, ensure_ascii=False)
        print(f"\n✔ 成功创建 schema 文件 '{args.output_file}'，包含 {len(all_schemas)} 个数据库。")
    except IOError as e:
        print(f"\n✖ 写入输出文件失败: {e}")


if __name__ == '__main__':
    main()
