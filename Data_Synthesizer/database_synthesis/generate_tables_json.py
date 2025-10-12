import json
import sqlite3
import sqlite_vec
import os
import re
import traceback
from tqdm import tqdm

def obtain_db_ddls(db_file_dir):
    conn = sqlite3.connect(db_file_dir)
    cursor = conn.cursor()
        
    # load sqlite-vec
    conn.enable_load_extension(True)
    sqlite_vec.load(conn) 

    cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    create_statements = []
    for table in tables:
        _, create_statement = table
        create_statements.append(create_statement)

    cursor.close()
    conn.close()

    return create_statements

def obtain_pks(db_file_dir, table_name):
    conn = sqlite3.connect(db_file_dir)
    cursor = conn.cursor()

    # print(f"\ntable name: {table_name}")
    # load sqlite-vec
    conn.enable_load_extension(True)
    sqlite_vec.load(conn) 

    cursor.execute("SELECT name, type, pk FROM PRAGMA_TABLE_INFO('{}')".format(table_name))
    results = cursor.fetchall()
    # print("pk: ",results)

    column_names = [result[0] for result in results]
    column_types = [result[1] for result in results]
    pk_indicators = [result[2] for result in results]
    pk_columns = [column_name for column_name, pk_indicator in zip(column_names, pk_indicators) if pk_indicator == 1]
    
    # cursor.execute("""
    #     SELECT type, sql FROM sqlite_master 
    #     WHERE name='{}'
    # """.format(table_name))


    # results = cursor.fetchall()
    # print("table: ",results)
    # print("----------------------------------------\n")

    return [f'"{table_name}"."{pk_column}"' for pk_column in pk_columns]

def obtain_fks(db_file_dir, table_name):
    conn = sqlite3.connect(db_file_dir)
    cursor = conn.cursor()

    # load sqlite-vec
    conn.enable_load_extension(True)
    sqlite_vec.load(conn) 

    # obtain foreign keys in the current table
    cursor.execute("SELECT * FROM pragma_foreign_key_list('{}');".format(table_name))
    results = cursor.fetchall()

    foreign_keys = []
    for result in results:
        if None not in [result[3], result[2], result[4]]:
            foreign_keys.append([f'"{table_name}"."{result[3]}"', f'"{result[2]}"."{result[4]}"'])

    return foreign_keys
def check_ddl(ddl: str) -> bool:
    """
    检查 DDL 字符串是否包含以下内容（不区分大小写）：
    - "virtual"
    - "using" and "vec0"
    - "["
    - "]"
    
    参数:
        ddl (str): 要检查的 SQL DDL 字符串
        
    返回:
        bool: 如果所有条件都满足返回 True，否则返回 False
    """
    # 转换为小写以便不区分大小写匹配
    ddl_lower = ddl.lower()
    
    # 检查是否包含 "virtual"
    has_virtual = "virtual" in ddl_lower
    
    # 检查是否包含 "using vec0"
    has_using_vec0 = "using" in ddl_lower and "vec0" in ddl_lower
    
    # 检查是否包含 "[" 和 "]"
    has_brackets = "[" in ddl and "]" in ddl
    
    # 所有条件都必须满足
    return has_virtual and has_using_vec0 and has_brackets

if __name__ == "__main__":
    db_ids = os.listdir("./synthesis_data")

    tables = []
    for db_id in tqdm(db_ids):
        table = dict()
        table["db_id"] = db_id
        table["ddls"] = []
        table["column_names"] = [[-1, "*"]]
        table["column_names_original"] = [[-1, "*"]]
        table["column_types"] = ["text"]
        table["table_names"] = []
        table["table_names_original"] = []
        table["foreign_keys"] = []
        table["primary_keys"] = []

        db_file_dir = os.path.join("synthesis_data", db_id, db_id + ".sqlite")
        ddls = obtain_db_ddls(db_file_dir)
        # print("\n\n".join(ddls))

        primary_keys_info = []
        foreign_keys_info = []

        table_column_names = ["*"]
        for table_idx, ddl in enumerate(ddls):
            if ddl.count("PRIMARY KEY") > 1:
                print("PRIMARY KEY more than one: ",ddl)
            table["ddls"].append(ddl)
            table_name_match = re.search(
                r'CREATE (?:VIRTUAL )?TABLE\s+"([^"]+)"', 
                ddl, 
                re.IGNORECASE
            )
            # if check_ddl(ddl):
            #     table_name_match = re.search(r'CREATE VIRTUAL TABLE\s+"([^"]+)"', ddl)
            # else:
            #     table_name_match = re.search(r'CREATE TABLE\s+"([^"]+)"', ddl)
            table_name = table_name_match.group(1) if table_name_match else None
            # print("table_name: ",table_name)
            if table_name is None:
                continue

            table["table_names"].append(table_name)
            table["table_names_original"].append(table_name)

            columns_part = re.search(r'\(([\s\S]*?)\);?$', ddl)
            if columns_part:
                column_text = columns_part.group(1)  # 提取括号内的内容
                # print("column_text: ",column_text)
                column_pattern = r'''
                    \s*                                 # 允许前置空白（包括制表符）
                    "?([^"\s,]+)"?                      # 列名（更宽松的引号处理）
                    \s+                                 # 类型前分隔符
                    ([^\s,/*]+(?:\[\d+\])?)            # 类型（支持FLOAT[3]格式）
                    (?:\s*/\*\s*(.*?)\s*\*/)?          # 可选注释（调整位置）
                    (?:\s+(?:PRIMARY\s+KEY|NOT\s+NULL|UNIQUE)\b)*  # 可出现在注释后的约束
                    \s*                                 # 允许后置空白
                '''
                column_infos = re.findall(column_pattern, column_text, re.VERBOSE)

            # print("column_infos: ",column_infos)
            # print(f"Table Name: {table_name}")
            for column_name, column_type, comment in column_infos:
                # print(f"Column Name: {column_name}, Type: {column_type}, Comment: {comment}")
                table["column_names"].append([table_idx, comment]) # column_names is the semantic names (i.e., descriptions) of columns
                table["column_names_original"].append([table_idx, column_name]) # column_names_original is the original names used in DDLs
                table["column_types"].append(column_type)
                table_column_names.append(f'"{table_name}"."{column_name}"')
            
            # 添加隐式rowid列（如果表中没有显式定义）
            # if not any(col[1].lower() == 'rowid' for col in table["column_names"]):
            #     table["column_names"].append([table_idx, "Row ID"])
            #     table["column_names_original"].append([table_idx, "rowid"])
            #     table["column_types"].append("INTEGER")
            #     table_column_names.append(f'"{table_name}"."rowid"')
            # print("\ntable_column_names: ",table_column_names)
            primary_keys_info.append(obtain_pks(db_file_dir, table_name))
            foreign_keys_info.extend(obtain_fks(db_file_dir, table_name))


        # print("\ntable_column_names: ",table_column_names)
        # print("\nprimary_keys_info: ",primary_keys_info)

        for primary_key_info in primary_keys_info:
            try:
                if len(primary_key_info) == 1:
                    # if "rowid" in primary_key_info:
                    #     pass
                    # print("table_name: ",table_name)
                    # print("\nprimary_key_info: ",primary_key_info)
                    # print("____________________\n")
                    table["primary_keys"].append(table_column_names.index(primary_key_info[0]))
                elif len(primary_key_info) > 1:
                    pk_idx_list = []
                    for primary_key_info_str in primary_key_info:
                        pk_idx_list.append(table_column_names.index(primary_key_info_str))
                    table["primary_keys"].append(pk_idx_list)
            except Exception as e:
                # print(primary_key_info)
                # print(db_id)
                # print("primary key error: ", e)
                # print("table_column_names.index: ",column_infos)
                # traceback.print_exc()
                continue

        for foreign_key_info in foreign_keys_info:
            try:
                table["foreign_keys"].append(
                    [table_column_names.index(foreign_key_info[0]), table_column_names.index(foreign_key_info[1])]
                )
            except Exception as e:
                print(foreign_key_info)
                # print(db_id)
                print("foreign key error: ", e)
        
        tables.append(table)

    with open("tables.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(tables, ensure_ascii=False, indent=2))
