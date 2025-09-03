import sqlite3
import datetime
import traceback
from collections import defaultdict
from sentence_transformers import SentenceTransformer

def export_to_single_sql_file(db_path, output_file):
    """将所有数据库内容导出到单个SQL文件"""
    conn = sqlite3.connect(db_path)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        try:
            # 文件头部信息
            f.write(f"-- SQLite Database Export\n")
            f.write(f"-- Exported at: {datetime.datetime.now()}\n")
            f.write(f"-- Source: {db_path}\n\n")
            
            # 1. 基础设置
            f.write("PRAGMA foreign_keys = OFF;\n")
            f.write("BEGIN TRANSACTION;\n\n")
            
            # 2. 获取所有用户对象
            objects = conn.execute("""
                SELECT type, name, tbl_name, sql 
                FROM sqlite_master
                WHERE type IN ('table', 'view', 'trigger', 'index')
                  AND name NOT LIKE 'sqlite_%'
                ORDER BY
                    CASE type 
                        WHEN 'table' THEN 1
                        WHEN 'view' THEN 4
                        ELSE 3
                    END
            """).fetchall()
            
            # 3. 创建表（包含外键约束）
            created_tables = []
            for obj_type, name, tbl_name, sql in objects:
                if obj_type == 'table':
                    f.write(f"-- Table: {name}\n")
                    f.write(sql + ";\n\n")
                    created_tables.append(name)
            
            # 4. 插入数据
            f.write("\n-- DATA INSERTION --\n\n")
            for table in created_tables:
                # 获取实际数据列名（不是表结构元数据）
                cursor = conn.execute(f'SELECT * FROM "{table}" LIMIT 0')  # 仅获取列名
                col_names = [desc[0] for desc in cursor.description]
                
                f.write(f"-- Data for table: {table}\n")
                data = conn.execute(f'SELECT * FROM "{table}"').fetchall()
                
                if not data:
                    f.write(f"-- No data in table {table}\n\n")
                    continue
                    
                # 开始插入语句
                f.write(f'INSERT INTO "{table}" (')
                f.write(', '.join([f'"{col}"' for col in col_names]))
                f.write(') VALUES\n')
                
                # 插入所有行
                for i, row in enumerate(data):
                    # 生成行值
                    values = []
                    for val in row:
                        if val is None:
                            values.append("NULL")
                        elif isinstance(val, (int, float)):
                            values.append(str(val))
                        elif isinstance(val, bytes):
                            values.append(f"X'{val.hex()}'")  # BLOB转为十六进制
                        else:
                            # 字符串转义和引号处理
                            s = str(val).replace("'", "''")
                            values.append(f"'{s}'")
                    
                    # 写入行
                    f.write("(" + ", ".join(values) + ")")
                    
                    # 行结束符（最后一行用分号）
                    if i < len(data) - 1:
                        f.write(",\n")
                    else:
                        f.write(";\n\n")
            
            # 5. 创建其他对象（索引、视图、触发器）
            f.write("\n-- ADDITIONAL DATABASE OBJECTS --\n\n")
            for obj_type, name, tbl_name, sql in objects:
                if obj_type != 'table':
                    obj_title = {
                        'index': f'Index on {tbl_name}',
                        'view': 'View',
                        'trigger': f'Trigger on {tbl_name}'
                    }[obj_type]
                    
                    f.write(f"-- {obj_title}: {name}\n")
                    f.write(sql + ";\n\n")
            
            # 6. SQLite 不支持单独添加外键，所以这一步去除
            # （外键已经包含在表创建语句中）
            
            # 7. 完成事务
            f.write("COMMIT;\n")
            f.write("PRAGMA foreign_keys = ON;\n")
            
            # 8. 数据库信息
            f.write("\n-- DATABASE INFO --\n\n")
            f.write(f"-- SQLite version: {conn.execute('SELECT sqlite_version()').fetchone()[0]}\n")
            f.write(f"-- Encoding: {conn.execute('PRAGMA encoding').fetchone()[0]}\n")
            f.write(f"-- Creation time: {datetime.datetime.now()}\n")
            
            return True, f"Successfully exported {len(created_tables)} tables to {output_file}"
        
        except Exception as e:
            error_msg = f"Export failed: {str(e)}\n{traceback.format_exc()}"
            return False, error_msg
        
        finally:
            conn.close()

# 使用示例
if __name__ == "__main__":
    db_path = "train/train_databases/world/world.sqlite"
    output_file = "complete_database_export_fixed.sql"
    
    success, message = export_to_single_sql_file(db_path, output_file)
    
    if success:
        print(f"✔ {message}")
        print("导入SQLite命令:")
        print(f"  sqlite3 new_database.db < {output_file}")
    else:
        print(f"✖ 导出失败!\n{message}")
