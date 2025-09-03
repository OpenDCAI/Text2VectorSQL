import sqlite3
import os
import re
import logging
from tqdm import tqdm
import sqlite_vec  # 关键：导入 sqlite_vec 库

def process_sql_file(sql_file_path, db_path):
    """
    高效处理SQL文件，通过 sqlite_vec 库加载扩展，并正确处理所有语句。
    """
    # 删除旧数据库
    if os.path.exists(db_path):
        os.remove(db_path)
        logger.info(f"已删除旧数据库: {db_path}")

    # 创建新数据库连接
    conn = sqlite3.connect(db_path)
    
    try:
        # =================================================================
        # 【关键修正】使用 sqlite_vec 库加载扩展
        # =================================================================
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        logger.info(f"通过 sqlite_vec 库成功加载向量扩展。")
        # =================================================================

        # 优化性能
        cursor = conn.cursor()
        cursor.execute("PRAGMA synchronous = OFF;")
        cursor.execute("PRAGMA journal_mode = MEMORY;")
        cursor.execute("PRAGMA cache_size = -100000;")  # 100MB 缓存

        file_size = os.path.getsize(sql_file_path)
        statement_buffer = ""

        with open(sql_file_path, 'r', encoding='utf-8') as f, \
             tqdm(total=file_size, unit='B', unit_scale=True, desc="导入进度") as pbar:
            
            for line in f:
                pbar.update(len(line.encode('utf-8')))
                
                # 跳过注释行
                if line.strip().startswith('--'):
                    continue
                
                statement_buffer += line
                
                # 如果行尾有分号，说明一条语句结束了
                if line.strip().endswith(';'):
                    try:
                        cursor.execute(statement_buffer)
                        statement_buffer = ""
                    except sqlite3.Error as e:
                        logger.error(f"SQL执行失败: {e}")
                        logger.error(f"问题语句 (前200字符): {statement_buffer.strip()[:200]}...")
                        raise  # 抛出异常以停止脚本

        # 处理文件末尾可能剩余的最后一条语句
        if current_statement := statement_buffer.strip():
            cursor.execute(current_statement)
        
        # 确保所有操作都被提交
        conn.commit()
        logger.info("SQL 脚本导入完成。")
        
        # 验证导入
        cursor.execute("SELECT count(*) FROM sqlite_master WHERE type='table'")
        table_count = cursor.fetchone()[0]
        logger.info(f"验证成功，数据库中包含 {table_count} 张表。")
            
    except Exception as e:
        logger.error(f"导入过程中发生严重错误: {e}")
        conn.close() # 确保在出错时关闭连接
        if os.path.exists(db_path):
             os.remove(db_path) # 删除可能损坏的数据库文件
        raise
    finally:
        if conn:
            conn.close()

# if __name__ == '__main__':
def build_vector_database(SQL_FILE, DB_FILE):
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('sql_import.log', mode='w'), # 'w' to overwrite log each run
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # SQL_FILE = "complete_database_export.sql"
    # DB_FILE = "optimized_database.db"
    
    try:
        # 检查是否已安装 sqlite_vec
        # 这只是一个友好的提醒，如果未安装，导入时就会报错
        import sqlite_vec
    except ImportError:
        logger.error("致命错误: 'sqlite_vec' 库未安装。")
        logger.error("请先通过 'pip install sqlite-vec' 命令进行安装。")
        exit()

    try:
        logger.info(f"开始从 {SQL_FILE} 导入到 {DB_FILE}...")
        process_sql_file(SQL_FILE, DB_FILE)
        logger.info("🎉 导入成功完成！")
    except Exception:
        logger.error("❌ 导入最终失败。请检查上面的日志获取详细信息。")
