import sqlite3
import os
import re
import logging
import mmap
from tqdm import tqdm
import io
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sql_import.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SQLImportMonitor:
    """导入状态监控器"""
    def __init__(self):
        self.last_update = time.time()
        self.last_bytes = 0
        self.stuck_count = 0
    
    def check_stuck(self, current_bytes, total_bytes):
        """检查是否卡住"""
        now = time.time()
        if current_bytes == self.last_bytes:
            if now - self.last_update > 30:  # 30秒无进度
                self.stuck_count += 1
                return True
        else:
            self.last_update = now
            self.last_bytes = current_bytes
            self.stuck_count = 0
        return False
    
    def is_really_stuck(self):
        """确认是否真的卡住（超过3次检测）"""
        return self.stuck_count >= 3

def stream_large_insert(conn, table_name, columns, values_stream, monitor):
    """带超时检测的流式插入"""
    cursor = conn.cursor()
    batch_size = 500  # 减小批量大小避免卡住
    insert_template = f'INSERT INTO "{table_name}" ({columns}) VALUES ({",".join(["?"]*len(columns.split(",")))})'
    
    batch = []
    row_count = 0
    last_progress = 0
    
    for line in values_stream:
        if monitor.check_stuck(row_count, None):
            logger.warning(f"插入操作卡住，当前批次大小: {len(batch)}")
            if monitor.is_really_stuck():
                raise RuntimeError("插入操作已卡死，终止执行")
        
        line = line.strip()
        if not line or line.startswith('--'):
            continue
        
        # 解析值部分
        if line.startswith('(') and line.endswith(')'):
            values = line[1:-1].split(',')
            batch.append(tuple(v.strip() for v in values))
            row_count += 1
            
            if len(batch) >= batch_size:
                try:
                    cursor.executemany(insert_template, batch)
                    conn.commit()
                    batch = []
                except sqlite3.Error as e:
                    logger.error(f"批量插入失败: {str(e)}")
                    conn.rollback()
                    raise
    
    # 插入剩余批次
    if batch:
        cursor.executemany(insert_template, batch)
        conn.commit()
    
    return row_count

def process_sql_file(sql_file_path, db_path):
    """带完整监控的SQL导入"""
    if os.path.exists(db_path):
        os.remove(db_path)
    
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode = MEMORY")
    conn.execute("PRAGMA synchronous = OFF")
    conn.execute("PRAGMA cache_size = -50000")  # 50MB缓存
    
    monitor = SQLImportMonitor()
    file_size = os.path.getsize(sql_file_path)
    processed_bytes = 0
    
    try:
        with open(sql_file_path, 'r') as f, \
             tqdm(total=file_size, unit='B', unit_scale=True, desc="导入进度") as pbar:
            
            # 使用内存映射但分块处理
            chunk_size = 1024 * 1024  # 1MB
            buffer = ""
            in_insert = False
            insert_header = ""
            
            while processed_bytes < file_size:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                
                processed_bytes += len(chunk)
                buffer += chunk
                
                # 处理非INSERT语句
                if not in_insert:
                    while ';' in buffer:
                        pos = buffer.find(';')
                        stmt = buffer[:pos+1].strip()
                        buffer = buffer[pos+1:]
                        
                        if stmt and not stmt.startswith('--'):
                            if stmt.upper().startswith('INSERT'):
                                # 开始处理大INSERT
                                match = re.match(
                                    r'INSERT\s+INTO\s+"?([^"]+)"?\s*\((.*?)\)\s*VALUES\s*(.*)',
                                    stmt,
                                    re.IGNORECASE|re.DOTALL
                                )
                                if match:
                                    in_insert = True
                                    insert_header = f"{match.group(1)}|{match.group(2)}"
                                    remaining = match.group(3)
                                    if remaining:
                                        buffer = remaining + buffer
                                else:
                                    conn.execute(stmt)
                            else:
                                try:
                                    conn.execute(stmt)
                                except sqlite3.Error as e:
                                    logger.error(f"执行失败: {str(e)}")
                                    raise
                
                # 处理大INSERT语句
                if in_insert:
                    table_name, columns = insert_header.split('|')
                    
                    # 找到语句结束
                    if ');' in buffer:
                        end_pos = buffer.find(');') + 1
                        values_part = buffer[:end_pos]
                        buffer = buffer[end_pos:]
                        in_insert = False
                        
                        # 使用StringIO模拟流
                        values_stream = io.StringIO(values_part)
                        row_count = stream_large_insert(
                            conn, table_name, columns, values_stream, monitor
                        )
                        logger.info(f"成功插入 {row_count} 行到 {table_name}")
                    
                pbar.update(len(chunk))
                
                # 检查是否卡住
                if monitor.check_stuck(processed_bytes, file_size):
                    logger.warning(f"处理卡住，已处理 {processed_bytes}/{file_size} 字节")
                    if monitor.is_really_stuck():
                        raise RuntimeError("导入进程已卡死")
        
        # 处理剩余内容
        if buffer.strip():
            conn.execute(buffer.strip())
        
        logger.info("SQL导入完成")
        
        # 验证导入
        cursor = conn.cursor()
        cursor.execute("SELECT count(*) FROM sqlite_master WHERE type='table'")
        table_count = cursor.fetchone()[0]
        logger.info(f"导入成功，共 {table_count} 张表")
    
    except Exception as e:
        logger.error(f"导入失败: {str(e)}")
        raise
    finally:
        conn.close()

if __name__ == '__main__':
    SQL_FILE = "complete_database_export.sql"
    DB_FILE = "optimized_database.db"
    
    try:
        logger.info("开始最终优化导入...")
        start_time = time.time()
        
        process_sql_file(SQL_FILE, DB_FILE)
        
        elapsed = time.time() - start_time
        logger.info(f"导入总耗时: {elapsed:.2f}秒")
        
    except Exception as e:
        logger.error(f"最终导入失败: {str(e)}")
        # 生成诊断报告
        with open('import_diagnostic.txt', 'w') as f:
            f.write(f"失败时间: {time.ctime()}\n")
            f.write(f"错误信息: {str(e)}\n")
            f.write(f"已处理文件大小: {os.path.getsize(SQL_FILE)/(1024 * 1024):.2f}MB\n")
            if os.path.exists(DB_FILE):
                db_size = os.path.getsize(DB_FILE)/(1024 * 1024)
                f.write(f"当前数据库大小: {db_size:.2f}MB\n")
