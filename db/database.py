import sqlite3
from .connection_manager import connection_manager

def register_database(db_name: str, db_path: str):
    """
    如果一个同名连接不存在，则创建一个新的数据库连接并将其注册到全局管理器中。

    Args:
        db_name: 要为数据库分配的名称。
        db_path: SQLite 数据库文件的路径。

    Returns:
        成功时返回数据库连接对象，否则返回 None。
    """
    # 检查连接是否已存在
    if connection_manager.get_connection(db_name):
        print(f"连接 '{db_name}' 已存在，将复用现有连接。")
        return connection_manager.get_connection(db_name)

    print(f"--- 正在为 '{db_name}' 创建新连接 ---")
    try:
        # 创建一个到 SQLite 数据库的连接
        conn = sqlite3.connect(db_path)

        # 应用性能优化 PRAGMA
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA cache_size=-20000;") # ~20MB cache
        print("已应用性能优化 PRAGMA (WAL, Normal Sync, 20MB Cache)。")
        
        # 加载 sqlite-vec 扩展
        conn.enable_load_extension(True)
        try:
            import sqlite_vec
            import sqlite_lembed
            sqlite_vec.load(conn)
            sqlite_lembed.load(conn)
            print("sqlite-vec 扩展已成功加载。")
        except (ImportError, sqlite3.Error) as e:
            print(f"警告: 无法加载 sqlite-vec 扩展: {e}")
            print("向量搜索查询可能会失败。")

        # 在管理器中注册连接
        connection_manager.register_connection(db_name, conn)
        return conn
    except sqlite3.Error as e:
        print(f"数据库错误: {e}")
        return None
