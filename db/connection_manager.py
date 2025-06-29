import sqlite3

class ConnectionManager:
    """管理并缓存数据库连接的单例类。"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConnectionManager, cls).__new__(cls)
            # 私有变量，用于存储 name -> connection 的映射
            cls._instance._connections = {}
        return cls._instance

    def get_connection(self, db_name: str):
        """按名称获取一个已缓存的连接。"""
        return self._connections.get(db_name)

    def register_connection(self, db_name: str, connection):
        """注册并缓存一个新的连接。"""
        if db_name in self._connections:
            # 在这个实现中，我们允许重新注册以更新连接，
            # 但在某些用例中您可能希望引发错误。
            print(f"警告: 名为 '{db_name}' 的连接已注册。将被覆盖。")
        self._connections[db_name] = connection
        print(f"连接 '{db_name}' 已注册并缓存。")

    def close_connection(self, db_name: str):
        """关闭并移除一个指定的连接。"""
        if db_name in self._connections:
            self._connections[db_name].close()
            del self._connections[db_name]
            print(f"连接 '{db_name}' 已关闭并从缓存中移除。")

    def close_all(self):
        """关闭所有受管的连接。"""
        for name, conn in self._connections.items():
            conn.close()
            print(f"连接 '{name}' 已关闭。")
        self._connections = {}

# 创建一个全局可用的单例实例
connection_manager = ConnectionManager()
