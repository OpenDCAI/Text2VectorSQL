import sqlite3

class VectorSQLClient:
    """
    一个用于演示的 `vectorsql` 客户端。
    它现在使用全局的 ConnectionManager 来获取数据库连接。
    """
    def __init__(self, manager):
        self._manager = manager

    def execute(self, db_name: str, sql: str):
        """在指定的数据库上执行 SQL 查询。"""
        conn = self._manager.get_connection(db_name)
        if not conn:
            raise ValueError(f"数据库 '{db_name}' 未在 ConnectionManager 中注册。")
        
        print(f"在 '{db_name}' 上执行查询: {sql}")
        cursor = conn.cursor()
        cursor.execute(sql)
        return cursor.fetchall()
