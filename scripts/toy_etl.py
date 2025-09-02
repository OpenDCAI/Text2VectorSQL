import sqlite3
import os
import sqlite_lembed

# --- 配置 ---
DB_PATH = 'data/vector_database_2.sqlite'
MODEL_PATH = 'embed-model/all-MiniLM-L6-v2.e4ce9877.q8_0.gguf'
MODEL_NAME = 'all-MiniLM-L6-v2' # 这个名称将用于 lembed() 函数

# --- 示例数据 (NPR headlines from 2024-06-04) ---
sample_headlines = [
    ("Shohei Ohtani's ex-interpreter pleads guilty to charges related to gambling and theft",),
    ("The jury has been selected in Hunter Biden's gun trial",),
    ("Larry Allen, a Super Bowl champion and famed Dallas Cowboy, has died at age 52",),
    ("After saying Charlotte, a lone stingray, was pregnant, aquarium now says she's sick",),
    ("An Epoch Times executive is facing money laundering charge",)
]

def setup_database():
    """
    设置 SQLite 数据库，创建表，并用新闻标题及其嵌入填充它们。
    """
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    print(f"正在连接数据库: {DB_PATH}...")
    db = sqlite3.connect(DB_PATH)
    db.enable_load_extension(True)
    import sqlite_vec
    try:
        sqlite_vec.load(db)
        print("sqlite-vec 扩展已加载。")
    except ImportError as e:
        print(f"警告: 无法加载 sqlite-vec 扩展: {e}")
        print("向量搜索查询可能会失败。")
    sqlite_lembed.load(db)
    print("sqlite-lembed 扩展已加载。")

    cursor = db.cursor()

    print(f"正在注册模型 '{MODEL_NAME}' 从 '{MODEL_PATH}'...")
    cursor.execute(
        "INSERT OR IGNORE INTO temp.lembed_models(name, model) SELECT ?, lembed_model_from_file(?)",
        (MODEL_NAME, MODEL_PATH)
    )
    print("模型已注册。")

    # --- 创建表 ---
    print("正在创建表: 'articles' 和 'vec_articles'...")
    cursor.execute("DROP TABLE IF EXISTS articles;")
    cursor.execute("DROP TABLE IF EXISTS vec_articles;")

    cursor.execute("CREATE TABLE articles(headline TEXT);")

    # 使用 vec0 虚拟表进行向量搜索
    # 嵌入的维度 (384) 取决于 all-MiniLM-L6-v2 模型
    cursor.execute("""
    CREATE VIRTUAL TABLE vec_articles USING vec0(
        headline_embedding FLOAT[384]
    );
    """)
    print("表已创建。")

    # --- 填充表 ---
    print("正在插入新闻标题...")
    cursor.executemany("INSERT INTO articles (headline) VALUES (?)", sample_headlines)
    print(f"{cursor.rowcount} 个标题已插入。")

    print("正在为标题生成并插入嵌入...")
    # 使用注册的模型名称 (MODEL_NAME) 来生成嵌入
    insert_query = f"""
    INSERT INTO vec_articles(rowid, headline_embedding)
      SELECT rowid, lembed('{MODEL_NAME}', headline)
      FROM articles;
    """
    cursor.execute(insert_query)
    print(f"{cursor.rowcount} 个嵌入已生成并插入。")

    db.commit()
    db.close()
    print("数据库设置完成。")

if __name__ == '__main__':
    setup_database()
