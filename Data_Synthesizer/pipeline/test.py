import pysqlite3 as sqlite3
import sqlite_vec
import sqlite_lembed

DB_PATH = ":memory:"  # 使用内存数据库，避免文件问题
MODEL_NAME = "CLIP-ViT-B-32-laion2B-s34B-b79K"
MODEL_PATH ="/mnt/b_public/data/ydw/Text2VectorSQL/model/CLIP-ViT-B-32-laion2B-s34B-b79K_ggml-model-q8_0.gguf"
REGISTER_MODEL_SQL = "INSERT OR IGNORE INTO main.lembed_models (name, model) VALUES (?, lembed_model_from_file(?))"

try:
    print(f"尝试加载模型: {MODEL_PATH}")
    conn = sqlite3.connect(DB_PATH)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    sqlite_lembed.load(conn)

    # 核心测试代码
    conn.execute(REGISTER_MODEL_SQL, (MODEL_NAME, MODEL_PATH))
    conn.commit()
    
    print("✅ 模型加载成功！")

    # 尝试使用模型
    cursor = conn.cursor()
    query = f"SELECT lembed_version(), lembed_text_embedding('{MODEL_NAME}', 'hello world')"
    cursor.execute(query)
    result = cursor.fetchone()
    print(f"✅ 模型使用成功: {result}")

except sqlite3.OperationalError as e:
    print(f"❌ 操作失败: {e}")
except Exception as e:
    print(f"❌ 发生未知错误: {e}")
finally:
    if 'conn' in locals() and conn:
        conn.close()
