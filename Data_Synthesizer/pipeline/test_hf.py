import os
from sentence_transformers import SentenceTransformer

# 配置hugging face代理
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 注意：我们使用你错误信息中提到的模型，以确保测试的一致性
model_name = "seeklhy/OmniSQL-7B"
cache_dir = "/mnt/b_public/data/yaodongwen/model"

print(f"--- Starting Download Test ---")
print(f"Model to download: '{model_name}'")
print(f"Cache directory: '{cache_dir}'")

try:
    # 确保目录存在
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Cache directory exists.")

    # 检查写入权限
    if not os.access(cache_dir, os.W_OK):
        print(f"[ERROR] The script does not have write permissions for the cache directory: {cache_dir}")
    else:
        print(f"Permissions check: Writable.")

        # 核心下载/加载步骤
        print("Attempting to initialize SentenceTransformer...")
        model = SentenceTransformer(model_name, cache_folder=cache_dir)
        print("\n[SUCCESS] Successfully loaded the model!")
        print(model)

except Exception as e:
    print(f"\n[FAILURE] An error occurred during the process.")
    # 打印详细的错误信息
    import traceback
    traceback.print_exc()

print("--- Test Finished ---")
