注意：输入SQL的lembed部分不包含双引号，单引号内无单引号

A. 作为命令行工具使用
python execution_engine.py   --db-type clickhouse   --db-identifier "musical"   --sql "SELECT Musical_ID,Name FROM musical m order by L2Distance(Category_embedding, lembed('all-MiniLM-L6-v2','xxx')) + L2Distance(Category_embedding, lembed('all-MiniLM-L6-v2','xxxx')) LIMIT 5;"

B. 作为Python库调用
这是在您的主项目中（如Data Synthesizer或Model Evaluation Framework）使用它的方式。

```
from execution_engine import ExecutionEngine

# 1. 初始化引擎（可以在您的应用启动时完成）
try:
    engine = ExecutionEngine(config_path="path/to/engine_config.yaml")
except Exception as e:
    print(f"Failed to initialize engine: {e}")
    # 处理初始化失败

# 2. 在需要时调用执行方法
my_sql_query = "SELECT name FROM products ORDER BY embedding <-> lembed('bge-base', 'high quality headphones') LIMIT 3"
db_name = "e_commerce_db"
db_type = "postgresql"

result = engine.execute(sql=my_sql_query, db_type=db_type, db_identifier=db_name)

# 3. 处理结果
if result['status'] == 'success':
    print("Execution successful!")
    print("Columns:", result['columns'])
    for row in result['data']:
        print(row)
else:
    print("Execution failed!")
    print("Error:", result['message'])
```