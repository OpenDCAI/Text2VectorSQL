# 执行引擎 (Execution Engine)

执行引擎负责解析并执行"VectorSQL"查询. 它在用户通过扩展SQL方言表达的意图与各种数据库系统的原生功能之间架起了一座桥梁.

## 核心功能

该引擎专门用于处理包含特殊函数 `lembed(model, text)` 的VectorSQL查询. 其主要职责包括:

1.  **SQL解析**: 解析输入的VectorSQL查询, 查找所有 `lembed(model, text)` 函数的实例.

2.  **向量化**: 对于找到的每一个唯一的 `(model, text)` 组合, 它会向一个外部的**Embedding服务**发起网络请求. 该服务负责使用指定的嵌入模型将文本转换为高维向量.

3.  **SQL翻译**: 收到向量后, 引擎会将原始的VectorSQL翻译成与目标数据库兼容的原生查询. 它会用相应的向量字面量替换 `lembed(...)` 调用, 并确保格式符合数据库要求.

4.  **数据库执行**: 引擎连接到指定的目标数据库 (PostgreSQL, ClickHouse, 或 SQLite), 并执行翻译后的原生查询.

5.  **结果处理**: 获取查询结果, 并以结构化的JSON格式返回.

6.  **超时与错误管理**: 引擎为网络请求, 数据库连接和查询执行实现了健壮的超时机制, 以防止无限期挂起. 同时, 它为失败的操作提供清晰的错误信息.

## 依赖项

引擎的正常运行依赖于多个Python库. 这些库已在 `requirements.txt` 文件中列出.

-   `psycopg2-binary`: 用于连接到PostgreSQL数据库.
-   `requests`: 用于向Embedding服务发出HTTP请求.
-   `PyYAML`: 用于从YAML文件加载引擎配置.
-   `clickhouse-connect`: 用于连接到ClickHouse数据库.
-   `sqlite-vec`: 用于提供向量搜索能力的自定义SQLite扩展.
-   `sqlite-lembed`: 用于处理 `lembed` 函数的自定义SQLite扩展.

## 配置

引擎通过一个YAML文件 (例如 `engine_config.yaml`) 进行配置. 该文件必须指定Embedding服务的URL, 并且可以定义不同数据库的连接参数和各种超时设置.

**`engine_config.yaml` 示例:**
```yaml
embedding_service:
  url: "http://127.0.0.1:8000/embed"

database_connections:
  postgresql:
    user: "postgres"
    password: "postgres"
    host: "localhost"
    port: 5432
  clickhouse:
    host: "localhost"
    port: 8123
    user: "default"
    password: ""

timeouts:
  embedding_service: 30  # 秒
  database_connection: 10 # 秒
  sql_execution: 60     # 秒
  total_execution: 120  # 秒
```

## 用法

`ExecutionEngine` 可以作为一个命令行工具使用，也可以作为一个Python类调用。

注意：输入VectorSQL的lembed部分不包含双引号，单引号内无单引号。

### 命令行界面

```bash
# 安装依赖
pip install -r requirements.txt

# 运行查询
python execution_engine.py \
    --sql "SELECT Musical_ID,Name FROM musical m order by L2Distance(Category_embedding, lembed('all-MiniLM-L6-v2','xxx')) + L2Distance(Category_embedding, lembed('all-MiniLM-L6-v2','yyy')) LIMIT 5;" \
    --db-type "clickhouse" \
    --db-identifier "musical" \
    --config "engine_config.yaml"
```

### 命令行参数

-   `--sql`: (必需) 要执行的VectorSQL查询语句.
-   `--db-type`: (必需) 目标数据库的类型. 可选值: `postgresql`, `clickhouse`, `sqlite`.
-   `--db-identifier`: (必需) 数据库标识符 (例如, PostgreSQL/ClickHouse的数据库名, 或SQLite的文件路径).
-   `--config`: 引擎的YAML配置文件路径 (默认为 `engine_config.yaml`).
-   `--...-timeout`: 可选参数, 用于覆盖配置文件中的超时设置 (例如, `--sql-execution-timeout 90`).

### Python类调用

```python
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