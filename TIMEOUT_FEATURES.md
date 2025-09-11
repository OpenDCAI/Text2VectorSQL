# ExecutionEngine 超时功能说明

## 概述

`execution_engine.py` 现在支持多种超时机制，确保长时间运行的操作不会无限期阻塞系统。

## 超时类型

### 1. Embedding服务调用超时
- **配置项**: `timeouts.embedding_service`
- **默认值**: 30秒
- **作用**: 限制调用外部Embedding服务的时间
- **命令行参数**: `--embedding-timeout`

### 2. 数据库连接超时
- **配置项**: `timeouts.database_connection`
- **默认值**: 10秒
- **作用**: 限制建立数据库连接的时间
- **命令行参数**: `--db-connection-timeout`

### 3. SQL执行超时
- **配置项**: `timeouts.sql_execution`
- **默认值**: 60秒
- **作用**: 限制SQL查询执行的时间
- **命令行参数**: `--sql-execution-timeout`

### 4. 总执行超时
- **配置项**: `timeouts.total_execution`
- **默认值**: 120秒
- **作用**: 限制整个执行过程的总时间
- **命令行参数**: `--total-timeout`

## 配置文件示例

```yaml
# engine_config.yaml
embedding_service:
  url: "http://localhost:8000/embeddings"

database_connections:
  postgresql:
    host: "localhost"
    port: 5432
    user: "postgres"
    password: "password"
  clickhouse:
    host: "localhost"
    port: 8123
    user: "default"
    password: ""

# 超时配置（单位：秒）
timeouts:
  embedding_service: 30      # Embedding服务调用超时
  database_connection: 10    # 数据库连接超时
  sql_execution: 60          # SQL执行超时
  total_execution: 120       # 总执行超时
```

## 命令行使用

### 基本用法
```bash
python execution_engine/execution_engine.py \
  --sql "SELECT * FROM table WHERE lembed('model', 'text') = '[1,2,3]'" \
  --db-type postgresql \
  --db-identifier mydb
```

### 使用超时参数
```bash
python execution_engine/execution_engine.py \
  --sql "SELECT * FROM table WHERE lembed('model', 'text') = '[1,2,3]'" \
  --db-type postgresql \
  --db-identifier mydb \
  --embedding-timeout 60 \
  --sql-execution-timeout 120 \
  --total-timeout 300
```

## 超时异常处理

当发生超时时，系统会：

1. 记录详细的超时日志
2. 返回包含错误信息的JSON响应：
   ```json
   {
     "status": "error",
     "message": "操作超时 (60 秒)"
   }
   ```
3. 自动清理资源（关闭数据库连接等）

## 实现细节

### 超时机制
- 使用 `signal.SIGALRM` 实现超时控制
- 通过上下文管理器 `timeout_context()` 包装关键操作
- 支持嵌套超时（内层超时不会影响外层超时）

### 支持的数据库
- **PostgreSQL**: 支持连接超时和查询超时
- **ClickHouse**: 支持连接超时和查询超时
- **SQLite**: 支持连接超时和查询超时

### 错误恢复
- 超时后自动回滚数据库事务
- 自动关闭数据库连接
- 提供详细的错误信息用于调试

## 测试

运行测试脚本验证超时功能：

```bash
python test_timeout.py
```

## 注意事项

1. **信号处理**: 超时功能使用信号机制，在Windows上可能不完全支持
2. **嵌套超时**: 内层超时会覆盖外层超时，确保使用合理的超时值
3. **资源清理**: 超时后会自动清理资源，但建议在应用层也做好异常处理
4. **日志记录**: 所有超时事件都会记录到日志中，便于调试和监控

## 性能影响

超时机制对性能的影响很小：
- 信号处理开销极小
- 只在超时时才产生额外开销
- 正常执行时几乎没有性能损失
