# /etc/init.d/postgresql start # 启动postgresql
# clickhouse  start # 启动clickhouse

# 清空clickhouse数据库
# clickhouse-client --query="SELECT name FROM system.databases WHERE name NOT IN ('system', 'default', 'INFORMATION_SCHEMA', 'information_schema')" | xargs -I {} clickhouse-client --query="DROP DATABASE IF EXISTS {}"