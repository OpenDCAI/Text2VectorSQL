# /usr/lib/postgresql/14/bin/pg_ctl -D /mnt/b_public/data/wangzr/pgdata/ -l logfile start
# clickhouse  start # 启动clickhouse http://localhost:8123/play

# 清空clickhouse数据库
# clickhouse-client --query="SELECT name FROM system.databases WHERE name NOT IN ('system', 'default', 'INFORMATION_SCHEMA', 'information_schema')" | xargs -I {} clickhouse-client --query="DROP DATABASE IF EXISTS {}"

# 清空postgresql数据库
sudo -u postgres psql -t -c "SELECT datname FROM pg_database WHERE datistemplate = false AND datname <> 'postgres';" | while read dbname; do
  # 如果读取到非空行（即数据库名），则执行删除
  if [ -n "$dbname" ]; then
    echo "Dropping database: $dbname"
    sudo -u postgres dropdb "$dbname"
  fi
done
echo "All user databases have been dropped."

# /etc/init.d/postgresql start # 启动postgresql sudo -u postgres psql
