如果你想要迁移sqlite数据库和sql到其他的数据库，比如clickhouse,postgre,myscale，那么你得先启动clickhouse服务。
运行：
```bash
sudo -u clickhouse /usr/bin/clickhouse-server --config-file=/etc/clickhouse-server/my-clean-config.xml --daemon
```

检查postgre是否运行：
```bash
## 里面的路径是postgre数据库保存路径中的文件
head -n 1 /mnt/DataFlow/ydw/data/pgdata/postmaster.pid
## 下面的<PID>是上面的结果
ps -p <PID> -f

## 如果显示有 postgres 进程：说明数据库已经在运行了，无需再次启动。
## 如果报错说进程不存在：说明数据库之前崩了，你需要手动删除这个锁文件才能重新启动：
rm /mnt/DataFlow/ydw/data/pgdata/postmaster.pid
```

启动postgre，首先切换用户，因为PostgreSQL 严禁使用 root 用户启动。
```bash
## 确保权限正确 你需要把这个数据目录的所有权给到 postgres 用户（或者你想用来运行数据库的非 root 用户）：
chown -R postgres:postgres /mnt/DataFlow/ydw/data/pgdata/
## 切换用户
su - postgres
## 启动
/usr/lib/postgresql/14/bin/pg_ctl -D /mnt/DataFlow/ydw/data/pgdata/ -l /tmp/pg_logfile.log start
## 或者使用更通用的命令：
## pg_ctl -D /mnt/DataFlow/ydw/data/pgdata/ -l /tmp/pg_logfile.log start

## 验证启动
# 查看日志确认 success
tail -f /tmp/pg_logfile.log

# 尝试连接（默认端口通常是 5432）
psql -h localhost -p 5432 -d postgres
```
