import sqlite3, subprocess, sys, os
print("python-sqlite3 version :", sqlite3.sqlite_version)

# 如果你安装了命令行 sqlite3
print("cli sqlite3 version   :", 
      subprocess.check_output(["sqlite3", "-version"]).decode().split()[0])
