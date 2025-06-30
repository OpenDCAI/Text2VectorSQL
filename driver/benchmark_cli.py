import argparse
import sqlite3
import sqlite_vec
from core.benchmark_runner import run_benchmark

def main():
    """解析命令行参数并启动基准测试运行器。"""
    parser = argparse.ArgumentParser(description="运行一个基于配置文件的 VectorSQL 基准测试。")
    parser.add_argument(
        '-c',
        '--config',
        type=str,
        required=True,
        help='基准测试配置文件的路径 (例如, config/benchmark_config.yaml)'
    )
    args = parser.parse_args()

    run_benchmark(args.config)

if __name__ == '__main__':
    main()
