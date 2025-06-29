from setuptools import setup, find_packages

setup(
    name='vectorsql-benchmark',
    version='0.1.0',  # Incremented version
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'vectorsql-cli=interfaces.sql_interface:sql_interface',
            'vectorsql-benchmark=interfaces.benchmark_cli:main',
        ],
    },
    install_requires=[
        'PyYAML',
        'sqlite-vec',  # Added dependency for automatic extension loading
        'sqlite-lembed',  # Added dependency for lembed support
        'sqlite3',  # Ensure sqlite3 is available
    ],

    description="一个用于类 VectorSQL 系统的基准测试和 CLI 工具。",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',

    author="Bozhou LI",  # 添加作者姓名
    author_email="libozhou@pku.edu.cn",  # 添加作者邮箱
)
