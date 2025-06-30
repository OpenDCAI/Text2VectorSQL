from setuptools import setup, find_packages

setup(
    name='vectorsql-bench',
    version='0.1.0',  # Incremented version
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'vectorsql-client=driver.client_cli:main',
            'vectorsql-bench=driver.bench_cli:main',
        ],
    },
    install_requires=[
        'PyYAML',
        'sqlite-vec',  # Added dependency for automatic extension loading
        'sqlite-lembed',  # Added dependency for lembed support
        # 'sqlite3',  # Ensure sqlite3 is available
        'litellm', # Added dependency for litellm
    ],

    description="一个用于类 VectorSQL 系统的基准测试和 CLI 工具。",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',

    author="Bozhou LI, Dongwen Yao, Zhengren Wang",  # 添加作者姓名
    author_email="libozhou@pku.edu.cn",  # 添加作者邮箱
)
