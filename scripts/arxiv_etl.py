import sqlite3
import json
import re
from tqdm import tqdm

# --- 配置 ---
JSON_FILE_PATH = 'arxiv-metadata-oai-snapshot.json'
DB_FILE_PATH = 'arxiv.db'
BATCH_SIZE = 50000

# --- 数据库Schema创建函数 ---
def create_schema(cursor):
    """根据设计的Schema创建数据库表"""
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS articles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        arxiv_id TEXT UNIQUE NOT NULL,
        submitter_id INTEGER,
        title TEXT NOT NULL,
        comments TEXT,
        journal_ref TEXT,
        doi TEXT,
        report_no TEXT,
        license TEXT,
        abstract TEXT NOT NULL,
        update_date DATE,
        FOREIGN KEY (submitter_id) REFERENCES submitters (id)
    )
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS authors (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL
    )
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS submitters (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL
    )
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS categories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        code TEXT UNIQUE NOT NULL
    )
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS article_authors (
        article_id INTEGER,
        author_id INTEGER,
        PRIMARY KEY (article_id, author_id),
        FOREIGN KEY (article_id) REFERENCES articles (id),
        FOREIGN KEY (author_id) REFERENCES authors (id)
    )
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS article_categories (
        article_id INTEGER,
        category_id INTEGER,
        PRIMARY KEY (article_id, category_id),
        FOREIGN KEY (article_id) REFERENCES articles (id),
        FOREIGN KEY (category_id) REFERENCES categories (id)
    )
    ''')
    print("数据库Schema已创建或已存在。")

# --- Get-or-Create 辅助函数 ---
def get_or_create(cursor, table, column, value):
    """查询或创建记录，并返回其ID"""
    cursor.execute(f"SELECT id FROM {table} WHERE {column} =?", (value,))
    result = cursor.fetchone()
    if result:
        return result[0]
    else:
        cursor.execute(f"INSERT INTO {table} ({column}) VALUES (?)", (value,))
        return cursor.lastrowid

# --- 作者姓名解析函数 ---
def parse_authors(authors_str):
    """从复杂的作者字符串中解析出作者列表"""
    # 移除括号内的机构信息等
    authors_cleaned = re.sub(r'\s*\([^)]*\)', '', authors_str)
    # 按逗号和'and'分割
    authors_list = re.split(r',\s*|\s+and\s+', authors_cleaned)
    return [name.strip() for name in authors_list if name.strip()]

# --- 主ETL流程 ---
def run_etl():
    conn = sqlite3.connect(DB_FILE_PATH)
    cursor = conn.cursor()

    # 1. 创建Schema
    create_schema(cursor)

    # 2. 逐行读取JSON并处理
    with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
        # 使用tqdm获取文件总行数以显示进度
        total_lines = sum(1 for line in f)
        f.seek(0) # 重置文件指针
        # total_lines = 2818564 # 预设总行数以避免重复计算

        records_in_batch = 0
        for line in tqdm(f, total=total_lines, desc="Processing Articles"):
            try:
                record = json.loads(line)

                # 提取并清洗数据
                arxiv_id = record.get('id')
                submitter = record.get('submitter')
                title = record.get('title', '').strip()
                authors_str = record.get('authors', '')
                categories_str = record.get('categories', '')
                abstract = record.get('abstract', '').strip()

                if not all([arxiv_id, title, authors_str, categories_str, abstract, submitter]):
                    continue # 跳过关键字段缺失的记录

                # 3. 转换与加载 (Transform & Load)
                # 处理 submitter
                submitter_id = get_or_create(cursor, 'submitters', 'name', submitter)

                # 插入 article
                cursor.execute('''
                INSERT INTO articles (arxiv_id, submitter_id, title, comments, journal_ref, doi, report_no, license, abstract, update_date)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                ''', (
                    arxiv_id, submitter_id, title, record.get('comments'), record.get('journal-ref'),
                    record.get('doi'), record.get('report-no'), record.get('license'), abstract, record.get('update_date')
                ))
                article_id = cursor.lastrowid

                # 处理 authors
                authors = parse_authors(authors_str)
                for author_name in authors:
                    author_id = get_or_create(cursor, 'authors', 'name', author_name)
                    cursor.execute("INSERT OR IGNORE INTO article_authors (article_id, author_id) VALUES (?,?)", (article_id, author_id))

                # 处理 categories
                categories = categories_str.split()
                for cat_code in categories:
                    category_id = get_or_create(cursor, 'categories', 'code', cat_code)
                    cursor.execute("INSERT OR IGNORE INTO article_categories (article_id, category_id) VALUES (?,?)", (article_id, category_id))

                records_in_batch += 1
                # 4. 事务管理
                if records_in_batch >= BATCH_SIZE:
                    conn.commit()
                    records_in_batch = 0

            except json.JSONDecodeError:
                print(f"警告：跳过一行无法解析的JSON。")
                continue
            except Exception as e:
                print(f"错误：处理记录时发生异常: {e}")
                continue
    
    # 提交最后一批数据
    if records_in_batch > 0:
        conn.commit()

    print("ETL流程完成。")
    conn.close()

if __name__ == '__main__':
    run_etl()