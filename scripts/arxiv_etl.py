import sqlite3
import json
import re
from tqdm import tqdm

# --- 配置 ---
JSON_FILE_PATH = '../database/arxiv/arxiv-metadata-oai-snapshot.json'
DB_FILE_PATH = 'arxiv.db'
BATCH_SIZE = 50000

# --- 数据库Schema创建函数 ---
def create_schema(cursor):
    """根据最终设计的Schema创建数据库表，包含versions表"""
    print("正在创建数据库Schema...")
    # 核心表
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
    # 维度表
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
    # 新增的versions表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS versions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        article_id INTEGER NOT NULL,
        version_num TEXT NOT NULL,
        created TEXT NOT NULL,
        FOREIGN KEY (article_id) REFERENCES articles (id)
    )
    ''')
    # 连接表
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

def create_indexes(cursor):
    """为外键和常用查询列创建索引以加速查询"""
    print("正在创建索引以优化查询性能...")
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_articles_submitter_id ON articles (submitter_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_versions_article_id ON versions (article_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_article_authors_author_id ON article_authors (author_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_article_categories_category_id ON article_categories (category_id)')
    print("索引已创建或已存在。")

# --- Get-or-Create 辅助函数 ---
def get_or_create(cursor, table, column, value):
    """查询或创建记录，并返回其ID。这是ETL流程的核心辅助函数。"""
    cursor.execute(f"SELECT id FROM {table} WHERE {column} =?", (value,))
    result = cursor.fetchone()
    if result:
        return result[0]
    else:
        cursor.execute(f"INSERT INTO {table} ({column}) VALUES (?)", (value,))
        return cursor.lastrowid

# --- 作者姓名解析函数 ---
def parse_authors(authors_str):
    """从复杂的作者字符串中解析出作者列表，移除括号内的机构信息等。"""
    authors_cleaned = re.sub(r'\s*\([^)]*\)', '', authors_str)
    authors_list = re.split(r',\s*|\s+and\s+', authors_cleaned)
    return [name.strip() for name in authors_list if name.strip()]

# --- 主ETL流程 ---
def run_etl():
    conn = sqlite3.connect(DB_FILE_PATH)
    cursor = conn.cursor()
    # 开启PRAGMA优化以提高写入速度
    cursor.execute('PRAGMA journal_mode = WAL;')
    cursor.execute('PRAGMA synchronous = NORMAL;')
    cursor.execute('PRAGMA cache_size = -1000000;') # ~1GB cache

    # 1. 创建Schema和索引
    create_schema(cursor)

    # 2. 逐行读取JSON并处理
    try:
        with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for line in f)
            f.seek(0)
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
                    versions = record.get('versions',)

                    if not all([arxiv_id, title, authors_str, categories_str, abstract, submitter, versions]):
                        continue

                    # 3. 转换与加载 (Transform & Load)
                    # 处理 submitter
                    submitter_id = get_or_create(cursor, 'submitters', 'name', submitter)

                    # 插入 article (使用 INSERT OR IGNORE 保证幂等性)
                    cursor.execute('''
                    INSERT OR IGNORE INTO articles (arxiv_id, submitter_id, title, comments, journal_ref, doi, report_no, license, abstract, update_date)
                    VALUES (?,?,?,?,?,?,?,?,?,?)
                    ''', (
                        arxiv_id, submitter_id, title, record.get('comments'), record.get('journal-ref'),
                        record.get('doi'), record.get('report-no'), record.get('license'), abstract, record.get('update_date')
                    ))
                    
                    # 如果文章是新插入的，获取其ID
                    if cursor.rowcount > 0:
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
                        
                        # 处理 versions
                        for version_info in versions:
                            cursor.execute("INSERT INTO versions (article_id, version_num, created) VALUES (?,?,?)",
                                           (article_id, version_info.get('version'), version_info.get('created')))

                    records_in_batch += 1
                    # 4. 事务管理
                    if records_in_batch >= BATCH_SIZE:
                        conn.commit()
                        records_in_batch = 0

                except json.JSONDecodeError:
                    print(f"警告：跳过一行无法解析的JSON。")
                    continue
                except Exception as e:
                    print(f"错误：处理记录 {arxiv_id if 'arxiv_id' in locals() else ''} 时发生异常: {e}")
                    continue
        
        # 提交最后一批数据
        if records_in_batch > 0:
            conn.commit()
        
        print("ETL数据加载完成。")
        
        # 5. 在数据加载完成后创建索引
        create_indexes(cursor)
        conn.commit()

    finally:
        conn.close()
        print("数据库连接已关闭。")

if __name__ == '__main__':
    run_etl()
