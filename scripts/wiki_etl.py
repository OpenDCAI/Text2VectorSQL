import os
import json
import sqlite3
import numpy as np
from bs4 import BeautifulSoup
import logging
from tqdm import tqdm

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_connection_and_schema(db_file):
    """ 创建数据库连接和表结构 """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        logging.info(f"成功连接到SQLite数据库: {db_file}")
        c = conn.cursor()

        # 创建 Articles 表
        c.execute('''
        CREATE TABLE IF NOT EXISTS Articles (
            article_id INTEGER PRIMARY KEY,
            wiki_id INTEGER NOT NULL UNIQUE,
            title TEXT NOT NULL,
            url TEXT,
            raw_html TEXT,
            raw_wikitext TEXT
        );
        ''')

        # 创建 Paragraphs 表
        c.execute('''
        CREATE TABLE IF NOT EXISTS Paragraphs (
            paragraph_id INTEGER PRIMARY KEY,
            article_id INTEGER NOT NULL,
            paragraph_index INTEGER NOT NULL,
            text TEXT,
            FOREIGN KEY (article_id) REFERENCES Articles (article_id)
        );
        ''')

        # 创建 Images 表
        c.execute('''
        CREATE TABLE IF NOT EXISTS Images (
            image_id INTEGER PRIMARY KEY,
            article_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            image_title TEXT,
            parsed_title TEXT,
            url TEXT,
            is_icon BOOLEAN,
            on_commons BOOLEAN,
            description TEXT,
            caption TEXT,
            FOREIGN KEY (article_id) REFERENCES Articles (article_id)
        );
        ''')

        # 创建 Headings 表
        c.execute('''
        CREATE TABLE IF NOT EXISTS Headings (
            heading_id INTEGER PRIMARY KEY,
            heading_text TEXT NOT NULL,
            parent_heading_id INTEGER,
            FOREIGN KEY (parent_heading_id) REFERENCES Headings (heading_id),
            UNIQUE (heading_text, parent_heading_id)
        );
        ''')

        # 创建 Image_Headings 关联表
        c.execute('''
        CREATE TABLE IF NOT EXISTS Image_Headings (
            image_id INTEGER NOT NULL,
            heading_id INTEGER NOT NULL,
            PRIMARY KEY (image_id, heading_id),
            FOREIGN KEY (image_id) REFERENCES Images (image_id),
            FOREIGN KEY (heading_id) REFERENCES Headings (heading_id)
        );
        ''')
        
        conn.commit()
        logging.info("数据库表结构已创建或确认存在。")
        return conn
    except sqlite3.Error as e:
        logging.error(f"数据库错误: {e}")
    return conn

def split_into_paragraphs(html_content):
    """ 从HTML中提取并拆分段落 """
    if not html_content:
        return
    soup = BeautifulSoup(html_content, 'html.parser')
    paragraphs = [p.get_text().strip() for p in soup.find_all('p') if p.get_text().strip()]
    return paragraphs

def process_article_directory(article_path):
    """ 解析单个文章目录，返回结构化数据 """
    text_json_path = os.path.join(article_path, 'text.json')
    meta_json_path = os.path.join(article_path, 'img/meta.json')

    if not os.path.exists(text_json_path):
        return None, None

    # 解析 text.json
    with open(text_json_path, 'r', encoding='utf-8') as f:
        text_data = json.load(f)
    text_data = json.loads(text_data)
    
    article_info = {
        'wiki_id': text_data.get('id'),
        'title': text_data.get('title'),
        'url': text_data.get('url'),
        'raw_html': text_data.get('html'),
        'raw_wikitext': text_data.get('wikitext'),
        'paragraphs': split_into_paragraphs(text_data.get('html'))
    }

    # 解析 meta.json
    images_info = []
    if os.path.exists(meta_json_path):
        with open(meta_json_path, 'r', encoding='utf-8') as f:
            meta_data = json.load(f)
            meta_data = json.loads(meta_data)
            for img_meta in meta_data.get('img_meta',):
                images_info.append({
                    'filename': img_meta.get('filename'),
                    'image_title': img_meta.get('title'),
                    'parsed_title': img_meta.get('parsed_title'),
                    'url': img_meta.get('url'),
                    'is_icon': img_meta.get('is_icon'),
                    'on_commons': img_meta.get('on_commons'),
                    'description': img_meta.get('description'),
                    'caption': img_meta.get('caption'),
                    'headings': img_meta.get('headings',)
                })
    
    return article_info, images_info

def load_headings_cache(conn):
    """ 从数据库加载现有的 headings 到内存缓存中 """
    cache = {}
    try:
        cur = conn.cursor()
        cur.execute("SELECT heading_id, heading_text, parent_heading_id FROM Headings")
        rows = cur.fetchall()
        for row in rows:
            # key is (heading_text, parent_id), value is heading_id
            cache[(row, row)] = row
        logging.info(f"已加载 {len(cache)} 个标题到缓存。")
    except sqlite3.Error as e:
        logging.error(f"加载标题缓存失败: {e}")
    return cache

def main(dataset_root, db_file):
    """ 主ETL流程 """
    conn = create_connection_and_schema(db_file)
    if not conn:
        return

    headings_cache = load_headings_cache(conn)
    cur = conn.cursor()

    article_dirs = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]
    
    # 最多处理100个目录以防止过载
    article_dirs = article_dirs[:100]

    for article_dir_name in tqdm(article_dirs, desc="处理文章"):
        article_path = os.path.join(dataset_root, article_dir_name)
        
        try:
            article_info, images_info = process_article_directory(article_path)
            if not article_info:
                logging.warning(f"跳过目录 {article_dir_name}，未找到 text.json。")
                continue

            # 使用事务以提高性能
            conn.execute('BEGIN TRANSACTION')

            # 插入 Article
            cur.execute('''
            INSERT INTO Articles (wiki_id, title, url, raw_html, raw_wikitext)
            VALUES (?,?,?,?,?)
            ''', (article_info['wiki_id'], article_info['title'], article_info['url'], article_info['raw_html'], article_info['raw_wikitext']))
            article_id = cur.lastrowid

            # 插入 Paragraphs
            if article_info['paragraphs']:
                for i, p_text in enumerate(article_info['paragraphs']):
                    cur.execute('''
                    INSERT INTO Paragraphs (article_id, paragraph_index, text)
                    VALUES (?,?,?)
                    ''', (article_id, i, p_text))

            # 插入 Images 和 Headings
            for img in images_info:
                cur.execute('''
                INSERT INTO Images (article_id, filename, image_title, parsed_title, url, is_icon, on_commons, description, caption)
                VALUES (?,?,?,?,?,?,?,?,?)
                ''', (article_id, img['filename'], img['image_title'], img['parsed_title'], img['url'], img['is_icon'], img['on_commons'], img['description'], img['caption']))
                image_id = cur.lastrowid

                # 处理 Headings 的层级关系
                parent_id = None
                if img['headings']:
                    for heading_text in img['headings']:
                        cache_key = (heading_text, parent_id)
                        if cache_key not in headings_cache:
                            cur.execute('INSERT INTO Headings (heading_text, parent_heading_id) VALUES (?,?)', (heading_text, parent_id))
                            heading_id = cur.lastrowid
                            headings_cache[cache_key] = heading_id
                        else:
                            heading_id = headings_cache[cache_key]
                        
                        # 链接图片和标题
                        cur.execute('INSERT OR IGNORE INTO Image_Headings (image_id, heading_id) VALUES (?,?)', (image_id, heading_id))
                        
                        # 为下一个标题设置父ID
                        parent_id = heading_id
            
            conn.commit()

        except Exception as e:
            conn.rollback()
            logging.error(f"处理目录 {article_dir_name} 时发生错误: {e}")

    conn.close()
    logging.info("ETL流程完成。")

if __name__ == '__main__':
    # 使用示例：请将下面的路径替换为您的实际路径
    DATASET_ROOT_PATH = '/mnt/b_public/data/wangzr/Text2VectorSQL/database/wiki'
    DATABASE_FILE_PATH = 'wikipedia_multimodal.db'
    
    if not os.path.isdir(DATASET_ROOT_PATH) or DATASET_ROOT_PATH == 'path/to/your/extended-wikipedia-multimodal-dataset':
        print("请在脚本中设置正确的数据集根目录路径 (DATASET_ROOT_PATH)。")
    else:
        main(DATASET_ROOT_PATH, DATABASE_FILE_PATH)