# Scripts 目录说明

此目录包含用于为基准测试设置数据库的 ETL (Extract, Transform, Load) 脚本。

## 数据库 Schema

以下是由这些脚本创建的数据库的 Schema。

### 1. Toy 数据库 (`toy_etl.py`)

这是一个小型的示例数据库，用于快速测试和演示。

-   **`articles`**: 存储基础的文章信息。
    ```sql
    CREATE TABLE articles(
        headline TEXT
    );
    ```
-   **`vec_articles`**: 一个虚拟表，用于存储由 `all-MiniLM-L6-v2` 模型生成的标题嵌入向量，并支持向量搜索。
    ```sql
    CREATE VIRTUAL TABLE vec_articles USING vec0(
        headline_embedding FLOAT[384]
    );
    ```

---

### 2. ArXiv 数据库 (`arxiv_etl.py`)

该数据库存储了从 ArXiv 数据集提取的学术论文元数据。

-   **`articles`**: 存储论文的核心信息。
    ```sql
    CREATE TABLE articles (
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
    );
    ```
-   **`authors`**: 论文作者。
    ```sql
    CREATE TABLE authors (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL
    );
    ```
-   **`submitters`**: 论文提交者。
    ```sql
    CREATE TABLE submitters (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL
    );
    ```
-   **`categories`**: 论文分类。
    ```sql
    CREATE TABLE categories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        code TEXT UNIQUE NOT NULL
    );
    ```
-   **`versions`**: 论文的版本历史。
    ```sql
    CREATE TABLE versions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        article_id INTEGER NOT NULL,
        version_num TEXT NOT NULL,
        created TEXT NOT NULL,
        FOREIGN KEY (article_id) REFERENCES articles (id)
    );
    ```
-   **`article_authors`**: 连接 `articles` 和 `authors` 的多对多关系表。
-   **`article_categories`**: 连接 `articles` 和 `categories` 的多对多关系表。

---

### 3. Wikipedia 数据库 (`wiki_etl.py`)

该数据库存储了从维基百科多模态数据集提取的文章、段落、图片和标题等信息。

-   **`Articles`**: 存储维基百科文章的基础信息。
    ```sql
    CREATE TABLE Articles (
        article_id INTEGER PRIMARY KEY,
        wiki_id INTEGER NOT NULL UNIQUE,
        title TEXT NOT NULL,
        url TEXT,
        raw_html TEXT,
        raw_wikitext TEXT
    );
    ```
-   **`Paragraphs`**: 存储从文章中提取的段落。
    ```sql
    CREATE TABLE Paragraphs (
        paragraph_id INTEGER PRIMARY KEY,
        article_id INTEGER NOT NULL,
        paragraph_index INTEGER NOT NULL,
        text TEXT,
        FOREIGN KEY (article_id) REFERENCES Articles (article_id)
    );
    ```
-   **`Images`**: 存储文章中图片的相关元数据。
    ```sql
    CREATE TABLE Images (
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
    ```
-   **`Headings`**: 存储文章中的标题层级结构。
    ```sql
    CREATE TABLE Headings (
        heading_id INTEGER PRIMARY KEY,
        heading_text TEXT NOT NULL,
        parent_heading_id INTEGER,
        FOREIGN KEY (parent_heading_id) REFERENCES Headings (heading_id),
        UNIQUE (heading_text, parent_heading_id)
    );
    ```
-   **`Image_Headings`**: 连接 `Images` 和 `Headings` 的多对多关系表，用于表示图片所属的标题章节。
