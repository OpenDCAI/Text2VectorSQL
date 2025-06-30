-- gold_sql: 直接通过 ID 获取我们期望作为最相关结果的文章。
-- 这代表了向量搜索应该找到的“正确”答案。
SELECT rowid FROM vec_articles WHERE rowid = 4;
