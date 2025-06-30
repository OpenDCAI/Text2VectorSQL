-- test_sql: 使用 lembed() 和 MATCH 语法在 FTS 表中执行向量搜索。
-- 我们正在寻找与“what are vector databases?”在语义上最相似的文章。
-- 这应该匹配到关于向量数据库历史的文章 (id=4)。
SELECT rowid
FROM vec_articles
WHERE headline_embedding MATCH lembed("embed-model","the jury has been selected in Hunter")
LIMIT 2;
