这个文件夹负责合成Text2VecSql数据。

# bird_vectoriation
负责将bird数据集变成向量数据集（添加embedding列，并使用sqlite_vec中的虚拟表来存储向量）

# database_vectorize（目前弃用）
可以为bird和spider1.0添加description列，来增加语义丰富列的数量。同时将其变成向量数据集。

# sql_synthesis
负责基于向量数据集合成vecsql。

# question_synthesis
负责基于vecsql合成对应的问题。
