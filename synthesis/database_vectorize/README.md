## bird数据集结构如下：
database: The database should be stored under the ./data/dev_databases/. In each database folder, it has two components:
1. database_description: the csv files are manufactured to describe database schema and its values for models to explore or references.
2. sqlite: The database contents in BIRD.

data: Each text-to-SQL pairs with the oracle knowledge evidence is stored as a json file, i.e., dev.json is stored on ./data/dev.json. In each json file, it has three main parts:
1. db_id: the names of databases
2. question: the questions curated by human crowdsourcing according to database descriptions, database contents.
3. evidence: the external knowledge evidence annotated by experts for assistance of models or SQL annotators.

SQL: SQLs annotated by crowdsource referring to database descriptions, database contents, to answer the questions accurately.

## 数据库向量化流程
1. 为了增加train_tables.json的信息，有利于大模型推理。先根据bird数据集的结构，先将每个database_description文件夹下的描述内容和sqlite文件中的前两行示例数据导入train_tables.json的每个对应元素中，新增字段分别为table_description和table_samples。
```bash
python enhance_tables_json.py
```
2. 将增强后的enhanced_train_tables.json给大模型，让大模型判断哪些表格适合添加description信息，在table.json中添加new_description字段，其值为一个字典。这个字典的键为表格名，值为要添加的description和可以这么添加的原因。设置好.env文件后直接运行
```bash
python add_descriptions_schema.py
```
接下来大模型会根据上一步得到的results/add_description_table.json批量为数据库添加description列，为了提高效率修改将直接更改原始数据库，建议您提前使用cp命令备份好您的数据库。设置好.env文件后直接运行
```bash
python add_new_description.py
```
根据新数据库重新生成schema来供后续使用，修改下面文件中的参数后，执行：
```bash
python generate_vector_schema.py
```
3. 将上一步得到的table.json给大模型，大模型判断语义丰富的列，输出到字段column_alter。
```bash
python find_semantic_rich_column.py --model gpt-4o --api_key sk-xxx  --api_url http://123.129.219.111:3000/v1
```
4. 根据第二步的输出结果，先生成向量数据库创建脚本，保存在results/sql_script目录下。这主要使用了vecotr_database_generate.py中的generate_database_script。其读取单个bird数据库文件，将这个数据库导出为sql文件。这个sql文件包含原数据库的信息，并且会根据情况批量为数据库添加新向量列（原列名加"_embedding"），同时使用embedding模型批量往insert语句里面填充embedding，最终就得到一个后缀为sql的创建向量数据库的脚本，接下来会使用vecotr_database_generate.py中的build_vector_database函数来创建向量数据库。
```bash
python batch_vectorize_databases.py
```
4. 根据新的数据库，重新合成final_table.json文件。
```bash
python generate_vector_schema.py
```

## 修改sql-question问答对
1. 筛选原始问答对： 从BIRD数据集中，筛选出那些WHERE子句对阶段一中已向量化的列进行精确文本匹配的问题-SQL对。
```bash
python filter_bird.py
```
2. 在使用大模型修改问题-SQL对前，先增强一下schema文件，把表格的样例加进去。这样方便大模型参考样例中的描述信息，生成更高质量的描述的匹配。
```bash
python enhance_tables_json.py
```
3. 利用LLM进行问题重写 (Prompt Engineering)：
输入 (Input):
    原始问题 (Original Question)
    原始SQL (Original SQL)
    作为WHERE条件的精确值 (e.g., a specific paper's title)
    该精确值对应的描述性文本 (e.g., the paper's abstract)
指令 (Prompt):
    "你是一个NL2SQL数据集增强专家。请基于以下信息，将一个精确匹配的问题改写成一个需要语义搜索的模糊问题。
    [原则]
    新问题必须更模糊、更概念化。
    新问题不能包含原始的精确匹配关键词。
    新问题查询的最终目标（SELECT部分）应与原问题保持一致。
    新问题需要听起来自然流畅。
    [原始信息]
    原始问题: {original_question}
    原始SQL: {original_sql}
    精确匹配的值: {exact_match_value}
    用于语义化的参考文本: {vectorized_column_text}
    [任务]
    请生成一个新的、符合上述原则的语义化问题。
    请生成与新问题对应的VectorSQL查询语句。"
生成VectorSQL语法：
    LLM也需要根据指令生成对应的VectorSQL。你需要为LLM定义一个统一的VectorSQL语法模板。一个通用的语法可能如下：
    SQL
    SELECT ... FROM ... WHERE ... -- (other conditions)
    ORDER BY COSINE_SIMILARITY(table.vector_column, EMBED('semantic query text')) DESC
    LIMIT 5;
    或者使用一个更简洁的函数：
    SQL
    SELECT ... FROM VECTOR_SEARCH(
        TABLE table,
        COLUMN vector_column,
        QUERY EMBED('semantic query text'),
        TOP_K 5
    )
    WHERE ...;
    EMBED(...) 是一个示意函数，代表将引号内的文本通过嵌入模型转换为向量的过程。
```bash
python sql_vectorize.py
```
