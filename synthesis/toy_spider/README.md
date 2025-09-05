这个文件用来将arxiv数据集向量化,你只需要配置.env文件里面的api_key，vllm可以不用设置。你需要先设置hugging face国内镜像。
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

# 生成schema
```bash
python generate_schema.py --db-dir train/toy_spider --output-file train/table.json
```
产生的结果文件在results/table.json

# 加强schema
先使用enhance_tables_toy.py往schema加入每个数据表格的示例信息，运行：
```bash
python enhance_tables_toy.py
```

# 找到语言信息很丰富的列
注意：要排除地名和人名这种语义信息不丰富的列
运行：
```bash
python find_semantic_rich_column.py
```

# 为这些列生成embedding
```bash
export HF_ENDPOINT=https://hf-mirror.com
```
修改.env相关变量后，运行：
```bash
python batch_vectorize_databases.py
```

# 生成新的schema并且填入样例数据
运行：
```bash
python generate_vector_schema.py
#python enhance_tables_json.py #可以省略这一步，也就不用修改.env了
```

# 合成sql
## Step 1: SQL Query Generation

Generate SQL queries by leveraging database schemas, database values, query complexity, and SQLite-supported functions.

1. Execute `python3 generate_sql_synthesis_prompts.py` to create prompts for SQL query generation.
2. Run `python3 synthesize_sql.py` to generate SQL queries using LLMs. 

## Step 2: Post-Processing

Refine the generated SQL queries to ensure quality and remove invalid or redundant queries:

1. Run `python3 post_process_sqls.py` to:
   - Discard non-SELECT queries.
   - Remove queries with syntax errors or execution timeouts.
   - Deduplicate queries based on their templates.

2. The final synthetic SQL queries will be saved in `./results/synthetic_sqls.json`.

# 合成问题
## Step 1: Question Generation
Generate stylized natural language questions.
1. Run `python3 generate_question_synthesis_prompts.py` to create prompts for question generation.
2. Execute `python3 synthesize_question.py` to generate questions for the synthesized SQL queries. 

## Step 2: Post-Processing
1. Execute `python3 post_process_questions.py` to perform semantic consistency selection, ensuring the generated questions align closely with their corresponding SQL queries.
2. The final synthetic `<question, SQL>` pairs will be saved to `./results/question_and_sql_pairs.json`.

## Step 3: Add Candidate Sql
```bash
python synthesize_candidate.py --api_key <your api key>
```
1. Generate candidate sql.
2. 2. The final synthetic file will be saved to `./results/candidate_sql.json`.
