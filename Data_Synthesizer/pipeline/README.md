# 说明
这个目录用于展示整个数据合成过程。
这里提供一个openai的配置文件示例config.yaml.example。
你可以删除文件名中的.example后，配置api-key字段，来使用默认的openai api方法调用大模型。
也可以修改脚本，将openai的调用换成ollama或者vllm（这需要你自己修改每一个文件中的大模型调用的实现）。
配置好大模型调用api-key后，可以直接运行：
```bash
python toy_pipeline.py
```
日志在logging/out.log
就可以在"sqlite/results/toy_spider"得到最终文件(synthetic_text2sql_dataset.json)和向量数据库，包括一系列中间文件。

下面将介绍这个pipeline的每一步，详细描述每个文件的作用。你也可以单独调用，精细调整其中的每一步操作。

—————————————————————————————————————————————————————————————————————————————————————————————————————————————
如果你希望自己控制其中的每一步过程，那么可以参考下面的执行。下面介绍了本目录下每个文件的具体作用。
# 生成schema
```bash
python generate_schema.py --db-dir train/toy_spider --output-file train/table.json
```

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


# Stylized Natural Language Question Synthesis
This is the final step in our data synthesis framework, focused on generating step-by-step chain-of-thought (CoT) solutions for `<database, question, SQL query>` triplets.
## Step 1: Chain-of-Thought Generation
Create CoT solutions for each data sample.
1. Run `python3 generate_cot_synthesis_prompts.py` to prepare prompts for CoT generation.
2. Execute `python3 synthesize_cot.py` to generate CoT solutions for `<database, question, SQL query>` samples. (Note: For each prompt, we sample multiple CoT solutions with a temperature of `0.8`.)
## Step 2: Post-Processing
1. Run `python3 post_process_cot.py` to perform execution-based major voting, selecting the most reliable CoT solutions.
2. Save the final synthetic `<database, question, SQL query, CoT solution>` samples to `./results/synthetic_text2sql_dataset.json`.
