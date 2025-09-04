# Complexity-Aware SQL Query Generation

This is the second step in our data synthesis framework, focused on generating complexity-aware SQL queries based on synthetic databases.

## Step 1: SQL Query Generation

Generate SQL queries by leveraging database schemas, database values, query complexity, and SQLite-supported functions.

1. Execute `python3 generate_sql_synthesis_prompts.py` to create prompts for SQL query generation.
2. Run `python3 synthesize_sql.py` to generate SQL queries using LLMs. (Note: Implement the `llm_inference()` function to integrate your preferred LLM.)

## Step 2: Post-Processing

Refine the generated SQL queries to ensure quality and remove invalid or redundant queries:

1. Run `python3 post_process_sqls.py` to:
   - Discard non-SELECT queries.
   - Remove queries with syntax errors or execution timeouts.
   - Deduplicate queries based on their templates.

2. The final synthetic SQL queries will be saved in `./results/synthetic_sqls.json`.
