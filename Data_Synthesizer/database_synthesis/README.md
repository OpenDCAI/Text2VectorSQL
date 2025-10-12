# Web Table-Driven Database Synthesis

This is the first step in our data synthesis framework, designed to generate realistic databases using web tables.

## Prepare Web Tables
Unzip `web_tables.json.zip` to access 19,935 high-quality web tables from [Tablib](https://arxiv.org/pdf/2310.07875).

## Step 1: Initial Database Generation
Generate an initial database from the web tables.

1. Run `python3 generate_schema_synthesis_prompts.py` to create prompts for database generation.
2. Run `python3 synthesize_schema.py` to generate initial database schemas. (Implement the `llm_inference()` function to use your preferred LLMs.)

## Step 2: Database Enhancement
Enhance the initially generated databases to increase complexity and realism.

1. Run `python3 generate_schema_enhancement_prompts.py` to create prompts for database enhancement.
2. Run `python3 enhance_schema.py` to generate enhanced database schemas. (Implement the `llm_inference()` function to use your preferred LLMs.)

<!-- ## Step 3: Generate vector database schemas
1. Run `python3 generate_schema_embedding_prompts.py` to create prompts for database vector table generation.
2. Run `python3 embedding_schema.py` to generate vector database schemas. (Implement the `llm_inference()` function to use your preferred LLMs.) -->

## Step 4: Building SQLite Databases
Build SQLite databases based on the enhanced database schemas.

1. Run `python3 build_sqlite_databases.py` to construct SQLite databases, which are stored in the `synthesis_data` folder.
2. Run `python3 generate_tables_json.py` to create the `tables.json` file, containing detailed information about the synthetic databases, aligning with previous text-to-SQL datasets.
