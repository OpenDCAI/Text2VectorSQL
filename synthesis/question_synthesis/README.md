# Stylized Natural Language Question Synthesis

This is the third step in our data synthesis framework, dedicated to generating stylized natural language questions for synthetic SQL queries.

## Step 1: Question Generation

Generate stylized natural language questions.

1. Run `python3 generate_question_synthesis_prompts.py` to create prompts for question generation.
2. Execute `python3 synthesize_question.py` to generate questions for the synthesized SQL queries. Note: Ensure the `llm_inference()` function is implemented to integrate your preferred LLM. For each prompt (SQL query), we sample multiple responses (questions) with a temperature of `0.8`.

## Step 2: Post-Processing

1. Execute `python3 post_process_questions.py` to perform semantic consistency selection, ensuring the generated questions align closely with their corresponding SQL queries.
2. The final synthetic `<question, SQL>` pairs will be saved to `./results/question_and_sql_pairs.json`.