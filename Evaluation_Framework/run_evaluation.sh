#!/bin/bash
# This script is auto-generated to run the evaluation pipeline.
set -e  # Exit immediately if a command exits with a non-zero status.

#################################################
#    Evaluation for DB Backend: SQLITE
#################################################

# Ensure output directory for sqlite/arxiv exists
mkdir -p ./results/sqlite/arxiv

# --- API Models on Dataset: arxiv for DB: sqlite ---
echo "==> Evaluating API Model [gpt-4o] on Dataset [arxiv] for DB [sqlite]"
python run_eval_pipeline.py --all \
    --base_dir "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/sqlite/results/arxiv/vector_databases" \
    --evaluation_report_file "./results/sqlite/arxiv/evaluation_report_api_gpt-4o.json" \
    --db_type "sqlite" \
    --eval_data_file "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/results/sqlite/arxiv/out_llm_api_gpt-4o.json" \
    --execution_results_file "./results/sqlite/arxiv/sql_execution_results_api_gpt-4o.json"

echo "==> Evaluating API Model [gpt-4o-mini] on Dataset [arxiv] for DB [sqlite]"
python run_eval_pipeline.py --all \
    --base_dir "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/sqlite/results/arxiv/vector_databases" \
    --evaluation_report_file "./results/sqlite/arxiv/evaluation_report_api_gpt-4o-mini.json" \
    --db_type "sqlite" \
    --eval_data_file "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/results/sqlite/arxiv/out_llm_api_gpt-4o-mini.json" \
    --execution_results_file "./results/sqlite/arxiv/sql_execution_results_api_gpt-4o-mini.json"

echo "==> Evaluating API Model [gpt-4-turbo] on Dataset [arxiv] for DB [sqlite]"
python run_eval_pipeline.py --all \
    --base_dir "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/sqlite/results/arxiv/vector_databases" \
    --evaluation_report_file "./results/sqlite/arxiv/evaluation_report_api_gpt-4-turbo.json" \
    --db_type "sqlite" \
    --eval_data_file "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/results/sqlite/arxiv/out_llm_api_gpt-4-turbo.json" \
    --execution_results_file "./results/sqlite/arxiv/sql_execution_results_api_gpt-4-turbo.json"

# Ensure output directory for sqlite/bird exists
mkdir -p ./results/sqlite/bird

# --- API Models on Dataset: bird for DB: sqlite ---
echo "==> Evaluating API Model [gpt-4o] on Dataset [bird] for DB [sqlite]"
python run_eval_pipeline.py --all \
    --base_dir "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/sqlite/results/bird/vector_databases" \
    --evaluation_report_file "./results/sqlite/bird/evaluation_report_api_gpt-4o.json" \
    --db_type "sqlite" \
    --eval_data_file "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/results/sqlite/bird/out_llm_api_gpt-4o.json" \
    --execution_results_file "./results/sqlite/bird/sql_execution_results_api_gpt-4o.json"

echo "==> Evaluating API Model [gpt-4o-mini] on Dataset [bird] for DB [sqlite]"
python run_eval_pipeline.py --all \
    --base_dir "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/sqlite/results/bird/vector_databases" \
    --evaluation_report_file "./results/sqlite/bird/evaluation_report_api_gpt-4o-mini.json" \
    --db_type "sqlite" \
    --eval_data_file "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/results/sqlite/bird/out_llm_api_gpt-4o-mini.json" \
    --execution_results_file "./results/sqlite/bird/sql_execution_results_api_gpt-4o-mini.json"

echo "==> Evaluating API Model [gpt-4-turbo] on Dataset [bird] for DB [sqlite]"
python run_eval_pipeline.py --all \
    --base_dir "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/sqlite/results/bird/vector_databases" \
    --evaluation_report_file "./results/sqlite/bird/evaluation_report_api_gpt-4-turbo.json" \
    --db_type "sqlite" \
    --eval_data_file "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/results/sqlite/bird/out_llm_api_gpt-4-turbo.json" \
    --execution_results_file "./results/sqlite/bird/sql_execution_results_api_gpt-4-turbo.json"

# Ensure output directory for sqlite/wikipedia_multimodal exists
mkdir -p ./results/sqlite/wikipedia_multimodal

# --- API Models on Dataset: wikipedia_multimodal for DB: sqlite ---
echo "==> Evaluating API Model [gpt-4o] on Dataset [wikipedia_multimodal] for DB [sqlite]"
python run_eval_pipeline.py --all \
    --base_dir "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/sqlite/results/wikipedia_multimodal/vector_databases" \
    --evaluation_report_file "./results/sqlite/wikipedia_multimodal/evaluation_report_api_gpt-4o.json" \
    --db_type "sqlite" \
    --eval_data_file "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/results/sqlite/wikipedia_multimodal/out_llm_api_gpt-4o.json" \
    --execution_results_file "./results/sqlite/wikipedia_multimodal/sql_execution_results_api_gpt-4o.json"

echo "==> Evaluating API Model [gpt-4o-mini] on Dataset [wikipedia_multimodal] for DB [sqlite]"
python run_eval_pipeline.py --all \
    --base_dir "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/sqlite/results/wikipedia_multimodal/vector_databases" \
    --evaluation_report_file "./results/sqlite/wikipedia_multimodal/evaluation_report_api_gpt-4o-mini.json" \
    --db_type "sqlite" \
    --eval_data_file "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/results/sqlite/wikipedia_multimodal/out_llm_api_gpt-4o-mini.json" \
    --execution_results_file "./results/sqlite/wikipedia_multimodal/sql_execution_results_api_gpt-4o-mini.json"

echo "==> Evaluating API Model [gpt-4-turbo] on Dataset [wikipedia_multimodal] for DB [sqlite]"
python run_eval_pipeline.py --all \
    --base_dir "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/sqlite/results/wikipedia_multimodal/vector_databases" \
    --evaluation_report_file "./results/sqlite/wikipedia_multimodal/evaluation_report_api_gpt-4-turbo.json" \
    --db_type "sqlite" \
    --eval_data_file "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/results/sqlite/wikipedia_multimodal/out_llm_api_gpt-4-turbo.json" \
    --execution_results_file "./results/sqlite/wikipedia_multimodal/sql_execution_results_api_gpt-4-turbo.json"

# Ensure output directory for sqlite/spider exists
mkdir -p ./results/sqlite/spider

# --- API Models on Dataset: spider for DB: sqlite ---
echo "==> Evaluating API Model [gpt-4o] on Dataset [spider] for DB [sqlite]"
python run_eval_pipeline.py --all \
    --base_dir "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/sqlite/results/spider/vector_databases" \
    --evaluation_report_file "./results/sqlite/spider/evaluation_report_api_gpt-4o.json" \
    --db_type "sqlite" \
    --eval_data_file "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/results/sqlite/spider/out_llm_api_gpt-4o.json" \
    --execution_results_file "./results/sqlite/spider/sql_execution_results_api_gpt-4o.json"

echo "==> Evaluating API Model [gpt-4o-mini] on Dataset [spider] for DB [sqlite]"
python run_eval_pipeline.py --all \
    --base_dir "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/sqlite/results/spider/vector_databases" \
    --evaluation_report_file "./results/sqlite/spider/evaluation_report_api_gpt-4o-mini.json" \
    --db_type "sqlite" \
    --eval_data_file "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/results/sqlite/spider/out_llm_api_gpt-4o-mini.json" \
    --execution_results_file "./results/sqlite/spider/sql_execution_results_api_gpt-4o-mini.json"

echo "==> Evaluating API Model [gpt-4-turbo] on Dataset [spider] for DB [sqlite]"
python run_eval_pipeline.py --all \
    --base_dir "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/sqlite/results/spider/vector_databases" \
    --evaluation_report_file "./results/sqlite/spider/evaluation_report_api_gpt-4-turbo.json" \
    --db_type "sqlite" \
    --eval_data_file "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/results/sqlite/spider/out_llm_api_gpt-4-turbo.json" \
    --execution_results_file "./results/sqlite/spider/sql_execution_results_api_gpt-4-turbo.json"
