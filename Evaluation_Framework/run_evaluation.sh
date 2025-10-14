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


#################################################
#    Evaluation for DB Backend: CLICKHOUSE
#################################################

# Ensure output directory for clickhouse/arxiv exists
mkdir -p ./results/clickhouse/arxiv

# --- API Models on Dataset: arxiv for DB: clickhouse ---
echo "==> Evaluating API Model [gpt-4o] on Dataset [arxiv] for DB [clickhouse]"
python run_eval_pipeline.py --all \
    --base_dir "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/clickhouse/results/arxiv/vector_databases" \
    --evaluation_report_file "./results/clickhouse/arxiv/evaluation_report_api_gpt-4o.json" \
    --db_type "clickhouse" \
    --eval_data_file "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/results/clickhouse/arxiv/out_llm_api_gpt-4o.json" \
    --execution_results_file "./results/clickhouse/arxiv/sql_execution_results_api_gpt-4o.json"

echo "==> Evaluating API Model [gpt-4o-mini] on Dataset [arxiv] for DB [clickhouse]"
python run_eval_pipeline.py --all \
    --base_dir "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/clickhouse/results/arxiv/vector_databases" \
    --evaluation_report_file "./results/clickhouse/arxiv/evaluation_report_api_gpt-4o-mini.json" \
    --db_type "clickhouse" \
    --eval_data_file "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/results/clickhouse/arxiv/out_llm_api_gpt-4o-mini.json" \
    --execution_results_file "./results/clickhouse/arxiv/sql_execution_results_api_gpt-4o-mini.json"

echo "==> Evaluating API Model [gpt-4-turbo] on Dataset [arxiv] for DB [clickhouse]"
python run_eval_pipeline.py --all \
    --base_dir "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/clickhouse/results/arxiv/vector_databases" \
    --evaluation_report_file "./results/clickhouse/arxiv/evaluation_report_api_gpt-4-turbo.json" \
    --db_type "clickhouse" \
    --eval_data_file "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/results/clickhouse/arxiv/out_llm_api_gpt-4-turbo.json" \
    --execution_results_file "./results/clickhouse/arxiv/sql_execution_results_api_gpt-4-turbo.json"

# Ensure output directory for clickhouse/bird exists
mkdir -p ./results/clickhouse/bird

# --- API Models on Dataset: bird for DB: clickhouse ---
echo "==> Evaluating API Model [gpt-4o] on Dataset [bird] for DB [clickhouse]"
python run_eval_pipeline.py --all \
    --base_dir "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/clickhouse/results/bird/vector_databases" \
    --evaluation_report_file "./results/clickhouse/bird/evaluation_report_api_gpt-4o.json" \
    --db_type "clickhouse" \
    --eval_data_file "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/results/clickhouse/bird/out_llm_api_gpt-4o.json" \
    --execution_results_file "./results/clickhouse/bird/sql_execution_results_api_gpt-4o.json"

echo "==> Evaluating API Model [gpt-4o-mini] on Dataset [bird] for DB [clickhouse]"
python run_eval_pipeline.py --all \
    --base_dir "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/clickhouse/results/bird/vector_databases" \
    --evaluation_report_file "./results/clickhouse/bird/evaluation_report_api_gpt-4o-mini.json" \
    --db_type "clickhouse" \
    --eval_data_file "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/results/clickhouse/bird/out_llm_api_gpt-4o-mini.json" \
    --execution_results_file "./results/clickhouse/bird/sql_execution_results_api_gpt-4o-mini.json"

echo "==> Evaluating API Model [gpt-4-turbo] on Dataset [bird] for DB [clickhouse]"
python run_eval_pipeline.py --all \
    --base_dir "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/clickhouse/results/bird/vector_databases" \
    --evaluation_report_file "./results/clickhouse/bird/evaluation_report_api_gpt-4-turbo.json" \
    --db_type "clickhouse" \
    --eval_data_file "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/results/clickhouse/bird/out_llm_api_gpt-4-turbo.json" \
    --execution_results_file "./results/clickhouse/bird/sql_execution_results_api_gpt-4-turbo.json"

# Ensure output directory for clickhouse/wikipedia_multimodal exists
mkdir -p ./results/clickhouse/wikipedia_multimodal

# --- API Models on Dataset: wikipedia_multimodal for DB: clickhouse ---
echo "==> Evaluating API Model [gpt-4o] on Dataset [wikipedia_multimodal] for DB [clickhouse]"
python run_eval_pipeline.py --all \
    --base_dir "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/clickhouse/results/wikipedia_multimodal/vector_databases" \
    --evaluation_report_file "./results/clickhouse/wikipedia_multimodal/evaluation_report_api_gpt-4o.json" \
    --db_type "clickhouse" \
    --eval_data_file "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/results/clickhouse/wikipedia_multimodal/out_llm_api_gpt-4o.json" \
    --execution_results_file "./results/clickhouse/wikipedia_multimodal/sql_execution_results_api_gpt-4o.json"

echo "==> Evaluating API Model [gpt-4o-mini] on Dataset [wikipedia_multimodal] for DB [clickhouse]"
python run_eval_pipeline.py --all \
    --base_dir "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/clickhouse/results/wikipedia_multimodal/vector_databases" \
    --evaluation_report_file "./results/clickhouse/wikipedia_multimodal/evaluation_report_api_gpt-4o-mini.json" \
    --db_type "clickhouse" \
    --eval_data_file "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/results/clickhouse/wikipedia_multimodal/out_llm_api_gpt-4o-mini.json" \
    --execution_results_file "./results/clickhouse/wikipedia_multimodal/sql_execution_results_api_gpt-4o-mini.json"

echo "==> Evaluating API Model [gpt-4-turbo] on Dataset [wikipedia_multimodal] for DB [clickhouse]"
python run_eval_pipeline.py --all \
    --base_dir "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/clickhouse/results/wikipedia_multimodal/vector_databases" \
    --evaluation_report_file "./results/clickhouse/wikipedia_multimodal/evaluation_report_api_gpt-4-turbo.json" \
    --db_type "clickhouse" \
    --eval_data_file "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/results/clickhouse/wikipedia_multimodal/out_llm_api_gpt-4-turbo.json" \
    --execution_results_file "./results/clickhouse/wikipedia_multimodal/sql_execution_results_api_gpt-4-turbo.json"

# Ensure output directory for clickhouse/spider exists
mkdir -p ./results/clickhouse/spider

# --- API Models on Dataset: spider for DB: clickhouse ---
echo "==> Evaluating API Model [gpt-4o] on Dataset [spider] for DB [clickhouse]"
python run_eval_pipeline.py --all \
    --base_dir "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/clickhouse/results/spider/vector_databases" \
    --evaluation_report_file "./results/clickhouse/spider/evaluation_report_api_gpt-4o.json" \
    --db_type "clickhouse" \
    --eval_data_file "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/results/clickhouse/spider/out_llm_api_gpt-4o.json" \
    --execution_results_file "./results/clickhouse/spider/sql_execution_results_api_gpt-4o.json"

echo "==> Evaluating API Model [gpt-4o-mini] on Dataset [spider] for DB [clickhouse]"
python run_eval_pipeline.py --all \
    --base_dir "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/clickhouse/results/spider/vector_databases" \
    --evaluation_report_file "./results/clickhouse/spider/evaluation_report_api_gpt-4o-mini.json" \
    --db_type "clickhouse" \
    --eval_data_file "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/results/clickhouse/spider/out_llm_api_gpt-4o-mini.json" \
    --execution_results_file "./results/clickhouse/spider/sql_execution_results_api_gpt-4o-mini.json"

echo "==> Evaluating API Model [gpt-4-turbo] on Dataset [spider] for DB [clickhouse]"
python run_eval_pipeline.py --all \
    --base_dir "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/clickhouse/results/spider/vector_databases" \
    --evaluation_report_file "./results/clickhouse/spider/evaluation_report_api_gpt-4-turbo.json" \
    --db_type "clickhouse" \
    --eval_data_file "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/results/clickhouse/spider/out_llm_api_gpt-4-turbo.json" \
    --execution_results_file "./results/clickhouse/spider/sql_execution_results_api_gpt-4-turbo.json"


#################################################
#    Evaluation for DB Backend: POSTGRESQL
#################################################

# Ensure output directory for postgresql/arxiv exists
mkdir -p ./results/postgresql/arxiv

# --- API Models on Dataset: arxiv for DB: postgresql ---
echo "==> Evaluating API Model [gpt-4o] on Dataset [arxiv] for DB [postgresql]"
python run_eval_pipeline.py --all \
    --base_dir "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/postgresql/results/arxiv/vector_databases" \
    --evaluation_report_file "./results/postgresql/arxiv/evaluation_report_api_gpt-4o.json" \
    --db_type "postgresql" \
    --eval_data_file "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/results/postgresql/arxiv/out_llm_api_gpt-4o.json" \
    --execution_results_file "./results/postgresql/arxiv/sql_execution_results_api_gpt-4o.json"

echo "==> Evaluating API Model [gpt-4o-mini] on Dataset [arxiv] for DB [postgresql]"
python run_eval_pipeline.py --all \
    --base_dir "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/postgresql/results/arxiv/vector_databases" \
    --evaluation_report_file "./results/postgresql/arxiv/evaluation_report_api_gpt-4o-mini.json" \
    --db_type "postgresql" \
    --eval_data_file "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/results/postgresql/arxiv/out_llm_api_gpt-4o-mini.json" \
    --execution_results_file "./results/postgresql/arxiv/sql_execution_results_api_gpt-4o-mini.json"

echo "==> Evaluating API Model [gpt-4-turbo] on Dataset [arxiv] for DB [postgresql]"
python run_eval_pipeline.py --all \
    --base_dir "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/postgresql/results/arxiv/vector_databases" \
    --evaluation_report_file "./results/postgresql/arxiv/evaluation_report_api_gpt-4-turbo.json" \
    --db_type "postgresql" \
    --eval_data_file "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/results/postgresql/arxiv/out_llm_api_gpt-4-turbo.json" \
    --execution_results_file "./results/postgresql/arxiv/sql_execution_results_api_gpt-4-turbo.json"

# Ensure output directory for postgresql/bird exists
mkdir -p ./results/postgresql/bird

# --- API Models on Dataset: bird for DB: postgresql ---
echo "==> Evaluating API Model [gpt-4o] on Dataset [bird] for DB [postgresql]"
python run_eval_pipeline.py --all \
    --base_dir "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/postgresql/results/bird/vector_databases" \
    --evaluation_report_file "./results/postgresql/bird/evaluation_report_api_gpt-4o.json" \
    --db_type "postgresql" \
    --eval_data_file "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/results/postgresql/bird/out_llm_api_gpt-4o.json" \
    --execution_results_file "./results/postgresql/bird/sql_execution_results_api_gpt-4o.json"

echo "==> Evaluating API Model [gpt-4o-mini] on Dataset [bird] for DB [postgresql]"
python run_eval_pipeline.py --all \
    --base_dir "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/postgresql/results/bird/vector_databases" \
    --evaluation_report_file "./results/postgresql/bird/evaluation_report_api_gpt-4o-mini.json" \
    --db_type "postgresql" \
    --eval_data_file "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/results/postgresql/bird/out_llm_api_gpt-4o-mini.json" \
    --execution_results_file "./results/postgresql/bird/sql_execution_results_api_gpt-4o-mini.json"

echo "==> Evaluating API Model [gpt-4-turbo] on Dataset [bird] for DB [postgresql]"
python run_eval_pipeline.py --all \
    --base_dir "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/postgresql/results/bird/vector_databases" \
    --evaluation_report_file "./results/postgresql/bird/evaluation_report_api_gpt-4-turbo.json" \
    --db_type "postgresql" \
    --eval_data_file "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/results/postgresql/bird/out_llm_api_gpt-4-turbo.json" \
    --execution_results_file "./results/postgresql/bird/sql_execution_results_api_gpt-4-turbo.json"

# Ensure output directory for postgresql/wikipedia_multimodal exists
mkdir -p ./results/postgresql/wikipedia_multimodal

# --- API Models on Dataset: wikipedia_multimodal for DB: postgresql ---
echo "==> Evaluating API Model [gpt-4o] on Dataset [wikipedia_multimodal] for DB [postgresql]"
python run_eval_pipeline.py --all \
    --base_dir "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/postgresql/results/wikipedia_multimodal/vector_databases" \
    --evaluation_report_file "./results/postgresql/wikipedia_multimodal/evaluation_report_api_gpt-4o.json" \
    --db_type "postgresql" \
    --eval_data_file "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/results/postgresql/wikipedia_multimodal/out_llm_api_gpt-4o.json" \
    --execution_results_file "./results/postgresql/wikipedia_multimodal/sql_execution_results_api_gpt-4o.json"

echo "==> Evaluating API Model [gpt-4o-mini] on Dataset [wikipedia_multimodal] for DB [postgresql]"
python run_eval_pipeline.py --all \
    --base_dir "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/postgresql/results/wikipedia_multimodal/vector_databases" \
    --evaluation_report_file "./results/postgresql/wikipedia_multimodal/evaluation_report_api_gpt-4o-mini.json" \
    --db_type "postgresql" \
    --eval_data_file "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/results/postgresql/wikipedia_multimodal/out_llm_api_gpt-4o-mini.json" \
    --execution_results_file "./results/postgresql/wikipedia_multimodal/sql_execution_results_api_gpt-4o-mini.json"

echo "==> Evaluating API Model [gpt-4-turbo] on Dataset [wikipedia_multimodal] for DB [postgresql]"
python run_eval_pipeline.py --all \
    --base_dir "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/postgresql/results/wikipedia_multimodal/vector_databases" \
    --evaluation_report_file "./results/postgresql/wikipedia_multimodal/evaluation_report_api_gpt-4-turbo.json" \
    --db_type "postgresql" \
    --eval_data_file "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/results/postgresql/wikipedia_multimodal/out_llm_api_gpt-4-turbo.json" \
    --execution_results_file "./results/postgresql/wikipedia_multimodal/sql_execution_results_api_gpt-4-turbo.json"

# Ensure output directory for postgresql/spider exists
mkdir -p ./results/postgresql/spider

# --- API Models on Dataset: spider for DB: postgresql ---
echo "==> Evaluating API Model [gpt-4o] on Dataset [spider] for DB [postgresql]"
python run_eval_pipeline.py --all \
    --base_dir "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/postgresql/results/spider/vector_databases" \
    --evaluation_report_file "./results/postgresql/spider/evaluation_report_api_gpt-4o.json" \
    --db_type "postgresql" \
    --eval_data_file "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/results/postgresql/spider/out_llm_api_gpt-4o.json" \
    --execution_results_file "./results/postgresql/spider/sql_execution_results_api_gpt-4o.json"

echo "==> Evaluating API Model [gpt-4o-mini] on Dataset [spider] for DB [postgresql]"
python run_eval_pipeline.py --all \
    --base_dir "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/postgresql/results/spider/vector_databases" \
    --evaluation_report_file "./results/postgresql/spider/evaluation_report_api_gpt-4o-mini.json" \
    --db_type "postgresql" \
    --eval_data_file "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/results/postgresql/spider/out_llm_api_gpt-4o-mini.json" \
    --execution_results_file "./results/postgresql/spider/sql_execution_results_api_gpt-4o-mini.json"

echo "==> Evaluating API Model [gpt-4-turbo] on Dataset [spider] for DB [postgresql]"
python run_eval_pipeline.py --all \
    --base_dir "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/postgresql/results/spider/vector_databases" \
    --evaluation_report_file "./results/postgresql/spider/evaluation_report_api_gpt-4-turbo.json" \
    --db_type "postgresql" \
    --eval_data_file "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/results/postgresql/spider/out_llm_api_gpt-4-turbo.json" \
    --execution_results_file "./results/postgresql/spider/sql_execution_results_api_gpt-4-turbo.json"
