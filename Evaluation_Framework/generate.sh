#!/bin/bash
# This script is auto-generated to run generation tasks across different backends, modes, and models.
set -e  # Exit immediately if a command exits with a non-zero status.

#################################################
#    Commands for Backend: SQLITE
#################################################

### API Mode Commands ###

# Ensure output directory exists for arxiv on sqlite
mkdir -p ./results/sqlite/arxiv

# echo "--- Running API [gpt-4o] on DATASET [arxiv] for BACKEND [sqlite] ---"
# python generate.py \
#     --mode "api" \
#     --dataset "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/sqlite/results/arxiv/input_llm.json" \
#     --output "./results/sqlite/arxiv/out_llm_api_gpt-4o.json" \
#     --model_name "gpt-4o" \
#     --config "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/generate_config.yaml"

# # Ensure output directory exists for arxiv on sqlite
# mkdir -p ./results/sqlite/arxiv

# echo "--- Running API [gpt-4o-mini] on DATASET [arxiv] for BACKEND [sqlite] ---"
# python generate.py \
#     --mode "api" \
#     --dataset "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/sqlite/results/arxiv/input_llm.json" \
#     --output "./results/sqlite/arxiv/out_llm_api_gpt-4o-mini.json" \
#     --model_name "gpt-4o-mini" \
#     --config "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/generate_config.yaml"

# # Ensure output directory exists for arxiv on sqlite
# mkdir -p ./results/sqlite/arxiv

# echo "--- Running API [gpt-4-turbo] on DATASET [arxiv] for BACKEND [sqlite] ---"
# python generate.py \
#     --mode "api" \
#     --dataset "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/sqlite/results/arxiv/input_llm.json" \
#     --output "./results/sqlite/arxiv/out_llm_api_gpt-4-turbo.json" \
#     --model_name "gpt-4-turbo" \
#     --config "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/generate_config.yaml"

# # Ensure output directory exists for bird on sqlite
# mkdir -p ./results/sqlite/bird

# echo "--- Running API [gpt-4o] on DATASET [bird] for BACKEND [sqlite] ---"
# python generate.py \
#     --mode "api" \
#     --dataset "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/sqlite/results/bird/input_llm.json" \
#     --output "./results/sqlite/bird/out_llm_api_gpt-4o.json" \
#     --model_name "gpt-4o" \
#     --config "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/generate_config.yaml"

# # Ensure output directory exists for bird on sqlite
# mkdir -p ./results/sqlite/bird

# echo "--- Running API [gpt-4o-mini] on DATASET [bird] for BACKEND [sqlite] ---"
# python generate.py \
#     --mode "api" \
#     --dataset "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/sqlite/results/bird/input_llm.json" \
#     --output "./results/sqlite/bird/out_llm_api_gpt-4o-mini.json" \
#     --model_name "gpt-4o-mini" \
#     --config "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/generate_config.yaml"

# # Ensure output directory exists for bird on sqlite
# mkdir -p ./results/sqlite/bird

# echo "--- Running API [gpt-4-turbo] on DATASET [bird] for BACKEND [sqlite] ---"
# python generate.py \
#     --mode "api" \
#     --dataset "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/sqlite/results/bird/input_llm.json" \
#     --output "./results/sqlite/bird/out_llm_api_gpt-4-turbo.json" \
#     --model_name "gpt-4-turbo" \
#     --config "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/generate_config.yaml"

# # Ensure output directory exists for spider on sqlite
# mkdir -p ./results/sqlite/spider

# echo "--- Running API [gpt-4o] on DATASET [spider] for BACKEND [sqlite] ---"
# python generate.py \
#     --mode "api" \
#     --dataset "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/sqlite/results/spider/input_llm.json" \
#     --output "./results/sqlite/spider/out_llm_api_gpt-4o.json" \
#     --model_name "gpt-4o" \
#     --config "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/generate_config.yaml"

# # Ensure output directory exists for spider on sqlite
# mkdir -p ./results/sqlite/spider

# echo "--- Running API [gpt-4o-mini] on DATASET [spider] for BACKEND [sqlite] ---"
# python generate.py \
#     --mode "api" \
#     --dataset "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/sqlite/results/spider/input_llm.json" \
#     --output "./results/sqlite/spider/out_llm_api_gpt-4o-mini.json" \
#     --model_name "gpt-4o-mini" \
#     --config "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/generate_config.yaml"

# # Ensure output directory exists for spider on sqlite
# mkdir -p ./results/sqlite/spider

# echo "--- Running API [gpt-4-turbo] on DATASET [spider] for BACKEND [sqlite] ---"
# python generate.py \
#     --mode "api" \
#     --dataset "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/sqlite/results/spider/input_llm.json" \
#     --output "./results/sqlite/spider/out_llm_api_gpt-4-turbo.json" \
#     --model_name "gpt-4-turbo" \
#     --config "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/generate_config.yaml"

# # Ensure output directory exists for wikipedia_multimodal on sqlite
# mkdir -p ./results/sqlite/wikipedia_multimodal

# echo "--- Running API [gpt-4o] on DATASET [wikipedia_multimodal] for BACKEND [sqlite] ---"
# python generate.py \
#     --mode "api" \
#     --dataset "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/sqlite/results/wikipedia_multimodal/input_llm.json" \
#     --output "./results/sqlite/wikipedia_multimodal/out_llm_api_gpt-4o.json" \
#     --model_name "gpt-4o" \
#     --config "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/generate_config.yaml"

# # Ensure output directory exists for wikipedia_multimodal on sqlite
# mkdir -p ./results/sqlite/wikipedia_multimodal

# echo "--- Running API [gpt-4o-mini] on DATASET [wikipedia_multimodal] for BACKEND [sqlite] ---"
# python generate.py \
#     --mode "api" \
#     --dataset "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/sqlite/results/wikipedia_multimodal/input_llm.json" \
#     --output "./results/sqlite/wikipedia_multimodal/out_llm_api_gpt-4o-mini.json" \
#     --model_name "gpt-4o-mini" \
#     --config "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/generate_config.yaml"

# # Ensure output directory exists for wikipedia_multimodal on sqlite
# mkdir -p ./results/sqlite/wikipedia_multimodal

# echo "--- Running API [gpt-4-turbo] on DATASET [wikipedia_multimodal] for BACKEND [sqlite] ---"
# python generate.py \
#     --mode "api" \
#     --dataset "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/sqlite/results/wikipedia_multimodal/input_llm.json" \
#     --output "./results/sqlite/wikipedia_multimodal/out_llm_api_gpt-4-turbo.json" \
#     --model_name "gpt-4-turbo" \
#     --config "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/generate_config.yaml"


# #################################################
# #    Commands for Backend: CLICKHOUSE
# #################################################

# ### API Mode Commands ###

# # Ensure output directory exists for arxiv on clickhouse
# mkdir -p ./results/clickhouse/arxiv

# echo "--- Running API [gpt-4o] on DATASET [arxiv] for BACKEND [clickhouse] ---"
# python generate.py \
#     --mode "api" \
#     --dataset "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/clickhouse/results/arxiv/input_llm.json" \
#     --output "./results/clickhouse/arxiv/out_llm_api_gpt-4o.json" \
#     --model_name "gpt-4o" \
#     --config "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/generate_config.yaml"

# # Ensure output directory exists for arxiv on clickhouse
# mkdir -p ./results/clickhouse/arxiv

# echo "--- Running API [gpt-4o-mini] on DATASET [arxiv] for BACKEND [clickhouse] ---"
# python generate.py \
#     --mode "api" \
#     --dataset "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/clickhouse/results/arxiv/input_llm.json" \
#     --output "./results/clickhouse/arxiv/out_llm_api_gpt-4o-mini.json" \
#     --model_name "gpt-4o-mini" \
#     --config "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/generate_config.yaml"

# # Ensure output directory exists for arxiv on clickhouse
# mkdir -p ./results/clickhouse/arxiv

# echo "--- Running API [gpt-4-turbo] on DATASET [arxiv] for BACKEND [clickhouse] ---"
# python generate.py \
#     --mode "api" \
#     --dataset "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/clickhouse/results/arxiv/input_llm.json" \
#     --output "./results/clickhouse/arxiv/out_llm_api_gpt-4-turbo.json" \
#     --model_name "gpt-4-turbo" \
#     --config "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/generate_config.yaml"

# # Ensure output directory exists for bird on clickhouse
# mkdir -p ./results/clickhouse/bird

# echo "--- Running API [gpt-4o] on DATASET [bird] for BACKEND [clickhouse] ---"
# python generate.py \
#     --mode "api" \
#     --dataset "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/clickhouse/results/bird/input_llm.json" \
#     --output "./results/clickhouse/bird/out_llm_api_gpt-4o.json" \
#     --model_name "gpt-4o" \
#     --config "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/generate_config.yaml"

# # Ensure output directory exists for bird on clickhouse
# mkdir -p ./results/clickhouse/bird

# echo "--- Running API [gpt-4o-mini] on DATASET [bird] for BACKEND [clickhouse] ---"
# python generate.py \
#     --mode "api" \
#     --dataset "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/clickhouse/results/bird/input_llm.json" \
#     --output "./results/clickhouse/bird/out_llm_api_gpt-4o-mini.json" \
#     --model_name "gpt-4o-mini" \
#     --config "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/generate_config.yaml"

# # Ensure output directory exists for bird on clickhouse
# mkdir -p ./results/clickhouse/bird

# echo "--- Running API [gpt-4-turbo] on DATASET [bird] for BACKEND [clickhouse] ---"
# python generate.py \
#     --mode "api" \
#     --dataset "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/clickhouse/results/bird/input_llm.json" \
#     --output "./results/clickhouse/bird/out_llm_api_gpt-4-turbo.json" \
#     --model_name "gpt-4-turbo" \
#     --config "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/generate_config.yaml"

# # Ensure output directory exists for spider on clickhouse
# mkdir -p ./results/clickhouse/spider

# echo "--- Running API [gpt-4o] on DATASET [spider] for BACKEND [clickhouse] ---"
# python generate.py \
#     --mode "api" \
#     --dataset "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/clickhouse/results/spider/input_llm.json" \
#     --output "./results/clickhouse/spider/out_llm_api_gpt-4o.json" \
#     --model_name "gpt-4o" \
#     --config "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/generate_config.yaml"

# # Ensure output directory exists for spider on clickhouse
# mkdir -p ./results/clickhouse/spider

# echo "--- Running API [gpt-4o-mini] on DATASET [spider] for BACKEND [clickhouse] ---"
# python generate.py \
#     --mode "api" \
#     --dataset "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/clickhouse/results/spider/input_llm.json" \
#     --output "./results/clickhouse/spider/out_llm_api_gpt-4o-mini.json" \
#     --model_name "gpt-4o-mini" \
#     --config "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/generate_config.yaml"

# # Ensure output directory exists for spider on clickhouse
# mkdir -p ./results/clickhouse/spider

# echo "--- Running API [gpt-4-turbo] on DATASET [spider] for BACKEND [clickhouse] ---"
# python generate.py \
#     --mode "api" \
#     --dataset "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/clickhouse/results/spider/input_llm.json" \
#     --output "./results/clickhouse/spider/out_llm_api_gpt-4-turbo.json" \
#     --model_name "gpt-4-turbo" \
#     --config "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/generate_config.yaml"

# # Ensure output directory exists for wikipedia_multimodal on clickhouse
# mkdir -p ./results/clickhouse/wikipedia_multimodal

# echo "--- Running API [gpt-4o] on DATASET [wikipedia_multimodal] for BACKEND [clickhouse] ---"
# python generate.py \
#     --mode "api" \
#     --dataset "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/clickhouse/results/wikipedia_multimodal/input_llm.json" \
#     --output "./results/clickhouse/wikipedia_multimodal/out_llm_api_gpt-4o.json" \
#     --model_name "gpt-4o" \
#     --config "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/generate_config.yaml"

# # Ensure output directory exists for wikipedia_multimodal on clickhouse
# mkdir -p ./results/clickhouse/wikipedia_multimodal

# echo "--- Running API [gpt-4o-mini] on DATASET [wikipedia_multimodal] for BACKEND [clickhouse] ---"
# python generate.py \
#     --mode "api" \
#     --dataset "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/clickhouse/results/wikipedia_multimodal/input_llm.json" \
#     --output "./results/clickhouse/wikipedia_multimodal/out_llm_api_gpt-4o-mini.json" \
#     --model_name "gpt-4o-mini" \
#     --config "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/generate_config.yaml"

# # Ensure output directory exists for wikipedia_multimodal on clickhouse
# mkdir -p ./results/clickhouse/wikipedia_multimodal

# echo "--- Running API [gpt-4-turbo] on DATASET [wikipedia_multimodal] for BACKEND [clickhouse] ---"
# python generate.py \
#     --mode "api" \
#     --dataset "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/clickhouse/results/wikipedia_multimodal/input_llm.json" \
#     --output "./results/clickhouse/wikipedia_multimodal/out_llm_api_gpt-4-turbo.json" \
#     --model_name "gpt-4-turbo" \
#     --config "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/generate_config.yaml"


#################################################
#    Commands for Backend: POSTGRESQL
#################################################

### API Mode Commands ###

# Ensure output directory exists for arxiv on postgresql
mkdir -p ./results/postgresql/arxiv

echo "--- Running API [gpt-4o] on DATASET [arxiv] for BACKEND [postgresql] ---"
python generate.py \
    --mode "api" \
    --dataset "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/postgresql/results/arxiv/input_llm.json" \
    --output "./results/postgresql/arxiv/out_llm_api_gpt-4o.json" \
    --model_name "gpt-4o" \
    --config "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/generate_config.yaml"

# Ensure output directory exists for arxiv on postgresql
mkdir -p ./results/postgresql/arxiv

echo "--- Running API [gpt-4o-mini] on DATASET [arxiv] for BACKEND [postgresql] ---"
python generate.py \
    --mode "api" \
    --dataset "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/postgresql/results/arxiv/input_llm.json" \
    --output "./results/postgresql/arxiv/out_llm_api_gpt-4o-mini.json" \
    --model_name "gpt-4o-mini" \
    --config "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/generate_config.yaml"

# Ensure output directory exists for arxiv on postgresql
mkdir -p ./results/postgresql/arxiv

echo "--- Running API [gpt-4-turbo] on DATASET [arxiv] for BACKEND [postgresql] ---"
python generate.py \
    --mode "api" \
    --dataset "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/postgresql/results/arxiv/input_llm.json" \
    --output "./results/postgresql/arxiv/out_llm_api_gpt-4-turbo.json" \
    --model_name "gpt-4-turbo" \
    --config "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/generate_config.yaml"

# Ensure output directory exists for bird on postgresql
mkdir -p ./results/postgresql/bird

echo "--- Running API [gpt-4o] on DATASET [bird] for BACKEND [postgresql] ---"
python generate.py \
    --mode "api" \
    --dataset "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/postgresql/results/bird/input_llm.json" \
    --output "./results/postgresql/bird/out_llm_api_gpt-4o.json" \
    --model_name "gpt-4o" \
    --config "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/generate_config.yaml"

# Ensure output directory exists for bird on postgresql
mkdir -p ./results/postgresql/bird

echo "--- Running API [gpt-4o-mini] on DATASET [bird] for BACKEND [postgresql] ---"
python generate.py \
    --mode "api" \
    --dataset "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/postgresql/results/bird/input_llm.json" \
    --output "./results/postgresql/bird/out_llm_api_gpt-4o-mini.json" \
    --model_name "gpt-4o-mini" \
    --config "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/generate_config.yaml"

# Ensure output directory exists for bird on postgresql
mkdir -p ./results/postgresql/bird

echo "--- Running API [gpt-4-turbo] on DATASET [bird] for BACKEND [postgresql] ---"
python generate.py \
    --mode "api" \
    --dataset "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/postgresql/results/bird/input_llm.json" \
    --output "./results/postgresql/bird/out_llm_api_gpt-4-turbo.json" \
    --model_name "gpt-4-turbo" \
    --config "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/generate_config.yaml"

# Ensure output directory exists for spider on postgresql
mkdir -p ./results/postgresql/spider

echo "--- Running API [gpt-4o] on DATASET [spider] for BACKEND [postgresql] ---"
python generate.py \
    --mode "api" \
    --dataset "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/postgresql/results/spider/input_llm.json" \
    --output "./results/postgresql/spider/out_llm_api_gpt-4o.json" \
    --model_name "gpt-4o" \
    --config "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/generate_config.yaml"

# Ensure output directory exists for spider on postgresql
mkdir -p ./results/postgresql/spider

echo "--- Running API [gpt-4o-mini] on DATASET [spider] for BACKEND [postgresql] ---"
python generate.py \
    --mode "api" \
    --dataset "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/postgresql/results/spider/input_llm.json" \
    --output "./results/postgresql/spider/out_llm_api_gpt-4o-mini.json" \
    --model_name "gpt-4o-mini" \
    --config "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/generate_config.yaml"

# Ensure output directory exists for spider on postgresql
mkdir -p ./results/postgresql/spider

echo "--- Running API [gpt-4-turbo] on DATASET [spider] for BACKEND [postgresql] ---"
python generate.py \
    --mode "api" \
    --dataset "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/postgresql/results/spider/input_llm.json" \
    --output "./results/postgresql/spider/out_llm_api_gpt-4-turbo.json" \
    --model_name "gpt-4-turbo" \
    --config "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/generate_config.yaml"

# Ensure output directory exists for wikipedia_multimodal on postgresql
mkdir -p ./results/postgresql/wikipedia_multimodal

echo "--- Running API [gpt-4o] on DATASET [wikipedia_multimodal] for BACKEND [postgresql] ---"
python generate.py \
    --mode "api" \
    --dataset "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/postgresql/results/wikipedia_multimodal/input_llm.json" \
    --output "./results/postgresql/wikipedia_multimodal/out_llm_api_gpt-4o.json" \
    --model_name "gpt-4o" \
    --config "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/generate_config.yaml"

# Ensure output directory exists for wikipedia_multimodal on postgresql
mkdir -p ./results/postgresql/wikipedia_multimodal

echo "--- Running API [gpt-4o-mini] on DATASET [wikipedia_multimodal] for BACKEND [postgresql] ---"
python generate.py \
    --mode "api" \
    --dataset "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/postgresql/results/wikipedia_multimodal/input_llm.json" \
    --output "./results/postgresql/wikipedia_multimodal/out_llm_api_gpt-4o-mini.json" \
    --model_name "gpt-4o-mini" \
    --config "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/generate_config.yaml"

# Ensure output directory exists for wikipedia_multimodal on postgresql
mkdir -p ./results/postgresql/wikipedia_multimodal

echo "--- Running API [gpt-4-turbo] on DATASET [wikipedia_multimodal] for BACKEND [postgresql] ---"
python generate.py \
    --mode "api" \
    --dataset "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/postgresql/results/wikipedia_multimodal/input_llm.json" \
    --output "./results/postgresql/wikipedia_multimodal/out_llm_api_gpt-4-turbo.json" \
    --model_name "gpt-4-turbo" \
    --config "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/generate_config.yaml"
