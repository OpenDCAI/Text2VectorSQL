#!/bin/bash
# This script is auto-generated to run generation tasks across different backends, modes, and models.
set -e  # Exit immediately if a command exits with a non-zero status.

#################################################
#    Commands for Backend: SQLITE
#################################################

### API Mode Commands ###

# Ensure output directory exists for wikipedia_multimodal on sqlite
mkdir -p ./results/sqlite/wikipedia_multimodal

echo "--- Running API [gpt-4o] on DATASET [wikipedia_multimodal] for BACKEND [sqlite] ---"
python generate.py \
    --mode "api" \
    --dataset "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/sqlite/results/wikipedia_multimodal/input_llm.json" \
    --output "./results/sqlite/wikipedia_multimodal/out_llm_api_gpt-4o.json" \
    --model_name "gpt-4o" \
    --config "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/generate_config.yaml"

# Ensure output directory exists for wikipedia_multimodal on sqlite
mkdir -p ./results/sqlite/wikipedia_multimodal

echo "--- Running API [gpt-4o-mini] on DATASET [wikipedia_multimodal] for BACKEND [sqlite] ---"
python generate.py \
    --mode "api" \
    --dataset "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/sqlite/results/wikipedia_multimodal/input_llm.json" \
    --output "./results/sqlite/wikipedia_multimodal/out_llm_api_gpt-4o-mini.json" \
    --model_name "gpt-4o-mini" \
    --config "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/generate_config.yaml"

# Ensure output directory exists for wikipedia_multimodal on sqlite
mkdir -p ./results/sqlite/wikipedia_multimodal

echo "--- Running API [gpt-4-turbo] on DATASET [wikipedia_multimodal] for BACKEND [sqlite] ---"
python generate.py \
    --mode "api" \
    --dataset "/mnt/b_public/data/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/sqlite/results/wikipedia_multimodal/input_llm.json" \
    --output "./results/sqlite/wikipedia_multimodal/out_llm_api_gpt-4-turbo.json" \
    --model_name "gpt-4-turbo" \
    --config "/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/generate_config.yaml"
