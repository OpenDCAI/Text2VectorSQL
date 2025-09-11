#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
python server.py --config config.yaml

# curl http://localhost:8000/health
# curl -X POST http://localhost:8000/embed \
# -H "Content-Type: application/json" \
# -d '{
#       "model": "all-MiniLM-L6-v2",
#       "texts": [
#         "Hello, world!",
#         "This is a test of the embedding service."
#       ]
#     }'