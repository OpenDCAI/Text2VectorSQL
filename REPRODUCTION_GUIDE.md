# Text2VectorSQL 完整复现指南 (Complete Reproduction Guide)

[English Version](#english-version) | [中文版本](#chinese-version)

---

## English Version

### 📋 Table of Contents

1. [Overview](#overview)
2. [System Requirements](#system-requirements)
3. [Installation Guide](#installation-guide)
4. [Module 1: Embedding Service](#module-1-embedding-service)
5. [Module 2: Execution Engine](#module-2-execution-engine)
6. [Module 3: Evaluation Framework](#module-3-evaluation-framework)
7. [Module 4: Data Synthesizer](#module-4-data-synthesizer)
8. [Complete Workflow Example](#complete-workflow-example)
9. [Troubleshooting](#troubleshooting)
10. [FAQ](#faq)

---

### Overview

**Text2VectorSQL** is a unified natural language interface for querying both structured and unstructured data. This guide provides step-by-step instructions to reproduce all functionalities of the project.

**Project Architecture:**
- **Embedding Service**: High-performance API for text/image vectorization
- **Execution Engine**: Parses and executes VectorSQL queries
- **Evaluation Framework**: Evaluates model performance on Text2VectorSQL tasks
- **Data Synthesizer**: Generates training data for Text2VectorSQL models

**Key Features:**
- Supports SQLite, PostgreSQL, and ClickHouse databases
- Multi-model embedding support (text and image)
- Execution-based evaluation (compares query results, not SQL strings)
- Automated data synthesis pipeline

---

### System Requirements

**Minimum Requirements:**
- **OS**: Linux, macOS, or Windows (WSL recommended)
- **Python**: 3.8 or higher
- **RAM**: 8GB minimum (16GB+ recommended for model inference)
- **Disk Space**: 10GB+ for models and databases

**Optional (for GPU acceleration):**
- **GPU**: NVIDIA GPU with CUDA support
- **CUDA**: 11.0 or higher
- **GPU Memory**: 4GB+ (8GB+ recommended for larger models)

**Software Dependencies:**
- Git
- pip or conda
- (Optional) Docker for database setup

---

### Installation Guide

#### Step 1: Clone the Repository

```bash
git clone https://github.com/OpenDCAI/Text2VectorSQL.git --depth 1
cd Text2VectorSQL
```

#### Step 2: Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n text2vectorsql python=3.9
conda activate text2vectorsql
```

#### Step 3: Install Dependencies by Module

Each module has its own `requirements.txt`. Install based on your needs:

```bash
# For Embedding Service
cd Embedding_Service
pip install -r requirements.txt
cd ..

# For Execution Engine
cd Execution_Engine
pip install -r requirements.txt
cd ..

# For Evaluation Framework
cd Evaluation_Framework
pip install -r requirements.txt
cd ..

# For Data Synthesizer (optional, for data generation)
cd Data_Synthesizer
pip install -r requirements.txt
cd ..
```

#### Step 3.1: Verify sqlite-vec and sqlite-lembed Installation

The Execution Engine and Evaluation Framework depend on `sqlite-vec` (provides `vec0` virtual tables for vector search) and `sqlite-lembed` (provides the `lembed()` function). These are installed via `requirements.txt`, but you should verify they work correctly:

```bash
python -c "
import sqlite3, sqlite_vec, sqlite_lembed
conn = sqlite3.connect(':memory:')
conn.enable_load_extension(True)
sqlite_vec.load(conn)
sqlite_lembed.load(conn)
print('vec_version:', conn.execute('select vec_version()').fetchone()[0])
print('sqlite-vec and sqlite-lembed loaded successfully!')
"
```

**Expected Output:**
```
vec_version: v0.1.7
sqlite-vec and sqlite-lembed loaded successfully!
```

If this fails, try:
```bash
pip install --upgrade sqlite-vec sqlite-lembed
```

**Note**: `sqlite-vec` provides the `vec0` virtual table type used in all vector databases. Without it, you will see errors like `no such module: vec0` when querying databases.

---

### Module 1: Embedding Service

The Embedding Service provides vectorization capabilities for text and images. It's a **prerequisite** for running the Execution Engine and Evaluation Framework.

#### 1.1 Configuration

Create a configuration file `Embedding_Service/config.yaml`:

```yaml
# config.yaml
server:
  host: "0.0.0.0"
  port: 8000

models:
  - name: "all-MiniLM-L6-v2"
    hf_model_path: "sentence-transformers/all-MiniLM-L6-v2"
    local_model_path: "./models/all-MiniLM-L6-v2"
    trust_remote_code: true
    max_model_len: 512
```

**Configuration Options:**
- `server.host`: Server host address (use "0.0.0.0" for all interfaces)
- `server.port`: Server port (default: 8000)
- `models`: List of embedding models to load
  - `name`: Model identifier used in queries
  - `hf_model_path`: Hugging Face model path
  - `local_model_path`: Local cache directory
  - `trust_remote_code`: Allow custom model code
  - `max_model_len`: Maximum sequence length

#### 1.2 Start the Service

```bash
cd Embedding_Service
python server.py --config config.yaml
```

**Expected Output:**
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Loading model: all-MiniLM-L6-v2
INFO:     Model loaded successfully
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**First Run**: Models will be downloaded from Hugging Face (may take several minutes depending on your internet speed).

#### 1.3 Test the Service

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "models": ["all-MiniLM-L6-v2"]
}
```

**Get Text Embeddings:**
```bash
curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '{
    "model": "all-MiniLM-L6-v2",
    "texts": ["Hello, world!", "Machine learning is amazing"]
  }'
```

**Expected Response:**
```json
{
  "embeddings": [
    [0.123, -0.456, 0.789, ...],
    [0.234, -0.567, 0.890, ...]
  ]
}
```

#### 1.4 Advanced Configuration

**Multi-GPU Support:**
```yaml
models:
  - name: "large-model"
    hf_model_path: "sentence-transformers/all-mpnet-base-v2"
    local_model_path: "./models/all-mpnet-base-v2"
    trust_remote_code: true
    tensor_parallel_size: 2  # Use 2 GPUs
```

**Multiple Models:**
```yaml
models:
  - name: "all-MiniLM-L6-v2"
    hf_model_path: "sentence-transformers/all-MiniLM-L6-v2"
    local_model_path: "./models/all-MiniLM-L6-v2"
    trust_remote_code: true
  
  - name: "gte-large"
    hf_model_path: "thenlper/gte-large"
    local_model_path: "./models/gte-large"
    trust_remote_code: true
```

**Using Hugging Face Mirror (for users in China):**
```bash
export HF_ENDPOINT=https://hf-mirror.com
python server.py --config config.yaml
```

---

### Module 2: Execution Engine

The Execution Engine parses VectorSQL queries containing `lembed()` functions and executes them on target databases.

#### 2.1 Prerequisites

- Embedding Service must be running (see Module 1)
- Target database must be set up (SQLite, PostgreSQL, or ClickHouse)

#### 2.2 Configuration

Create `Execution_Engine/engine_config.yaml`:

```yaml
# Embedding Service Configuration
embedding_service:
  url: "http://127.0.0.1:8000/embed"

# Database Connection Settings
database_connections:
  postgresql:
    user: "postgres"
    password: "postgres"
    host: "localhost"
    port: 5432
  
  clickhouse:
    host: "localhost"
    port: 8123
    user: "default"
    password: ""

# Timeout Settings (in seconds)
timeouts:
  embedding_service: 30
  database_connection: 10
  sql_execution: 60
  total_execution: 120
```

#### 2.3 Prepare a Test Database

**For SQLite (Easiest for Testing):**

The project includes test databases. You can use them directly or create your own:

```bash
# Test databases are typically located in:
# Data_Synthesizer/pipeline/sqlite/results/test/vector_databases/
```

#### 2.4 Execute a VectorSQL Query

**Basic Example:**

```bash
cd Execution_Engine

python execution_engine.py \
  --sql "SELECT Name FROM musical ORDER BY L2Distance(Category_embedding, lembed('all-MiniLM-L6-v2', 'opera')) LIMIT 5;" \
  --db-type "sqlite" \
  --db-identifier "../Data_Synthesizer/pipeline/sqlite/results/test/vector_databases/musical.db" \
  --config "engine_config.yaml"
```

**Command-line Arguments:**
- `--sql`: VectorSQL query to execute
- `--db-type`: Database type (`sqlite`, `postgresql`, or `clickhouse`)
- `--db-identifier`: Database name (for PostgreSQL/ClickHouse) or file path (for SQLite)
- `--config`: Path to engine configuration file

**Expected Output:**
```json
{
  "status": "success",
  "columns": ["Name"],
  "data": [
    ["La Traviata"],
    ["Carmen"],
    ["The Magic Flute"],
    ["Tosca"],
    ["Madama Butterfly"]
  ],
  "execution_time": 0.234
}
```

#### 2.5 Using the Engine in Python

```python
from execution_engine import ExecutionEngine

# Initialize engine
engine = ExecutionEngine(config_path="engine_config.yaml")

# Execute query
result = engine.execute(
    sql="SELECT name FROM products ORDER BY embedding <-> lembed('all-MiniLM-L6-v2', 'laptop') LIMIT 3",
    db_type="sqlite",
    db_identifier="products.db"
)

# Process results
if result['status'] == 'success':
    print("Query successful!")
    for row in result['data']:
        print(row)
else:
    print(f"Query failed: {result['message']}")
```

---

### Module 3: Evaluation Framework

The Evaluation Framework evaluates Text2VectorSQL models by comparing execution results rather than SQL strings.

#### 3.1 Overview

**Evaluation Pipeline:**
1. **SQL Generation** (`generate.py`): Generate predicted SQL from model
2. **SQL Execution** (`sql_executor.py`): Execute predicted and gold SQL queries
3. **Result Evaluation** (`evaluate_results.py`): Compare results and compute metrics
4. **Result Aggregation** (`aggregate_results.py`): Aggregate multiple evaluation reports

#### 3.2 Prepare Test Data

The project includes test data in `Evaluation_Framework/test_data/input_output.json`.

> **Important: Choosing the Right Dataset**
>
> The project includes several datasets for different purposes:
>
> | Dataset | DBs | Purpose |
> |---------|-----|---------|
> | `spider` | 167 | Evaluation benchmark (VectorSQLBench) |
> | `bird` | 70 | Evaluation benchmark (VectorSQLBench) |
> | `arxiv` | - | Domain-specific evaluation |
> | `wikipedia_multimodal` | - | Multi-modal evaluation |
> | `synthesis_data` | 1865 | Training data (NOT for evaluation) |
> | `test` | 2 | **Debug only** - minimal sanity check |
> | `toy_spider` | - | **Debug only** - reduced Spider |
>
> The default `evaluation_config.yaml` uses the `test/` dataset (only 2 databases, 10 cases) for quick sanity checks. **For real evaluation, change `base_dir` to point to `spider/`, `bird/`, or other benchmark datasets.**
>
> To download all datasets:
> ```bash
> cd Data_Synthesizer/tools
> python download_data.py
> ```

**Test Data Format:**
```json
[
  {
    "query_id": "unique_id",
    "db_id": "database_name",
    "db_identifier": "database_name",
    "db_type": "sqlite",
    "question": "Natural language question",
    "predicted_sql": "Model's predicted SQL",
    "sql_candidate": ["Gold SQL 1", "Gold SQL 2", ...]
  }
]
```

#### 3.3 Configuration

Create `Evaluation_Framework/evaluation_config.yaml`:

```yaml
# Database Configuration
base_dir: ../Data_Synthesizer/pipeline/sqlite/results/test/vector_databases
db_type: 'sqlite'

# Project root directory
project_dir: ../

# Execution Engine Configuration
engine_config_path: Execution_Engine/engine_config.yaml

# Input/Output Files
eval_data_file: ./test_data/input_output.json
execution_results_file: sql_execution_results.json
evaluation_report_file: evaluation_report.json

# Embedding Service Configuration
embedding_service:
  auto_manage: false  # Set to true to auto-start service
  host: "127.0.0.1"
  port: 8000
  models:
    - name: "all-MiniLM-L6-v2"
      hf_model_path: "sentence-transformers/all-MiniLM-L6-v2"
      trust_remote_code: true

# Evaluation Metrics
metrics:
  # Set-based metrics
  - name: 'exact_match'
  - name: 'f1_score'
  - name: 'precision'
  - name: 'recall'
  
  # Rank-based metrics
  - name: 'map'
  - name: 'mrr'
  - name: 'ndcg'
    k: 10

# LLM Evaluation (optional)
llm_evaluation:
  enabled: false
  api_url: "https://api.openai.com/v1/chat/completions"
  api_key: "your-api-key"
  model_name: "gpt-4"
  timeout: 60
```

#### 3.4 Run Evaluation

**Prerequisites:**
- Embedding Service is running
- Test databases are available
- Evaluation data file exists

**Option 1: Run Complete Pipeline**

```bash
cd Evaluation_Framework

python run_eval_pipeline.py --all --config evaluation_config.yaml
```

**Option 2: Run Step-by-Step**

```bash
# Step 1: Execute SQL queries
python run_eval_pipeline.py --execute --config evaluation_config.yaml

# Step 2: Evaluate results
python run_eval_pipeline.py --evaluate --config evaluation_config.yaml
```

#### 3.5 Understanding Results

**Execution Results** (`sql_execution_results.json`):
```json
{
  "query_id": {
    "predicted": {
      "status": "success",
      "columns": ["col1", "col2"],
      "data": [[val1, val2], ...]
    },
    "gold": [
      {
        "status": "success",
        "columns": ["col1", "col2"],
        "data": [[val1, val2], ...]
      }
    ]
  }
}
```

**Evaluation Report** (`evaluation_report.json`):
```json
{
  "overall_metrics": {
    "exact_match": 0.75,
    "f1_score": 0.82,
    "precision": 0.85,
    "recall": 0.80,
    "map": 0.78,
    "mrr": 0.81,
    "ndcg@10": 0.79
  },
  "per_query_results": {
    "query_id": {
      "exact_match": 1,
      "f1_score": 1.0,
      ...
    }
  }
}
```

#### 3.6 Aggregate Multiple Results

If you have multiple evaluation reports from different models or datasets:

```bash
python aggregate_results.py \
  --results-dir ./results \
  --output summary.csv
```

**Output CSV Format:**
```
model,dataset,exact_match,f1_score,precision,recall,map,mrr,ndcg@10
model_a,test_set,0.75,0.82,0.85,0.80,0.78,0.81,0.79
model_b,test_set,0.72,0.80,0.83,0.78,0.76,0.79,0.77
```

#### 3.7 Generate Predictions (Optional)

If you need to generate predictions from a model:

```bash
cd Evaluation_Framework

# Using vLLM (local inference)
python generate.py --config generate_config.yaml --mode vllm

# Using API (OpenAI, Claude, etc.)
python generate.py --config generate_config.yaml --mode api
```

**Generate Config Example** (`generate_config.yaml`):
```yaml
# Input data
input_file: ./test_data/questions.json
output_file: ./predictions.json

# Model configuration
mode: api  # or vllm
model_name: gpt-4
api_url: https://api.openai.com/v1/chat/completions
api_key: your-api-key

# For vLLM mode
# model_path: /path/to/local/model
# tensor_parallel_size: 1
```

---

### Module 4: Data Synthesizer

The Data Synthesizer generates training data for Text2VectorSQL models through an automated pipeline.

#### 4.1 Overview

**Synthesis Pipeline:**
1. **Database Synthesis**: Generate structured databases from web tables
2. **Database Vectorization**: Add vector embeddings to semantic columns
3. **SQL Synthesis**: Generate VectorSQL queries
4. **Question Synthesis**: Generate natural language questions
5. **Chain-of-Thought Synthesis**: Generate reasoning steps

#### 4.2 Configuration

Create `Data_Synthesizer/pipeline/config.yaml`:

```yaml
# LLM API Configuration
llm:
  api_key: "your-openai-api-key"
  base_url: "https://api.openai.com/v1"
  model: "gpt-4"
  temperature: 0.7
  max_tokens: 2048

# Embedding Model Configuration
embedding:
  model_name: "all-MiniLM-L6-v2"
  batch_size: 32

# Pipeline Configuration
pipeline:
  # Source dataset
  source_dataset: "wikitables"  # or "spider", "bird"
  
  # Output paths
  result_path: "./results"
  
  # Synthesis settings
  num_samples: 100
  complexity_levels: ["simple", "moderate", "challenging"]
  
# Database Backend
database:
  type: "sqlite"  # or "postgresql", "clickhouse"
```

#### 4.3 Download Pre-synthesized Data (Recommended)

Instead of synthesizing from scratch, you can download pre-generated data:

```bash
cd Data_Synthesizer/tools

# Download databases
python download_data.py

# Download trained models (if available)
python download_model.py
```

#### 4.4 Run Synthesis Pipeline (Advanced)

**Prerequisites:**
- LLM API access (OpenAI, Claude, etc.)
- Sufficient API credits
- Source datasets downloaded

```bash
cd Data_Synthesizer

# Run complete pipeline
python pipeline/general_pipeline.py
```

**Note**: Full synthesis can be time-consuming and expensive (requires many LLM API calls).

#### 4.5 Synthesis Stages

**Stage 1: Database Synthesis**
```bash
cd Data_Synthesizer/database_synthesis
python synthesize_schema.py --config ../pipeline/config.yaml
```

**Stage 2: Vectorization**
```bash
cd Data_Synthesizer/vectorization
python batch_vectorize_databases.py --config ../pipeline/config.yaml
```

**Stage 3: SQL Synthesis**
```bash
cd Data_Synthesizer/synthesis_sql
python synthesize_sql.py --config ../pipeline/config.yaml
```

---

### Complete Workflow Example

Here's a complete end-to-end workflow for evaluating a Text2VectorSQL system:

#### Step 1: Start Embedding Service

```bash
# Terminal 1
cd Embedding_Service
python server.py --config config.yaml
```

#### Step 2: Prepare Test Database

```bash
# Use provided test database or download
cd Data_Synthesizer/tools
python download_data.py
```

#### Step 3: Test Execution Engine

```bash
# Terminal 2
cd Execution_Engine

python execution_engine.py \
  --sql "SELECT * FROM employee WHERE employee_description_embedding MATCH lembed('all-MiniLM-L6-v2', 'software engineer') LIMIT 5;" \
  --db-type "sqlite" \
  --db-identifier "../Data_Synthesizer/pipeline/sqlite/results/test/vector_databases/company_1.db" \
  --config "engine_config.yaml"
```

#### Step 4: Run Evaluation

```bash
cd Evaluation_Framework

# Run complete evaluation
python run_eval_pipeline.py --all --config evaluation_config.yaml

# View results
cat evaluation_report.json
```

#### Step 5: Analyze Results

```bash
# If you have multiple experiments
python aggregate_results.py --results-dir ./results --output summary.csv
cat summary.csv
```

---

### Troubleshooting

#### Issue 1: Embedding Service Connection Error

**Error**: `Connection refused to http://localhost:8000`

**Solution**:
- Ensure Embedding Service is running
- Check if port 8000 is available
- Verify firewall settings
- Try using `127.0.0.1` instead of `localhost`

#### Issue 2: Model Download Fails

**Error**: `Failed to download model from Hugging Face`

**Solution**:
```bash
# Use Hugging Face mirror (for users in China)
export HF_ENDPOINT=https://hf-mirror.com

# Or manually download and specify local path
# Download from: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
```

#### Issue 3: SQLite Vector Extension Not Found

**Error**: `sqlite3.OperationalError: no such function: lembed` or `no such module: vec0`

**Solution**:
```bash
# Install sqlite-vec and sqlite-lembed
pip install sqlite-vec sqlite-lembed
```

**Verify installation**:
```bash
python -c "
import sqlite3, sqlite_vec, sqlite_lembed
conn = sqlite3.connect(':memory:')
conn.enable_load_extension(True)
sqlite_vec.load(conn)
sqlite_lembed.load(conn)
print('vec_version:', conn.execute('select vec_version()').fetchone()[0])
"
```

**Common causes**:
- `sqlite-vec` and `sqlite-lembed` not in `Evaluation_Framework/requirements.txt` (fixed in latest version)
- Python's SQLite was compiled without extension loading support (recompile Python or use conda)
- System SQLite version too old (need >= 3.31.0 for `enable_load_extension`)

#### Issue 3.5: Disk I/O Error on virtiofs (Linux VM/Container)

**Error**: `disk I/O error` when accessing SQLite databases on shared/mounted filesystems

**Cause**: virtiofs (or similar shared filesystems) has known compatibility issues with SQLite's locking mechanism

**Solution**:
```bash
# Copy database to local /tmp directory
cp /path/to/database.db /tmp/database.db

# Use the /tmp copy in your commands
python execution_engine.py \
  --sql "YOUR_SQL" \
  --db-type "sqlite" \
  --db-identifier "/tmp/database.db" \
  --config "engine_config.yaml"
```

**For evaluation framework**, update database paths:
```bash
# Create script to copy databases to /tmp
mkdir -p /tmp/vector_databases
cp -r /path/to/vector_databases/* /tmp/vector_databases/

# Update evaluation_config.yaml
# base_dir: /tmp/vector_databases
```

#### Issue 4: Database Connection Timeout

**Error**: `Database connection timeout`

**Solution**:
- Increase timeout in `engine_config.yaml`
- Check database server is running
- Verify connection credentials
- Test connection manually

#### Issue 5: Out of Memory During Evaluation

**Error**: `CUDA out of memory` or `MemoryError`

**Solution**:
- Reduce batch size in config
- Use smaller embedding models
- Enable CPU-only mode
- Process data in smaller chunks

#### Issue 6: Permission Denied on Database Files

**Error**: `Permission denied: database.db`

**Solution**:
```bash
# Fix file permissions
chmod 644 database.db
chmod 755 database_directory/
```

---

### FAQ

#### Q1: Can I run this without GPU?

**A**: Yes! The embedding models can run on CPU, though it will be slower. The system automatically detects available hardware.

#### Q2: Which embedding models are supported?

**A**: Any model from Hugging Face's `sentence-transformers` library. Popular choices:
- `all-MiniLM-L6-v2` (fast, lightweight)
- `all-mpnet-base-v2` (balanced)
- `gte-large` (high quality)
- `CLIP` models (for multi-modal)

#### Q3: How do I evaluate UniVectorSQL models without GPU?

**A**: You can:
1. Use the API mode with cloud providers (OpenAI, Anthropic, etc.)
2. Use smaller models that fit in CPU memory
3. Use the pre-computed results from the paper

#### Q4: Can I use my own database?

**A**: Yes! Follow these steps:
1. Create your database with vector columns
2. Populate embeddings using the Embedding Service
3. Update `engine_config.yaml` with connection details
4. Execute queries using the Execution Engine

#### Q5: What's the difference between `lembed()` and pre-computed embeddings?

**A**: 
- `lembed()`: Computes embeddings at query time (flexible, slower)
- Pre-computed: Embeddings stored in database (faster, requires storage)

Both approaches are supported.

#### Q6: How long does data synthesis take?

**A**: Depends on dataset size and LLM API speed:
- Small dataset (100 samples): 1-2 hours
- Medium dataset (1000 samples): 10-20 hours
- Large dataset (10000+ samples): Several days

Recommendation: Download pre-synthesized data instead.

#### Q7: Can I use databases other than SQLite/PostgreSQL/ClickHouse?

**A**: The current implementation supports these three. To add others, you'll need to:
1. Implement a database connector in `execution_engine.py`
2. Handle vector syntax for that database
3. Update configuration schema

---

### Performance Tips

1. **Use GPU acceleration** for embedding models when available
2. **Cache embeddings** in the database to avoid recomputation
3. **Batch queries** when evaluating multiple samples
4. **Use smaller models** for faster inference (e.g., MiniLM vs MPNet)
5. **Enable connection pooling** for database connections
6. **Adjust timeout values** based on your hardware

---

### Citation

If you use this project in your research, please cite:

```bibtex
@article{wang2025text2vectorsql,
  title={Text2VectorSQL: Towards a Unified Interface for Vector Search and SQL Queries},
  author={Wang, Zhengren and Yao, Dongwen and Li, Bozhou and Ma, Dongsheng and Li, Bo and Li, Zhiyu and Xiong, Feiyu and Cui, Bin and Tang, Linpeng and Zhang, Wentao},
  journal={arXiv preprint arXiv:2506.23071},
  year={2025}
}
```

---

### Resources

- **Paper**: https://arxiv.org/abs/2506.23071
- **Hugging Face**: https://huggingface.co/VectorSQL
- **GitHub**: https://github.com/OpenDCAI/Text2VectorSQL
- **Issues**: https://github.com/OpenDCAI/Text2VectorSQL/issues

---

### Contact

For questions or feedback, contact: wzr@stu.pku.edu.cn

---

## Chinese Version

### 📋 目录

1. [概述](#概述)
2. [系统要求](#系统要求)
3. [安装指南](#安装指南)
4. [模块1：嵌入服务](#模块1嵌入服务)
5. [模块2：执行引擎](#模块2执行引擎)
6. [模块3：评估框架](#模块3评估框架)
7. [模块4：数据合成器](#模块4数据合成器)
8. [完整工作流示例](#完整工作流示例)
9. [故障排除](#故障排除)
10. [常见问题](#常见问题)

---

### 概述

**Text2VectorSQL** 是一个用于查询结构化和非结构化数据的统一自然语言接口。本指南提供了逐步说明，以复现项目的所有功能。

**项目架构：**
- **嵌入服务**：用于文本/图像向量化的高性能API
- **执行引擎**：解析并执行VectorSQL查询
- **评估框架**：评估Text2VectorSQL任务的模型性能
- **数据合成器**：为Text2VectorSQL模型生成训练数据

**主要特性：**
- 支持SQLite、PostgreSQL和ClickHouse数据库
- 多模型嵌入支持（文本和图像）
- 基于执行的评估（比较查询结果，而非SQL字符串）
- 自动化数据合成流水线

---

### 系统要求

**最低要求：**
- **操作系统**：Linux、macOS或Windows（推荐WSL）
- **Python**：3.8或更高版本
- **内存**：最低8GB（推荐16GB+用于模型推理）
- **磁盘空间**：10GB+用于模型和数据库

**可选（用于GPU加速）：**
- **GPU**：支持CUDA的NVIDIA GPU
- **CUDA**：11.0或更高版本
- **GPU内存**：4GB+（推荐8GB+用于较大模型）

**软件依赖：**
- Git
- pip或conda
- （可选）Docker用于数据库设置

---

### 安装指南

#### 步骤1：克隆仓库

```bash
git clone https://github.com/OpenDCAI/Text2VectorSQL.git --depth 1
cd Text2VectorSQL
```

#### 步骤2：创建虚拟环境（推荐）

```bash
# 使用venv
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 或使用conda
conda create -n text2vectorsql python=3.9
conda activate text2vectorsql
```

#### 步骤3：按模块安装依赖

每个模块都有自己的`requirements.txt`。根据需要安装：

```bash
# 嵌入服务
cd Embedding_Service
pip install -r requirements.txt
cd ..

# 执行引擎
cd Execution_Engine
pip install -r requirements.txt
cd ..

# 评估框架
cd Evaluation_Framework
pip install -r requirements.txt
cd ..

# 数据合成器（可选，用于数据生成）
cd Data_Synthesizer
pip install -r requirements.txt
cd ..
```

#### 步骤3.1：验证 sqlite-vec 和 sqlite-lembed 安装

执行引擎和评估框架依赖 `sqlite-vec`（提供 `vec0` 虚拟表用于向量搜索）和 `sqlite-lembed`（提供 `lembed()` 函数）。它们已包含在 `requirements.txt` 中，但建议验证安装是否正确：

```bash
python -c "
import sqlite3, sqlite_vec, sqlite_lembed
conn = sqlite3.connect(':memory:')
conn.enable_load_extension(True)
sqlite_vec.load(conn)
sqlite_lembed.load(conn)
print('vec_version:', conn.execute('select vec_version()').fetchone()[0])
print('sqlite-vec 和 sqlite-lembed 加载成功！')
"
```

**预期输出：**
```
vec_version: v0.1.7
sqlite-vec 和 sqlite-lembed 加载成功！
```

如果失败，请尝试：
```bash
pip install --upgrade sqlite-vec sqlite-lembed
```

**注意**：`sqlite-vec` 提供了所有向量数据库中使用的 `vec0` 虚拟表类型。缺少它将导致查询时出现 `no such module: vec0` 错误。

---

### 模块1：嵌入服务

嵌入服务为文本和图像提供向量化能力。这是运行执行引擎和评估框架的**先决条件**。

#### 1.1 配置

创建配置文件 `Embedding_Service/config.yaml`：

```yaml
# config.yaml
server:
  host: "0.0.0.0"
  port: 8000

models:
  - name: "all-MiniLM-L6-v2"
    hf_model_path: "sentence-transformers/all-MiniLM-L6-v2"
    local_model_path: "./models/all-MiniLM-L6-v2"
    trust_remote_code: true
    max_model_len: 512
```

**配置选项：**
- `server.host`：服务器主机地址（使用"0.0.0.0"监听所有接口）
- `server.port`：服务器端口（默认：8000）
- `models`：要加载的嵌入模型列表
  - `name`：查询中使用的模型标识符
  - `hf_model_path`：Hugging Face模型路径
  - `local_model_path`：本地缓存目录
  - `trust_remote_code`：允许自定义模型代码
  - `max_model_len`：最大序列长度

#### 1.2 启动服务

```bash
cd Embedding_Service
python server.py --config config.yaml
```

**预期输出：**
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Loading model: all-MiniLM-L6-v2
INFO:     Model loaded successfully
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**首次运行**：模型将从Hugging Face下载（根据网速可能需要几分钟）。

#### 1.3 测试服务

**健康检查：**
```bash
curl http://localhost:8000/health
```

**预期响应：**
```json
{
  "status": "healthy",
  "models": ["all-MiniLM-L6-v2"]
}
```

**获取文本嵌入：**
```bash
curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '{
    "model": "all-MiniLM-L6-v2",
    "texts": ["你好，世界！", "机器学习很棒"]
  }'
```

#### 1.4 高级配置

**多GPU支持：**
```yaml
models:
  - name: "large-model"
    hf_model_path: "sentence-transformers/all-mpnet-base-v2"
    local_model_path: "./models/all-mpnet-base-v2"
    trust_remote_code: true
    tensor_parallel_size: 2  # 使用2个GPU
```

**使用Hugging Face镜像（中国用户）：**
```bash
export HF_ENDPOINT=https://hf-mirror.com
python server.py --config config.yaml
```

---

### 模块2：执行引擎

执行引擎解析包含`lembed()`函数的VectorSQL查询并在目标数据库上执行。

#### 2.1 前提条件

- 嵌入服务必须运行（见模块1）
- 目标数据库必须设置（SQLite、PostgreSQL或ClickHouse）

#### 2.2 配置

创建 `Execution_Engine/engine_config.yaml`：

```yaml
# 嵌入服务配置
embedding_service:
  url: "http://127.0.0.1:8000/embed"

# 数据库连接设置
database_connections:
  postgresql:
    user: "postgres"
    password: "postgres"
    host: "localhost"
    port: 5432
  
  clickhouse:
    host: "localhost"
    port: 8123
    user: "default"
    password: ""

# 超时设置（秒）
timeouts:
  embedding_service: 30
  database_connection: 10
  sql_execution: 60
  total_execution: 120
```

#### 2.3 准备测试数据库

**对于SQLite（最简单的测试方式）：**

项目包含测试数据库，可以直接使用：

```bash
# 测试数据库通常位于：
# Data_Synthesizer/pipeline/sqlite/results/test/vector_databases/
```

#### 2.4 执行VectorSQL查询

**基本示例：**

```bash
cd Execution_Engine

python execution_engine.py \
  --sql "SELECT Name FROM musical ORDER BY L2Distance(Category_embedding, lembed('all-MiniLM-L6-v2', '歌剧')) LIMIT 5;" \
  --db-type "sqlite" \
  --db-identifier "../Data_Synthesizer/pipeline/sqlite/results/test/vector_databases/musical.db" \
  --config "engine_config.yaml"
```

**命令行参数：**
- `--sql`：要执行的VectorSQL查询
- `--db-type`：数据库类型（`sqlite`、`postgresql`或`clickhouse`）
- `--db-identifier`：数据库名称（PostgreSQL/ClickHouse）或文件路径（SQLite）
- `--config`：引擎配置文件路径

#### 2.5 在Python中使用引擎

```python
from execution_engine import ExecutionEngine

# 初始化引擎
engine = ExecutionEngine(config_path="engine_config.yaml")

# 执行查询
result = engine.execute(
    sql="SELECT name FROM products ORDER BY embedding <-> lembed('all-MiniLM-L6-v2', '笔记本电脑') LIMIT 3",
    db_type="sqlite",
    db_identifier="products.db"
)

# 处理结果
if result['status'] == 'success':
    print("查询成功！")
    for row in result['data']:
        print(row)
else:
    print(f"查询失败：{result['message']}")
```

---

### 模块3：评估框架

评估框架通过比较执行结果而非SQL字符串来评估Text2VectorSQL模型。

#### 3.1 概述

**评估流水线：**
1. **SQL生成** (`generate.py`)：从模型生成预测SQL
2. **SQL执行** (`sql_executor.py`)：执行预测和黄金SQL查询
3. **结果评估** (`evaluate_results.py`)：比较结果并计算指标
4. **结果聚合** (`aggregate_results.py`)：聚合多个评估报告

#### 3.2 准备测试数据

项目在`Evaluation_Framework/test_data/input_output.json`中包含测试数据。

> **重要：选择正确的数据集**
>
> 项目包含多个用于不同目的的数据集：
>
> | 数据集 | 数据库数量 | 用途 |
> |--------|-----------|------|
> | `spider` | 167 | 评测基准 (VectorSQLBench) |
> | `bird` | 70 | 评测基准 (VectorSQLBench) |
> | `arxiv` | - | 领域特定评测 |
> | `wikipedia_multimodal` | - | 多模态评测 |
> | `synthesis_data` | 1865 | 训练数据（不用于评测） |
> | `test` | 2 | **仅调试用** - 最小化检查 |
> | `toy_spider` | - | **仅调试用** - 缩减版Spider |
>
> 默认 `evaluation_config.yaml` 使用 `test/` 数据集（仅2个数据库、10个样例），仅供快速验证。**进行正式评测时，请将 `base_dir` 改为指向 `spider/`、`bird/` 等基准数据集。**
>
> 下载所有数据集：
> ```bash
> cd Data_Synthesizer/tools
> python download_data.py
> ```

#### 3.3 配置

创建 `Evaluation_Framework/evaluation_config.yaml`：

```yaml
# 数据库配置
base_dir: ../Data_Synthesizer/pipeline/sqlite/results/test/vector_databases
db_type: 'sqlite'

# 项目根目录
project_dir: ../

# 执行引擎配置
engine_config_path: Execution_Engine/engine_config.yaml

# 输入/输出文件
eval_data_file: ./test_data/input_output.json
execution_results_file: sql_execution_results.json
evaluation_report_file: evaluation_report.json

# 嵌入服务配置
embedding_service:
  auto_manage: false
  host: "127.0.0.1"
  port: 8000

# 评估指标
metrics:
  - name: 'exact_match'
  - name: 'f1_score'
  - name: 'precision'
  - name: 'recall'
  - name: 'map'
  - name: 'mrr'
  - name: 'ndcg'
    k: 10
```

#### 3.4 运行评估

**前提条件：**
- 嵌入服务正在运行
- 测试数据库可用
- 评估数据文件存在

**选项1：运行完整流水线**

```bash
cd Evaluation_Framework
python run_eval_pipeline.py --all --config evaluation_config.yaml
```

**选项2：分步运行**

```bash
# 步骤1：执行SQL查询
python run_eval_pipeline.py --execute --config evaluation_config.yaml

# 步骤2：评估结果
python run_eval_pipeline.py --evaluate --config evaluation_config.yaml
```

#### 3.5 理解结果

**评估报告** (`evaluation_report.json`)：
```json
{
  "overall_metrics": {
    "exact_match": 0.75,
    "f1_score": 0.82,
    "precision": 0.85,
    "recall": 0.80,
    "map": 0.78,
    "mrr": 0.81,
    "ndcg@10": 0.79
  }
}
```


#### 3.6 聚合多个结果

如果有来自不同模型或数据集的多个评估报告：

```bash
python aggregate_results.py \
  --results-dir ./results \
  --output summary.csv
```

---

### 模块4：数据合成器

数据合成器通过自动化流水线为Text2VectorSQL模型生成训练数据。

#### 4.1 概述

**合成流水线：**
1. **数据库合成**：从Web表格生成结构化数据库
2. **数据库向量化**：向语义列添加向量嵌入
3. **SQL合成**：生成VectorSQL查询
4. **问题合成**：生成自然语言问题
5. **思维链合成**：生成推理步骤

#### 4.2 下载预合成数据（推荐）

无需从头合成，可以下载预生成的数据：

```bash
cd Data_Synthesizer/tools

# 下载数据库
python download_data.py

# 下载训练好的模型（如果可用）
python download_model.py
```

#### 4.3 运行合成流水线（高级）

**前提条件：**
- LLM API访问（OpenAI、Claude等）
- 足够的API额度
- 源数据集已下载

```bash
cd Data_Synthesizer

# 运行完整流水线
python pipeline/general_pipeline.py
```

**注意**：完整合成可能耗时且昂贵（需要大量LLM API调用）。

---

### 完整工作流示例

以下是评估Text2VectorSQL系统的完整端到端工作流：

#### 步骤1：启动嵌入服务

```bash
# 终端1
cd Embedding_Service
python server.py --config config.yaml
```

#### 步骤2：准备测试数据库

```bash
# 使用提供的测试数据库或下载
cd Data_Synthesizer/tools
python download_data.py
```

#### 步骤3：测试执行引擎

```bash
# 终端2
cd Execution_Engine

python execution_engine.py \
  --sql "SELECT * FROM employee WHERE employee_description_embedding MATCH lembed('all-MiniLM-L6-v2', '软件工程师') LIMIT 5;" \
  --db-type "sqlite" \
  --db-identifier "../Data_Synthesizer/pipeline/sqlite/results/test/vector_databases/company_1.db" \
  --config "engine_config.yaml"
```

#### 步骤4：运行评估

```bash
cd Evaluation_Framework

# 运行完整评估
python run_eval_pipeline.py --all --config evaluation_config.yaml

# 查看结果
cat evaluation_report.json
```

#### 步骤5：分析结果

```bash
# 如果有多个实验
python aggregate_results.py --results-dir ./results --output summary.csv
cat summary.csv
```

---

### 故障排除

#### 问题1：嵌入服务连接错误

**错误**：`Connection refused to http://localhost:8000`

**解决方案**：
- 确保嵌入服务正在运行
- 检查端口8000是否可用
- 验证防火墙设置
- 尝试使用`127.0.0.1`而不是`localhost`

#### 问题2：模型下载失败

**错误**：`Failed to download model from Hugging Face`

**解决方案**：
```bash
# 使用Hugging Face镜像（中国用户）
export HF_ENDPOINT=https://hf-mirror.com

# 或手动下载并指定本地路径
```

#### 问题3：SQLite向量扩展未找到

**错误**：`sqlite3.OperationalError: no such function: lembed` 或 `no such module: vec0`

**解决方案**：
```bash
# 安装sqlite-vec和sqlite-lembed
pip install sqlite-vec sqlite-lembed
```

**验证安装**：
```bash
python -c "
import sqlite3, sqlite_vec, sqlite_lembed
conn = sqlite3.connect(':memory:')
conn.enable_load_extension(True)
sqlite_vec.load(conn)
sqlite_lembed.load(conn)
print('vec_version:', conn.execute('select vec_version()').fetchone()[0])
"
```

**常见原因**：
- `sqlite-vec` 和 `sqlite-lembed` 未包含在依赖中（最新版本已修复）
- Python 编译时未启用扩展加载支持（重新编译 Python 或使用 conda）
- 系统 SQLite 版本过旧（需要 >= 3.31.0 才能使用 `enable_load_extension`）

#### 问题3.5：virtiofs磁盘I/O错误（Linux VM/容器）

**错误**：`disk I/O error` 访问共享/挂载文件系统上的SQLite数据库时

**原因**：virtiofs（或类似的共享文件系统）与SQLite的锁机制存在已知兼容性问题

**解决方案**：
```bash
# 将数据库复制到本地/tmp目录
cp /path/to/database.db /tmp/database.db

# 在命令中使用/tmp副本
python execution_engine.py \
  --sql "YOUR_SQL" \
  --db-type "sqlite" \
  --db-identifier "/tmp/database.db" \
  --config "engine_config.yaml"
```

**对于评估框架**，更新配置中的数据库路径：
```bash
# 创建脚本将数据库复制到/tmp
mkdir -p /tmp/vector_databases
cp -r /path/to/vector_databases/* /tmp/vector_databases/

# 更新evaluation_config.yaml
# base_dir: /tmp/vector_databases
```

#### 问题4：数据库连接超时

**错误**：`Database connection timeout`

**解决方案**：
- 在`engine_config.yaml`中增加超时时间
- 检查数据库服务器是否运行
- 验证连接凭据

#### 问题5：评估期间内存不足

**错误**：`CUDA out of memory`或`MemoryError`

**解决方案**：
- 减少配置中的批处理大小
- 使用较小的嵌入模型
- 启用仅CPU模式
- 分块处理数据

---

### 常见问题

#### Q1：可以在没有GPU的情况下运行吗？

**A**：可以！嵌入模型可以在CPU上运行，尽管速度较慢。系统会自动检测可用硬件。

#### Q2：支持哪些嵌入模型？

**A**：Hugging Face的`sentence-transformers`库中的任何模型。热门选择：
- `all-MiniLM-L6-v2`（快速、轻量）
- `all-mpnet-base-v2`（平衡）
- `gte-large`（高质量）
- CLIP模型（多模态）

#### Q3：如何在没有GPU的情况下评估UniVectorSQL模型？

**A**：您可以：
1. 使用云提供商的API模式（OpenAI、Anthropic等）
2. 使用适合CPU内存的较小模型
3. 使用论文中的预计算结果

#### Q4：可以使用自己的数据库吗？

**A**：可以！按照以下步骤：
1. 创建带有向量列的数据库
2. 使用嵌入服务填充嵌入
3. 使用连接详细信息更新`engine_config.yaml`
4. 使用执行引擎执行查询

#### Q5：数据合成需要多长时间？

**A**：取决于数据集大小和LLM API速度：
- 小数据集（100个样本）：1-2小时
- 中等数据集（1000个样本）：10-20小时
- 大数据集（10000+样本）：几天

建议：下载预合成数据。

---

### 性能提示

1. **使用GPU加速**（如果可用）用于嵌入模型
2. **在数据库中缓存嵌入**以避免重新计算
3. **批量查询**评估多个样本时
4. **使用较小模型**以加快推理速度
5. **启用连接池**用于数据库连接
6. **根据硬件调整超时值**

---

### 引用

如果您在研究中使用此项目，请引用：

```bibtex
@article{wang2025text2vectorsql,
  title={Text2VectorSQL: Towards a Unified Interface for Vector Search and SQL Queries},
  author={Wang, Zhengren and Yao, Dongwen and Li, Bozhou and Ma, Dongsheng and Li, Bo and Li, Zhiyu and Xiong, Feiyu and Cui, Bin and Tang, Linpeng and Zhang, Wentao},
  journal={arXiv preprint arXiv:2506.23071},
  year={2025}
}
```

---

### 资源

- **论文**：https://arxiv.org/abs/2506.23071
- **Hugging Face**：https://huggingface.co/VectorSQL
- **GitHub**：https://github.com/OpenDCAI/Text2VectorSQL
- **问题反馈**：https://github.com/OpenDCAI/Text2VectorSQL/issues

---

### 联系方式

如有问题或反馈，请联系：wzr@stu.pku.edu.cn

---

## 附录：快速参考

### 常用命令速查

```bash
# 启动嵌入服务
cd Embedding_Service && python server.py --config config.yaml

# 执行VectorSQL查询
cd Execution_Engine && python execution_engine.py \
  --sql "YOUR_SQL" --db-type sqlite --db-identifier "path/to/db.db" \
  --config engine_config.yaml

# 运行评估
cd Evaluation_Framework && python run_eval_pipeline.py --all \
  --config evaluation_config.yaml

# 下载数据
cd Data_Synthesizer/tools && python download_data.py
```

### 配置文件模板

所有配置文件模板可在各模块目录中找到：
- `Embedding_Service/config.yaml`
- `Execution_Engine/engine_config.yaml`
- `Evaluation_Framework/evaluation_config.yaml`
- `Data_Synthesizer/pipeline/config.yaml`

---

**文档版本**：v1.0  
**最后更新**：2026-04-01  
**维护者**：Text2VectorSQL团队
