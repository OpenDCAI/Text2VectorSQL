# Text2VectorSQL 项目复现指南

[![arXiv](https://img.shields.io/badge/arXiv-2506.23071-b31b1b.svg)](https://arxiv.org/abs/2506.23071)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-VectorSQL-blue)](https://huggingface.co/VectorSQL)

本指南提供 Text2VectorSQL 项目的完整复现步骤，包括环境配置、数据准备、模型下载和评测运行。

## 目录

- [系统要求](#系统要求)
- [快速开始](#快速开始)
- [详细步骤](#详细步骤)
- [常见问题](#常见问题)
- [评测结果](#评测结果)

---

## 系统要求

### 硬件要求
- **存储空间**: 至少 200GB 可用空间
  - 数据集: ~146GB
  - UniVectorSQL 模型: ~29GB
  - 其他文件: ~25GB
- **内存**: 建议 16GB 以上
- **GPU**: 
  - 嵌入服务: 可选（CPU 也可运行）
  - UniVectorSQL 评测: 需要 GPU（建议 24GB+ 显存）

### 软件要求
- Python 3.8+
- CUDA 11.8+ (如使用 GPU)
- Git
- 数据库: SQLite 3.35+

---

## 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/OpenDCAI/Text2VectorSQL.git
cd Text2VectorSQL

# 2. 安装依赖
cd Embedding_Service && pip install -r requirements.txt
cd ../Evaluation_Framework && pip install -r requirements.txt
cd ../Execution_Engine && pip install -r requirements.txt

# 3. 下载数据和模型
cd ../Data_Synthesizer/tools
python download_data.py    # 下载评测数据
python download_model.py   # 下载 UniVectorSQL 模型

# 4. 启动嵌入服务
cd ../../Embedding_Service
bash run.sh

# 5. 运行评测（见详细步骤）
```

---

## 详细步骤

### 步骤 1: 环境准备

#### 1.1 安装 Python 依赖

每个模块都有独立的 `requirements.txt`：

```bash
# 嵌入服务
cd Embedding_Service
pip install -r requirements.txt

# 执行引擎
cd ../Execution_Engine
pip install -r requirements.txt

# 评测框架
cd ../Evaluation_Framework
pip install -r requirements.txt
```

#### 1.2 配置嵌入服务

创建 `Embedding_Service/config.yaml`：

```yaml
server:
  host: "0.0.0.0"
  port: 8000

models:
  - name: "all-MiniLM-L6-v2"
    hf_model_path: "sentence-transformers/all-MiniLM-L6-v2"
    local_model_path: "./models/all-MiniLM-L6-v2"
    trust_remote_code: true
```

启动服务：
```bash
cd Embedding_Service
bash run.sh
```

验证服务：
```bash
curl http://localhost:8000/health
```

---

### 步骤 2: 下载数据和模型

#### 2.1 下载评测数据

```bash
cd Data_Synthesizer/tools
python download_data.py
```

这将下载约 146GB 的数据，包括：
- Spider 数据集
- Bird 数据集
- Arxiv 数据集
- Wikipedia 多模态数据集

#### 2.2 下载 UniVectorSQL 模型

```bash
python download_model.py
```

模型将下载到 `Data_Synthesizer/model/UniVectorSQL-7B-LoRA/`


#### 2.3 解压数据

```bash
cd ../pipeline/sqlite
tar -xzf results.tar.gz
```

---

### 步骤 3: 配置执行引擎

编辑 `Execution_Engine/engine_config.yaml`：

```yaml
embedding_service:
  url: "http://127.0.0.1:8000/embed"

database_connections:
  postgresql:
    host: "localhost"
    port: 5432
    user: "postgres"
    password: "your_password"
  
  clickhouse:
    host: "localhost"
    port: 8123
    user: "default"
    password: ""

timeouts:
  embedding_service: 10
  sql_execution: 60
  total_execution: 120
```

---

### 步骤 4: 运行评测

#### 4.1 使用 API 模型（如 GPT-4o）

1. 创建配置文件 `Evaluation_Framework/generate_config.yaml`：

```yaml
api:
  base_url: "https://api.openai.com/v1"
  api_key: "your-api-key"
  timeout: 60
  max_workers: 4
  temperature: 0.7
  max_tokens: 2048
```

2. 生成预测 SQL：

```bash
cd Evaluation_Framework
python generate.py \
  --mode api \
  --dataset "../Data_Synthesizer/pipeline/sqlite/results/spider/input_llm.json" \
  --model_name "gpt-4o" \
  --output "predictions_gpt4o.json" \
  --config "generate_config.yaml"
```


3. 运行评测：

```bash
python run_eval_pipeline.py --all \
  --base_dir "../Data_Synthesizer/pipeline/sqlite/results/spider/vector_databases" \
  --db_type "sqlite" \
  --eval_data_file "predictions_gpt4o.json" \
  --evaluation_report_file "report_gpt4o.json" \
  --execution_results_file "execution_gpt4o.json"
```

#### 4.2 使用 UniVectorSQL 模型（需要 GPU）

1. 生成预测 SQL：

```bash
python generate.py \
  --mode vllm \
  --dataset "../Data_Synthesizer/pipeline/sqlite/results/spider/input_llm.json" \
  --model_path "../Data_Synthesizer/model/UniVectorSQL-7B-LoRA" \
  --output "predictions_univectorsql.json" \
  --tensor_parallel_size 1
```

2. 运行评测：

```bash
python run_eval_pipeline.py --all \
  --base_dir "../Data_Synthesizer/pipeline/sqlite/results/spider/vector_databases" \
  --db_type "sqlite" \
  --eval_data_file "predictions_univectorsql.json" \
  --evaluation_report_file "report_univectorsql.json" \
  --execution_results_file "execution_univectorsql.json"
```

#### 4.3 批量评测

使用提供的脚本生成批量评测命令：

```bash
# 修改参数
python create_generate_script.py
python create_eval_script.py

# 运行批量评测
bash generate.sh
bash run_evaluation.sh
```


---

## 常见问题

### Q1: SQLite 报 "disk I/O error"

**问题**: 在某些文件系统（如 virtiofs）上运行时，SQLite 数据库访问失败。

**解决方案**: 将数据库复制到本地文件系统（如 /tmp）：

```bash
# 创建本地目录
mkdir -p /tmp/vector_databases

# 复制数据库
cp -r Data_Synthesizer/pipeline/sqlite/results/spider/vector_databases/* /tmp/vector_databases/

# 使用本地路径运行评测
python run_eval_pipeline.py --all \
  --base_dir "/tmp/vector_databases" \
  --db_type "sqlite" \
  ...
```

### Q2: 嵌入服务无法启动

**检查步骤**:
1. 确认端口未被占用：`lsof -i :8000`
2. 检查模型是否下载：`ls Embedding_Service/models/`
3. 查看日志：`tail -f Embedding_Service/embedding_service.log`

### Q3: 模型下载失败

**问题**: UniVectorSQL 模型下载报 401 错误。

**解决方案**: 
- 确保模型仓库已设为 public
- 如果是 gated repo，需要申请访问权限


### Q4: GPU 内存不足

**问题**: 运行 UniVectorSQL 时 GPU 内存不足。

**解决方案**:
- 使用更小的 batch size
- 使用 tensor parallelism: `--tensor_parallel_size 2`
- 使用量化版本（如果可用）

### Q5: 评测报告显示评估失败

**问题**: 评测报告中所有案例都显示 "eval_status: unknown"。

**原因**: 直接使用 `input_llm.json` 文件，该文件缺少 `predicted_sql` 字段。

**解决方案**: 必须先用模型生成预测 SQL，然后再评测。

---

## 评测结果

### 预期结果

在 Spider 数据集上的参考结果：

| 模型 | Exact Match | F1 Score | Precision | Recall |
|------|-------------|----------|-----------|--------|
| GPT-4o | ~40% | ~47% | ~52% | ~43% |
| UniVectorSQL-7B | TBD | TBD | TBD | TBD |

*注: 具体结果可能因配置和数据版本而异*

---

## 项目结构

```
Text2VectorSQL/
├── Data_Synthesizer/          # 数据合成模块
│   ├── tools/                 # 下载脚本
│   ├── pipeline/              # 合成流程
│   └── model/                 # 模型存储
├── Embedding_Service/         # 嵌入服务
│   ├── models/                # 嵌入模型
│   └── config.yaml            # 服务配置
├── Execution_Engine/          # 执行引擎
│   └── engine_config.yaml     # 引擎配置
└── Evaluation_Framework/      # 评测框架
    ├── generate.py            # SQL 生成
    ├── run_eval_pipeline.py   # 评测流程
    └── evaluate_results.py    # 结果评估
```


---

## 参考资源

- **论文**: [Text2VectorSQL: Towards a Unified Interface for Vector Search and SQL Queries](https://arxiv.org/abs/2506.23071)
- **Hugging Face**: [VectorSQL Models](https://huggingface.co/VectorSQL)
- **GitHub**: [Text2VectorSQL Repository](https://github.com/OpenDCAI/Text2VectorSQL)

---

## 贡献指南

欢迎提交 Issue 和 Pull Request！

### 报告问题
- 使用 GitHub Issues
- 提供详细的错误信息和复现步骤
- 包含系统环境信息

### 提交代码
- Fork 仓库并创建新分支
- 遵循现有代码风格
- 添加必要的测试
- 提交 Pull Request

---

## 许可证

本项目采用 Apache 2.0 许可证。详见 [LICENSE](LICENSE) 文件。

---

## 联系方式

如有任何问题或反馈，请联系：
- Email: wzr@stu.pku.edu.cn
- GitHub Issues: https://github.com/OpenDCAI/Text2VectorSQL/issues

---

**最后更新**: 2026-04-01
**版本**: v1.0

