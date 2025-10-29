# Text2VectorSQL: 向量搜索与SQL查询的统一自然语言接口

本项目提出了**Text2VectorSQL** ，旨在为查询**结构化数据**和**非结构化数据**建立统一自然语言接口的新任务。

传统的Text2SQL系统在访问表格等结构化数据方面取得了显著进展，但无法理解语义或多模态查询。与此同时，向量搜索已成为查询非结构化数据（如文本、图像）的标准，但将其与 SQL集成（称为 VectorSQL）仍然依赖于手动的、易错的查询构建，并且缺乏标准化的评估方法。

Text2VectorSQL 旨在弥合这一“根本差距”。我们提供了一个全面的基础生态系统，不仅用于训练和评估强大的 Text2VectorSQL 模型（如 **UniVectorSQL** ），还包括一个完整的工具链，用于合成数据、执行查询和评估结果。

-----

## 🚀 项目架构与核心模块

本仓库包含一个完整的生态系统，分为四个核心模块：

1.  **数据合成 (Data\_Synthesizer)**:
    一个可扩展的流水线，用于从公开的基础表格开始，自动合成包含（数据库、自然语言问题、VectorSQL 查询、思维链）的“四元组”训练数据。
2.  **执行引擎 (Execution\_Engine)**:
    一个负责解析和执行 VectorSQL 查询的后端引擎。它通过调用**嵌入服务**来处理特殊的 `lembed(model, text)` 函数，并将其转换为与目标数据库（SQLite、PostgreSQL、ClickHouse）兼容的原生查询。
3.  **嵌入服务 (Embedding\_Service)**:
    一个基于 FastAPI 的高性能 API 服务，为**执行引擎**提供按需的文本和图像向量化能力。它支持多模型、多GPU，并能自动缓存模型。
4.  **评估框架 (Evaluation\_Framework)**:
    一个用于全面评估 Text2VectorSQL 模型性能的框架。它通过**执行**模型生成的 SQL 和黄金 SQL，并比较两者的**执行结果**（而非SQL字符串）来进行准确评估。

-----


## 🔧 安装

1.  克隆本仓库：

    ```bash
    git clone https://github.com/OpenDCAI/Text2VectorSQL.git --depth 1
    cd Text2VectorSQL
    ```

2.  根据您需要使用的模块，安装其独立的依赖文件。每个模块（`Data_Synthesizer`, `Execution_Engine`, `Embedding_Service`, `Evaluation_Framework`）的目录下都有一个 `requirements.txt` 文件。

    例如，要安装执行引擎的依赖：

    ```bash
    cd Execution_Engine
    pip install -r requirements.txt
    ```

-----

## ⚡ 快速开始

根据您的目标，可以按以下场景使用本项目的工具链：

### 场景一：运行嵌入服务 (所有执行的前提)

`Execution_Engine` 依赖此服务来获取向量。

1.  **配置服务**:
    进入 `Embedding_Service/` 目录，创建 `config.yaml` 文件，指定您要使用的模型。
    ```yaml
    # config.yaml 示例
    server:
      host: "0.0.0.0"
      port: 8000
    
    models:
      - name: "all-MiniLM-L6-v2"
        hf_model_path: "sentence-transformers/all-MiniLM-L6-v2"
        local_model_path: "./models/all-MiniLM-L6-v2"
        trust_remote_code: true
      # ... 其他模型
    ```
2.  **启动服务**:
    ```bash
    cd Embedding_Service/
    bash run.sh
    ```
    服务将在 `http://0.0.0.0:8000` 上运行。首次运行会自动下载模型。

### 场景二：执行一个 VectorSQL 查询

确保场景一中的**嵌入服务**正在运行。

`Execution_Engine` 可以作为命令行工具使用：

1.  **配置引擎**:
    进入 `Execution_Engine/` 目录，创建 `engine_config.yaml`，指定嵌入服务的地址和数据库连接信息。
    ```yaml
    embedding_service:
      url: "http://127.0.0.1:8000/embed" # 对应场景一的服务
    
    database_connections:
      clickhouse:
        host: "localhost"
        port: 8123
        # ...
    
    timeouts:
      sql_execution: 60
    ```
2.  **运行查询**:
    ```bash
    cd Execution_Engine/
    python execution_engine.py \
        --sql "SELECT Name FROM musical m ORDER BY L2Distance(Category_embedding, lembed('all-MiniLM-L6-v2','opera')) LIMIT 5;" \
        --db-type "clickhouse" \
        --db-identifier "musical" \
        --config "engine_config.yaml"
    ```

### 场景三：合成全新的 Text2VectorSQL 数据集

使用 `Data_Synthesizer` 模块。

1.  **配置流水线**:
    进入 `Data_Synthesizer/` 目录，复制 `pipeline/config.yaml.example` 为 `config.yaml`。
    在 `config.yaml` 中填入您的 LLM API-Key、Base-URL 等。
2.  **选择数据集**:
    编辑 `pipeline/general_pipeline.py`，修改顶部的 `DATASET_BACKEND` 和 `DATASET_TO_LOAD` 变量。
3.  **运行流水线**:
    ```bash
    cd Data_Synthesizer/
    python pipeline/general_pipeline.py
    ```
    最终的合成数据集将保存在 `config.yaml` 中配置的 `result_path` 路径下。

### 场景四：评估一个 Text2VectorSQL 模型

使用 `Evaluation_Framework` 模块。

1.  **准备数据**: 确保您有一个包含模型预测 SQL 的评估文件（`eval_data_file`）。如果还没有，请运行 `generate.py` 来生成。
2.  **配置评估**:
    进入 `Evaluation_Framework/` 目录，创建 `evaluation_config.yaml`。
    配置数据库类型 (`db_type`)、数据库文件根目录 (`base_dir`)、输入文件 (`eval_data_file`) 和评估指标 (`metrics`)。
3.  **运行评估流水线**:
    
    ```bash
    cd Evaluation_Framework/
    
    # 运行完整流程（SQL执行 + 结果评估）
    python run_eval_pipeline.py --all --config evaluation_config.yaml
    
    # 或分步运行
    # python run_eval_pipeline.py --execute --config evaluation_config.yaml
    # python run_eval_pipeline.py --evaluate --config evaluation_config.yaml
    ```
    评估报告（JSON文件）将保存在配置的输出路径中。您可以使用 `aggregate_results.py` 将多个报告汇总为 CSV。

-----

## 🧩 模块详解

### 1\. 数据合成 (Data\_Synthesizer)

此模块是训练强大 Text2VectorSQL 模型（如 UniVectorSQL）的基础 。它通过一个自动化的流水线，生成高质量的 Text2VectorSQL 数据集。

**核心流程**：

1.  **数据库合成与增强 (`database_synthesis`)**: 基于 Web 表格生成结构化数据库。
2.  **数据库向量化 (`vectorization`)**: 识别“语义丰富”的列（如描述），使用 Sentence Transformer 生成向量嵌入，并构建新的支持向量查询的数据库。
3.  **VectorSQL与问题合成 (`synthesis_sql`, `synthesis_nl`)**: 自动生成不同复杂度的 VectorSQL 查询，并反向翻译生成对应的自然语言问题。
4.  **思维链合成 (`synthesis_cot`)**: 为每个数据样本生成详细的推理步骤（Chain-of-Thought），解释从问题到 VectorSQL 的推导过程。

您可以通过 `pipeline/general_pipeline.py` 脚本一键运行完整的端到端合成流程。

### 2\. 执行引擎 (Execution\_Engine)

这是 Text2VectorSQL 的运行时核心。它充当一个桥梁，解析包含语义搜索意图的 VectorSQL，并将其转换为数据库可以理解的原生查询。

**核心功能**：

  * **解析 `lembed` 函数**: 引擎专门用于处理 `lembed(model, text)` 语法的查询。
  * **动态向量化**:
    1.  引擎解析查询，提取所有唯一的 `(model, text)` 组合。
    2.  它向一个外部的 **Embedding 服务** 发起网络请求，获取这些文本的向量表示。
  * **SQL 翻译与执行**:
    1.  收到向量后，引擎将 `lembed(...)` 调用替换为数据库原生的向量字面量（例如，在 PostgreSQL 中为 `[0.1, 0.2, ...]`）。
    2.  它连接到目标数据库（支持 PostgreSQL, ClickHouse, SQLite）并执行翻译后的原生查询。
  * **健壮性**: 引擎为网络请求和数据库执行实现了健壮的超时和错误管理。

### 3\. 嵌入服务 (Embedding\_Service)

这是一个独立的高性能 API 服务，充当 `Execution_Engine` 的向量化后端。

**主要特性**：

  * **高性能**: 基于 FastAPI 和 Uvicorn，提供异步处理能力。
  * **多模型与多GPU**: 支持通过 `config.yaml` 同时加载和管理多个模型（如 `all-MiniLM-L6-v2` 或 `CLIP`），并支持张量并行。
  * **自动缓存**: 启动时自动从 Hugging Face Hub 下载模型并缓存到本地，避免重复下载。
  * **核心接口**:
      * `/embed`: 接收模型名称和文本/图像列表，返回向量。
      * `/health`: 健康检查。

### 4\. 评估框架 (Evaluation\_Framework)

为了客观评估 Text2VectorSQL 模型的真实能力，我们构建了一个专用的评估框架。评估框架对比的是预测 SQL 和标准 SQL 的**执行结果**，而非仅仅比较 SQL 字符串。这对于向量搜索至关重要，因为近似结果是可接受的，且不同语法的 SQL 可能产生相同的结果 。

**评估流水线**：

1.  **SQL 生成 (`generate.py`)**: 调用模型（通过 vLLM 或 API）为评估问题生成预测的 SQL 查询。
2.  **SQL 执行 (`sql_executor.py`)**: 在沙箱化进程中（带超时）分别执行预测 SQL 和所有黄金 SQL，并缓存两者的执行结果。
3.  **结果评估 (`evaluate_results.py`)**: 对比预测结果和黄金结果，计算一系列评估指标。
      * **集合指标 (Set-based)**: Precision, Recall, F1-Score。
      * **排序指标 (Rank-based)**: nDCG@k, MAP, MRR。
      * **分解指标 (Decomposed)**: $ACC_{SQL}$ (SQL骨架正确性) 和 $ACC_{Vec}$ (向量部分正确性) 。
4.  **结果聚合 (`aggregate_results.py`)**: 将多个实验的 JSON 报告汇总为易于比较的 CSV 文件。