# 评估框架 (Evaluation Framework)

本模块专门用于评估模型生成的SQL（特别是向量SQL）查询的准确性。

该框架通过一个分为两个核心阶段的流水线来工作：**SQL执行** 和 **结果评估**。

## 核心功能

1.  **SQL生成 (`generate.py`)**
    *   根据输入的问题、数据库Schema和相关元数据，调用大语言模型（LLM）生成预测的SQL查询。
    *   支持两种生成模式：
        *   **vLLM离线推理**：利用vLLM库在本地进行高效的批量推理，支持多GPU张量并行。
        *   **API在线调用**：通过HTTP API调用外部模型服务（如OpenAI、Claude等），支持多线程并发请求。
    *   具备强大的缓存机制，能够根据输入数据和模型自动缓存生成结果，实现断点续传。

2.  **SQL执行 (`sql_executor.py`)**
    *   连接到指定的数据库（支持 SQLite、PostgreSQL、ClickHouse 等）并执行SQL查询。
    *   **沙箱化执行**：在独立的进程中执行每个SQL查询，并强制实施超时，防止恶意或低效的查询卡死整个评估流程。
    *   同时执行模型生成的预测SQL和所有标准的（Ground Truth）SQL，并保存两者的执行结果（数据、列名、状态等）。
    *   集成了嵌入服务（Embedding Service）的自动管理功能，可以在执行前自动启动所需的服务。
    *   同样具备缓存机制，可以跳过已成功执行的查询。

3.  **结果评估 (`evaluate_results.py`)**
    *   对比预测SQL和标准SQL的**执行结果**，而非仅仅比较SQL字符串。
    *   支持多种评估指标：
        *   **精确匹配 (Exact Match)**: 预测结果与任一标准结果完全一致。
        *   **集合指标 (Set-based Metrics)**:
            *   **Precision, Recall, F1-Score**: 基于结果集的交集计算，不考虑行顺序。
        *   **排序指标 (Ranking Metrics)**:
            *   **nDCG@k**: 评估返回结果的排序质量。
            *   **MAP (Mean Average Precision)**, **MRR (Mean Reciprocal Rank)**.
    *   **基于LLM的评估**: 调用另一个LLM来从语义层面评估预测SQL的“SQL骨架”和“向量部分”的正确性。
    *   评估过程同样支持并发处理和断点续传。

4.  **结果聚合 (`aggregate_results.py`)**
    *   一个实用工具，用于从多个模型、多个数据集的评估报告（JSON文件）中收集、汇总评估指标。
    *   将分散的结果聚合成一个结构化的CSV文件，便于横向对比不同模型的性能。

## 使用流程

评估过程通过 `run_eval_pipeline.py` 脚本进行统一调度，该脚本通过读取 `evaluation_config.yaml` 配置文件来驱动整个流程。

1.  **准备配置文件 (`evaluation_config.yaml`)**
    *   配置数据库类型 (`db_type`)、数据库文件根目录 (`base_dir`)。
    *   指定包含预测SQL的输入文件 (`eval_data_file`)。
    *   定义中间结果和最终报告的输出路径。
    *   配置需要计算的评估指标 (`metrics`)。
    *   （可选）配置嵌入服务的地址和模型。

2.  **运行SQL生成 (如果需要)**
    *   执行 `generate.py` 脚本，为数据集生成预测SQL。
    *   `python generate.py --config generate_config.yaml`

3.  **运行评估流水线**
    *   **完整流程** (执行 + 评估):
        ```bash
        python run_eval_pipeline.py --all --config evaluation_config.yaml
        ```
    *   **仅执行SQL**:
        ```bash
        python run_eval_pipeline.py --execute --config evaluation_config.yaml
        ```
    *   **仅评估结果** (需要已有的执行结果文件):
        ```bash
        python run_eval_pipeline.py --evaluate --config evaluation_config.yaml
        ```

4.  **聚合多个实验的结果**
    *   将所有实验的报告（JSON文件）按约定目录结构存放。
    *   运行 `aggregate_results.py` 生成对比表格。
        ```bash
        python aggregate_results.py --results-dir ./results --output summary.csv
        ```

## 依赖安装

要运行此评估框架，请安装所需的Python包：

```bash
pip install -r requirements.txt
```