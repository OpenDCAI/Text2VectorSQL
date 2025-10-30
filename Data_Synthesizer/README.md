# 数据合成 (Data_Synthesizer)

## 简介

Data_Synthesizer是一个功能强大的数据合成流水线，专为生成高质量Text2VectorSQL数据集而设计。本模块从基础数据库出发，通过一系列自动化步骤，最终产出包含数据库、自然语言问题、VectorSQL查询以及思维链（Chain-of-Thought）的完整数据集，为训练和评测先进的Text2VectorSQL模型提供支持。

该流水线支持多种数据库后端，如 SQLite、PostgreSQL 和 ClickHouse，并集成了向量化能力，使模型能够理解和利用数据中的语义信息。

## 核心功能

- **数据库合成与增强 (`database_synthesis`)**: 从零开始或基于现有Web表格，自动生成结构化的数据库，并可对数据库模式进行增强，增加其复杂性和真实性。
- **数据库向量化 (`vectorization`)**: 识别数据库中的“语义丰富”列（如描述性文本），利用Sentence Transformer模型为这些列生成向量嵌入，并构建支持向量查询的新数据库。这是实现语义搜索的关键。
- **VectorSQL与问题合成 (`synthesis_sql`, `synthesis_nl`)**: 基于（向量化的）数据库模式，自动生成不同复杂度的VectorSQL查询，并为每个VectorSQL查询生成对应的自然语言问题。
- **思维链合成 (`synthesis_cot`)**: 为每一个“数据库-问题-VectorSQL”三元组，生成详细的推理步骤（即思维链），解释从问题到VectorSQL的推导过程。这对于训练具有更强推理能力的模型至关重要。
- **统一流水线 (`pipeline`)**: 提供一个总控脚本 `general_pipeline.py`，通过简单的配置即可运行完整的端到端数据合成流程，同时也支持对流程中每一步的独立调用和微调。

## 目录结构

```
Data_Synthesizer/
├─── database_synthesis/   # 从Web表格合成数据库
├─── pipeline/             # 统一流水线和配置文件
├─── synthesis_cot/        # 思维链(CoT)合成
├─── synthesis_eval/       # 为模型生成训练和评估数据
├─── synthesis_nl/         # 自然语言问题(NL)合成
├─── synthesis_sql/        # VectorSQL查询合成
├─── tools/                # 数据集迁移、混合等辅助工具
└─── vectorization/        # 数据库向量化
```

## 快速开始

通过运行统一流水线，您可以最便捷地完成整个数据合成过程。

1.  **安装依赖**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **配置环境**:
    - 复制 `pipeline/config.yaml.example` 并重命名为 `pipeline/config.yaml`。
    - 在 `config.yaml` 中填入您的 LLM API-Key、Base-URL 以及其他相关配置。
    - 在服务器中启动embedding服务，参考文件 `../../Embedding_Service/README.md`

3.  **配置流水线**:
    - 打开 `pipeline/general_pipeline.py` 文件。
    - 修改顶部的 `DATASET_BACKEND` 和 `DATASET_TO_LOAD` 变量，以选择您要使用的数据库类型和具体的数据集配置。
    ```python
    # 示例：选择 clickhouse 后端和名为 synthesis_data_deversity 的数据集
    DATASET_BACKEND = "clickhouse"
    DATASET_TO_LOAD = "synthesis_data_deversity"
    ```

4.  **运行流水线**:
    ```bash
    python pipeline/general_pipeline.py
    ```
    脚本将自动执行所有步骤，包括数据库向量化、VectorSQL生成、问题生成等。最终的合成数据集和向量数据库将保存在 `config.yaml` 中为该数据集配置的 `result_path` 路径下。

## 分步执行

如果您希望更精细地控制每一步，可以按照以下顺序手动执行各个子模块的脚本。

### 第1步: 数据库向量化 (`vectorization`)

此步骤为现有数据库添加向量信息。

1.  **生成基础Schema**:
    ```bash
    python vectorization/generate_schema.py --db-dir <数据库目录> --output-file <输出的tables.json路径>
    ```
2.  **（可选）为Schema填充样本数据**:
    ```bash
    python vectorization/enhance_tables_json.py ...
    ```
3.  **寻找语义丰富的列**:
    ```bash
    python vectorization/find_semantic_rich_column.py ...
    ```
4.  **批量向量化**:
    为语义列生成向量嵌入，并创建初始的向量数据库脚本。
    ```bash
    python vectorization/batch_vectorize_databases.py ...
    ```
5.  **生成最终向量数据库**:
    使用上一步生成的脚本，构建最终的SQLite向量数据库。
    ```bash
    python vectorization/vector_database_generate.py ...
    ```

### 第2步: SQL查询合成 (`synthesis_sql`)

1.  **生成VectorSQL合成提示**:
    ```bash
    python synthesis_sql/generate_sql_synthesis_prompts.py ...
    ```
2.  **调用LLM合成VectorSQL**:
    ```bash
    python synthesis_sql/synthesize_sql.py ...
    ```
3.  **后处理与筛选**:
    验证VectorSQL的正确性，去除无效和重复的查询。
    ```bash
    python synthesis_sql/post_process_sqls.py ...
    ```

### 第3步: 自然语言问题合成 (`synthesis_nl`)

1.  **生成问题合成提示**:
    ```bash
    python synthesis_nl/generate_question_synthesis_prompts.py ...
    ```
2.  **调用LLM合成问题**:
    ```bash
    python synthesis_nl/synthesize_question.py ...
    ```
3.  **后处理与筛选**:
    通过语义一致性筛选，确保问题与VectorSQL查询高度匹配。
    ```bash
    python synthesis_nl/post_process_questions.py ...
    ```

### 第4步: 思维链合成 (`synthesis_cot`)

1.  **生成CoT合成提示**:
    ```bash
    python synthesis_cot/generate_cot_synthesis_prompts.py ...
    ```
2.  **调用LLM合成CoT**:
    ```bash
    python synthesis_cot/synthesize_cot.py ...
    ```
3.  **后处理与筛选**:
    通过执行验证和投票机制，选出最可靠的思维链。
    ```bash
    python synthesis_cot/post_process_cot.py ...
    ```

最终，您将得到一个包含“数据库、问题、VectorSQL、思维链”的完整数据集，可用于模型训练。

## 依赖安装

在运行任何脚本之前，请确保已安装所有必需的Python库。

```bash
pip install -r requirements.txt
```
