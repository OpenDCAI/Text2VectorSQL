# Text2VectorSQL 评测框架技术文档

## 概述

Text2VectorSQL 评测框架是一个自动化的、端到端的解决方案，专门用于科学地评估 Text-to-Vector SQL 模型的性能。该框架采用**执行与测评分离**的设计，通过统一的配置文件驱动，分两个独立阶段完成整个评测流程。

### 核心特性

- **分离式架构**: 执行和测评完全分离，支持独立运行和调试
- **统一配置**: 通过单一YAML文件控制整个评测流程
- **灵活的数据库标识**: 支持 SQLite 文件路径和 PostgreSQL/ClickHouse 数据库名
- **多指标支持**: 支持基于集合和基于排名的多种评测指标
- **LLM 辅助评估**: 支持使用大语言模型评估 SQL 骨架和向量组件的正确性
- **灵活扩展**: 模块化设计，易于添加新指标和功能
- **详细报告**: 生成包含汇总和详细信息的综合JSON报告
- **中间结果可复用**: SQL执行结果可被多次评测

## 系统架构

### 核心组件

```
Evaluation_Framework/
├── run_eval_pipeline.py        # 主控制器 - 统一入口
├── sql_executor.py             # 阶段1: SQL执行器（集成服务管理）
├── evaluate_results.py         # 阶段2: 结果评测器
├── evaluation_config.yaml      # 统一配置文件
├── metrics.py                  # 指标计算模块
├── eval_queries.json           # 评测查询文件
├── ground_truth.json           # Ground Truth 查询文件
├── sql_execution_results.json  # 中间结果文件（自动生成）
├── evaluation_report.json      # 最终评测报告（自动生成）
└── DOCUMENTATION.md            # 技术文档
```

### 两阶段执行流程

```mermaid
graph LR
    A[评测查询文件] --> B[SQL执行器]
    C[Ground Truth文件] --> B
    D[配置文件] --> B
    B --> E[SQL执行结果]
    E --> F[结果评测器]
    D --> F
    F --> G[评测报告]
```

### 依赖模块

- **Execution_Engine/**: 负责SQL查询的执行和翻译
- **Embedding_Service/**: 提供文本向量化服务

## 使用指南

### 快速开始

#### 方法1: 运行完整流程（推荐）
```bash
cd Evaluation_Framework
python run_eval_pipeline.py --all
```
**注意**: 默认情况下会自动启动 Embedding Service，无需手动管理。

#### 方法2: 分阶段运行
```bash
# 阶段1: 执行SQL查询（自动启动服务）
python run_eval_pipeline.py --execute

# 阶段2: 计算评测指标
python run_eval_pipeline.py --evaluate
```

#### 方法3: 直接使用单独脚本
```bash
# 仅执行SQL（自动启动服务）
python sql_executor.py

# 仅评测结果
python evaluate_results.py
```

#### 禁用自动服务管理
如果您希望手动管理 Embedding Service，可以使用：
```bash
# 禁用自动服务管理
python run_eval_pipeline.py --all --no-service-management
```

### 1. 配置评测任务

编辑 `evaluation_config.yaml` 文件：

```yaml
# --- 数据库配置 ---
# 数据库文件的根目录（仅对 SQLite 有效）
# 对于 SQLite: db_identifier 是相对于此目录的相对路径
# 对于 PostgreSQL/ClickHouse: db_identifier 是数据库名，不使用 base_dir
base_dir: ./databases

# --- SQL执行配置 ---
engine_config_path: ../Execution_Engine/engine_config.yaml
db_type: 'sqlite'  # 可选: 'sqlite', 'postgresql', 'clickhouse'
eval_queries_file: eval_queries.json
ground_truth_file: ground_truth.json
execution_results_file: sql_execution_results.json

# --- 嵌入服务配置 ---
embedding_service:
  auto_manage: true
  host: "127.0.0.1"
  port: 8000
  models:
    - name: "all-MiniLM-L6-v2"
      hf_model_path: "sentence-transformers/all-MiniLM-L6-v2"
      trust_remote_code: true
      tensor_parallel_size: 1
      max_model_len: 256

# --- LLM-based Evaluation Configuration ---
# Configuration for using LLM to evaluate VectorSQL query correctness
llm_evaluation:
  # Enable or disable LLM-based evaluation
  enabled: true  # 设置为 true 启用 LLM 评估
  
  # LLM API configuration
  api_url: "http://123.129.219.111:3000/v1/chat/completions"
  api_key: "sk-your-api-key-here"  # 替换为您的 API Key
  model_name: "gpt-3.5-turbo"  # 可选: gpt-4, gpt-3.5-turbo 等
  
  # Request timeout in seconds
  timeout: 60
  
  # Optional: Whether to include LLM evaluation details in the report
  include_details: true

# --- 评测指标配置 ---
metrics:
  - name: 'f1_score'
  - name: 'precision'
  - name: 'recall'
  - name: 'exact_match'
  - name: 'map'
  - name: 'mrr'
  - name: 'ndcg'
    k: 10

# --- 输出配置 ---
evaluation_report_file: evaluation_report.json
```

#### 评测查询文件 (eval_queries.json)

**SQLite 示例：**
```json
[
  {
    "query_id": "q1",
    "db_identifier": "arxiv/arxiv.db",
    "sql": "SELECT title FROM papers WHERE author = 'John Doe'"
  }
]
```

**PostgreSQL/ClickHouse 示例：**
```json
[
  {
    "query_id": "q1",
    "db_identifier": "arxiv_database",
    "sql": "SELECT title FROM papers WHERE author = 'John Doe'"
  }
]
```

**重要说明**:
- **SQLite**: `db_identifier` 是相对于配置文件中 `base_dir` 的相对路径。如果 `base_dir` 设置为 `./databases`，则实际数据库路径为 `./databases/arxiv/arxiv.db`。如果 `db_identifier` 是绝对路径，则直接使用该路径。
- **PostgreSQL/ClickHouse**: `db_identifier` 是数据库名称，不受 `base_dir` 影响，直接作为数据库连接参数使用。

#### Ground Truth 文件 (ground_truth.json)

**SQLite 示例：**
```json
{
  "q1": {
    "db_identifier": "arxiv/arxiv.db",
    "sqls": [
      "SELECT title FROM papers WHERE author_name = 'John Doe'",
      "SELECT paper_title FROM papers WHERE main_author = 'John Doe'"
    ]
  }
}
```

**PostgreSQL/ClickHouse 示例：**
```json
{
  "q1": {
    "db_identifier": "arxiv_database",
    "sqls": [
      "SELECT title FROM papers WHERE author_name = 'John Doe'",
      "SELECT paper_title FROM papers WHERE main_author = 'John Doe'"
    ]
  }
}
```

## 代码执行流程详解

### 阶段1: SQL执行 (sql_executor.py)

#### 执行步骤
1. **配置加载**: 读取YAML配置和输入文件
2. **引擎初始化**: 初始化ExecutionEngine
3. **查询执行**: 
   - 遍历每个测试用例
   - 执行测试SQL查询
   - 执行对应的Ground Truth查询
4. **结果保存**: 将所有执行结果保存到中间文件

#### 输出格式 (sql_execution_results.json)
```json
[
  {
    "eval_case": {
      "query_id": "q1",
      "db_identifier": "arxiv/arxiv.db", 
      "sql": "SELECT ..."
    },
    "eval_execution": {
      "status": "success",
      "columns": ["title"],
      "data": [["Paper 1"], ["Paper 2"]],
      "row_count": 2
    },
    "ground_truth_executions": [
      {
        "sql": "SELECT title FROM papers WHERE author_name = 'John'",
        "execution": {
          "status": "success",
          "columns": ["title"],
          "data": [["Paper 1"], ["Paper 3"]],
          "row_count": 2
        }
      }
    ]
  }
]
```

### 阶段2: 结果评测 (evaluate_results.py)

#### 执行步骤
1. **配置和结果加载**: 读取配置文件和SQL执行结果
2. **数据提取**: 从执行结果中提取成功的数据
3. **黄金标准集构建**:
   - 并集黄金标准集（用于大部分指标）
   - 分级黄金标准集（用于NDCG）
4. **指标计算**: 根据配置计算各种评测指标
5. **结果聚合**: 计算所有测试用例的平均分数
6. **报告生成**: 生成详细的评测报告

## 指标计算详解

### 列对齐和去重策略

评测框架采用智能的列对齐策略来处理待测评SQL和Ground Truth之间的列差异，确保评估的准确性和公平性。

#### 列匹配策略
**问题**: 待测评SQL可能返回比Ground Truth更多的列（冗余列），如何公平比较？

**解决方案**: 基于列名的智能匹配
1. **列名提取**: 从SQL执行结果中获取列名信息
2. **列匹配**: 找到待测评结果中与Ground Truth列名相同的列
3. **列提取**: 只提取匹配的列，按Ground Truth的列顺序重新排列
4. **去重处理**: 对提取后的结果行进行去重，保持原始顺序
5. **回退机制**: 如果没有列名信息，自动回退到位置匹配

#### 工作示例
```python
# Ground Truth结果: 
data = [(1, 'Alice'), (2, 'Bob')]
columns = ['id', 'name']

# 待测评SQL结果:
test_data = [(1, 'Alice', 25, 'Engineer'), (2, 'Bob', 30, 'Manager'), (1, 'Alice', 25, 'Engineer')]
test_columns = ['id', 'name', 'age', 'job']

# 处理步骤:
# 1. 识别匹配列: ['id', 'name'] 
# 2. 提取对应数据: [(1, 'Alice'), (2, 'Bob'), (1, 'Alice')]
# 3. 去重处理: [(1, 'Alice'), (2, 'Bob')]
# 4. 进行评估比较
```

#### 优势
- **准确性**: 避免冗余列影响评估结果
- **公平性**: 只比较语义相关的列
- **鲁棒性**: 自动处理重复数据
- **智能回退**: 兼容没有列名信息的情况

### 基于集合的指标 (不考虑顺序)

这类指标将查询结果视为无序集合，使用**并集黄金标准集**进行评判。

#### 1. 精确率 (Precision)
**衡量内容**: 返回结果的"纯度" - 返回的结果中有多少是正确的

**计算公式**:
```
Precision = |Test_Set ∩ Golden_Set| / |Test_Set|
```

#### 2. 召回率 (Recall)
**衡量内容**: 查询的"完整性" - 所有应该找到的结果中，实际找到了多少

**计算公式**:
```
Recall = |Test_Set ∩ Golden_Set| / |Golden_Set|
```

#### 3. F1分数 (F1-Score)
**衡量内容**: 精确率和召回率的调和平均数，综合评估准确性

**计算公式**:
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

#### 4. 完全准确率 (Exact Match)
**衡量内容**: 测试结果是否与黄金标准完全一致

**计算公式**:
```
Exact_Match = 1 if Test_Set == Golden_Set else 0
```

### 基于排名的指标 (考虑顺序)

这类指标将测试结果视为有序列表，排名越靠前的正确结果价值越高。

#### 1. 平均精度均值 (Mean Average Precision, MAP)
**衡量内容**: 整个排序列表的平均表现，是搜索质量的黄金标准

**计算步骤**:
1. 遍历测试结果的有序列表
2. 每遇到一个正确结果，计算当前位置的精确率
3. 将所有精确率相加，除以相关项总数

**示例**: 
- 测试结果: [A, B, C, D, E]
- 黄金标准: {A, C, E}
- 计算过程:
  - 位置1: A正确, P@1 = 1/1 = 1.0
  - 位置3: C正确, P@3 = 2/3 = 0.67
  - 位置5: E正确, P@5 = 3/5 = 0.6
- AP = (1.0 + 0.67 + 0.6) / 3 = 0.76

#### 2. 平均倒数排名 (Mean Reciprocal Rank, MRR)
**衡量内容**: 多快能找到第一个正确答案

**计算公式**:
```
RR = 1/k (k是第一个正确结果的排名)
MRR = Average(RR_all_queries)
```

**适用场景**: 特别适合问答系统，用户通常只需要一个正确答案。

#### 3. 归一化折扣累计收益 (NDCG@k)
**衡量内容**: 最科学的排序质量评估指标，考虑相关性等级和位置折扣

**核心概念**:
- **相关性分级**: 根据结果在Ground Truth中的出现次数确定相关性等级
- **位置折扣**: 排名越靠后，价值越小
- **归一化**: 与理想排序比较，得出0-1之间的分数

**计算步骤**:

1. **计算DCG@k (Discounted Cumulative Gain)**:
```python
dcg = 0.0
for i, result in enumerate(test_results[:k]):
    relevance = graded_golden_set.get(result, 0)  # 相关性等级
    if relevance > 0:
        dcg += relevance / np.log2(i + 2)  # 位置折扣
```

2. **计算IDCG@k (Ideal DCG)**:
```python
ideal_relevances = sorted(graded_golden_set.values(), reverse=True)
idcg = 0.0
for i, relevance in enumerate(ideal_relevances[:k]):
    idcg += relevance / np.log2(i + 2)
```

3. **归一化**:
```python
ndcg = dcg / idcg if idcg > 0 else 0.0
```

### LLM 辅助评估（可选）

框架支持使用大语言模型（LLM）来评估 VectorSQL 查询的正确性，提供两个维度的评分：

#### 1. SQL 骨架正确性 (SQL Skeleton Accuracy)

评估 SQL 查询中非向量部分的正确性，包括：
- **SELECT**: 是否选择了正确的列和聚合函数
- **FROM/JOIN**: 是否使用了正确的表和连接条件
- **WHERE**: 非向量过滤条件是否正确
- **GROUP BY/HAVING**: 分组和聚合过滤逻辑是否正确
- **ORDER BY**: 非向量排序逻辑是否正确

**评分**: 1（完全正确）或 0（存在错误）

#### 2. 向量组件正确性 (Vector Component Accuracy)

评估语义搜索部分的正确性，包括：
- **向量列**: 是否使用了正确的向量列
- **向量操作**: 是否使用了正确的距离/相似度函数（如 `<->`、`L2Distance`）
- **查询文本**: 用于嵌入的文本是否与 Ground Truth **语义等价**（最关键的检查）
- **Top-K (LIMIT)**: 返回结果数量是否符合问题要求

**评分**: 1（完全正确）或 0（存在错误）

#### 3. 综合得分 (Overall Score)

综合得分 = (SQL 骨架得分 + 向量组件得分) / 2

#### LLM 评估的优势

- **语义理解**: 能够判断查询文本的语义等价性（如 "AI research" ≈ "artificial intelligence papers"）
- **结构分析**: 深入分析 SQL 查询的各个组成部分
- **详细反馈**: 提供每个组件的详细评估原因
- **补充传统指标**: 与执行结果指标互补，提供更全面的评估

#### 使用要求

要使用 LLM 评估功能，需要：
1. 在配置文件中启用 `llm_evaluation.enabled = true`
2. 配置正确的 API URL 和密钥
3. 评测数据中包含 `question` 字段
4. 数据库 schema 可以被自动提取（SQLite）或手动提供

## 多Ground Truth的消偏策略

### 问题背景
当一个查询对应多个Ground Truth时，如何公平地聚合它们的结果是关键问题。

### 解决方案

#### 1. 并集策略 (用于大部分指标)
```python
golden_set_union = set(tuple(row) for row in all_gt_results_data)
```

**优势**:
- 包容性强，承认"正确"有多种形式
- 为召回率提供真实的上限
- 避免因匹配特定GT而受惩罚

#### 2. 多级相关性策略 (用于NDCG)
```python
graded_golden_set = defaultdict(int)
for row in all_gt_results_data:
    graded_golden_set[tuple(row)] += 1
```

**优势**:
- 将出现频次转化为置信度
- 奖励高共识度的结果
- 充分利用多GT的信息

### 策略对比

| 方法 | 适用指标 | 优势 | 应用场景 |
|------|----------|------|----------|
| 并集 | F1, MAP, MRR | 全面性、公平性 | 大部分评测指标 |
| 交集 | 补充指标 | 高置信度 | 共识度分析 |
| 多级相关性 | NDCG | 科学量化置信度 | 排序质量评估 |

## 输出报告结构

### 最终报告格式 (evaluation_report.json)
```json
{
  "evaluation_summary": {
    "total_cases": 100,
    "successful_evaluations": 95,
    "evaluation_success_rate": 0.95,
    "average_f1_score": 0.85,
    "average_map": 0.78,
    "average_ndcg@10": 0.82,
    "average_llm_sql_skeleton_score": 0.92,
    "average_llm_vector_component_score": 0.88,
    "average_llm_overall_score": 0.90,
    "count_f1_score": 95,
    "count_llm_overall_score": 95
  },
  "individual_results": [
    {
      "eval_case": {
        "query_id": "q1",
        "db_identifier": "arxiv/arxiv.db",
        "question": "Find papers on AI...",
        "sql": "SELECT ..."
      },
      "execution_summary": {
        "eval_status": "success",
        "eval_row_count": 5,
        "ground_truth_summary": [...]
      },
      "scores": {
        "f1_score": 0.67,
        "precision": 0.5,
        "recall": 1.0,
        "map": 0.75,
        "ndcg@10": 0.8,
        "llm_sql_skeleton_score": 1.0,
        "llm_vector_component_score": 1.0,
        "llm_overall_score": 1.0
      },
      "llm_evaluation_details": {
        "sql_skeleton_evaluation": {
          "reasoning": "All SQL components are correct...",
          "select_correct": true,
          "from_join_correct": true,
          "where_correct": true,
          "groupby_having_correct": true,
          "orderby_correct": true,
          "score": 1
        },
        "vector_component_evaluation": {
          "reasoning": "Vector search is semantically correct...",
          "vector_column_correct": true,
          "vector_operation_correct": true,
          "query_text_semantically_correct": true,
          "top_k_correct": true,
          "score": 1
        }
      }
    }
  ]
}
```

### 报告解读

#### Evaluation Summary部分
- **total_cases**: 总测试用例数
- **successful_evaluations**: 成功评测的用例数  
- **evaluation_success_rate**: 评测成功率
- **average_***: 各指标的平均分数
- **count_***: 参与计算的用例数量

#### Individual Results部分
- **eval_case**: 原始测试用例信息
- **execution_summary**: 执行状态摘要
- **scores**: 该用例的各项指标得分

## 高级功能

### 1. 分离式运行的优势

#### 独立调试
```bash
# 只重新运行评测部分，无需重复执行SQL
python run_eval_pipeline.py --evaluate
```

#### 多次评测
```bash
# 修改指标配置后，基于同一组执行结果重新评测
python evaluate_results.py --config new_metrics_config.yaml
```

#### 并行处理
```bash
# 可以在不同机器上分别运行执行和评测
# 机器A: 执行SQL查询
python sql_executor.py

# 机器B: 评测结果
python evaluate_results.py --execution-results results_from_machine_a.json
```

### 2. 错误处理与容错

#### 执行阶段错误处理
- SQL执行失败的用例会被标记但不影响其他用例
- 提供详细的错误信息用于调试
- 支持部分成功的批量处理

#### 评测阶段错误处理  
- 跳过执行失败的用例
- 提供执行状态摘要
- 计算有效用例的统计信息

### 3. 可扩展性

#### 添加新指标
1. 在 `metrics.py` 中实现计算函数
2. 在 `evaluate_results.py` 中添加调用逻辑
3. 在配置文件中启用新指标

#### 自定义聚合策略
可以通过修改Golden Set构建逻辑来实现不同的消偏策略：
- 加权并集
- 投票机制  
- 置信度阈值过滤

### 4. 结果汇总工具

框架提供了一个强大的结果汇总工具，可以将多个评估报告合并到一个 CSV 文件中，方便对比分析。

#### 使用场景
- 对比不同模型的性能
- 跟踪模型在不同时间点的进化
- 分析不同配置的影响
- 生成实验报告和可视化数据

#### 基本用法

**1. 汇总指定的多个文件**：
```bash
python aggregate_results.py --input report1.json report2.json report3.json --output summary.csv
```

**2. 汇总目录中的所有 JSON 文件**：
```bash
python aggregate_results.py --input-dir ./reports --output summary.csv
```

**3. 使用通配符模式**：
```bash
python aggregate_results.py --input model_*_report.json --output comparison.csv
```

**4. 按指标排序**：
```bash
# 按 F1 分数降序排列
python aggregate_results.py --input *.json --output summary.csv --sort-by average_f1_score

# 按 LLM 综合得分排序
python aggregate_results.py --input *.json --output summary.csv --sort-by average_llm_overall_score
```

**5. 静默模式（仅输出错误）**：
```bash
python aggregate_results.py --input-dir ./reports --output summary.csv --quiet
```

#### 输出格式

**CSV 文件示例**：
```csv
report_name,total_cases,successful_evaluations,evaluation_success_rate,average_f1_score,average_precision,average_recall,average_map,average_mrr,average_ndcg@10,average_llm_sql_skeleton_score,average_llm_vector_component_score,average_llm_overall_score
model_v1_report,100,95,0.95,0.85,0.82,0.88,0.78,0.72,0.82,0.92,0.88,0.90
model_v2_report,100,98,0.98,0.88,0.85,0.91,0.82,0.75,0.85,0.95,0.91,0.93
baseline_report,100,90,0.90,0.75,0.70,0.80,0.68,0.65,0.72,0.85,0.80,0.82
```

**控制台输出示例**：
```
Found 3 JSON file(s) to process
Processing: model_v1_report.json
Processing: model_v2_report.json
Processing: baseline_report.json

✅ Successfully wrote aggregated results to: summary.csv
   Total reports: 3
   Total metrics: 15

================================================================================
AGGREGATED EVALUATION SUMMARY
================================================================================
Report Name                    Total Cases  Success Rate      F1 Score     Precision        Recall
--------------------------------------------------------------------------------
model_v2_report                         100        0.9800        0.8800        0.8500        0.9100
model_v1_report                         100        0.9500        0.8500        0.8200        0.8800
baseline_report                         100        0.9000        0.7500        0.7000        0.8000
================================================================================

✅ Aggregation complete! Results saved to: summary.csv
```

#### 高级选项

| 选项 | 说明 |
|------|------|
| `--input`, `-i` | 指定一个或多个 JSON 文件 |
| `--input-dir`, `-d` | 指定包含 JSON 文件的目录 |
| `--output`, `-o` | 输出 CSV 文件路径（默认: aggregated_results.csv） |
| `--sort-by`, `-s` | 按指定指标排序（降序） |
| `--no-table` | 禁用控制台表格输出 |
| `--quiet`, `-q` | 静默模式，仅显示错误 |

#### 实际应用示例

**场景 1：模型版本对比**
```bash
# 对比不同版本的模型性能
python aggregate_results.py \
  --input gpt3.5_report.json gpt4_report.json claude_report.json \
  --output model_comparison.csv \
  --sort-by average_f1_score
```

**场景 2：实验结果收集**
```bash
# 收集所有实验结果
python aggregate_results.py \
  --input-dir ./experiments/2025-01 \
  --output january_experiments.csv
```

**场景 3：批量分析**
```bash
# 分析特定配置的所有运行结果
python aggregate_results.py \
  --input config_a_run*.json \
  --output config_a_summary.csv \
  --sort-by average_llm_overall_score
```

#### 与其他工具集成

**Excel 分析**：
生成的 CSV 文件可以直接在 Excel 中打开，进行进一步的数据透视分析、图表绘制等。

**Python 数据分析**：
```python
import pandas as pd

# 读取汇总结果
df = pd.read_csv('summary.csv')

# 计算统计信息
print(df[['average_f1_score', 'average_precision', 'average_recall']].describe())

# 绘制对比图
df.plot(x='report_name', y=['average_f1_score', 'average_precision', 'average_recall'], kind='bar')
```

**自动化流程**：
```bash
#!/bin/bash
# 自动化评估和汇总脚本

# 运行多个模型的评估
for model in gpt3.5 gpt4 claude; do
    python run_eval_pipeline.py --all --config ${model}_config.yaml
    mv evaluation_report.json ${model}_report.json
done

# 汇总所有结果
python aggregate_results.py --input *_report.json --output final_comparison.csv
```

### 5. 性能优化建议

#### 执行优化
- 合理设置数据库连接超时
- 对大量查询进行批次处理
- 监控SQL执行性能

#### 评测优化
- 对于大数据集，考虑采样评测
- 缓存中间计算结果
- 并行计算不同指标

## 故障排除

### 常见问题

#### 1. SQL执行失败
- 检查数据库连接配置
- 验证SQL语法正确性
- 确认数据库中存在相关表和数据

#### 2. 指标计算异常
- 检查Ground Truth文件格式
- 验证测试用例与GT的query_id匹配
- 确认执行结果数据格式正确

#### 3. 配置文件错误
- 验证YAML语法正确性
- 检查文件路径是否存在
- 确认指标配置格式正确

### 调试技巧

#### 查看中间结果
```bash
# 检查SQL执行结果
cat sql_execution_results.json | python -m json.tool

# 检查特定用例的执行情况
jq '.[] | select(.eval_case.query_id == "q1")' sql_execution_results.json
```

#### 逐步调试
```bash
# 单独运行各阶段
python sql_executor.py --config debug_config.yaml
python evaluate_results.py --config debug_config.yaml
```

---

*本文档涵盖了Text2VectorSQL评测框架的完整技术细节。如需了解更多实现细节，请参考源代码注释和相关学术文献。*