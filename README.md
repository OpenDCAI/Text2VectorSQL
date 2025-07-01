# VectorSQL Benchmark Tool

## 简介
VectorSQL Benchmark Tool 是一个专为测试和评估基于 Text-to-Vector SQL 模型生成的 VectorSQL 查询正确性而设计的工具。它支持多种性能和准确性指标的计算，包括准确率、平均倒数排名 (MRR)、平均精度均值 (MAP) 和归一化折扣累计收益 (NDCG)。此外，该工具目前支持 SQLite-Vec 语法。

## 配置文件
工具使用 YAML 格式的配置文件来定义测试环境、查询和评估指标。以下是配置文件的主要部分：

### 数据库配置
```yaml
database:
  name: 'test'
  path: 'data/vector_database_2.sqlite'
```
- `name`: 数据库名称。
- `path`: 数据库文件路径。

### 嵌入模型配置
嵌入模型用于计算文本向量嵌入。模型需转换为 GGUF 格式，并在基准测试过程中用于所有查询的嵌入计算。
将模型转换为GGUF，参见[llama.cpp](https://github.com/ggml-org/llama.cpp)
```yaml
embedding:
  name: 'all-MiniLM-L6-v2'
  model_path: 'embed-model/all-MiniLM-L6-v2.e4ce9877.q8_0.gguf'
```
- `name`: 嵌入模型名称。
- `model_path`: 嵌入模型文件路径。

### 查询配置
```yaml
queries:
  - name: 'gold_standard'
    path: 'queries/sample_gold.sql'
    is_gold: true
  - name: 'test'
    path: 'queries/sample_test.sql'
    is_gold: false
```
- `name`: 查询名称。
- `path`: 查询文件路径。
- `is_gold`: 是否为黄金标准查询。

在基准测试过程中，工具会执行 `gold_standard` 查询和测试集中的其他查询，并将测试结果与黄金标准结果进行比较以计算各项指标。

### 评估指标
```yaml
metrics:
  - execution_time
  - result_count
  - accuracy
  - name: mrr
    k: 100
  - name: map
    k: 100
  - name: ndcg
    k: 100
```
- 支持的基本指标包括 `execution_time`（执行时间）、`result_count`（结果数量）和 `accuracy`（准确率）。
- 高级指标如 `mrr`（平均倒数排名）、`map`（平均精度均值）和 `ndcg`（归一化折扣累计收益）可以通过 `k` 参数指定计算范围。

### 输出配置
```yaml
output:
  dir: 'results/'
  format: 'json'
```
- `dir`: 输出结果保存的目录。
- `format`: 输出结果的格式（如 `json`）。

## 使用方法
1. **安装依赖**:
   确保安装了工具所需的依赖项，例如 `PyYAML` 和 `sqlite-vec`。
   ```bash
   pip install -e .
   ```

2. **运行基准测试**:
   使用以下命令运行基准测试工具：
   ```bash
   vectorsql-benchmark --config config/benchmark_config.yaml
   ```

3. **查看结果**:
   基准测试结果将保存到配置文件中指定的输出目录，格式为 JSON。

## 示例
以下是一个运行基准测试的完整示例：
```bash
vectorsql-benchmark --config config/benchmark_config.yaml
```
运行后，结果将保存到 `results/` 目录中，文件格式为 JSON。

## 添加新指标

### 步骤1：在 `core/metrics.py` 中实现指标函数

```python
def calculate_your_metric(test_results: List[Any], gold_results: List[Any], **kwargs) -> float:
    """
    计算您的自定义指标。
    
    Args:
        test_results: 测试结果列表
        gold_results: 黄金标准结果列表
        **kwargs: 其他参数（如 k 值）
    
    Returns:
        指标计算结果
    """
    # 实现您的指标计算逻辑
    pass
```

### 步骤2：在配置文件中声明指标

#### 简单指标（无参数）
```yaml
metrics:
  - your_metric_name
```

#### 带参数的指标
```yaml
metrics:
  - name: your_metric_name
    k: 100
    custom_param: value
```

### 步骤3：添加指标评估
在core/benchmark_runner.py中计算meric的部分（222行）添加对应meric计算。



## 贡献
欢迎提交问题和贡献代码！请确保遵循项目的贡献指南。

## 许可证
本项目遵循 MIT 许可证。
