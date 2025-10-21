# Evaluation_Framework/metrics.py

import numpy as np

def _to_comparable_set(results: list[tuple] | list[list]) -> set:
    """
    将数据库行列表转换为可比较的集合。
    支持 list[tuple] 和 list[list] 两种格式。
    
    Args:
        results: 可能是 list[tuple] 或 list[list] 的数据
        
    Returns:
        set: 转换后的集合（元素为 tuple）
    """
    if not results:
        return set()
    
    # 检查第一个元素的类型
    if results and isinstance(results[0], list):
        # 如果是 list[list]，转换为 tuple 后再创建集合
        return set(tuple(row) for row in results)
    
    # 已经是 list[tuple]，直接创建集合
    return set(results)

def _get_gt_column_count(individual_gt_results: list[list[tuple]]) -> int:
    """
    Get the standard column count from ground truth results.
    Assumes all ground truth results have consistent column counts.
    
    Args:
        individual_gt_results: List of lists, each containing tuples from one GT execution
        
    Returns:
        int: The column count from ground truth (0 if no valid GT found)
    """
    if not individual_gt_results:
        return 0
    
    # Find the first non-empty GT result to get column count
    for gt_results in individual_gt_results:
        if gt_results and len(gt_results) > 0:
            return len(gt_results[0])
    
    return 0





def _extract_columns_by_name(test_results: list[tuple], test_columns: list[str], gt_results: list[tuple], gt_columns: list[str]) -> list[tuple]:
    """
    从测试结果中提取与ground truth列名匹配的列，并重新排序以匹配GT的列顺序。
    提取后对结果进行去重。
    
    Args:
        test_results: 测试SQL的执行结果数据
        test_columns: 测试SQL的列名列表
        gt_results: Ground truth的执行结果数据
        gt_columns: Ground truth的列名列表
    
    Returns:
        去重后只包含匹配列的测试结果
    """
    if not test_results or not gt_results or not test_columns or not gt_columns:
        return test_results
    
    # 找到测试结果中与GT列名匹配的列索引
    matched_indices = []
    for gt_col in gt_columns:
        try:
            test_col_idx = test_columns.index(gt_col)
            matched_indices.append(test_col_idx)
        except ValueError:
            # GT中的列在测试结果中不存在，跳过这个列
            continue
    
    if not matched_indices:
        # 没有匹配的列，返回空结果
        return []
    
    # 提取匹配的列数据
    aligned_results = []
    for test_row in test_results:
        if len(test_row) > max(matched_indices):  # 确保行数据足够长
            aligned_row = tuple(test_row[i] for i in matched_indices)
            aligned_results.append(aligned_row)
    
    # 对提取后的结果进行去重，保持原始顺序
    deduplicated_results = []
    seen_rows = set()
    for row in aligned_results:
        if row not in seen_rows:
            deduplicated_results.append(row)
            seen_rows.add(row)
    
    return deduplicated_results

def _extract_columns_by_name_without_dedup(test_results: list[tuple], test_columns: list[str], gt_results: list[tuple], gt_columns: list[str]) -> list[tuple]:
    """
    从测试结果中提取与ground truth列名匹配的列，并重新排序以匹配GT的列顺序。
    提取后不去重。
    
    Args:
        test_results: 测试SQL的执行结果数据
        test_columns: 测试SQL的列名列表
        gt_results: Ground truth的执行结果数据
        gt_columns: Ground truth的列名列表
    
    Returns:
        不去重的匹配列的测试结果
    """
    if not test_results or not gt_results or not test_columns or not gt_columns:
        return test_results
    
    # 找到测试结果中与GT列名匹配的列索引
    matched_indices = []
    for gt_col in gt_columns:
        try:
            test_col_idx = test_columns.index(gt_col)
            matched_indices.append(test_col_idx)
        except ValueError:
            # GT中的列在测试结果中不存在，跳过这个列
            continue
    
    if not matched_indices:
        # 没有匹配的列，返回空结果
        return []
    
    # 提取匹配的列数据
    aligned_results = []
    for test_row in test_results:
        if len(test_row) > max(matched_indices):  # 确保行数据足够长
            aligned_row = tuple(test_row[i] for i in matched_indices)
            aligned_results.append(aligned_row)
    
    return aligned_results

def _get_gt_columns(individual_gt_results: list[dict]) -> list[str]:
    """
    获取ground truth的标准列名列表。
    假定所有ground truth都有一致的列结构，使用第一个成功的GT作为标准。
    
    Args:
        individual_gt_results: 包含执行结果的字典列表，每个字典包含'execution'字段
        
    Returns:
        列名列表，如果没有找到有效的GT则返回空列表
    """
    for gt_result in individual_gt_results:
        gt_execution = gt_result.get('execution', {})
        if gt_execution.get('status') == 'success' and gt_execution.get('columns'):
            return gt_execution['columns']
    return []









def calculate_exact_match_any_gt_with_columns(test_results: list[tuple], test_columns: list[str], individual_gt_results: list[dict]) -> float:
    """
    Calculate exact match against any individual ground truth using column name matching.
    Returns 1.0 if test results exactly match any single ground truth, 0.0 otherwise.
    
    边界情况处理：
    - 所有 GT 和 test 都为空：返回 1.0（完全匹配）
    - 只有所有 GT 为空：返回 0.0
    - 只有 test 为空：检查是否有 GT 也为空，如果有则返回 1.0
    
    Args:
        test_results: List of tuples from test execution
        test_columns: Column names from test execution
        individual_gt_results: List of dicts containing GT execution results with columns
    
    Returns:
        1.0 if exact match with any GT, 0.0 otherwise
    """
    # 边界情况：test 为空且所有 GT 也为空
    if not test_results and all(not gt.get('execution', {}).get('data', []) for gt in individual_gt_results):
        return 1.0
    
    for gt_result in individual_gt_results:
        gt_execution = gt_result.get('execution', {})
        if gt_execution.get('status') != 'success':
            continue
            
        gt_data = gt_execution.get('data', [])
        gt_columns = gt_execution.get('columns', [])
        gt_columns=[col for col in gt_columns if col !='distance']
        
        if not gt_data:
            # If this GT is empty and test is also empty, it's a match
            if not test_results:
                return 1.0
            continue
        
        # 如果 test 为空但 GT 不为空，继续检查其他 GT
        if not test_results:
            continue
        
        # Extract matching columns from test results
        aligned_test_results = _extract_columns_by_name(test_results, test_columns, gt_data, gt_columns)
        gt_data = _extract_columns_by_name(gt_data, gt_columns, gt_data, gt_columns)
        
        # Compare aligned results
        aligned_test_set = _to_comparable_set(aligned_test_results)
        gt_set = _to_comparable_set(gt_data)
        
        if aligned_test_set == gt_set:
            return 1.0
    
    return 0.0

def calculate_set_metrics_with_columns(test_results: list[tuple], test_columns: list[str], golden_data: list[tuple], golden_columns: list[str]) -> dict:
    """
    Calculate set-based metrics using column name matching.
    
    边界情况处理：
    - 两者都为空：precision=1.0, recall=1.0, F1=1.0（完全匹配）
    - 只有 golden_data 为空：precision=0.0, recall=1.0, F1=0.0（假阳性）
    - 只有 test_results 为空：precision=1.0, recall=0.0, F1=0.0（假阴性）
    
    Args:
        test_results: List of tuples from test execution
        test_columns: Column names from test execution
        golden_data: List of tuples from golden/reference execution
        golden_columns: Column names from golden execution
        
    Returns:
        dict: Dictionary containing precision, recall, and F1 scores
    """
    # 边界情况 1：两者都为空 - 完全匹配
    if not test_results and not golden_data:
        return {
            'precision': 1.0,
            'recall': 1.0,
            'f1': 1.0
        }
    
    # 边界情况 2：只有 golden_data 为空，test_results 不为空 - 假阳性
    if not golden_data and test_results:
        return {
            'precision': 0.0,
            'recall': 1.0,
            'f1': 0.0
        }
    
    # 边界情况 3：只有 test_results 为空，golden_data 不为空 - 假阴性
    if not test_results and golden_data:
        return {
            'precision': 1.0,
            'recall': 0.0,
            'f1': 0.0
        }
    
    # Extract matching columns from test results
    aligned_test_results = _extract_columns_by_name(test_results, test_columns, golden_data, golden_columns)
    golden_data = _extract_columns_by_name(golden_data, golden_columns, golden_data, golden_columns)
    
    # Convert to sets for set operations
    test_set = set(aligned_test_results) if aligned_test_results else set()
    golden_set = set(golden_data) if golden_data else set()
    
    # Calculate intersection (true positives)
    intersection = test_set & golden_set
    
    # Calculate precision (what fraction of retrieved items are relevant)
    precision = len(intersection) / len(test_set) if len(test_set) > 0 else 0.0
    
    # Calculate recall (what fraction of relevant items are retrieved)
    recall = len(intersection) / len(golden_set) if len(golden_set) > 0 else 0.0
    
    # Calculate F1 score (harmonic mean of precision and recall)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def calculate_ranking_metrics_with_columns(test_results: list[tuple], test_columns: list[str], golden_data: list[tuple], golden_columns: list[str], metric_type: str, k: int | None = None) -> float:
    """
    Calculate ranking-based metrics using column name matching.
    
    边界情况处理：
    - 两者都为空：返回 1.0（完全匹配）
    - 只有 golden_data 为空：返回 0.0（没有相关项）
    - 只有 test_results 为空：返回 0.0（没有检索到任何项）
    
    Args:
        test_results: List of tuples from test execution
        test_columns: Column names from test execution
        golden_data: List of tuples from golden execution
        golden_columns: Column names from golden execution
        metric_type: Type of metric ('map', 'mrr', 'ndcg')
        k: Parameter for NDCG@k
        
    Returns:
        float: Calculated metric value
    """
    # 边界情况 1：两者都为空 - 完全匹配
    if not test_results and not golden_data:
        return 1.0
    
    # 边界情况 2：只有 golden_data 为空
    if not golden_data:
        return 0.0
    
    # 边界情况 3：只有 test_results 为空
    if not test_results:
        return 0.0
    
    # Extract matching columns from test results
    aligned_test_results = _extract_columns_by_name(test_results, test_columns, golden_data, golden_columns)
    
    # For NDCG, we need to preserve the original golden_data before deduplication
    # to correctly count occurrences for graded relevance
    if metric_type == 'ndcg':
        # 先提取列对齐（不去重），用于统计原始出现次数
        original_golden_data = _extract_columns_by_name_without_dedup(golden_data, golden_columns, golden_data, golden_columns)
        
        # Build graded golden set from original data (count occurrences)
        graded_golden_set = {}
        for row in original_golden_data:
            graded_golden_set[row] = graded_golden_set.get(row, 0) + 1
        
        # Calculate DCG@k
        dcg = 0.0
        for i, result in enumerate(aligned_test_results[:k]):
            relevance = graded_golden_set.get(result, 0)
            if relevance > 0:
                dcg += relevance / np.log2(i + 2)
                
        if dcg == 0:
            return 0.0

        # Calculate IDCG@k
        ideal_relevances = sorted(graded_golden_set.values(), reverse=True)
        idcg = 0.0
        for i, relevance in enumerate(ideal_relevances[:k]):
            idcg += relevance / np.log2(i + 2)
            
        return dcg / idcg if idcg > 0 else 0.0
    
    # For MAP and MRR, use deduplicated golden data (existing behavior)
    golden_data_dedup = _extract_columns_by_name(golden_data, golden_columns, golden_data, golden_columns)
    golden_set = _to_comparable_set(golden_data_dedup)

    if not aligned_test_results or not golden_set:
        return 0.0

    if metric_type == 'map':
        hits = 0
        sum_precisions = 0.0
        for i, result in enumerate(aligned_test_results):
            if result in golden_set:
                hits += 1
                precision_at_k = hits / (i + 1)
                sum_precisions += precision_at_k
        return sum_precisions / len(golden_set) if hits > 0 else 0.0
    
    elif metric_type == 'mrr':
        for i, result in enumerate(aligned_test_results):
            if result in golden_set:
                return 1.0 / (i + 1)
        return 0.0
    
    return 0.0

# ==============================================================================
# LLM-based VectorSQL Query Evaluation
# ==============================================================================

from pyparsing import col
import requests
import json
import re
from typing import Dict, Any, Optional

def extract_and_parse_json(model_output_text: str) -> Dict[str, Any]:
    """
    从可能包含无关文本或 Markdown 标记的字符串中提取并解析 JSON 对象。
    
    Args:
        model_output_text: 模型返回的原始字符串内容
        
    Returns:
        解析后的 Python 字典
        
    Raises:
        ValueError: 如果在文本中找不到有效的 JSON 对象
        json.JSONDecodeError: 如果找到的字符串不是有效的 JSON
    """
    # 使用正则表达式查找从 '{' 开始到 '}' 结束的最大可能块
    json_match = re.search(r'\{.*\}', model_output_text, re.DOTALL)
    
    if not json_match:
        raise ValueError("在模型的输出中未能找到有效的 JSON 对象。")
        
    json_string = json_match.group(0)
    
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        print("❌ 解析提取出的 JSON 字符串时失败。")
        print("提取出的内容:", json_string)
        raise e


def evaluate_vectorsql_with_llm(
    nl_question: str,
    db_schema: str,
    ground_truth_query: str,
    predicted_query: str,
    api_config: Dict[str, str],
    timeout: int = 60
) -> Optional[Dict[str, Any]]:
    """
    使用 LLM 评估 VectorSQL 查询的正确性。
    
    Args:
        nl_question: 自然语言问题
        db_schema: 数据库 schema (DDL)
        ground_truth_query: Ground truth SQL 查询
        predicted_query: 待评估的预测 SQL 查询
        api_config: API 配置，包含 'url', 'api_key', 'model'
        timeout: API 请求超时时间（秒）
        
    Returns:
        包含评估结果的字典，如果评估失败则返回 None
        结构：{
            "sql_skeleton_evaluation": {...},
            "vector_component_evaluation": {...}
        }
    """
    
    # 构建评估 prompt
    prompt = """
You are an expert SQL analyst and data scientist, specializing in evaluating the correctness of complex database queries that combine structured SQL predicates with semantic vector search. Your task is to meticulously evaluate a predicted VectorSQL query against a ground-truth query, considering the user's natural language question and the database schema.

Your evaluation must be decomposed into two independent parts: **SQL Skeleton Accuracy** and **Vector Component Accuracy**.

**1. SQL Skeleton Accuracy (`ACC_SQL`)**:
Evaluate the correctness of the standard, non-vector parts of the SQL query. The final score is 1 only if **ALL** structural components are logically equivalent to the ground truth, otherwise it is 0.
-   **SELECT**: Are the correct columns and aggregations selected?
-   **FROM/JOIN**: Are the correct tables and join conditions used?
-   **WHERE**: Are all non-vector filtering conditions correct?
-   **GROUP BY/HAVING**: Is the grouping and aggregation filtering logic correct?
-   **ORDER BY**: Is the non-vector sorting logic correct?

**2. Vector Component Accuracy (`ACC_Vec`)**:
Evaluate the correctness of the semantic search part of the query. The final score is 1 only if the vector search is semantically correct and will retrieve the intended results, otherwise it is 0.
-   **Vector Column**: Is the correct vector column used for the search?
-   **Vector Operation**: Is the correct distance/similarity function used (e.g., `<->`, `L2Distance`)?
-   **Query Text**: Is the text used for embedding **semantically equivalent** to the one in the ground truth? For example, "AI research" is equivalent to "papers on artificial intelligence". This is the most critical check.
-   **Top-K (LIMIT)**: Is the number of results to retrieve correct as per the user's question?

Here is the information for your evaluation:

**Natural Language Question:**

{nl_question}

**Database Schema:**

```sql
{db_schema}
```

**Ground Truth VectorSQL Query:**

```sql
{ground_truth_query}
```

**Predicted VectorSQL Query to Evaluate:**

```sql
{predicted_query}
```

Based on your analysis, provide the evaluation in a single JSON object. Do not include any text or explanations outside of the JSON object.

**JSON Output Format:**

```json
{{
  "sql_skeleton_evaluation": {{
    "reasoning": "Provide a brief explanation for the SQL skeleton score.",
    "select_correct": <True_or_False>,
    "from_join_correct": <True_or_False>,
    "where_correct": <True_or_False>,
    "groupby_having_correct": <True_or_False>,
    "orderby_correct": <True_or_False>,
    "score": <1_or_0>
  }},
  "vector_component_evaluation": {{
    "reasoning": "Provide a brief explanation for the vector component score, focusing on the semantic similarity of the query text.",
    "vector_column_correct": <True_or_False>,
    "vector_operation_correct": <True_or_False>,
    "query_text_semantically_correct": <True_or_False>,
    "top_k_correct": <True_or_False>,
    "score": <1_or_0>
  }}
}}
```
"""
    
    # 填充 prompt
    final_prompt = prompt.format(
        nl_question=nl_question,
        db_schema=db_schema,
        ground_truth_query=ground_truth_query,
        predicted_query=predicted_query
    )
    
    # 准备 API 请求
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_config['api_key']}"
    }
    
    payload = {
        "model": api_config['model'],
        "messages": [
            {"role": "user", "content": final_prompt}
        ],
        "temperature": 0.0,
        "stream": False
    }
    
    try:
        # 发送 API 请求
        response = requests.post(
            api_config['url'],
            headers=headers,
            json=payload,
            timeout=timeout
        )
        response.raise_for_status()
        api_response_data = response.json()
        
        # 提取响应内容
        message_content = api_response_data['choices'][0]['message']['content']
        
        # 解析 JSON 结果
        evaluation_result = extract_and_parse_json(message_content)
        
        return evaluation_result
        
    except requests.exceptions.RequestException as e:
        print(f"❌ LLM API 请求错误: {e}")
        return None
    except (KeyError, IndexError) as e:
        print(f"❌ 解析 API 响应失败: {e}")
        return None
    except ValueError as e:
        print(f"❌ {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ JSON 解析失败: {e}")
        return None


def calculate_llm_based_scores(evaluation_result: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """
    从 LLM 评估结果中提取分数。
    
    Args:
        evaluation_result: LLM 返回的评估结果字典
        
    Returns:
        包含各项分数的字典
    """
    if not evaluation_result:
        return {
            'llm_sql_skeleton_score': 0.0,
            'llm_vector_component_score': 0.0,
            'llm_overall_score': 0.0
        }
    
    try:
        sql_score = evaluation_result.get('sql_skeleton_evaluation', {}).get('score', 0)
        vec_score = evaluation_result.get('vector_component_evaluation', {}).get('score', 0)
        
        return {
            'llm_sql_skeleton_score': float(sql_score),
            'llm_vector_component_score': float(vec_score),
            'llm_overall_score': (float(sql_score) + float(vec_score)) / 2.0
        }
    except (KeyError, ValueError, TypeError) as e:
        print(f"❌ 提取 LLM 评分失败: {e}")
        return {
            'llm_sql_skeleton_score': 0.0,
            'llm_vector_component_score': 0.0,
            'llm_overall_score': 0.0
        }
