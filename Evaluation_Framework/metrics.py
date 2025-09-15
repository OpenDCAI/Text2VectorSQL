# Evaluation_Framework/metrics.py

import numpy as np

def _to_comparable_set(results: list[tuple]) -> set:
    """Converts a list of tuples (database rows) to a set of tuples for efficient comparison."""
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

def _extract_gt_relevant_columns(test_results: list[tuple], gt_column_count: int) -> list[tuple]:
    """
    Extract only the first N columns from test results to match GT structure.
    
    Args:
        test_results: List of tuples from test execution (may have extra columns)
        gt_column_count: Number of columns in ground truth
    
    Returns:
        List of tuples with only the relevant columns extracted
    """
    if not test_results or gt_column_count <= 0:
        return test_results
    
    # Extract only the first gt_column_count columns from test results
    aligned_results = []
    for test_row in test_results:
        if len(test_row) >= gt_column_count:
            # Extract the first gt_column_count columns
            aligned_row = tuple(test_row[:gt_column_count])
            aligned_results.append(aligned_row)
        else:
            # If test row has fewer columns than GT, keep as is (will likely fail match)
            aligned_results.append(test_row)
    
    return aligned_results

def _find_common_column_count(individual_gt_results: list[list[tuple]]) -> list[int]:
    """
    Find the column count(s) that appear most frequently across all ground truth results.
    Returns all column counts that have the highest occurrence frequency.
    
    Args:
        individual_gt_results: List of lists, each containing tuples from one GT execution
        
    Returns:
        List of column counts that appear most frequently across all GT results
    """
    if not individual_gt_results:
        return []
    
    # Filter out empty GT results
    non_empty_gts = [gt for gt in individual_gt_results if gt]
    
    if not non_empty_gts:
        return []
    
    # Get column counts from each non-empty GT
    gt_col_counts = []
    for gt_results in non_empty_gts:
        if gt_results:  # Double check non-empty
            gt_col_counts.append(len(gt_results[0]))
    
    if not gt_col_counts:
        return []
    
    # Count the frequency of each column count
    from collections import Counter
    col_count_frequency = Counter(gt_col_counts)
    
    # Find the maximum frequency
    max_frequency = max(col_count_frequency.values())
    
    # Return all column counts that have the maximum frequency
    most_frequent_col_counts = [col_count for col_count, freq in col_count_frequency.items() if freq == max_frequency]
    return most_frequent_col_counts

def _extract_union_relevant_columns(test_results: list[tuple], union_gt_data: list[tuple], individual_gt_results: list[list[tuple]] | None = None) -> list[tuple]:
    """
    Extract relevant columns for union-based metrics.
    Uses columns that co-occur in all GT results to determine how many columns to extract.
    
    Args:
        test_results: List of tuples from test execution
        union_gt_data: Combined list of tuples from all GT executions
        individual_gt_results: List of individual GT results (used to find co-occurring columns)
        
    Returns:
        List of tuples with only relevant columns
    """
    if not test_results or not union_gt_data:
        return test_results
    
    # Determine column count based on individual GTs if provided
    if individual_gt_results is not None:
        gt_column_count = _get_gt_column_count(individual_gt_results)
        return _extract_gt_relevant_columns(test_results, gt_column_count)
    else:
        # Fallback to original logic if individual GTs not provided
        from collections import Counter
        col_counts = [len(row) for row in union_gt_data]
        if col_counts:
            common_col_count = Counter(col_counts).most_common(1)[0][0]
            return _extract_gt_relevant_columns(test_results, common_col_count)
    
    return test_results

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

def calculate_exact_match_any_gt(test_results: list[tuple], individual_gt_results: list[list[tuple]]) -> float:
    """
    Calculate exact match against any individual ground truth.
    Returns 1.0 if test results exactly match any single ground truth, 0.0 otherwise.
    Extracts only GT-relevant columns from test results for comparison.
    
    Args:
        test_results: List of tuples from test execution
        individual_gt_results: List of lists, each containing tuples from one GT execution
    
    Returns:
        1.0 if exact match with any GT, 0.0 otherwise
    """
    if not test_results and all(not gt for gt in individual_gt_results):
        return 1.0
    
    gt_column_count = _get_gt_column_count(individual_gt_results)
    
    for gt_results in individual_gt_results:
        if not gt_results:
            # If this GT is empty and test is also empty, it's a match
            if not test_results:
                return 1.0
            continue
        
        # Extract only GT-relevant columns from test results
        aligned_test_results = _extract_gt_relevant_columns(test_results, gt_column_count)
        aligned_test_set = _to_comparable_set(aligned_test_results)
        gt_set = _to_comparable_set(gt_results)
        
        # Check if aligned test results exactly match this ground truth
        if aligned_test_set == gt_set:
            return 1.0
    
    return 0.0

def calculate_set_metrics(test_results: list[tuple], golden_data: list[tuple], individual_gt_results: list[list[tuple]] | None = None) -> dict:
    """
    Calculate various set-based metrics (precision, recall, F1) for test results compared to golden data.
    
    Args:
        test_results: List of tuples from test execution
        golden_data: List of tuples from golden/reference execution
        individual_gt_results: Optional list of individual GT results for column count analysis
        
    Returns:
        dict: Dictionary containing precision, recall, and F1 scores
    """
    # Get GT column count and align test results if individual GT results provided
    if individual_gt_results is not None:
        gt_column_count = _get_gt_column_count(individual_gt_results)
        aligned_test_results = _extract_gt_relevant_columns(test_results, gt_column_count)
    else:
        aligned_test_results = test_results
    
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

def calculate_average_precision(test_results: list[tuple], golden_data: list[tuple], individual_gt_results: list[list[tuple]] | None = None) -> float:
    """
    Calculates Average Precision (AP) for a single query.
    Extracts only GT-relevant columns from test results for comparison.
    """
    # Get GT column count and align test results if individual GT results provided
    if individual_gt_results is not None:
        gt_column_count = _get_gt_column_count(individual_gt_results)
        aligned_test_results = _extract_gt_relevant_columns(test_results, gt_column_count)
    else:
        aligned_test_results = test_results
    
    golden_set = _to_comparable_set(golden_data)

    if not aligned_test_results or not golden_set:
        return 0.0

    hits = 0
    sum_precisions = 0.0
    for i, result in enumerate(aligned_test_results):
        if result in golden_set:
            hits += 1
            precision_at_k = hits / (i + 1)
            sum_precisions += precision_at_k
            
    if not hits:
        return 0.0

    return sum_precisions / len(golden_set)

def calculate_reciprocal_rank(test_results: list[tuple], golden_data: list[tuple], individual_gt_results: list[list[tuple]] | None = None) -> float:
    """
    Calculates Reciprocal Rank (RR) for a single query.
    Extracts only GT-relevant columns from test results for comparison.
    """
    # Get GT column count and align test results if individual GT results provided
    if individual_gt_results is not None:
        gt_column_count = _get_gt_column_count(individual_gt_results)
        aligned_test_results = _extract_gt_relevant_columns(test_results, gt_column_count)
    else:
        aligned_test_results = test_results
    
    golden_set = _to_comparable_set(golden_data)

    if not aligned_test_results or not golden_data:
        return 0.0

    for i, result in enumerate(aligned_test_results):
        if result in golden_set:
            return 1.0 / (i + 1)
    
    return 0.0

def calculate_ndcg(test_results: list[tuple], golden_data: list[tuple], k: int, individual_gt_results: list[list[tuple]] | None = None) -> float:
    """
    Calculates Normalized Discounted Cumulative Gain (NDCG@k).
    Extracts only GT-relevant columns from test results for comparison.
    """
    # Get GT column count and align test results if individual GT results provided
    if individual_gt_results is not None:
        gt_column_count = _get_gt_column_count(individual_gt_results)
        aligned_test_results = _extract_gt_relevant_columns(test_results, gt_column_count)
    else:
        aligned_test_results = test_results
    
    # Build graded golden set (count occurrences)
    graded_golden_set = {}
    for row in golden_data:
        graded_golden_set[row] = graded_golden_set.get(row, 0) + 1
    
    # Calculate DCG@k
    dcg = 0.0
    for i, result in enumerate(aligned_test_results[:k]):
        relevance = graded_golden_set.get(result, 0)
        if relevance > 0:
            dcg += relevance / np.log2(i + 2) # log2(i+1+1)
            
    if dcg == 0:
        return 0.0

    # Calculate IDCG@k
    ideal_relevances = sorted(graded_golden_set.values(), reverse=True)
    idcg = 0.0
    for i, relevance in enumerate(ideal_relevances[:k]):
        idcg += relevance / np.log2(i + 2)
        
    if idcg == 0:
        return 0.0 # Should not happen if dcg > 0

    return dcg / idcg

def execution_accuracy(test_results: list[tuple], golden_data: list[tuple], individual_gt_results: list[list[tuple]] | None = None) -> float:
    """
    Calculate execution accuracy by comparing column count and set similarity.
    
    Args:
        test_results: List of tuples from test execution
        golden_data: List of tuples from golden/reference execution  
        individual_gt_results: Optional list of individual GT results for column count analysis
        
    Returns:
        float: Execution accuracy score (0.0 to 1.0)
    """
    # If no test results, accuracy is 0
    if not test_results:
        return 0.0
    
    # If golden data is empty but test results exist, accuracy is 0
    if not golden_data:
        return 0.0
    
    # Get column counts
    test_col_count = len(test_results[0]) if test_results else 0
    golden_col_count = len(golden_data[0]) if golden_data else 0
    
    # Find common column count(s) from individual GT results
    common_col_counts = _find_common_column_count(individual_gt_results) if individual_gt_results else [golden_col_count]
    
    # Check if test column count matches any of the common column counts
    if test_col_count not in common_col_counts:
        return 0.0
    
    # If column counts match, calculate set-based similarity
    set_metrics = calculate_set_metrics(test_results, golden_data, individual_gt_results)
    return set_metrics['f1']

def calculate_exact_match_any_gt_with_columns(test_results: list[tuple], test_columns: list[str], individual_gt_results: list[dict]) -> float:
    """
    Calculate exact match against any individual ground truth using column name matching.
    Returns 1.0 if test results exactly match any single ground truth, 0.0 otherwise.
    
    Args:
        test_results: List of tuples from test execution
        test_columns: Column names from test execution
        individual_gt_results: List of dicts containing GT execution results with columns
    
    Returns:
        1.0 if exact match with any GT, 0.0 otherwise
    """
    if not test_results and all(not gt.get('execution', {}).get('data', []) for gt in individual_gt_results):
        return 1.0
    
    for gt_result in individual_gt_results:
        gt_execution = gt_result.get('execution', {})
        if gt_execution.get('status') != 'success':
            continue
            
        gt_data = gt_execution.get('data', [])
        gt_columns = gt_execution.get('columns', [])
        
        if not gt_data:
            # If this GT is empty and test is also empty, it's a match
            if not test_results:
                return 1.0
            continue
        
        # Extract matching columns from test results
        aligned_test_results = _extract_columns_by_name(test_results, test_columns, gt_data, gt_columns)
        
        # Compare aligned results
        aligned_test_set = _to_comparable_set(aligned_test_results)
        gt_set = _to_comparable_set(gt_data)
        
        if aligned_test_set == gt_set:
            return 1.0
    
    return 0.0

def calculate_set_metrics_with_columns(test_results: list[tuple], test_columns: list[str], golden_data: list[tuple], golden_columns: list[str]) -> dict:
    """
    Calculate set-based metrics using column name matching.
    
    Args:
        test_results: List of tuples from test execution
        test_columns: Column names from test execution
        golden_data: List of tuples from golden/reference execution
        golden_columns: Column names from golden execution
        
    Returns:
        dict: Dictionary containing precision, recall, and F1 scores
    """
    # Extract matching columns from test results
    aligned_test_results = _extract_columns_by_name(test_results, test_columns, golden_data, golden_columns)
    
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
    # Extract matching columns from test results
    aligned_test_results = _extract_columns_by_name(test_results, test_columns, golden_data, golden_columns)
    golden_set = _to_comparable_set(golden_data)

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
    
    elif metric_type == 'ndcg':
        # Build graded golden set (count occurrences)
        graded_golden_set = {}
        for row in golden_data:
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
    
    return 0.0
