import math
from typing import List, Any, Set
import collections.abc

def calculate_accuracy(test_results: List[Any], gold_results: List[Any]) -> float:
    """
    计算精确匹配准确率。
    如果测试结果与黄金标准结果完全相同（顺序无关），则准确率为 1.0，否则为 0.0。
    """
    if not isinstance(test_results, list) or not isinstance(gold_results, list):
        return 0.0

    # 将字典列表转换为可哈希的元组集合以进行比较
    try:
        test_set = set(tuple(sorted(d.items())) for d in test_results)
        gold_set = set(tuple(sorted(d.items())) for d in gold_results)
        return 1.0 if test_set == gold_set else 0.0
    except (TypeError, AttributeError):
        # 回退到处理非字典项的列表
        try:
            # 对于可哈希但无序的项目
            return 1.0 if set(test_results) == set(gold_results) else 0.0
        except TypeError:  # 对于不可哈希的项目
            # 这是一个慢速回退
            return 1.0 if sorted(str(x) for x in test_results) == sorted(str(x) for x in gold_results) else 0.0

def _to_comparable_set(results: List[Any]) -> Set[Any]:
    """将结果列表转换为可比较的集合。"""
    try:
        # 假设是字典列表
        return set(tuple(sorted(d.items())) for d in results)
    except (TypeError, AttributeError):
        try:
            # 假设是可哈希项的列表
            return set(results)
        except TypeError:
            # 慢速回退
            return set(str(x) for x in results)

def calculate_mrr(test_results: List[Any], gold_results: List[Any], k: int = 100) -> float:
    """
    计算平均倒数排名 (Mean Reciprocal Rank) @ k。
    """
    gold_set = _to_comparable_set(gold_results)
    if not gold_set:
        return 0.0
        
    for i, res in enumerate(test_results[:k]):
        try:
            item = tuple(sorted(res.items()))
        except (TypeError, AttributeError):
            try:
                item = res if isinstance(res, collections.abc.Hashable) else str(res)
            except TypeError:
                item = str(res)

        if item in gold_set:
            return 1.0 / (i + 1)
    return 0.0

def calculate_map(test_results: List[Any], gold_results: List[Any], k: int = 100) -> float:
    """
    计算平均精度均值 (Mean Average Precision) @ k。
    """
    gold_set = _to_comparable_set(gold_results)
    if not gold_set:
        return 0.0

    hits = 0
    sum_precisions = 0.0
    for i, res in enumerate(test_results[:k]):
        try:
            item = tuple(sorted(res.items()))
        except (TypeError, AttributeError):
            try:
                item = res if isinstance(res, collections.abc.Hashable) else str(res)
            except TypeError:
                item = str(res)

        if item in gold_set:
            hits += 1
            precision_at_i = hits / (i + 1)
            sum_precisions += precision_at_i
            
    if not gold_set:
        return 0.0
        
    return sum_precisions / len(gold_set)

def calculate_ndcg(test_results: List[Any], gold_results: List[Any], k: int = 100) -> float:
    """
    计算归一化折扣累计收益 (normalized Discounted Cumulative Gain) @ k。
    """
    gold_set = _to_comparable_set(gold_results)
    
    # DCG
    dcg = 0.0
    for i, res in enumerate(test_results[:k]):
        try:
            item = tuple(sorted(res.items()))
        except (TypeError, AttributeError):
            try:
                item = res if isinstance(res, collections.abc.Hashable) else str(res)
            except TypeError:
                item = str(res)
        
        if item in gold_set:
            dcg += 1.0 / math.log2(i + 2)
            
    # IDCG
    idcg = 0.0
    num_relevant = len(gold_set)
    for i in range(min(num_relevant, k)):
        idcg += 1.0 / math.log2(i + 2)
        
    if idcg == 0:
        return 1.0 if dcg == 0 else 0.0
        
    return dcg / idcg
