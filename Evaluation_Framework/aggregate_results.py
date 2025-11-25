#!/usr/bin/env python3
"""
aggregate_results.py - Aggregate Multiple Evaluation Reports

This script aggregates evaluation metrics from multiple evaluation report JSON files
and exports them to a structured CSV file for easy comparison and analysis.

Directory structure: results_dir/db_type/dataset_type/evaluation_report_{model_name}.json

Example:
    results/
        sqlite/
            bird/
                evaluation_report_gpt4.json
                evaluation_report_qwen2.5-7b.json
            spider/
                evaluation_report_gpt4.json
        postgresql/
            bird/
                evaluation_report_gpt4.json

Usage:
    python aggregate_results.py --results-dir ./results --output structured_results.csv
    python aggregate_results.py --results-dir ./results --metric1 average_exact_match --metric2 average_recall
"""

import argparse
import json
import csv
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import OrderedDict, defaultdict

# 默认模型配置 - 可以在代码中直接修改
DEFAULT_MODEL_CONFIG = {
    "closed_source": [
        # {"name": "api_gpt-4o-mini", "display_name": "GPT-4o-mini"},
        # {"name": "api_gpt-4-turbo", "display_name": "GPT-4-Turbo"},
        # {"name": "api_gpt-4o", "display_name": "GPT-4o"},
        # {"name": "api_claude-3-5-haiku-20241022", "display_name": "Claude-3.5-Haiku"},
        # {"name": "api_claude-3-7-sonnet-20250219", "display_name": "Claude-3.7-Sonnet"},
        # {"name": "api_claude-4-sonnet", "display_name": "Claude-4-Sonnet"},
        # {"name": "api_gemini-2.5-flash", "display_name": "Gemini-2.5-Flash"},
        # {"name": "api_gemini-2.5-pro", "display_name": "Gemini-2.5-Pro"},
        # {"name": "api_grok-3", "display_name": "Grok-3"},
        # {"name": "api_grok-4", "display_name": "Grok-4"},
        # {"name": "api_command-r-plus-08-2024", "display_name": "Command-R-Plus-08-2024"},

        # {"name": "vllm_OmniSQL-7B", "display_name": "OmniSQL-7B"},
        # {"name": "vllm_OmniSQL-14B", "display_name": "OmniSQL-14B"},
        # {"name": "vllm_OmniSQL-32B", "display_name": "OmniSQL-32B"},

        # {"name": "vllm_deepseek-coder-6.7b-instruct", "display_name": "DeepSeek-coder-6.7B-Instruct"},
        # {"name": "vllm_Qwen2.5-Coder-7B-Instruct", "display_name": "Qwen2.5-Coder-7B-Instruct"},
        # {"name": "vllm_Qwen2.5-7B-Instruct", "display_name": "Qwen2.5-7B-Instruct"},
        # {"name": "vllm_OpenCoder-8B-Instruct", "display_name": "OpenCoder-8B-Instruct"},
        # {"name": "vllm_Meta-Llama-3.1-8B-Instruct", "display_name": "Meta-Llama-3.1-8B-Instruct"},
        # {"name": "vllm_UniVectorSQL-7B-LoRA-Step600", "display_name": "UniVectorSQL-7B-LoRA-Step600"},
        # {"name": "vllm_UniVectorSQL-7B-LoRA-Step800", "display_name": "UniVectorSQL-7B-LoRA-Step800"},
        # {"name": "vllm_UniVectorSQL-7B-LoRA-Step1100", "display_name": "UniVectorSQL-7B-LoRA-Step1100"},

        {"name": "vllm_UniVectorSQL-7B-LoRA-Step200", "display_name": "UniVectorSQL-7B-Step200"},
        {"name": "vllm_UniVectorSQL-7B-LoRA-Step400", "display_name": "UniVectorSQL-7B-Step400"},
        {"name": "vllm_UniVectorSQL-7B-LoRA-Step600", "display_name": "UniVectorSQL-7B-Step600"},
        {"name": "vllm_UniVectorSQL-7B-LoRA-Step800", "display_name": "UniVectorSQL-7B-Step800"},
        {"name": "vllm_UniVectorSQL-7B-LoRA-Step1000", "display_name": "UniVectorSQL-7B-Step1000"},
        {"name": "vllm_UniVectorSQL-7B-LoRA-Step1200", "display_name": "UniVectorSQL-7B-Step1200"},
        {"name": "vllm_UniVectorSQL-7B-LoRA-Step1600", "display_name": "UniVectorSQL-7B-Step1600"},

        # {"name": "vllm_UniVectorSQL-7B-No_CoT_Step100", "display_name": "UniVectorSQL-7B-No_CoT_Step100"},
        # {"name": "vllm_UniVectorSQL-7B-No_CoT_Step300", "display_name": "UniVectorSQL-7B-No_CoT_Step300"},
        # {"name": "vllm_UniVectorSQL-7B-No_CoT_Step500", "display_name": "UniVectorSQL-7B-No_CoT_Step500"},
        # {"name": "vllm_UniVectorSQL-7B-No_CoT_Step800", "display_name": "UniVectorSQL-7B-No_CoT_Step800"},
        # {"name": "vllm_UniVectorSQL-7B-No_CoT_Step1000", "display_name": "UniVectorSQL-7B-No_CoT_Step1000"},

        # {"name": "vllm_UniVectorSQL-14B-LoRA-Step200", "display_name": "UniVectorSQL-14B-LoRA-Step200"},
        # {"name": "vllm_UniVectorSQL-14B-LoRA-Step400", "display_name": "UniVectorSQL-14B-LoRA-Step400"},
        # {"name": "vllm_UniVectorSQL-14B-LoRA-Step600", "display_name": "UniVectorSQL-14B-LoRA-Step600"}, 
        # {"name": "vllm_UniVectorSQL-14B-LoRA-Step900", "display_name": "UniVectorSQL-14B-LoRA-Step900"},
        # {"name": "vllm_UniVectorSQL-14B-LoRA-Step1000", "display_name": "UniVectorSQL-14B-LoRA-Step1000"},
        # {"name": "vllm_UniVectorSQL-14B-LoRA-Step1100", "display_name": "UniVectorSQL-14B-LoRA-Step1100"},

        # {"name": "vllm_UniVectorSQL-14B-No_CoT-Step500", "display_name": "UniVectorSQL-14B-No_CoT-Step500"},
        # {"name": "vllm_UniVectorSQL-14B-No_CoT-Step1000", "display_name": "UniVectorSQL-14B-No_CoT-Step1000"},
        # {"name": "vllm_UniVectorSQL-14B-No_CoT-Step1500", "display_name": "UniVectorSQL-14B-No_CoT-Step1500"},
        # {"name": "vllm_UniVectorSQL-14B-Step500", "display_name": "UniVectorSQL-14B-Step500"},
        # {"name": "vllm_UniVectorSQL-14B-Step1000", "display_name": "UniVectorSQL-14B-Step1000"},
        # {"name": "vllm_UniVectorSQL-14B-Step1500", "display_name": "UniVectorSQL-14B-Step1500"},

        # {"name": "vllm_Qwen2.5-Coder-14B-Instruct", "display_name": "Qwen2.5-Coder-14B-Instruct"},
        # {"name": "vllm_Qwen2.5-14B-Instruct", "display_name": "Qwen2.5-14B-Instruct"},
        # {"name": "vllm_starcoder2-15b-instruct-v0.1", "display_name": "StarCoder2-15B-Instruct-V0.1"},
        # {"name": "vllm_DeepSeek-Coder-V2-Lite-Instruct", "display_name": "DeepSeek-Coder-V2-Lite-Instruct"},
        # {"name": "vllm_Codestral-22B-v0.1", "display_name": "Codestral-22B-V0.1"},

        # {"name": "vllm_Qwen2.5-Coder-32B-Instruct", "display_name": "Qwen2.5-Coder-32B-Instruct"},
        # {"name": "api_qwen2.5-32b-instruct", "display_name": "Qwen2.5-32B-Instruct"},
        # {"name": "vllm_deepseek-coder-33b-instruct", "display_name": "DeepSeek-Coder-33B-Instruct"},
        # {"name": "vllm_Meta-Llama-3.1-70B-Instruct", "display_name": "Meta-Llama-3.1-70B-Instruct"},
        # {"name": "api_qwen2.5-72b-instruct", "display_name": "Qwen2.5-72B-Instruct"},
        # {"name": "api_deepseek-v3.1-250821", "display_name": "DeepSeek-V3.1 (671B, MoE)"},

        # {"name": "vllm_Mixtral-8x7B-Instruct-v0.1", "display_name": "Mixtral-8x7B-Instruct-v0.1"},
    ]
}

'''
    "open_7b": [
        {"name": "dsc-6.7b-instruct", "display_name": "DSC-6.7B-Instruct"},
        {"name": "qwen2.5-coder-7b-instruct", "display_name": "Qwen2.5-Coder-7B-Instruct"},
        {"name": "qwen2.5-7b-instruct", "display_name": "Qwen2.5-7B-Instruct"},
        {"name": "qwen2.5-7b", "display_name": "Qwen2.5-7B-Instruct"},
        {"name": "opencoder-8b-instruct", "display_name": "OpenCoder-8B-Instruct"},
        {"name": "meta-llama-3.1-8b-instruct", "display_name": "Meta-Llama-3.1-8B-Instruct"},
        {"name": "granite-8b-code-instruct", "display_name": "Granite-8B-Code-Instruct"},
        {"name": "granite-3.1-8b-instruct", "display_name": "Granite-3.1-8B-Instruct"},
    ],
    "custom_7b": [
        {"name": "vectorsql-7b-lora", "display_name": "VectorSQL-7B-LoRA"},
        {"name": "vectorsql-7b", "display_name": "VectorSQL-7B"},
    ],
    "open_14b_32b": [
        {"name": "qwen2.5-coder-14b-instruct", "display_name": "Qwen2.5-Coder-14B-Instruct"},
        {"name": "qwen2.5-14b-instruct", "display_name": "Qwen2.5-14B-Instruct"},
        {"name": "starcoder2-15b-instruct", "display_name": "Starcoder2-15B-Instruct"},
        {"name": "dsc-v2-lite-instruct", "display_name": "DSC-V2-Lite-In. (16B, MoE)"},
        {"name": "granite-20b-code-instruct", "display_name": "Granite-20B-Code-Instruct"},
        {"name": "codestral-22b", "display_name": "Codestral-22B"},
    ],
    "custom_14b": [
        {"name": "vectorsql-14b-lora", "display_name": "VectorSQL-14B-LoRA"},
        {"name": "vectorsql-14b", "display_name": "VectorSQL-14B"},
    ],
    "open_32b_plus": [
        {"name": "qwen2.5-coder-32b-instruct", "display_name": "Qwen2.5-Coder-32B-Instruct"},
        {"name": "qwen2.5-32b-instruct", "display_name": "Qwen2.5-32B-Instruct"},
        {"name": "dsc-33b-instruct", "display_name": "DSC-33B-Instruct"},
        {"name": "granite-34b-code-instruct", "display_name": "Granite-34B-Code-Instruct"},
        {"name": "mixtral-8x7b-instruct", "display_name": "Mixtral-8x7B-In. (47B, MoE)"},
        {"name": "meta-llama-3.1-70b-instruct", "display_name": "Meta-Llama-3.1-70B-Instruct"},
        {"name": "qwen2.5-72b-instruct", "display_name": "Qwen2.5-72B-Instruct"},
        {"name": "deepseek-v3", "display_name": "DeepSeek-V3 (671B, MoE)"},
    ],
    "custom_32b": [
        {"name": "vectorsql-32b", "display_name": "VectorSQL-32B"},
    ],
'''



def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load a JSON file and return its contents."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  Error loading {file_path}: {e}")
        return None


def load_results_by_model_name(root_dir: str, model_name: str, 
                               metric1_key: str, metric2_key: str) -> Tuple[Dict, Dict]:
    """
    根据模型名称从目录结构加载结果
    
    目录结构: root_dir/db_type/dataset_type/evaluation_report_{model_name}.json
    
    示例路径: results/sqlite/bird/evaluation_report_gpt4.json
    
    Args:
        root_dir: 根目录路径
        model_name: 模型名称（用于构建文件名）
        metric1_key: 第一个指标键名
        metric2_key: 第二个指标键名
        
    Returns:
        两个字典的元组: (metric1_dict, metric2_dict)
        每个字典格式: {db_type: {dataset_type: value}}
    """
    root_path = Path(root_dir)
    metric1_dict = {}
    metric2_dict = {}
    
    # 固定的数据库类型和数据集类型
    db_types = ['sqlite', 'postgresql', 'clickhouse', 'myscale']
    dataset_types = ['bird', 'spider', 'arxiv', 'wikipedia_multimodal']
    
    for db_type in db_types:
        metric1_dict[db_type] = {}
        metric2_dict[db_type] = {}
        
        for dataset_type in dataset_types:
            # 构建文件路径: results_dir/db_type/dataset_type/evaluation_report_model.json
            json_path = root_path / db_type / dataset_type / f"evaluation_report_{model_name}.json"
            
            
            
            if not json_path.exists():
                continue
            
            # 加载 JSON 文件
            report = load_json_file(str(json_path))
            if report is None:
                continue
            
            # 提取指标
            summary = report.get('evaluation_summary', {})
            
            metric1_value = summary.get(metric1_key, None)
            metric2_value = summary.get(metric2_key, None)
            
            if metric1_value is not None:
                metric1_dict[db_type][dataset_type] = metric1_value
            if metric2_value is not None:
                metric2_dict[db_type][dataset_type] = metric2_value
    
    return metric1_dict, metric2_dict


def load_all_results_from_config(root_dir: str, model_config: Dict[str, List[Dict[str, str]]],
                                 metric1_key: str, metric2_key: str) -> Tuple[Dict, Dict, Dict]:
    """
    根据模型配置加载所有结果
    
    Args:
        root_dir: 根目录路径
        model_config: 模型配置字典
        metric1_key: 第一个指标键名
        metric2_key: 第二个指标键名
        
    Returns:
        三个字典的元组:
        - metric1_results: {display_name: {db_type: {dataset_type: value}}}
        - metric2_results: {display_name: {db_type: {dataset_type: value}}}
        - model_categories: {category: [display_names]}
    """
    metric1_results = {}
    metric2_results = {}
    model_categories = {}
    
    # 类别映射
    category_map = {
        'closed_source': 'Closed-source LLMs (as a reference)',
        'open_7b': 'Open-source LLMs (~7B)',
        'custom_7b': 'custom_7b',
        'open_14b_32b': 'Open-source LLMs (14B-32B)',
        'custom_14b': 'custom_14b',
        'open_32b_plus': 'Open-source LLMs (≥32B)',
        'custom_32b': 'custom_32b'
    }
    
    for category_key, models in model_config.items():
        category_name = category_map.get(category_key, category_key)
        model_categories[category_name] = []
        
        for model_info in models:
            model_name = model_info.get('name')
            display_name = model_info.get('display_name', model_name)
            
            if not model_name:
                print(f"⚠️  Skipping model with missing 'name' field in category '{category_key}'")
                continue
            
            # 加载该模型的结果
            metric1_dict, metric2_dict = load_results_by_model_name(
                root_dir, model_name, metric1_key, metric2_key
            )
            
            # 检查是否有数据
            has_data = any(
                len(datasets) > 0 
                for datasets in metric1_dict.values()
            )
            
            if has_data:
                metric1_results[display_name] = metric1_dict
                metric2_results[display_name] = metric2_dict
                model_categories[category_name].append(display_name)
                print(f"  ✓ Loaded: {display_name} (from {model_name})")
            else:
                print(f"  ⚠️  No data found for: {display_name} (from {model_name})")
    
    return metric1_results, metric2_results, model_categories


def format_metric_pair(value1: float, value2: float, precision: int = 1) -> str:
    """
    格式化指标对为 "value1 / value2" 格式（百分比）
    
    Args:
        value1: 第一个指标值（0-1之间）
        value2: 第二个指标值（0-1之间）
        precision: 小数位数
        
    Returns:
        格式化的字符串，如 "45.6 / 78.9"
    """
    if value1 is None or value2 is None:
        return ""
    
    v1_percentage = value1 * 100
    v2_percentage = value2 * 100
    
    v1_str = f"{v1_percentage:.{precision}f}"
    v2_str = f"{v2_percentage:.{precision}f}"
    
    return f"{v1_str} / {v2_str}"


def calculate_average_metrics(model_metric1: Dict[str, Dict[str, float]], 
                              model_metric2: Dict[str, Dict[str, float]]) -> str:
    """
    计算模型在所有数据库和数据集上的平均指标对
    
    Args:
        model_metric1: {db_type: {dataset_type: metric1_value}}
        model_metric2: {db_type: {dataset_type: metric2_value}}
        
    Returns:
        格式化的平均值字符串
    """
    all_values1 = []
    all_values2 = []
    
    for db_type in model_metric1:
        for dataset_type in model_metric1[db_type]:
            val1 = model_metric1[db_type].get(dataset_type)
            val2 = model_metric2.get(db_type, {}).get(dataset_type)
            
            if val1 is not None:
                all_values1.append(val1)
            if val2 is not None:
                all_values2.append(val2)
    
    if not all_values1 or not all_values2:
        return ""
    
    avg1 = sum(all_values1) / len(all_values1)
    avg2 = sum(all_values2) / len(all_values2)
    
    return format_metric_pair(avg1, avg2)


def generate_structured_csv(metric1_results: Dict[str, Dict[str, Dict[str, float]]], 
                           metric2_results: Dict[str, Dict[str, Dict[str, float]]],
                           model_categories: Dict[str, List[str]],
                           output_path: str = 'structured_results.csv',
                           metric1_name: str = 'F1Score',
                           metric2_name: str = 'nDCG@10'):
    """
    生成结构化的CSV文件，格式类似LaTeX表格
    
    Args:
        metric1_results: 第一个指标结果 {display_name: {db_type: {dataset_type: value}}}
        metric2_results: 第二个指标结果 {display_name: {db_type: {dataset_type: value}}}
        model_categories: 模型分类 {category: [display_names]}
        output_path: 输出文件路径
        metric1_name: 第一个指标的显示名称
        metric2_name: 第二个指标的显示名称
    """
    db_types = ['sqlite', 'postgresql', 'clickhouse', 'myscale']
    dataset_types = ['bird', 'spider', 'arxiv', 'wikipedia_multimodal']
    
    # 准备CSV数据
    rows = []
    
    # 添加表头行1: 数据库类型
    header_row1 = ['LLM']
    for db in ['SQLite', 'PostgreSQL', 'ClickHouse', 'myscale']:
        header_row1.extend([db] + [''] * 3)
    header_row1.append('Average')
    rows.append(header_row1)
    
    # 添加表头行2: 数据集类型
    header_row2 = ['']
    for _ in range(3):  # 三个数据库
        header_row2.extend(['BIRD', 'Spider', 'arXiv', 'Wiki'])
    header_row2.append('')
    rows.append(header_row2)
    
    # 添加指标说明行
    #metric_row = [f'({metric1_name} / {metric2_name}, in %)'] + [''] * 12 + ['']
    #rows.append(metric_row)
    
    # 添加各个类别的模型
    category_order = [
        'Closed-source LLMs (as a reference)',
        'Open-source LLMs (~7B)',
        'custom_7b',
        'Open-source LLMs (14B-32B)',
        'custom_14b',
        'Open-source LLMs (≥32B)',
        'custom_32b',
    ]
    
    total_models = 0
    
    for cat_key in category_order:
        cat_models = model_categories.get(cat_key, [])
        if not cat_models:
            continue
        
        # 添加类别标题行（除了自定义模型类别）
        if not cat_key.startswith('custom_'):
            category_row = [cat_key] + [''] * 12 + ['']
            rows.append(category_row)
        
        for display_name in cat_models:
            if display_name not in metric1_results:
                continue
            
            model_metric1 = metric1_results[display_name]
            model_metric2 = metric2_results.get(display_name, {})
            
            # 构建数据行
            row_data = [display_name]
            
            for db_type in db_types:
                for dataset_type in dataset_types:
                    val1 = model_metric1.get(db_type, {}).get(dataset_type, None)
                    val2 = model_metric2.get(db_type, {}).get(dataset_type, None)
                    
                    cell = format_metric_pair(val1, val2)
                    row_data.append(cell)
            
            # 添加平均值
            avg_cell = calculate_average_metrics(model_metric1, model_metric2)
            row_data.append(avg_cell)
            
            rows.append(row_data)
            total_models += 1
    
    # 写入CSV文件
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        
        print(f"\n✅ Successfully generated structured CSV: {output_path}")
        print(f"   Total models: {total_models}")
        print(f"   Metrics: {metric1_name} / {metric2_name}")
        
    except Exception as e:
        print(f"❌ Error writing CSV file: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate evaluation metrics from hierarchical directory structure into a structured CSV file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Directory Structure:
  results_dir/
      sqlite/
          bird/
              evaluation_report_gpt4.json
              evaluation_report_qwen2.5-7b.json
          spider/
              evaluation_report_gpt4.json
          arxiv/
              evaluation_report_gpt4.json
          wiki/
              evaluation_report_gpt4.json
      postgresql/
          bird/
              evaluation_report_gpt4.json
          ...
      clickhouse/
          bird/
              evaluation_report_gpt4.json
          ...
      myscale/
          bird/
              evaluation_report_gpt4.json
          ...

Examples:
  # Generate structured CSV with default metrics (F1Score / nDCG@10)
  python aggregate_results.py --results-dir ./results --output structured_results.csv
  
  # Customize metrics
  python aggregate_results.py --results-dir ./results --output results.csv \\
      --metric1 average_exact_match --metric2 average_recall \\
      --metric1-name "EX" --metric2-name "Recall"
  
  # Use different results directory
  python aggregate_results.py --results-dir /path/to/results --output custom_output.csv

Note: Model configuration is set in the script code (DEFAULT_MODEL_CONFIG).
      Edit the script to modify which models are included in the output.
        """
    )
    
    # Required argument
    parser.add_argument(
        '--results-dir',
        default='/mnt/DataFlow/ydw/Text2VectorSQL/Evaluation_Framework/results',
        help='Root directory containing results (structure: results_dir/db_type/dataset_type/evaluation_report_model.json)'
    )
    
    # Output options
    parser.add_argument(
        '--output', '-o',
        default='structured_results.csv',
        help='Output CSV file path (default: structured_results.csv)'
    )
    
    # Metric options
    parser.add_argument(
        '--metric1',
        default='average_ndcg@10',
        help='First metric key (default: average_f1_score)'
    )
    
    parser.add_argument(
        '--metric2',
        default='evaluation_success_rate',
        help='Second metric key (default: average_ndcg@10)'
    )
    
    parser.add_argument(
        '--metric1-name',
        default='nDCG@10',
        help='Display name for first metric (default: F1Score)'
    )
    
    parser.add_argument(
        '--metric2-name',
        default='SR',
        help='Display name for second metric (default: nDCG@10)'
    )
    
    # Display options
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress all output except errors'
    )
    
    args = parser.parse_args()
    
    # Validate results directory exists
    if not Path(args.results_dir).exists():
        print(f"❌ Results directory not found: {args.results_dir}")
        sys.exit(1)
    
    if not args.quiet:
        print(f"📂 Loading results from: {args.results_dir}")
        print(f"   Expected structure: results_dir/db_type/dataset_type/evaluation_report_model.json")
        print(f"📊 Metrics: {args.metric1_name} ({args.metric1}) / {args.metric2_name} ({args.metric2})")
        print(f"📄 Output: {args.output}")
    
    # 使用默认模型配置
    model_config = DEFAULT_MODEL_CONFIG
    
    if not args.quiet:
        total_models = sum(len(models) for models in model_config.values())
        print(f"\n🔍 Configured models: {total_models} models across {len(model_config)} categories")
    
    # 加载所有结果
    if not args.quiet:
        print(f"\n📥 Loading results for models...")
    
    metric1_results, metric2_results, model_categories = load_all_results_from_config(
        args.results_dir,
        model_config,
        args.metric1,
        args.metric2
    )
    
    if not metric1_results:
        print("❌ No valid results found for any models.")
        print("\nℹ️  Please ensure:")
        print("   1. Results directory structure is: results_dir/db_type/dataset_type/evaluation_report_model.json")
        print("      Example: results/sqlite/bird/evaluation_report_gpt4.json")
        print("   2. Model names in DEFAULT_MODEL_CONFIG match the file names")
        print("   3. JSON files contain the specified metrics in 'evaluation_summary'")
        sys.exit(1)
    
    # 生成结构化CSV
    generate_structured_csv(
        metric1_results, 
        metric2_results,
        model_categories,
        args.output,
        args.metric1_name,
        args.metric2_name
    )
    
    if not args.quiet:
        print(f"\n✅ Structured CSV generation complete!")
        print(f"   Output saved to: {args.output}")


if __name__ == "__main__":
    main()
