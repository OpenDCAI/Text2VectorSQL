# evaluate_results.py - Stage 2: Evaluation Only

import argparse
import json
import yaml
import sys
import os
import shutil # 导入 shutil 库用于删除目录
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from sql_executor import resolve_db_identifier
# Import metrics calculation functions
from metrics import (
    calculate_set_metrics,
    calculate_average_precision,
    calculate_reciprocal_rank,
    calculate_ndcg,
    calculate_exact_match_any_gt,
    calculate_exact_match_any_gt_with_columns,
    calculate_set_metrics_with_columns,
    calculate_ranking_metrics_with_columns,
    _get_gt_columns,
    evaluate_vectorsql_with_llm,
    calculate_llm_based_scores
)

def _ensure_tuple_list(data):
    """确保数据格式为 list[tuple]。"""
    if not data or not isinstance(data, list): return []
    if not data: return []
    if isinstance(data[0], tuple): return data
    if isinstance(data[0], list): return [tuple(row) for row in data]
    return [tuple(row) if not isinstance(row, tuple) else row for row in data]

def load_yaml_config(path: str) -> dict:
    """Loads a YAML configuration file."""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_json_file(path: str):
    """Loads a JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_evaluation_report(report: dict, path: str):
    """Saves the final evaluation report to a JSON file."""
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    
    if "individual_results" in report:
        report["individual_results"].sort(key=lambda x: x.get("eval_case", {}).get("query_id", ""))

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nEvaluation report saved to '{path}'")

def extract_successful_data(execution_result: dict) -> list:
    """Extract data from a successful execution result."""
    if execution_result and execution_result.get('status') == 'success':
        data = execution_result.get('data', [])
        return _ensure_tuple_list(data)
    return []

def evaluate_single_case(case, metrics_to_run, metric_params, llm_enabled, llm_api_config, llm_timeout, llm_include_details):
    """
    Evaluates a single execution case. Designed to be run concurrently.
    """
    case_report = {
        "eval_case": case.get("eval_case", {}),
        "execution_summary": {},
        "scores": {}
    }
    
    if 'execution_error' in case:
        case_report["execution_summary"]["error"] = case['execution_error']
        return case_report
        
    eval_execution = case.get('eval_execution', {})
    eval_data = extract_successful_data(eval_execution)
    
    case_report["execution_summary"]["eval_status"] = eval_execution.get('status', 'unknown')
    case_report["execution_summary"]["eval_row_count"] = len(eval_data)
    
    gt_executions = case.get('ground_truth_executions', [])
    all_gt_data = []
    gt_summary = []
    
    for gt_exec in gt_executions:
        gt_result = gt_exec.get('execution', {})
        gt_data = extract_successful_data(gt_result)
        all_gt_data.extend(gt_data)
        gt_summary.append({
            "sql": gt_exec.get('sql', ''),
            "status": gt_result.get('status', 'unknown'),
            "row_count": len(gt_data)
        })
    
    case_report["execution_summary"]["ground_truth_summary"] = gt_summary
    case_report["execution_summary"]["total_gt_rows"] = len(all_gt_data)
    
    if eval_execution.get('status') != 'success':
        case_report["execution_summary"]["evaluation_skipped"] = "Evaluation execution failed"
        return case_report
        
    individual_gt_results = [extract_successful_data(gt_exec.get('execution', {})) for gt_exec in gt_executions if extract_successful_data(gt_exec.get('execution', {}))]

    try:
        eval_columns = eval_execution.get('columns', [])
        gt_columns = _get_gt_columns(gt_executions)
        use_column_matching = bool(eval_columns and gt_columns)
        
        if use_column_matching:
            if any(metric in metrics_to_run for metric in ['f1_score', 'precision', 'recall']):
                case_report['scores'].update(calculate_set_metrics_with_columns(eval_data, eval_columns, all_gt_data, gt_columns))
            if 'exact_match' in metrics_to_run:
                case_report['scores']['exact_match'] = calculate_exact_match_any_gt_with_columns(eval_data, eval_columns, gt_executions)
            if 'map' in metrics_to_run:
                case_report['scores']['map'] = calculate_ranking_metrics_with_columns(eval_data, eval_columns, all_gt_data, gt_columns, 'map')
            if 'mrr' in metrics_to_run:
                case_report['scores']['mrr'] = calculate_ranking_metrics_with_columns(eval_data, eval_columns, all_gt_data, gt_columns, 'mrr')
            if 'ndcg' in metrics_to_run:
                k = metric_params.get('ndcg', {}).get('k', 10)
                case_report['scores'][f'ndcg@{k}'] = calculate_ranking_metrics_with_columns(eval_data, eval_columns, all_gt_data, gt_columns, 'ndcg', k)
        else:
            if any(metric in metrics_to_run for metric in ['f1_score', 'precision', 'recall']):
                case_report['scores'].update(calculate_set_metrics(eval_data, all_gt_data, individual_gt_results))
            if 'exact_match' in metrics_to_run:
                case_report['scores']['exact_match'] = calculate_exact_match_any_gt(eval_data, individual_gt_results)
            if 'map' in metrics_to_run:
                case_report['scores']['map'] = calculate_average_precision(eval_data, all_gt_data, individual_gt_results)
            if 'mrr' in metrics_to_run:
                case_report['scores']['mrr'] = calculate_reciprocal_rank(eval_data, all_gt_data, individual_gt_results)
            if 'ndcg' in metrics_to_run:
                k = metric_params.get('ndcg', {}).get('k', 10)
                case_report['scores'][f'ndcg@{k}'] = calculate_ndcg(eval_data, all_gt_data, k, individual_gt_results)

        case_report["execution_summary"]["column_matching_used"] = use_column_matching
        if use_column_matching:
            case_report["execution_summary"]["eval_columns"] = eval_columns
            case_report["execution_summary"]["gt_columns"] = gt_columns
            
    except Exception as e:
        case_report["evaluation_error"] = str(e)

    if llm_enabled:
        try:
            eval_case_data = case.get("eval_case", {})
            nl_question = eval_case_data.get('question', '')
            predicted_query = eval_case_data.get('predicted_sql', '')
            ground_truth_query = gt_executions[0].get('sql', '') if gt_executions else ''
            db_schema = eval_case_data.get('schema', '')
            
            if nl_question and predicted_query and ground_truth_query and db_schema:
                llm_result = evaluate_vectorsql_with_llm(
                    nl_question=nl_question, db_schema=db_schema,
                    ground_truth_query=ground_truth_query, predicted_query=predicted_query,
                    api_config=llm_api_config, timeout=llm_timeout
                )
                if llm_result:
                    llm_scores = calculate_llm_based_scores(llm_result)
                    case_report['scores'].update(llm_scores)
                    if llm_include_details:
                        case_report['llm_evaluation_details'] = llm_result
                else:
                    case_report['llm_evaluation_error'] = "LLM evaluation failed"
            else:
                case_report['llm_evaluation_skipped'] = "Missing required fields for LLM evaluation."
                
        except Exception as e:
            case_report['llm_evaluation_error'] = str(e)
            
    return case_report


def main():
    parser = argparse.ArgumentParser(description="Text2VectorSQL Results Evaluator (Stage 2)")
    parser.add_argument("--config", default="evaluation_config.yaml", help="Path to the evaluation configuration YAML file.")
    parser.add_argument("--execution-results", help="Path to the SQL execution results file (overrides config).")
    # 新增: --no-cache 参数
    parser.add_argument("--no-cache", default=False, action="store_true", help="Force re-evaluation and ignore any existing cache.")
    args = parser.parse_args()

    # --- 1. Load Configuration ---
    print(f"Loading configuration from '{args.config}'...")
    try:
        config = load_yaml_config(args.config)
        db_type = config['db_type'] # 获取 db_type
        eval_data_file = config['eval_data_file'] # 获取数据集文件名
        execution_results_file = args.execution_results or config['execution_results_file']
        evaluation_report_file = config['evaluation_report_file']
        metrics_to_run = {m['name'] if isinstance(m, dict) else m for m in config.get('metrics', [])}
        metric_params = {m['name']: m for m in config.get('metrics', []) if isinstance(m, dict)}
        
        num_workers = config.get('num_workers', 8)

        llm_config = config.get('llm_evaluation', {})
        llm_enabled = llm_config.get('enabled', False)
        llm_api_config = {}
        llm_timeout = 60
        llm_include_details = True
        
        if llm_enabled:
            llm_api_config = {'url': llm_config.get('api_url', ''), 'api_key': llm_config.get('api_key', ''), 'model': llm_config.get('model_name', 'gpt-3.5-turbo')}
            llm_timeout = llm_config.get('timeout', 60)
            llm_include_details = llm_config.get('include_details', True)
            print(f"LLM Evaluation: ENABLED (Workers: {num_workers})")
            print(f"  Model: {llm_api_config['model']}")
        else:
            print(f"LLM Evaluation: DISABLED (Workers: {num_workers})")
        
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

    # --- 2. Load Execution Results ---
    print(f"Loading execution results from '{execution_results_file}'...")
    try:
        execution_results = load_json_file(execution_results_file)
    except Exception as e:
        print(f"Error loading execution results file: {e}")
        sys.exit(1)

    # --- 3. Interruption Recovery & Concurrent Evaluation ---
    # 新增: 构造新的缓存目录路径
    dataset_name, _ = os.path.splitext(os.path.basename(eval_data_file))
    cache_dir = os.path.join("cache", db_type, dataset_name, "evaluation")
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Using cache directory for recovery: '{cache_dir}'")
    
    # 新增: 处理 --no-cache 标志
    if args.no_cache:
        print("--- The --no-cache flag is set. Clearing cache and re-processing all items. ---")
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)

    cached_results = []
    processed_ids = set()
    if not args.no_cache:
        for filename in os.listdir(cache_dir):
            if filename.endswith('.json'):
                try:
                    with open(os.path.join(cache_dir, filename), 'r', encoding='utf-8') as f:
                        result = json.load(f)
                        query_id = result.get("eval_case", {}).get("query_id")
                        if query_id:
                            cached_results.append(result)
                            processed_ids.add(query_id)
                except (IOError, json.JSONDecodeError):
                    continue
    
    if cached_results:
        print(f"Loaded {len(cached_results)} completed evaluations from cache.")

    items_to_process = [case for case in execution_results if case.get("eval_case", {}).get("query_id") not in processed_ids]
    print(f"Total cases: {len(execution_results)}, To be evaluated: {len(items_to_process)}")

    new_results = []
    if items_to_process:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_case = {
                executor.submit(evaluate_single_case, case, metrics_to_run, metric_params, llm_enabled, llm_api_config, llm_timeout, llm_include_details): case
                for case in items_to_process
            }
            
            with tqdm(total=len(items_to_process), desc="Evaluating Cases") as pbar:
                for future in as_completed(future_to_case):
                    try:
                        result = future.result()
                        new_results.append(result)
                        query_id = result.get("eval_case", {}).get("query_id")
                        if query_id:
                            with open(os.path.join(cache_dir, f"{query_id}.json"), 'w', encoding='utf-8') as f:
                                json.dump(result, f, indent=2, ensure_ascii=False)
                    except Exception as e:
                        case = future_to_case[future]
                        print(f"\nError processing case {case.get('eval_case', {}).get('query_id', 'N/A')}: {e}")
                    pbar.update(1)
    
    evaluated_results = cached_results + new_results

    # --- 4. Compute Final Summary from all results ---
    aggregated_scores = defaultdict(list)
    for case_report in evaluated_results:
        for metric, score in case_report.get('scores', {}).items():
            if isinstance(score, (int, float)):
                aggregated_scores[metric].append(score)

    final_summary = {}
    for metric, scores in aggregated_scores.items():
        if scores:
            final_summary[f'average_{metric}'] = sum(scores) / len(execution_results) if execution_results else 0
            final_summary[f'count_{metric}'] = len(scores)

    total_cases = len(execution_results)
    successful_evaluations = sum(1 for r in evaluated_results if r.get('scores'))
    final_summary['total_cases'] = total_cases
    final_summary['successful_evaluations'] = successful_evaluations
    final_summary['evaluation_success_rate'] = successful_evaluations / total_cases if total_cases > 0 else 0

    # --- 5. Generate Final Report ---
    final_report = {
        "evaluation_summary": final_summary,
        "individual_results": evaluated_results
    }
    
    save_evaluation_report(final_report, evaluation_report_file)
    
    # --- 6. Print Summary to Console ---
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Total cases: {final_summary.get('total_cases', 0)}")
    print(f"Evaluated cases with scores: {final_summary.get('successful_evaluations', 0)}")
    print(f"Evaluation coverage: {final_summary.get('evaluation_success_rate', 0):.2%}")
    print("\nAverage Metrics (over all cases):")
    for key, value in sorted(final_summary.items()):
        if key.startswith('average_'):
            metric_name = key.replace('average_', '')
            print(f"  - {metric_name:<20}: {value:.4f}")

if __name__ == "__main__":
    main()
