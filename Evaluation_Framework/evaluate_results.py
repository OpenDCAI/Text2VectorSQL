# evaluate_results.py - Stage 2: Evaluation Only

import argparse
import json
import yaml
import sys
import os
from tqdm import tqdm
from collections import defaultdict

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
    _get_gt_columns
)

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
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nEvaluation report saved to '{path}'")

def extract_successful_data(execution_result: dict) -> list:
    """Extract data from a successful execution result."""
    if execution_result and execution_result.get('status') == 'success':
        return execution_result.get('data', [])
    return []

def main():
    """
    Stage 2: Evaluate execution results and compute metrics.
    This script reads SQL execution results and computes evaluation metrics.
    """
    parser = argparse.ArgumentParser(description="Text2VectorSQL Results Evaluator (Stage 2)")
    parser.add_argument(
        "--config",
        default="evaluation_config.yaml",
        help="Path to the evaluation configuration YAML file."
    )
    parser.add_argument(
        "--execution-results",
        help="Path to the SQL execution results file (overrides config)."
    )
    args = parser.parse_args()

    # --- 1. Load Configuration ---
    print(f"Loading configuration from '{args.config}'...")
    try:
        config = load_yaml_config(args.config)
        execution_results_file = args.execution_results or config['execution_results_file']
        evaluation_report_file = config['evaluation_report_file']
        
        # Parse metrics configuration
        metrics_to_run = {m['name'] if isinstance(m, dict) else m for m in config.get('metrics', [])}
        metric_params = {m['name']: m for m in config.get('metrics', []) if isinstance(m, dict)}
        
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

    # --- 3. Evaluate Each Case ---
    evaluated_results = []
    aggregated_scores = defaultdict(list)
    
    print(f"Evaluating {len(execution_results)} cases...")
    for case in tqdm(execution_results, desc="Computing Metrics"):
        case_report = {
            "eval_case": case.get("eval_case", {}),
            "execution_summary": {},
            "scores": {}
        }
        
        # Check for execution errors
        if 'execution_error' in case:
            case_report["execution_summary"]["error"] = case['execution_error']
            evaluated_results.append(case_report)
            continue
            
        # Extract evaluation results
        eval_execution = case.get('eval_execution', {})
        eval_data = extract_successful_data(eval_execution)
        
        case_report["execution_summary"]["eval_status"] = eval_execution.get('status', 'unknown')
        case_report["execution_summary"]["eval_row_count"] = len(eval_data)
        
        # Extract and aggregate ground truth results
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
        
        # Skip evaluation if eval failed or no ground truth data
        if eval_execution.get('status') != 'success':
            case_report["execution_summary"]["evaluation_skipped"] = "Evaluation execution failed"
            evaluated_results.append(case_report)
            continue
            
        if not all_gt_data:
            case_report["execution_summary"]["evaluation_skipped"] = "No successful ground truth executions"
            evaluated_results.append(case_report)
            continue

        # Individual GT results for exact match calculation
        individual_gt_results = []
        for gt_exec in gt_executions:
            gt_result = gt_exec.get('execution', {})
            gt_data = extract_successful_data(gt_result)
            if gt_data:  # Only include successful GT executions
                individual_gt_results.append(gt_data)

        # --- 5. Calculate Metrics ---
        try:
            # Get column information for evaluation and ground truth
            eval_columns = eval_execution.get('columns', [])
            gt_columns = _get_gt_columns(gt_executions)
            
            # Check if we have column information to use column-based matching
            use_column_matching = bool(eval_columns and gt_columns)
            
            if use_column_matching:
                # Use column-name-based matching
                
                # Set-based metrics (precision, recall, f1_score)
                if any(metric in metrics_to_run for metric in ['f1_score', 'precision', 'recall']):
                    set_scores = calculate_set_metrics_with_columns(eval_data, eval_columns, all_gt_data, gt_columns)
                    case_report['scores'].update(set_scores)

                # Exact match calculation - match any individual GT
                if 'exact_match' in metrics_to_run:
                    case_report['scores']['exact_match'] = calculate_exact_match_any_gt_with_columns(eval_data, eval_columns, gt_executions)

                # Ranking-based metrics
                if 'map' in metrics_to_run:
                    case_report['scores']['map'] = calculate_ranking_metrics_with_columns(eval_data, eval_columns, all_gt_data, gt_columns, 'map')
                
                if 'mrr' in metrics_to_run:
                    case_report['scores']['mrr'] = calculate_ranking_metrics_with_columns(eval_data, eval_columns, all_gt_data, gt_columns, 'mrr')

                if 'ndcg' in metrics_to_run:
                    k = metric_params.get('ndcg', {}).get('k', 10)
                    case_report['scores'][f'ndcg@{k}'] = calculate_ranking_metrics_with_columns(eval_data, eval_columns, all_gt_data, gt_columns, 'ndcg', k)
            
            else:
                # Fallback to position-based matching (original logic)
                
                # Set-based metrics (precision, recall, f1_score)
                if any(metric in metrics_to_run for metric in ['f1_score', 'precision', 'recall']):
                    set_scores = calculate_set_metrics(eval_data, all_gt_data, individual_gt_results)
                    case_report['scores'].update(set_scores)

                # Exact match calculation - match any individual GT
                if 'exact_match' in metrics_to_run:
                    case_report['scores']['exact_match'] = calculate_exact_match_any_gt(eval_data, individual_gt_results)

                # Ranking-based metrics
                if 'map' in metrics_to_run:
                    case_report['scores']['map'] = calculate_average_precision(eval_data, all_gt_data, individual_gt_results)
                
                if 'mrr' in metrics_to_run:
                    case_report['scores']['mrr'] = calculate_reciprocal_rank(eval_data, all_gt_data, individual_gt_results)

                if 'ndcg' in metrics_to_run:
                    k = metric_params.get('ndcg', {}).get('k', 10)
                    case_report['scores'][f'ndcg@{k}'] = calculate_ndcg(eval_data, all_gt_data, k, individual_gt_results)

            # Add matching method to report for debugging
            case_report["execution_summary"]["column_matching_used"] = use_column_matching
            if use_column_matching:
                case_report["execution_summary"]["eval_columns"] = eval_columns
                case_report["execution_summary"]["gt_columns"] = gt_columns

            # Aggregate scores for final summary
            for metric, score in case_report['scores'].items():
                if isinstance(score, (int, float)):
                    aggregated_scores[metric].append(score)
                    
        except Exception as e:
            case_report["evaluation_error"] = str(e)
            print(f"Error evaluating case {case_report['eval_case'].get('query_id', 'unknown')}: {e}")
        
        evaluated_results.append(case_report)

    # --- 6. Compute Final Summary ---
    final_summary = {}
    for metric, scores in aggregated_scores.items():
        if scores:
            final_summary[f'average_{metric}'] = sum(scores) / len(scores)
            final_summary[f'count_{metric}'] = len(scores)

    # Add overall statistics
    total_cases = len(execution_results)
    successful_evaluations = len([r for r in evaluated_results if 'scores' in r and r['scores']])
    final_summary['total_cases'] = total_cases
    final_summary['successful_evaluations'] = successful_evaluations
    final_summary['evaluation_success_rate'] = successful_evaluations / total_cases if total_cases > 0 else 0

    # --- 7. Generate Final Report ---
    final_report = {
        "evaluation_summary": final_summary,
        "individual_results": evaluated_results
    }
    
    save_evaluation_report(final_report, evaluation_report_file)
    
    # Print summary to console
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Total cases: {final_summary.get('total_cases', 0)}")
    print(f"Successful evaluations: {final_summary.get('successful_evaluations', 0)}")
    print(f"Success rate: {final_summary.get('evaluation_success_rate', 0):.2%}")
    print("\nAverage Metrics:")
    for key, value in final_summary.items():
        if key.startswith('average_'):
            metric_name = key.replace('average_', '')
            print(f"  {metric_name}: {value:.4f}")

if __name__ == "__main__":
    main()