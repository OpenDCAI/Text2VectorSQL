#!/usr/bin/env python3
"""
aggregate_results.py - Aggregate Multiple Evaluation Reports

This script aggregates evaluation metrics from multiple evaluation report JSON files
and exports them to a CSV file for easy comparison and analysis.

Usage:
    python aggregate_results.py --input report1.json report2.json report3.json --output summary.csv
    python aggregate_results.py --input-dir ./reports --output summary.csv
    python aggregate_results.py --input *.json --output summary.csv --sort-by average_f1_score
"""

import argparse
import json
import csv
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
from collections import OrderedDict


def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load a JSON file and return its contents."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def extract_summary_metrics(report: Dict[str, Any], report_name: str) -> Dict[str, Any]:
    """
    Extract evaluation summary metrics from a report.
    
    Args:
        report: The evaluation report dictionary
        report_name: Name/identifier for this report
        
    Returns:
        Dictionary containing the report name and all summary metrics
    """
    summary = report.get('evaluation_summary', {})
    
    # Start with report identifier
    metrics = OrderedDict()
    metrics['report_name'] = report_name
    
    # Add all metrics from the summary
    for key, value in sorted(summary.items()):
        if isinstance(value, (int, float)):
            metrics[key] = value
        elif isinstance(value, str):
            metrics[key] = value
    
    return metrics


def get_all_metric_names(all_metrics: List[Dict[str, Any]]) -> List[str]:
    """
    Get a sorted list of all unique metric names across all reports.
    
    Args:
        all_metrics: List of metric dictionaries from all reports
        
    Returns:
        Sorted list of unique metric names (excluding 'report_name')
    """
    metric_names = set()
    for metrics in all_metrics:
        metric_names.update(metrics.keys())
    
    # Remove 'report_name' and sort
    metric_names.discard('report_name')
    return sorted(metric_names)


def write_to_csv(all_metrics: List[Dict[str, Any]], output_path: str, sort_by: str = None):
    """
    Write aggregated metrics to a CSV file.
    
    Args:
        all_metrics: List of metric dictionaries
        output_path: Path to the output CSV file
        sort_by: Optional metric name to sort by (descending)
    """
    if not all_metrics:
        print("No metrics to write.")
        return
    
    # Sort if requested
    if sort_by and sort_by in all_metrics[0]:
        all_metrics = sorted(
            all_metrics, 
            key=lambda x: x.get(sort_by, 0), 
            reverse=True
        )
        print(f"Results sorted by '{sort_by}' (descending)")
    
    # Get all unique metric names
    metric_names = get_all_metric_names(all_metrics)
    
    # Define CSV header
    header = ['report_name'] + metric_names
    
    # Write to CSV
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            
            for metrics in all_metrics:
                # Ensure all fields are present (fill missing with empty string)
                row = {field: metrics.get(field, '') for field in header}
                writer.writerow(row)
        
        print(f"\n✅ Successfully wrote aggregated results to: {output_path}")
        print(f"   Total reports: {len(all_metrics)}")
        print(f"   Total metrics: {len(metric_names)}")
        
    except Exception as e:
        print(f"❌ Error writing CSV file: {e}")
        sys.exit(1)


def print_summary_table(all_metrics: List[Dict[str, Any]], top_n: int = 10):
    """
    Print a summary table of key metrics to console.
    
    Args:
        all_metrics: List of metric dictionaries
        top_n: Number of top reports to show
    """
    if not all_metrics:
        return
    
    print("\n" + "="*80)
    print("AGGREGATED EVALUATION SUMMARY")
    print("="*80)
    
    # Find common important metrics
    important_metrics = [
        'total_cases',
        'successful_evaluations', 
        'evaluation_success_rate',
        'average_f1_score',
        'average_precision',
        'average_recall',
        'average_exact_match',
        'average_map',
        'average_mrr',
        'average_ndcg@10',
        'average_llm_sql_skeleton_score',
        'average_llm_vector_component_score',
        'average_llm_overall_score'
    ]
    
    # Filter to only metrics that exist in the data
    existing_metrics = [m for m in important_metrics if any(m in metrics for metrics in all_metrics)]
    
    if not existing_metrics:
        print("No common metrics found across reports.")
        return
    
    # Print header
    col_width = 30
    print(f"{'Report Name':<{col_width}}", end='')
    for metric in existing_metrics[:5]:  # Show top 5 metrics
        metric_display = metric.replace('average_', '').replace('_', ' ').title()
        print(f"{metric_display:>15}", end='')
    print()
    print("-" * 80)
    
    # Print data rows
    for i, metrics in enumerate(all_metrics[:top_n]):
        report_name = metrics.get('report_name', 'Unknown')
        # Truncate long names
        if len(report_name) > col_width - 1:
            report_name = report_name[:col_width-4] + "..."
        
        print(f"{report_name:<{col_width}}", end='')
        for metric in existing_metrics[:5]:
            value = metrics.get(metric, '')
            if isinstance(value, float):
                print(f"{value:>15.4f}", end='')
            elif isinstance(value, int):
                print(f"{value:>15}", end='')
            else:
                print(f"{str(value):>15}", end='')
        print()
    
    if len(all_metrics) > top_n:
        print(f"... and {len(all_metrics) - top_n} more reports")
    
    print("="*80)


def find_json_files(directory: str) -> List[str]:
    """
    Find all JSON files in a directory.
    
    Args:
        directory: Path to directory to search
        
    Returns:
        List of JSON file paths
    """
    directory = Path(directory)
    if not directory.exists():
        print(f"Directory not found: {directory}")
        return []
    
    json_files = list(directory.glob("*.json"))
    return [str(f) for f in json_files]


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate evaluation metrics from multiple JSON reports into a CSV file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Aggregate specific files
  python aggregate_results.py --input report1.json report2.json --output summary.csv
  
  # Aggregate all JSON files in a directory
  python aggregate_results.py --input-dir ./reports --output summary.csv
  
  # Use wildcard pattern
  python aggregate_results.py --input model_*_report.json --output comparison.csv
  
  # Sort by a specific metric
  python aggregate_results.py --input *.json --output summary.csv --sort-by average_f1_score
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input', '-i',
        nargs='+',
        help='One or more JSON report files to aggregate'
    )
    input_group.add_argument(
        '--input-dir', '-d',
        help='Directory containing JSON report files'
    )
    
    # Output options
    parser.add_argument(
        '--output', '-o',
        default='aggregated_results.csv',
        help='Output CSV file path (default: aggregated_results.csv)'
    )
    
    # Sorting options
    parser.add_argument(
        '--sort-by', '-s',
        help='Sort results by this metric (descending). Example: average_f1_score'
    )
    
    # Display options
    parser.add_argument(
        '--no-table',
        action='store_true',
        help='Disable summary table output to console'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress all output except errors'
    )
    
    args = parser.parse_args()
    
    # Collect input files
    if args.input:
        input_files = args.input
    else:
        input_files = find_json_files(args.input_dir)
        if not input_files:
            print(f"No JSON files found in directory: {args.input_dir}")
            sys.exit(1)
    
    if not args.quiet:
        print(f"Found {len(input_files)} JSON file(s) to process")
    
    # Process each file
    all_metrics = []
    for file_path in input_files:
        if not args.quiet:
            print(f"Processing: {file_path}")
        
        report = load_json_file(file_path)
        if report is None:
            continue
        
        # Use filename (without extension) as report name
        report_name = Path(file_path).stem
        
        metrics = extract_summary_metrics(report, report_name)
        all_metrics.append(metrics)
    
    if not all_metrics:
        print("❌ No valid reports found. Exiting.")
        sys.exit(1)
    
    # Write to CSV
    write_to_csv(all_metrics, args.output, sort_by=args.sort_by)
    
    # Print summary table
    if not args.no_table and not args.quiet:
        print_summary_table(all_metrics)
    
    if not args.quiet:
        print(f"\n✅ Aggregation complete! Results saved to: {args.output}")


if __name__ == "__main__":
    main()
