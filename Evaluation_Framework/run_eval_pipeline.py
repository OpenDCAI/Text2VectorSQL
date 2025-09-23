# run_eval_pipeline.py - Main Pipeline Controller

import argparse
import subprocess
import sys
import os
import requests
import yaml

def run_command(command, description):
    """Run a shell command and handle errors."""
    print(f"\n{'='*60}")
    print(f"STAGE: {description}")
    print(f"COMMAND: {command}")
    print('='*60)
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=False)
        print(f"COMPLETED: {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"FAILED: {description} with exit code {e.returncode}")
        return False

def check_service_status(config_path="evaluation_config.yaml"):
    """Check and display embedding service status from config."""
    try:
        # Load config to get service settings
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        service_config = config.get('embedding_service', {})
        host = service_config.get('host', '127.0.0.1')
        port = service_config.get('port', 8000)
        
        response = requests.get(f"http://{host}:{port}/health", timeout=5)
        is_running = response.status_code == 200
        
        print(f"\nEmbedding Service 状态检查:")
        print(f"   服务地址: http://{host}:{port}")
        print(f"   运行状态: {'运行中' if is_running else '未运行'}")
        
        if is_running:
            try:
                health_data = response.json()
                models = health_data.get("loaded_models", [])
                if models:
                    print(f"   已加载模型: {', '.join(models)}")
            except:
                pass
        
        return is_running
        
    except Exception as e:
        print(f"无法检查服务状态: {e}")
        return False

def main():
    """
    Main pipeline controller for Text2VectorSQL evaluation.
    Can run individual stages or the complete pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Text2VectorSQL Evaluation Pipeline Controller",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline (embedding service auto-started by default)
  python run_eval_pipeline.py --all

  # Run only SQL execution (embedding service auto-started by default)
  python run_eval_pipeline.py --execute

  # Run only evaluation (requires existing execution results)
  python run_eval_pipeline.py --evaluate

  # Check service status
  python run_eval_pipeline.py --service-status

  # Disable automatic service management
  python run_eval_pipeline.py --all --no-service-management

  # Run with custom config
  python run_eval_pipeline.py --all --config my_config.yaml
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--all", 
        action="store_true", 
        help="Run complete pipeline (execute + evaluate)"
    )
    mode_group.add_argument(
        "--execute", 
        action="store_true", 
        help="Run only SQL execution stage"
    )
    mode_group.add_argument(
        "--evaluate", 
        action="store_true", 
        help="Run only evaluation stage"
    )
    mode_group.add_argument(
        "--service-status",
        action="store_true",
        help="Check embedding service status"
    )
    
    # Configuration
    parser.add_argument(
        "--config",
        default="evaluation_config.yaml",
        help="Path to the evaluation configuration file"
    )
    parser.add_argument(
        "--execution-results",
        help="Path to execution results file (for evaluate-only mode)"
    )
    parser.add_argument(
        "--no-service-management",
        action="store_true",
        help="Disable automatic service management for pipeline operations"
    )
    
    args = parser.parse_args()
    
    # Change to the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Handle service status command
    if args.service_status:
        check_service_status(args.config)
        return
    
    success = True
    
    if args.all or args.execute:
        # Stage 1: SQL Execution
        cmd = f"python sql_executor.py --config {args.config}"
        if args.no_service_management:
            cmd += " --no-service-management"
            
        success = run_command(cmd, "SQL Execution")
        
        if not success:
            print("\nPipeline stopped due to execution failure")
            sys.exit(1)
    
    if args.all or args.evaluate:
        # Stage 2: Evaluation
        cmd = f"python evaluate_results.py --config {args.config}"
        if args.execution_results:
            cmd += f" --execution-results {args.execution_results}"
        
        success = run_command(cmd, "Results Evaluation")
        
        if not success:
            print("\nPipeline stopped due to evaluation failure")
            sys.exit(1)
    
    if success:
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("="*60)
        if args.all:
            print("Both SQL execution and evaluation stages completed.")
        elif args.execute:
            print("SQL execution stage completed.")
            print("Run 'python run_eval_pipeline.py --evaluate' to compute metrics.")
        elif args.evaluate:
            print("Evaluation stage completed.")
        print(f"Check the output files specified in {args.config}")

if __name__ == "__main__":
    main()