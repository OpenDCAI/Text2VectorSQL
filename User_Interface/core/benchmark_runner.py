from litellm import modify_params
import yaml
import time
import json
import os
import re
import sqlite_lembed
from .db.mock_vsql_client import VectorSQLClient
from .db.database import register_database
from .db.connection_manager import connection_manager
from .metrics import calculate_accuracy, calculate_mrr, calculate_map, calculate_ndcg

def load_config(config_path: str) -> dict:
    """从 YAML 文件加载配置。"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def read_sql_queries(file_path: str) -> list[str]:
    """从文件中读取并解析 SQL 查询。"""
    if not os.path.exists(file_path):
        print(f"警告: SQL 文件未找到于 '{file_path}'")
        return []
    with open(file_path, 'r') as f:
        content = f.read()
    # 移除注释并规范化
    content = re.sub(r'--.*\n', '\n', content)
    clean_sql = content.replace('\n', ' ').strip()
    # 按分号分割SQL语句并移除空语句
    return [stmt.strip() for stmt in clean_sql.split(';') if stmt.strip()]

def setup_lembed(connection, embedding_config):
    """注册指定的 lembed 模型。
    假定 lembed 扩展已经由数据库连接加载。
    """
    if not embedding_config:
        return

    name = embedding_config.get('name')
    model_path = embedding_config.get('model_path')

    if not all([name, model_path]):
        print("警告: 嵌入配置不完整 (需要 name 和 model_path)。跳过 lembed 模型注册。")
        return

    try:
        print(f"正在注册模型 '{name}' 从 '{model_path}'...")
        cursor = connection.cursor()
        # 确保我们不会插入重复的模型名称，这是一个好习惯
        cursor.execute("DELETE FROM temp.lembed_models WHERE name = ?", (name,))
        cursor.execute(
            "INSERT INTO temp.lembed_models(name, model) SELECT ?, lembed_model_from_file(?) ",
            (name, model_path)
        )
        connection.commit()
        print(f"模型 '{name}' 已成功注册。")

    except Exception as e:
        print(f"错误: 注册 lembed 模型失败: {e}")
        raise

def process_sql_for_lembed(sql, embedding_config):
    """将 lembed() 调用中的模型名称替换为配置的名称。"""
    if not embedding_config or not sql or 'lembed(' not in sql.lower():
        return sql

    configured_name = embedding_config.get('name')
    if not configured_name:
        return sql

    # 正则表达式，用于查找 lembed() 的第一个参数（模型名称）并替换它。
    # 它不区分大小写，处理不同的间距，并保留原始的引号类型。
    # (lembed\s*\(s*) - 第1组：捕获 "lembed(" 和任何前导空格
    # (['"]) - 第2组：捕获开放引号（' 或 "）
    # (?:.*?) - 非捕获组：匹配模型名称，但不捕获它
    # (\2\s*,) - 第3组：捕获匹配的闭合引号和后面的逗号

    #print(sql)

    pattern = r"lembed\(['\"](.*?)['\"],\s*['\"].*?['\"]\)"

    def replacer(match):
        # Reconstruct the lembed string with the new model name
        # match.group(0) is the entire matched string, e.g., lembed("embed-model","...")
        # We need to rebuild it to ensure correct quoting and spacing.
        return f"lembed(\"{configured_name}\", {match.group(0).split(',', 1)[1]}"

    # Use re.sub with a function for the replacement
    modified_sql = re.sub(pattern, replacer, sql)
    # 这里使用了 re.sub 的函数形式来处理替换，
    
    # 替换第一个匹配项
    
    #print(f"处理后的 SQL: {modified_sql}")
    return modified_sql
    # 替换字符串使用反向引用来保留原始字符串的部分，
    # 并插入配置的模型名称。
    replacement = rf"\1\2{configured_name}\3"
    
    modified_sql = pattern.sub(replacement, sql)
        
    return modified_sql

def run_benchmark(config_path: str):
    """根据给定的配置文件运行完整的基准测试。"""
    config = load_config(config_path)
    db_config = config['database']
    output_config = config['output']
    queries_config = config['queries']
    metrics_config = config.get('metrics', [])
    embedding_config = config.get('embedding')

    # 1. 从配置中分离 gold 和 test 文件
    gold_file_config = None
    test_file_configs = []
    for qc in queries_config:
        if qc.get('is_gold', False):
            if gold_file_config:
                print("错误: 配置中找到多个 is_gold: true 的文件。只应有一个。")
                return
            gold_file_config = qc
        else:
            test_file_configs.append(qc)

    # 2. 读取 gold 查询
    gold_queries = []
    if gold_file_config:
        gold_queries = read_sql_queries(gold_file_config['path'])
    
    # 3. 解析指标配置
    parsed_metrics = []
    for m in metrics_config:
        if isinstance(m, str):
            parsed_metrics.append({'name': m})
        elif isinstance(m, dict) and 'name' in m:
            parsed_metrics.append(m)

    metrics_requiring_gold_names = {'accuracy', 'mrr', 'map', 'ndcg'}
    gold_metrics_enabled = any(m['name'] in metrics_requiring_gold_names for m in parsed_metrics)

    if gold_metrics_enabled and not gold_queries:
        print("错误: 已启用需要黄金标准文件的指标，但未在配置中提供 is_gold: true 的文件或文件为空。")
        return

    # 设置客户端和数据库
    client = VectorSQLClient(connection_manager)
    db_conn = register_database(
        db_config['name'],
        db_config['path'],
    )
    if not db_conn:
        print("数据库设置失败。正在中止。")
        return

    # 设置 lembed
    if embedding_config:
        try:
            setup_lembed(db_conn, embedding_config)
        except Exception as e:
            print(f"因 lembed 设置错误而中止基准测试: {e}")
            connection_manager.close_all()
            return

    results_by_file = {}
    overall_metrics_list = []

    for test_file in test_file_configs:
        test_file_name = test_file['name']
        test_file_path = test_file['path']
        print(f"\n===== 开始处理测试文件: {test_file_name} ({test_file_path}) =====")
        
        test_queries = read_sql_queries(test_file_path)
        if not test_queries:
            print(f"警告: 测试文件 {test_file_name} 为空或未找到。跳过。")
            continue

        if gold_metrics_enabled and len(gold_queries) != len(test_queries):
            print(f"错误: 为了计算指标，gold_sql ({len(gold_queries)} 个查询) 和测试文件 {test_file_name} ({len(test_queries)} 个查询) 必须包含相同数量的查询。")
            continue

        individual_results_for_file = []
        metrics_list_for_file = []
        successful_queries_for_file = 0
        failed_queries_for_file = 0

        for i, test_query in enumerate(test_queries):
            print(f"--- 正在运行查询 #{i+1} (来自 {test_file_name}) ---")
            print(f"  Test: {test_query[:70]}...")
            
            metrics = {}
            query_status = 'success'
            error_message = None
            processed_gold_query = None # 初始化
            
            try:
                # 为 lembed 处理 SQL
                processed_test_query = process_sql_for_lembed(test_query, embedding_config)
                if processed_test_query != test_query:
                    print(f"  (lembed SQL 已修改为使用模型 '{embedding_config['name']}')")

                # 执行测试查询并计时
                start_time = time.time()
                test_results = client.execute(db_config['name'], processed_test_query)
                end_time = time.time()

                # 计算简单指标
                for metric_conf in parsed_metrics:
                    if metric_conf['name'] == 'execution_time':
                        metrics['execution_time'] = end_time - start_time
                    if metric_conf['name'] == 'result_count':
                        metrics['result_count'] = len(test_results)

                # 如果需要，执行 gold 查询并计算指标
                if gold_metrics_enabled:
                    gold_query = gold_queries[i]
                    processed_gold_query = process_sql_for_lembed(gold_query, embedding_config)
                    if processed_gold_query != gold_query:
                         print(f"  (lembed Gold SQL 已修改为使用模型 '{embedding_config['name']}')")

                    print(f"  Gold: {processed_gold_query[:70]}...")
                    gold_results = client.execute(db_config['name'], processed_gold_query)
                    
                    for metric_conf in parsed_metrics:
                        metric_name = metric_conf['name']
                        if metric_name == 'accuracy':
                            metrics['accuracy'] = calculate_accuracy(test_results, gold_results)
                        elif metric_name in ('mrr', 'map', 'ndcg'):
                            k = metric_conf.get('k', 100)  # 默认 k
                            metric_key = f"{metric_name}_at_{k}"
                            if metric_name == 'mrr':
                                metrics[metric_key] = calculate_mrr(test_results, gold_results, k=k)
                            elif metric_name == 'map':
                                metrics[metric_key] = calculate_map(test_results, gold_results, k=k)
                            elif metric_name == 'ndcg':
                                metrics[metric_key] = calculate_ndcg(test_results, gold_results, k=k)
                
                metrics_list_for_file.append(metrics)
                overall_metrics_list.append(metrics) # 用于总体平均值
                successful_queries_for_file += 1

            except Exception as e:
                query_status = 'failure'
                error_message = str(e)
                failed_queries_for_file += 1
                print(f"查询失败: {e}")

            individual_results_for_file.append({
                'source_file': test_file_name,
                'original_test_query': test_query,
                'original_gold_query': gold_queries[i] if gold_metrics_enabled else None,
                'processed_test_query': processed_test_query,
                'processed_gold_query': processed_gold_query,
                'metrics': metrics,
                'status': query_status,
                'error': error_message
            })
            print(f"查询完成。状态: {query_status}。指标: {metrics}")

        # 计算当前文件的平均指标
        average_metrics_for_file = {}
        if metrics_list_for_file:
            metric_keys = set(key for metrics_dict in metrics_list_for_file for key in metrics_dict.keys())
            for key in metric_keys:
                values = [m[key] for m in metrics_list_for_file if key in m and m[key] is not None]
                if values:
                    average_metrics_for_file[f"average_{key}"] = sum(values) / len(values)
                else:
                    average_metrics_for_file[f"average_{key}"] = None
        
        results_by_file[test_file_name] = {
            "source_file_path": test_file_path,
            "summary_metrics": {
                **average_metrics_for_file,
                "total_queries": len(test_queries),
                "successful_queries": successful_queries_for_file,
                "failed_queries": failed_queries_for_file
            },
            "individual_query_results": individual_results_for_file
        }

    # 计算总体平均指标
    overall_average_metrics = {}
    if overall_metrics_list:
        metric_keys = set(key for metrics_dict in overall_metrics_list for key in metrics_dict.keys())
        for key in metric_keys:
            values = [m[key] for m in overall_metrics_list if key in m and m[key] is not None]
            if values:
                overall_average_metrics[f"average_{key}"] = sum(values) / len(values)
            else:
                overall_average_metrics[f"average_{key}"] = None

    total_queries_run = sum(len(v['individual_query_results']) for v in results_by_file.values())
    total_successful = sum(v['summary_metrics']['successful_queries'] for v in results_by_file.values())
    total_failed = sum(v['summary_metrics']['failed_queries'] for v in results_by_file.values())

    # 创建最终结果结构
    final_results_structure = {
        "benchmark_summary": {
            "gold_standard_file": gold_file_config['path'] if gold_file_config else None,
            "overall_metrics": {
                **overall_average_metrics,
                "total_queries_run": total_queries_run,
                "total_successful": total_successful,
                "total_failed": total_failed
            }
        },
        "results_by_file": results_by_file
    }

    # 保存结果
    save_results(final_results_structure, output_config)

    # 清理：关闭所有由管理器维护的连接
    connection_manager.close_all()
    print("\n所有数据库连接已关闭。基准测试完成。")

def save_results(results: dict, output_config: dict):
    """将基准测试结果保存到文件中。"""
    output_dir = output_config['dir']
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(output_dir, f"results_{timestamp}.{output_config['format']}")

    print(f"\n正在将结果保存到 '{file_path}'...")
    with open(file_path, 'w') as f:
        if output_config['format'] == 'json':
            json.dump(results, f, indent=4)
    
    print("结果已成功保存。")
