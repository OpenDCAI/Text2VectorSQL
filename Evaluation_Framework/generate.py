#!/usr/bin/env python3
"""
generate_predictions.py - VectorSQL 预测生成脚本

支持两种生成方式：
1. vLLM 离线推理（支持多 GPU 张量并行）
2. API 在线调用（支持多线程并发）

输出格式适配评测框架的输入要求。
"""

import json
import re
import argparse
import os
import sys
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from regex import P
from tqdm import tqdm
import time
import requests
import yaml  # 导入 yaml 库

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None
    SamplingParams = None


# def preprocess_dataset_item(item: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     对数据集中的每个项目进行预处理，按照指定顺序拼接提示词
    
#     参数:
#         item: 数据项字典，包含'input'和'question'等键
        
#     返回:
#         处理后的数据项
#     """
#     # 创建副本以避免修改原始数据
#     processed_item = item.copy()
    
#     # 确保包含必要的键
#     if 'schema' not in processed_item:
#         raise ValueError("数据项缺少 'schema' 键")
    
#     if 'db_type' not in processed_item:
#         raise ValueError("数据项缺少 'db_type' 键")
    
#     if 'embedding_model_name' not in processed_item:
#         raise ValueError("数据项缺少 'embedding_model_name' 键")
    
#     if 'database_note_prompt' not in processed_item:
#         raise ValueError("数据项缺少 'database_note_prompt' 键")
    
#     if 'question' not in processed_item:
#         raise ValueError("数据项缺少 'question' 键")
    
#     # 按照指定顺序拼接字段
#     prompt_parts = []
    
#     # 1. database_note_prompt
#     prompt_parts.append("### Database Note")
#     prompt_parts.append(str(processed_item['database_note_prompt']))
    
#     # 2. embedding_model_name
#     prompt_parts.append("### Embedding Model")
#     prompt_parts.append(str(processed_item['embedding_model_name']))
    
#     # 3. db_type
#     prompt_parts.append("### Database Type")
#     prompt_parts.append(str(processed_item['db_type']))
    
#     # 4. schema
#     prompt_parts.append("### Database Schema")
#     prompt_parts.append(str(processed_item['schema']))
    
#     # 5. 最后加入question
#     prompt_parts.append("### Question")
#     prompt_parts.append(str(processed_item['question']))
    
#     # 拼接所有部分
#     processed_prompt = "\n".join(prompt_parts)
    
#     # 更新item的input字段为拼接后的字符串
#     processed_item['input'] = processed_prompt
    
#     return processed_item

def parse_arguments():
    """解析命令行参数，支持从 YAML 文件加载配置"""
    parser = argparse.ArgumentParser(
        description='VectorSQL 预测生成脚本 - 支持 vLLM 和 API 两种方式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

1. vLLM 离线推理（单 GPU）:
   python generate_predictions.py \\
     --mode vllm \\
     --dataset data/bench.json \\
     --model_path /path/to/model \\
     --output results/predictions.json

2. vLLM 离线推理（多 GPU 张量并行）:
   python generate_predictions.py \\
     --mode vllm \\
     --dataset data/bench.json \\
     --model_path /path/to/model \\
     --output results/predictions.json \\
     --tensor_parallel_size 4

3. API 在线调用（多线程）:
   python generate_predictions.py \\
     --mode api \\
     --dataset data/bench.json \\
     --api_url https://api.openai.com/v1/chat/completions \\
     --api_key sk-xxx \\
     --model_name gpt-4 \\
     --output results/predictions.json \\
     --num_threads 10

4. 使用 YAML 配置文件:
   python generate_predictions.py --config config.yaml
   # 可以在命令行中覆盖 YAML 文件中的参数
   python generate_predictions.py --config config.yaml --num_threads 20
        """
    )
    
    parser.add_argument('--config', type=str,default='/mnt/b_public/data/ydw/Text2VectorSQL/Evaluation_Framework/generate_config.yaml', help='YAML 配置文件路径')
    
    # 通用参数
    parser.add_argument('--mode', type=str, choices=['vllm', 'api'], help='生成模式: vllm (离线推理) 或 api (在线调用)')
    parser.add_argument('--dataset', type=str, help='数据集文件路径 (JSON 格式)')
    parser.add_argument('--output', type=str, help='输出结果保存路径 (JSON 格式)')
    
    # vLLM 模式参数
    vllm_group = parser.add_argument_group('vLLM 模式参数')
    vllm_group.add_argument('--model_path', type=str, help='[vLLM] 本地模型路径或 HuggingFace 模型名称')
    vllm_group.add_argument('--tensor_parallel_size', type=int, help='[vLLM] 张量并行大小（GPU 数量）')
    vllm_group.add_argument('--gpu_memory_utilization', type=float, help='[vLLM] GPU 显存利用率 (0.0-1.0)')
    vllm_group.add_argument('--max_model_len', type=int, help='[vLLM] 模型最大序列长度')
    vllm_group.add_argument('--trust_remote_code', action='store_true', help='[vLLM] 是否信任远程代码')
    
    # API 模式参数
    api_group = parser.add_argument_group('API 模式参数')
    api_group.add_argument('--api_url', type=str, help='[API] API 端点 URL')
    api_group.add_argument('--api_key', type=str, help='[API] API 密钥')
    api_group.add_argument('--model_name', type=str, help='[API] 模型名称')
    api_group.add_argument('--num_threads', type=int, help='[API] 并发线程数')
    api_group.add_argument('--api_timeout', type=int, help='[API] API 请求超时时间（秒）')
    api_group.add_argument('--retry_times', type=int, help='[API] 失败重试次数')
    
    # 采样参数
    sampling_group = parser.add_argument_group('采样参数')
    sampling_group.add_argument('--max_tokens', type=int, help='最大生成 token 数')
    sampling_group.add_argument('--temperature', type=float, help='采样温度')
    sampling_group.add_argument('--top_p', type=float, help='nucleus 采样参数')
    sampling_group.add_argument('--top_k', type=int, help='top-k 采样参数')
    
    parser.add_argument('--save_interval', type=int, help='每处理多少条数据保存一次中间结果')
    
    # 解析命令行参数
    args, unknown = parser.parse_known_args()
    
    # 如果提供了 YAML 配置文件，则加载它
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # 将 YAML 文件中的值设置为默认值
        parser.set_defaults(**config)
    
    # 重新解析参数，以使命令行参数能够覆盖 YAML 文件中的值
    args = parser.parse_args()
    
    return args


def load_benchmark_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """加载 benchmark 数据集"""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"数据集文件不存在: {dataset_path}")
    
    print(f"正在加载数据集: {dataset_path}")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        dataset = data
    elif isinstance(data, dict) and 'data' in data:
        dataset = data['data']
    else:
        raise ValueError("不支持的 JSON 格式")
    
    print(f"✓ 成功加载 {len(dataset)} 条数据")
    
    required_field = 'input'
    for i, item in enumerate(dataset):
        if required_field not in item:
            raise ValueError(f"第 {i} 条数据缺少必需字段: {required_field}")
    
    return dataset


# def create_json_format_prompt(original_input: str) -> str:
#     """在原始输入后添加 JSON 格式输出要求"""
#     json_instruction = """

# ## Response Format

# Please respond with a JSON object in the following format:

# ```json
# {
#   "reasoning": "Your step-by-step analysis and reasoning process",
#   "sql": "The generated VectorSQL query"
# }
# ```

# **Important:**
# - The JSON must be valid and parseable
# - The "reasoning" field should contain your thought process
# - The "sql" field should contain ONLY the SQL query, without additional explanation
# - Do not include any text outside the JSON object
# """
    
#     return original_input + json_instruction


def extract_sql_from_json_response(response: str) -> Tuple[Optional[str], Optional[str]]:
    """从模型的 JSON 响应中提取 SQL 查询和推理过程"""
    if not response:
        return None, None
    
    json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL | re.IGNORECASE)
    if json_match:
        json_str = json_match.group(1).strip()
        try:
            result = json.loads(json_str)
            sql = result.get('sql', '').strip()
            reasoning = result.get('reasoning', '').strip()
            if sql:
                return sql, reasoning
        except json.JSONDecodeError:
            pass
    
    json_match = re.search(r'\{[^{}]*"sql"[^{}]*\}', response, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        try:
            result = json.loads(json_str)
            sql = result.get('sql', '').strip()
            reasoning = result.get('reasoning', '').strip()
            if sql:
                return sql, reasoning
        except json.JSONDecodeError:
            pass
    
    try:
        start_idx = response.find('{')
        end_idx = response.rfind('}')
        if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
            json_str = response[start_idx:end_idx+1]
            result = json.loads(json_str)
            sql = result.get('sql', '').strip()
            reasoning = result.get('reasoning', '').strip()
            if sql:
                return sql, reasoning
    except (json.JSONDecodeError, ValueError):
        pass
    
    sql = _fallback_extract_sql(response)
    if sql:
        return sql, None
    
    return None, None


def _fallback_extract_sql(response: str) -> Optional[str]:
    """备选的 SQL 提取方法"""
    sql_match = re.search(r'<sql>\s*(.*?)\s*</sql>', response, re.DOTALL | re.IGNORECASE)
    if sql_match:
        return sql_match.group(1).strip()
    
    sql_match = re.search(r'```sql\s*(.*?)\s*```', response, re.DOTALL | re.IGNORECASE)
    if sql_match:
        return sql_match.group(1).strip()
    
    sql_match = re.search(r'```\s*(SELECT.*?)\s*```', response, re.DOTALL | re.IGNORECASE)
    if sql_match:
        return sql_match.group(1).strip()
    
    sql_match = re.search(r'(SELECT\s+.*?;)', response, re.DOTALL | re.IGNORECASE)
    if sql_match:
        return sql_match.group(1).strip()
    
    sql_match = re.search(r'(SELECT\s+.+?)(?:\n\n|$)', response, re.DOTALL | re.IGNORECASE)
    if sql_match:
        sql = sql_match.group(1).strip()
        if not sql.endswith(';'):
            sql += ';'
        return sql
    
    return None


def format_for_eval_framework(item: Dict[str, Any], generated_sql: Optional[str], reasoning: Optional[str] = None) -> Dict[str, Any]:
    """格式化输出以适配评测框架"""
    if 'query_id' not in item:
        import hashlib
        query_text = f"{item.get('db_id', '')}_{item.get('question', '')}"
        query_id = hashlib.md5(query_text.encode()).hexdigest()[:16]
    else:
        query_id = item['query_id']
    
    result = {
        'query_id': query_id,
        'db_id': item.get('db_id', ''),
        'db_identifier': item.get('db_id', ''),
        'db_type': item.get('db_type', 'sqlite'),
        'sql': item.get('sql', ''),
        'question': item.get('question', ''),
        'scheme': item.get('scheme', ''),
        'syntax': item.get('syntax', ''),
        'embed': item.get('embed', ''),
        'predicted_sql': generated_sql if generated_sql else '',
        "sql_candidate": item.get("sql_candidate", []),
    }
    
    if reasoning:
        result['reasoning'] = reasoning
    
    for key in ['sql_complexity', 'vector_complexity', 'question_style', 'sql_explanation', 'external_knowledge', 'input']:
        if key in item:
            result[key] = item[key]
    
    return result


class VLLMGenerator:
    """使用 vLLM 生成 VectorSQL 查询"""
    
    def __init__(self, model_path: str, tensor_parallel_size: int = 1, gpu_memory_utilization: float = 0.9, max_model_len: Optional[int] = None, trust_remote_code: bool = False):
        if LLM is None or SamplingParams is None:
            raise ImportError("vLLM 未安装。请运行: pip install vllm")
        
        self.model_path = model_path
        
        print(f"\n正在初始化 vLLM 模型...")
        print(f"  模型路径: {model_path}")
        print(f"  张量并行大小: {tensor_parallel_size}")
        print(f"  GPU 显存利用率: {gpu_memory_utilization}")
        
        llm_kwargs = {
            'model': model_path,
            'tensor_parallel_size': tensor_parallel_size,
            'gpu_memory_utilization': gpu_memory_utilization,
            'trust_remote_code': trust_remote_code,
        }
        
        if max_model_len is not None:
            llm_kwargs['max_model_len'] = max_model_len
        
        self.llm = LLM(**llm_kwargs)
        print("✓ 模型加载完成\n")
    
    def generate(self, dataset: List[Dict[str, Any]], max_tokens: int = 2048, temperature: float = 0.1, top_p: float = 1.0, top_k: int = -1, save_interval: int = 50, output_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """批量生成预测"""
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k if top_k > 0 else -1,
            stop=["<|endoftext|>", "<|im_end|>"]
        )
        
        print(f"开始批量生成...")
        print(f"  数据集大小: {len(dataset)}")
        print(f"  采样参数: temperature={temperature}, max_tokens={max_tokens}\n")
        
        prompts=[]
        for item in dataset:
            if 'input' not in item:
                raise ValueError("数据项缺少 'input' 键")
            prompts.append(item['input'])
        
        print("正在进行批量推理...")
        outputs = self.llm.generate(prompts, sampling_params)
        
        results = []
        successful_count = 0
        
        print("\n正在处理生成结果...")
        for i, (item, output) in enumerate(tqdm(zip(dataset, outputs), total=len(dataset))):
            response_text = output.outputs[0].text
            extracted_sql, reasoning = extract_sql_from_json_response(response_text)
            
            result = format_for_eval_framework(item, extracted_sql, reasoning)
            results.append(result)
            
            if extracted_sql:
                successful_count += 1
            
            if output_path and (i + 1) % save_interval == 0:
                save_intermediate_results(results, output_path, i + 1)
        
        print(f"\n=== 生成完成 ===")
        print(f"总数据量: {len(dataset)}")
        print(f"成功提取 SQL: {successful_count}")
        print(f"成功率: {successful_count/len(dataset):.2%}")
        
        remove_intermediate_results(output_path)
        
        return results


class APIGenerator:
    """使用 API 生成 VectorSQL 查询（支持多线程）"""
    
    def __init__(self, api_url: str, api_key: str, model_name: str, timeout: int = 60, retry_times: int = 3):
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name
        self.timeout = timeout
        self.retry_times = retry_times
        
        print(f"\n初始化 API 生成器...")
        print(f"  API URL: {api_url}")
        print(f"  模型名称: {model_name}")
        print(f"  超时时间: {timeout}s")
        print(f"  重试次数: {retry_times}")
    
    def _call_api(self, prompt: str, max_tokens: int, temperature: float, top_p: float) -> Optional[str]:
        """调用 API 生成响应"""
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        payload = {"model": self.model_name, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": temperature, "top_p": top_p}
        
        for attempt in range(self.retry_times):
            try:
                response = requests.post(self.api_url, headers=headers, json=payload, timeout=self.timeout)
                response.raise_for_status()
                result = response.json()
                return result['choices'][0]['message']['content']
            except Exception as e:
                if attempt < self.retry_times - 1:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    print(f"  ✗ API 调用失败: {e}")
                    return None
        return None
    
    def _process_single_item(self, item: Dict[str, Any], index: int, max_tokens: int, temperature: float, top_p: float) -> Dict[str, Any]:
        """处理单个数据项"""
        if 'input' not in item:
            raise ValueError("数据项缺少 'schema' 键")
        prompt=item['input']
        response_text = self._call_api(prompt, max_tokens, temperature, top_p)
        
        extracted_sql = None
        reasoning = None
        if response_text:
            extracted_sql, reasoning = extract_sql_from_json_response(response_text)
        
        return format_for_eval_framework(item, extracted_sql, reasoning)
    
    def generate(self, dataset: List[Dict[str, Any]], max_tokens: int = 2048, temperature: float = 0.1, top_p: float = 1.0, top_k: int = -1, num_threads: int = 5, save_interval: int = 50, output_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """多线程批量生成"""
        print(f"\n开始多线程生成...")
        print(f"  数据集大小: {len(dataset)}")
        print(f"  并发线程数: {num_threads}")
        print(f"  采样参数: temperature={temperature}, max_tokens={max_tokens}\n")
        
        results = [None] * len(dataset)
        successful_count = 0
        lock = Lock()
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {executor.submit(self._process_single_item, item, i, max_tokens, temperature, top_p): i for i, item in enumerate(dataset)}
            
            with tqdm(total=len(dataset), desc="生成进度") as pbar:
                for future in as_completed(futures):
                    index = futures[future]
                    try:
                        result = future.result()
                        results[index] = result
                        
                        with lock:
                            if result['predicted_sql']:
                                successful_count += 1
                        
                        if output_path and (index + 1) % save_interval == 0:
                            with lock:
                                completed_results = [r for r in results if r is not None]
                                save_intermediate_results(completed_results, output_path, index + 1)
                    except Exception as e:
                        print(f"  ✗ 处理第 {index} 条数据时出错: {e}")
                    
                    pbar.update(1)
        
        print(f"\n=== 生成完成 ===")
        print(f"总数据量: {len(dataset)}")
        print(f"成功提取 SQL: {successful_count}")
        print(f"成功率: {successful_count/len(dataset):.2%}")
        
        return [r for r in results if r is not None]


def save_intermediate_results(results: List[Dict[str, Any]], output_path: str, count: int):
    """保存中间结果"""
    temp_path = output_path.replace('.json', f'.temp_{count}.json')
    with open(temp_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

def remove_intermediate_results(output_path: str):
    """
    Deletes intermediate result files generated by save_intermediate_results.

    It finds and removes all files in the same directory as the final output
    that match the pattern '.temp_COUNT.json'.

    Args:
        output_path: The path to the final output file, used to determine
                     the naming pattern and location of intermediate files.
    """
    if not output_path or not isinstance(output_path, str):
        print("Warning: Invalid output path provided for cleanup. Skipping.")
        return

    output_dir = os.path.dirname(output_path)
    if not output_dir:
        output_dir = '.' # Default to the current directory if path has no directory part

    if not os.path.isdir(output_dir):
        # If the directory doesn't exist, there's nothing to clean up.
        return

    # Extract the base name of the output file without the '.json' extension
    base_name = os.path.basename(output_path)
    file_prefix = base_name.replace('.json', '')

    # Create a regular expression to match the temporary files.
    # re.escape ensures that any special characters in the filename are treated literally.
    pattern = re.compile(f"^{re.escape(file_prefix)}\.temp_\d+\.json$")

    deleted_count = 0
    print("\nCleaning up intermediate result files...")

    # Iterate over all files in the directory
    for filename in os.listdir(output_dir):
        if pattern.match(filename):
            file_to_delete = os.path.join(output_dir, filename)
            try:
                os.remove(file_to_delete)
                print(f"  - Deleted: {filename}")
                deleted_count += 1
            except OSError as e:
                print(f"  ✗ Error deleting file {filename}: {e}")

    if deleted_count > 0:
        print(f"✓ Cleanup complete. Removed {deleted_count} temporary file(s).")
    else:
        print("✓ No intermediate files found to clean up.")

def save_results(results: List[Dict[str, Any]], output_path: str):
    """保存最终结果"""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n正在保存结果到: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 结果已保存: {output_path}")
    
    total = len(results)
    successful = sum(1 for r in results if r.get('predicted_sql'))
    print(f"\n最终统计:")
    print(f"  总查询数: {total}")
    print(f"  成功生成: {successful}")
    print(f"  失败数: {total - successful}")
    print(f"  成功率: {successful/total:.2%}")


def main():
    """主函数"""
    args = parse_arguments()
    
    try:
        if not args.dataset:
            raise ValueError("数据集文件路径 --dataset 未指定")
        dataset = load_benchmark_dataset(args.dataset)
        
        if args.mode == 'vllm':
            if not args.model_path:
                raise ValueError("vLLM 模式需要指定 --model_path")
            
            generator = VLLMGenerator(
                model_path=args.model_path,
                tensor_parallel_size=args.tensor_parallel_size,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=args.max_model_len,
                trust_remote_code=args.trust_remote_code
            )
            
            results = generator.generate(
                dataset=dataset,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                save_interval=args.save_interval,
                output_path=args.output
            )
            
        elif args.mode == 'api':
            if not args.api_url or not args.api_key or not args.model_name:
                raise ValueError("API 模式需要指定 --api_url, --api_key 和 --model_name")
            
            generator = APIGenerator(
                api_url=args.api_url,
                api_key=args.api_key,
                model_name=args.model_name,
                timeout=args.api_timeout,
                retry_times=args.retry_times
            )
            
            results = generator.generate(
                dataset=dataset,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                num_threads=args.num_threads,
                save_interval=args.save_interval,
                output_path=args.output
            )
        else:
            raise ValueError(f"未知模式: {args.mode}")
        
        save_results(results, args.output)
        
        print("\n✓ 全部完成！")
        print(f"\n接下来可以运行评测:")
        print(f"  python sql_executor.py --config evaluation_config.yaml")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠ 用户中断执行")
        return 1
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    # 在运行前，请确保已安装 PyYAML: pip install PyYAML
    sys.exit(main())
