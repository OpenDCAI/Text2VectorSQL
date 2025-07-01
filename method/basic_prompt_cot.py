import json
import re
import argparse
from typing import List, Dict, Any, Optional
from vllm import LLM, SamplingParams
import os

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='使用COT基础提示词方法生成VectorSQL查询')
    
    # 必需参数
    parser.add_argument('--dataset', type=str, required=True,
                       help='数据集文件路径 (JSON格式)')
    parser.add_argument('--model_path', type=str, required=True,
                       help='本地模型路径')
    parser.add_argument('--output', type=str, required=True,
                       help='输出结果保存路径')
    
    # 可选参数
    parser.add_argument('--max_tokens', type=int, default=1024,
                       help='最大生成token数 (默认: 1024)')
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='采样温度 (默认: 0.1)')
    parser.add_argument('--top_p', type=float, default=1.0,
                       help='nucleus采样参数 (默认: 1.0)')
    parser.add_argument('--top_k', type=int, default=-1,
                       help='top-k采样参数 (默认: -1, 表示不使用)')
    
    return parser.parse_args()

class BasicPromptCOTMethod:
    """使用COT基础提示词方法生成VectorSQL查询的类"""
    
    def __init__(self, model_path: str, max_tokens: int = 1024, temperature: float = 0.1):
        """
        初始化BasicPromptCOTMethod
        
        Args:
            model_path: 本地模型路径
            max_tokens: 最大生成token数
            temperature: 生成温度
        """
        self.model_path = model_path
        self.llm = LLM(model=model_path, trust_remote_code=True)
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
    def create_prompt(self, schema: str, query: str) -> str:
        """
        创建用于生成VectorSQL的提示词（使用COT）
        
        Args:
            schema: 数据库模式
            query: 自然语言查询
            
        Returns:
            格式化的提示词
        """
        prompt = f"""You are an expert in SQLite with vector extensions. Given a database schema and a natural language query, generate the corresponding VectorSQL query using SQLite-Vec and SQLite-Lembed syntax.

## Available Extensions:

### SQLite-Vec:
- **MATCH operator**: Use `vector_column MATCH query_vector` for similarity search
- Automatically returns results ordered by similarity (most similar first)
- **LIMIT k**: Controls the number of most similar results returned (e.g., LIMIT 5 returns top 5 matches)

### SQLite-Lembed:
- `lembed(model_name, text)` function to generate embeddings from text
- **Model name**: Use "embed-model" as the default registered embedding model name
- This corresponds to the embedding model registered in the database
- Example: `lembed("embed-model", "your text here")`

## Syntax Examples:

### Basic vector similarity search:
```sql
SELECT rowid 
FROM vec_articles 
WHERE headline_embedding MATCH lembed("embed-model", "search text")
LIMIT 5;
```

## Database Schema:
{schema}

## Natural Language Query:
{query}

## Instructions:
1. Analyze what the user is searching for
2. **Determine if vector search is needed**: Check if the query requires semantic similarity, fuzzy matching, or content-based search
3. If vector search is needed:
   - Determine the appropriate text to embed using lembed()
   - Use the MATCH operator for vector similarity search
   - Include appropriate LIMIT clause for top-k results
4. Join with main tables if additional columns are needed beyond rowid
5. Combine vector search with traditional SQL filters when appropriate

## Response Format:
<thinking>
[Analyze the query step by step:
- What is the user looking for?
- Which table contains the vectors?
- What text should be embedded for comparison?
- Do we need to join with other tables?
- What columns should be returned?]
</thinking>

<sql>
[Your generated VectorSQL query here]
</sql>"""
        
        return prompt
    
    def extract_sql_from_response(self, response: str) -> Optional[str]:
        """
        从模型响应中提取SQL查询
        
        Args:
            response: 模型的完整响应
            
        Returns:
            提取的SQL查询，如果提取失败返回None
        """
        # 尝试提取<sql>标签内的内容
        sql_match = re.search(r'<sql>\s*(.*?)\s*</sql>', response, re.DOTALL | re.IGNORECASE)
        if sql_match:
            return sql_match.group(1).strip()
        
        # 备选方案：寻找以SELECT开头的SQL语句
        sql_match = re.search(r'(SELECT.*?;?)', response, re.DOTALL | re.IGNORECASE)
        if sql_match:
            return sql_match.group(1).strip()
        
        return None
    
    def generate_vectorsql(self, queries: List[Dict[str, str]], output_path: str = None) -> List[Dict[str, Any]]:
        """
        为查询列表生成VectorSQL
        
        Args:
            queries: 包含schema和query的字典列表
            output_path: 输出文件路径（可选）
            
        Returns:
            包含原始查询、提示词、响应和提取SQL的结果列表
        """
        results = []
        
        # 准备所有提示词
        prompts = []
        for item in queries:
            schema = item.get('schema', '')
            query = item.get('query', '')
            prompt = self.create_prompt(schema, query)
            prompts.append(prompt)
        
        # 批量生成响应
        print(f"正在为 {len(queries)} 个查询生成VectorSQL...")
        outputs = self.llm.generate(prompts, self.sampling_params)
        
        # 处理结果
        for i, (item, output) in enumerate(zip(queries, outputs)):
            response = output.outputs[0].text
            extracted_sql = self.extract_sql_from_response(response)
            
            result = {
                'id': i + 1,
                'original_schema': item.get('schema', ''),
                'original_query': item.get('query', ''),
                'prompt': prompts[i],
                'full_response': response,
                'extracted_sql': extracted_sql,
                'success': extracted_sql is not None
            }
            
            results.append(result)
            
            # 打印进度
            status = "✓" if extracted_sql else "✗"
            print(f"{status} Query {i+1}/{len(queries)}: {item.get('query', '')[:50]}...")
        
        # 保存结果
        if output_path:
            self.save_results(results, output_path)
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]], output_path: str):
        """
        保存生成结果到文件
        
        Args:
            results: 生成结果列表
            output_path: 输出文件路径
        """
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 计算统计信息
        total_queries = len(results)
        successful_queries = sum(1 for r in results if r['success'])
        success_rate = successful_queries / total_queries if total_queries > 0 else 0
        
        # 仅保存SQL查询为.sql文件
        self.save_sql_queries(results, output_path)
        
        print(f"SQL查询已保存到: {output_path}")
        print(f"成功率: {success_rate:.2%} ({successful_queries}/{total_queries})")
    
    def save_sql_queries(self, results: List[Dict[str, Any]], sql_output_path: str):
        """
        将生成的SQL查询保存为.sql文件
        
        Args:
            results: 生成结果列表
            sql_output_path: SQL文件输出路径
        """
        with open(sql_output_path, 'w', encoding='utf-8') as f:
            for result in results:
                if result['extracted_sql']:
                    f.write(f"{result['extracted_sql']}")
                    if not result['extracted_sql'].rstrip().endswith(';'):
                        f.write(';')
                else:
                    # 生成失败时写入空查询
                    f.write("SELECT NULL;")
                f.write('\n')

    def extract_sql_queries_only(self, results: List[Dict[str, Any]]) -> List[str]:
        """
        从结果中仅提取SQL查询列表
        
        Args:
            results: generate_vectorsql的结果
            
        Returns:
            SQL查询字符串列表
        """
        return [r['extracted_sql'] for r in results if r['extracted_sql']]

def load_dataset(dataset_path: str) -> List[Dict[str, str]]:
    """
    从文件加载数据集
    
    Args:
        dataset_path: 数据集文件路径，支持JSON格式
        
    Returns:
        包含schema和query的字典列表
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"数据集文件不存在: {dataset_path}")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        if dataset_path.endswith('.json'):
            data = json.load(f)
            # 支持不同的JSON格式
            if isinstance(data, list):
                return data
            elif 'queries' in data:
                return data['queries']
            else:
                raise ValueError("JSON格式不正确，需要包含queries列表或直接为列表格式")
        else:
            raise ValueError("目前仅支持JSON格式的数据集文件")



# 使用示例
def main():
    # 解析命令行参数
    args = parse_arguments()
    
    try:
        # 加载数据集
        print(f"正在加载数据集: {args.dataset}")
        queries = load_dataset(args.dataset)
        print(f"成功加载 {len(queries)} 个查询")
        
        # 创建采样参数
        sampling_params = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            top_k=args.top_k if args.top_k > 0 else None,
            stop=["</sql>", "\n\n\n"]
        )
        
        # 初始化方法
        print(f"正在加载模型: {args.model_path}")
        method = BasicPromptCOTMethod(
            model_path=args.model_path,
            max_tokens=args.max_tokens,
            temperature=args.temperature
        )
        
        # 更新采样参数
        method.sampling_params = sampling_params
        
        # 生成VectorSQL
        results = method.generate_vectorsql(
            queries=queries,
            output_path=args.output
        )
        
        # 输出统计信息
        successful_queries = sum(1 for r in results if r['success'])
        print(f"\n=== 生成完成 ===")
        print(f"总查询数: {len(results)}")
        print(f"成功提取SQL: {successful_queries}")
        print(f"成功率: {successful_queries/len(results):.2%}")
        print(f"结果已保存到: {args.output}")
        
    except Exception as e:
        print(f"错误: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
