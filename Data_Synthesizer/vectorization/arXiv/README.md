这个文件用来将arxiv数据集向量化

# 生成schema
```bash
python generate_schema.py --db-dir train --output-file train/table.json
```
产生的结果文件在results/table.json

# 加强schema
先使用enhance_tables_json.py往schema加入每个数据表格的示例信息，将.env中的ENHANCE_TABLE_MODE设置为"enhance_arxiv"后运行：
```bash
python enhance_tabels_json.py
```

# 找到语言信息很丰富的列
注意：要排除地名和人名这种语义信息不丰富的列
运行：
```bash
python find_semantic_rich_column.py
```

# 为这些列生成embedding
```bash
export HF_ENDPOINT=https://hf-mirror.com
```
修改.env相关变量后，运行：
```bash
python batch_vectorize_databases.py
```

# 生成新的schema并且填入样例数据
修改.env中VECTOR_DB_ROOT_GENERATE_SCHEMA等参数。运行：
```bash
python generate_vector_schema.py
python enhance_tables_json.py #可以省略这一步，也就不用修改.env了
```

