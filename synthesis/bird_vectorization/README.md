这个文件用来将BRID数据集向量化

# 加强schema
先使用enhance_tables_json.py往schema加入每个数据表格的示例信息，将.env中的ENHANCE_TABLE_MODE设置为"enhance_bird"后运行：
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
运行：
```bash
python batch_vectorize_databases.py
```

# 生成新的schema并且填入样例数据
将.env中的ENHANCE_TABLE_MODE设置为"enhance_vec_bird",并且补充enhance_tables_json的其他参数后运行：
```bash
python generate_vector_schema.py
python enhance_tables_json.py #可以省略这一步，也就不用修改.env了
```

# 针对语言信息很丰富的列生成sql


# 使用sql合成question
