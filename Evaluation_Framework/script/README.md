这个目录用了为评估框架生成输入文件。

它读取Data_Synthesizer/pipeline/sqlite/results目录下，数据库目录中的candidate_sql.sql文件。然后产生评估框架需要的ground_truth.json和eval_queries.json文件。

你需要先去掉config.yaml.example的".example"，然后为其添加你的模型调用API。

然后需要修改Evaluation_Framework/script/api_pipeline.py文件中的DATASET_BACKEND = "sqlite" 
DATASET_TO_LOAD = "toy_spider"参数，来选择数据库后端和数据库。

运行：
```bash
python api_pipeline.py
```
即可得到最终的文件用于评估。
