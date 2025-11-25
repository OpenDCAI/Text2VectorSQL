python migrate_db.py --source /mnt/DataFlow/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/sqlite/results/arxiv/vector_databases
python migrate_db.py --source /mnt/DataFlow/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/sqlite/results/bird/vector_databases
# python migrate_db.py --source /mnt/DataFlow/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/sqlite/results/spider/vector_databases
python migrate_db.py --source /mnt/DataFlow/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/sqlite/results/wikipedia_multimodal/vector_databases

python migrate_db_myscale.py --source /mnt/DataFlow/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/sqlite/results/arxiv/vector_databases
python migrate_db_myscale.py --source /mnt/DataFlow/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/sqlite/results/bird/vector_databases
# python migrate_db_myscale.py --source /mnt/DataFlow/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/sqlite/results/spider/vector_databases
python migrate_db_myscale.py --source /mnt/DataFlow/ydw/Text2VectorSQL/Data_Synthesizer/pipeline/sqlite/results/wikipedia_multimodal/vector_databases

python migrate_main_sql_only.py 
