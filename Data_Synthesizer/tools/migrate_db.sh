python migrate_db.py --source ../pipeline/sqlite/results/arxiv/vector_databases
python migrate_db.py --source ../pipeline/sqlite/results/bird/vector_databases
# python migrate_db.py --source ../pipeline/sqlite/results/spider/vector_databases
python migrate_db.py --source ../pipeline/sqlite/results/wikipedia_multimodal/vector_databases

python migrate_db_myscale.py --source ../pipeline/sqlite/results/arxiv/vector_databases
python migrate_db_myscale.py --source ../pipeline/sqlite/results/bird/vector_databases
# python migrate_db_myscale.py --source ../pipeline/sqlite/results/spider/vector_databases
python migrate_db_myscale.py --source ../pipeline/sqlite/results/wikipedia_multimodal/vector_databases

python migrate_main_sql_only.py
