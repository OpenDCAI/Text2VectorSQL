import time
from client.mock_vsql_client import VectorSQLClient
import sqlite3
import sqlite_vec
def execute_query_and_benchmark(client: VectorSQLClient, db_name: str, sql: str):
    """
    Executes an SQL query and benchmarks the performance.

    Args:
        client: The VectorSQL client instance.
        db_name: The name of the registered database to execute the query on.
        sql: The SQL query string to execute.

    Returns:
        A list of results from the query, or None if an error occurs.
    """
    print(f"--- Steps 2 & 3: Executing SQL and Benchmarking ---")
    try:
        start_time = time.time()
        results = client.execute(db_name, sql)
        end_time = time.time()

        duration = end_time - start_time
        print("Query executed successfully.")
        print(f"Metrics:")
        print(f"  - Execution Time: {duration:.4f} seconds")
        print(f"  - Number of Results: {len(results)}")
        
        # You could add more complex benchmarking metrics here,
        # such as precision/recall if you have ground truth data.
        return results
    except Exception as e:
        print(f"An error occurred during query execution: {e}")
        return None
