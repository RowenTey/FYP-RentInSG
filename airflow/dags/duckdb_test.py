from airflow import DAG
from datetime import datetime
from airflow.operators.python import PythonOperator

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 7, 14),
    # 'email': ['kaiseong02@gmail.com'],
    # 'email_on_success': True,
    # 'email_on_failure': True,
    # 'email_on_retry': False,
}

# Define the DAG
dag = DAG(
    'duckdb_test',
    default_args=default_args,
    catchup=False, 
    description='A DAG to test DuckDB',
    schedule_interval='0 3 * * *',  # Run the DAG daily at 3 AM UTC
)

DUCKDB_CONN_ID = "duckdb_conn"
DUCKDB_TABLE_NAME = "supermarket_info"


def query_duckdb(my_table, conn_id, **kwargs):
    from duckdb_provider.hooks.duckdb_hook import DuckDBHook
    
    my_duck_hook = DuckDBHook.get_hook(conn_id)
    conn = my_duck_hook.get_conn()

    r = conn.execute(f"SELECT * FROM {my_table};").fetchall()
    print(r)

    return r

query_task = PythonOperator(
    task_id='query_duckdb',
    python_callable=query_duckdb,
    op_kwargs={
        'my_table': DUCKDB_TABLE_NAME,
        'conn_id': DUCKDB_CONN_ID
    },
    dag=dag
)