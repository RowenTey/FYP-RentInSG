from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

VOLUMES = ["99co_data", "propnex_data"]

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 7, 10),
    'email': ['kaiseong02@gmail.com'],
}

dag = DAG(
    'housekeeping',
    default_args=default_args,
    catchup=False,
    description='A DAG to perform housekeeping on the scraper data',
    schedule_interval='0 8 * * 1',
)


def housekeeping(volumes, **kwargs):
    from docker import from_env

    client = from_env()

    for v in volumes:
        list_command = 'ls -l /app'
        list_container = client.containers.run(
            'alpine',
            list_command,
            volumes={v: {'bind': '/app', 'mode': 'rw'}},
            remove=True
        )
        print(f"Contents of the volume {v} before housekeeping:\n")
        print(list_container.decode('utf-8'))

        probe_for_files_command = 'find /app -type f -mtime +10'
        probe_container = client.containers.run(
            'alpine',
            probe_for_files_command,
            volumes={v: {'bind': '/app', 'mode': 'rw'}},
            remove=True
        )
        print("Files to be deleted:\n")
        print(probe_container.decode('utf-8'))

        # Perform the cleanup
        cleanup_command = 'find /app -type f -mtime +10 -exec rm -f {} \\;'
        cleanup_container = client.containers.run(
            'alpine',
            cleanup_command,
            volumes={v: {'bind': '/app', 'mode': 'rw'}},
            remove=True
        )
        print("Files deleted:\n")
        print(cleanup_container.decode('utf-8'))

        print("Done for " + v)


housekeeping_task = PythonOperator(
    task_id='housekeeping',
    python_callable=housekeeping,
    op_kwargs={'volumes': VOLUMES},
    dag=dag,
)
