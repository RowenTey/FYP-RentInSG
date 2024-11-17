import os
from airflow import DAG
from airflow.datasets import Dataset
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.providers.telegram.operators.telegram import TelegramOperator
from airflow.operators.python import PythonOperator
from datetime import datetime
from docker.types import Mount

DATE_STR = datetime.today().strftime("%Y-%m-%d")
DOCKER_IMAGE = "rowentey/fyp-rent-in-sg:99co-scraper-latest"
DOCKER_TARGET_VOLUME = "99co_data"
DOCKER_INTERNAL_OUTPUT_DIR = "/app/pkg/rental_prices/ninety_nine"
S3_BUCKET = os.environ["S3_BUCKET"]
S3_KEY = f"airflow/ninety_nine/{DATE_STR}.parquet.gzip"
AWS_CONN_ID = "aws_conn"
DUCKDB_CONN_ID = "duckdb_conn"
LOCATION_INFO_TB_NAME = "plan_area_mapping"
TELEGRAM_TOKEN = os.environ['TELEGRAM_TOKEN']
TELEGRAM_CHAT_ID = os.environ['TELEGRAM_CHAT_ID']
TELEGRAM_CONN_ID = "telegram_conn"
INSERT_TBL = "property_listing"
CDC_TBL = "rental_price_history"
DATASET_URI = "duckdb://fyp_rent_in_sg/property_listing/"

INFOS = {
    "mrt": ["station_name", "latitude", "longitude"],
    "hawker_centre": ["name", "latitude", "longitude"],
    "supermarket": ["name", "latitude", "longitude"],
    "primary_school": ["name", "latitude", "longitude"],
    "mall": ["name", "latitude", "longitude"]
}

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2024, 7, 14),
    "email": ["kaiseong02@gmail.com"],
    "email_on_failure": True,
    # "retries": 1,
    # "retry_delay": datetime.timedelta(minutes=30),
}

dag = DAG(
    "99co_etl",
    default_args=default_args,
    catchup=False,
    description="A DAG to scrape data from 99.co, upload to S3 and load to DuckDB",
    schedule_interval="0 3 * * *",
)


def fetch_csv_from_volume(csv_dir, csv_filename, src_volume, **kwargs):
    """
    Fetches the content of a CSV file from a Docker container volume.

    Args:
        **kwargs: Additional keyword arguments.

    Returns:
        str: The content of the CSV file.

    Raises:
        None.

    This function uses the `docker` library to create a Docker client and run a container. It retrieves the content of a CSV file
    located in the `/app/output/{DATE_STR}.csv` path within the container. The `DATE_STR` variable is expected to be defined
    elsewhere in the code. The container is configured with the `DOCKER_TARGET_VOLUME` volume, which is mounted in read-only mode.
    The container is removed after the content is retrieved.

    Note:
        - The `docker` library is required and must be installed.
        - The `DATE_STR` variable must be defined and contain a valid date string.
    """
    from docker import from_env

    client = from_env()
    container = client.containers.run(
        "alpine",
        f"cat {csv_dir}/{csv_filename}.csv",
        volumes={src_volume: {"bind": csv_dir, "mode": "ro"}},
        remove=True
    )

    csv_content = container.decode("utf-8")
    return csv_content


def convert_csv_to_df(upstream_task, **kwargs):
    import pandas as pd
    from io import StringIO

    ti = kwargs["ti"]
    csv_content = ti.xcom_pull(task_ids=upstream_task)

    df = pd.read_csv(StringIO(csv_content))
    return (df, len(df))


def upload_to_s3(upstream_task, aws_conn_id, s3_bucket, s3_key, **kwargs):
    """
    Uploads a local file to an S3 bucket.

    Args:
        s3_bucket (str): The name of the S3 bucket.
        s3_key (str): The key of the file in the S3 bucket.
        **kwargs: Additional keyword arguments.

    Returns:
        None

    Raises:
        None

    This function uploads a local file to an S3 bucket using the provided S3 bucket and key.
    The local file path is obtained from the TaskInstance (ti) using the task_ids 'fetch_csv'.
    The function first prints the local file path being uploaded to S3.
    It then uses the S3Hook to load the file into the S3 bucket.
    Finally, it prints a message indicating that the file has been uploaded to S3.

    Note:
        - For more information on transferring files to and from an S3 bucket using Apache Airflow,
        refer to the blog post at https://blog.devgenius.io/transfer-files-to-and-from-s3-bucket-using-apache-airflow-e3790a3b47a2.
    """
    from lib.utils.parquet import parquet
    from airflow.providers.amazon.aws.operators.s3 import S3Hook

    ti = kwargs['ti']
    df = ti.xcom_pull(task_ids=upstream_task)[0]

    parquet_bytes = parquet(df)

    hook = S3Hook(aws_conn_id=aws_conn_id)
    hook.load_file_obj(parquet_bytes, s3_key, bucket_name=s3_bucket, replace=True)


def fetch_info(table_name, duckdb_conn_id, target_cols=None, **kwargs):
    from duckdb_provider.hooks.duckdb_hook import DuckDBHook
    import logging
    print(f"Fetching info for {table_name}...")

    duckdb_hook = DuckDBHook.get_hook(duckdb_conn_id)
    conn = duckdb_hook.get_conn()

    df = conn.sql(f"SELECT * FROM {table_name};").df()
    logging.info(df)

    return df[target_cols] if target_cols else df


def clean_and_transform(upstream_tasks: list[str], date_str: str, **kwargs):
    from lib.transformers.ninetynineco import transform

    ti = kwargs["ti"]
    df = ti.xcom_pull(task_ids=upstream_tasks[0])[0]

    augment_data = {task_id .replace("fetch_augmented_info_", "") .replace(
        "fetch_location_info", "plan_area_mapping"): ti.xcom_pull(task_ids=task_id) for task_id in upstream_tasks[1:]}

    return transform(df, augment_data, date_str, True)


def push_to_duckdb(
        duckdb_conn_id: str,
        upstream_task: str,
        insert_tbl: str,
        cdc_tbl: str,
        **kwargs):
    from duckdb_provider.hooks.duckdb_hook import DuckDBHook
    from lib.transformers.ninetynineco import insert_df
    from duckdb import DuckDBPyConnection

    ti = kwargs["ti"]
    df = ti.xcom_pull(task_ids=upstream_task)

    duckdb_hook = DuckDBHook.get_hook(duckdb_conn_id)
    conn: DuckDBPyConnection = duckdb_hook.get_conn()
    insert_df(conn, df, insert_tbl, cdc_tbl)
    conn.close()


docker_task = DockerOperator(
    task_id="scrape_data",
    image=DOCKER_IMAGE,
    api_version="auto",
    auto_remove=True,
    mounts=[
        Mount(source=DOCKER_TARGET_VOLUME, target=DOCKER_INTERNAL_OUTPUT_DIR, type="volume"),
    ],
    # Specify the Docker daemon socket
    docker_url="unix://var/run/docker.sock",
    retrieve_output=True,
    tty=True,
    force_pull=True,
    environment={
        "TELEGRAM_TOKEN": TELEGRAM_TOKEN,
        "TELEGRAM_CHAT_ID": TELEGRAM_CHAT_ID,
        "LOG_OUTPUT": "false"
    },
    dag=dag,
)

fetch_csv_task = PythonOperator(
    task_id="fetch_csv",
    python_callable=fetch_csv_from_volume,
    op_kwargs={
        "csv_dir": DOCKER_INTERNAL_OUTPUT_DIR,
        "csv_filename": DATE_STR,
        "src_volume": DOCKER_TARGET_VOLUME
    },
    dag=dag,
)

convert_csv_task = PythonOperator(
    task_id="convert_csv_to_df",
    python_callable=convert_csv_to_df,
    op_kwargs={
        "upstream_task": "fetch_csv",
    },
    dag=dag,
)

send_telegram_message_task = TelegramOperator(
    task_id="send_telegram_message",
    telegram_conn_id=TELEGRAM_CONN_ID,
    chat_id=TELEGRAM_CHAT_ID,
    text="99co scraper scraped {{ ti.xcom_pull(task_ids='convert_csv_to_df')[1] }} today.",
    dag=dag,
)

upload_to_s3_task = PythonOperator(
    task_id='upload_to_s3',
    python_callable=upload_to_s3,
    op_kwargs={
        'upstream_task': 'convert_csv_to_df',
        'aws_conn_id': AWS_CONN_ID,
        's3_bucket': S3_BUCKET,
        's3_key': S3_KEY
    },
    dag=dag,
)

fetch_augmented_info_task = []
for info_name in INFOS:
    fetch_augmented_info_task.append(
        PythonOperator(
            task_id=f"fetch_augmented_info_{info_name}",
            python_callable=fetch_info,
            op_kwargs={
                "table_name": f"{info_name}_info",
                "duckdb_conn_id": DUCKDB_CONN_ID,
                "target_cols": INFOS[info_name],
            },
            dag=dag,
        )
    )

fetch_augmented_info_task.append(
    PythonOperator(
        task_id="fetch_location_info",
        python_callable=fetch_info,
        op_kwargs={
            "table_name": LOCATION_INFO_TB_NAME,
            "duckdb_conn_id": DUCKDB_CONN_ID,
        },
        dag=dag,
    )
)

clean_and_transform_task = PythonOperator(
    task_id="clean_and_transform",
    python_callable=clean_and_transform,
    op_kwargs={
        "upstream_tasks": ["convert_csv_to_df", "fetch_location_info"] + [f"fetch_augmented_info_{info_name}" for info_name in INFOS],
        "date_str": DATE_STR
    },
    dag=dag,
)

push_to_duckdb_task = PythonOperator(
    task_id="push_to_duckdb",
    python_callable=push_to_duckdb,
    op_kwargs={
        "duckdb_conn_id": DUCKDB_CONN_ID,
        "upstream_task": clean_and_transform_task.task_id,
        "insert_tbl": INSERT_TBL,
        "cdc_tbl": CDC_TBL
    },
    outlets=[Dataset(DATASET_URI)],
    dag=dag,
)

docker_task >> fetch_csv_task
docker_task >> fetch_augmented_info_task

fetch_csv_task >> convert_csv_task

fetch_augmented_info_task >> clean_and_transform_task

convert_csv_task >> send_telegram_message_task
convert_csv_task >> clean_and_transform_task
convert_csv_task >> upload_to_s3_task

clean_and_transform_task >> push_to_duckdb_task
