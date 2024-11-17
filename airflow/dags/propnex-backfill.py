import os
import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator

S3_BUCKET = os.environ["S3_BUCKET"]
AWS_CONN_ID = "aws_conn"
DUCKDB_CONN_ID = "duckdb_conn"
LOCATION_INFO_TBL_NAME = "plan_area_mapping"
INSERT_TBL = "property_listing"
CDC_TBL = "rental_price_history"

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
    "email": ["kaiseong02@gmail.com"],
    "email_on_failure": True,
}

dag = DAG(
    "backfill_propnex_etl",
    default_args=default_args,
    catchup=False,
    schedule=None,
    description="A backfill DAG to scrape data from Propnex and load to DuckDB",
)


def process(**kwargs):
    import time
    import urllib3
    import requests
    import logging

    augment_data = {}
    for info in INFOS:
        df = fetch_info(f"{info}_info", duckdb_conn_id=DUCKDB_CONN_ID, target_cols=INFOS[info])
        augment_data[info] = df

    start_date = "2024-09-27"
    prev_end, end_date = None, None
    augment_data["plan_area_mapping"] = fetch_info(LOCATION_INFO_TBL_NAME, duckdb_conn_id=DUCKDB_CONN_ID)
    while start_date <= datetime.datetime.today().strftime("%Y-%m-%d"):
        # advance start_date
        prev_end = end_date
        end_date = start_date
        start_date = (
            datetime.datetime.strptime(
                start_date,
                "%Y-%m-%d") +
            datetime.timedelta(
                days=1)).strftime("%Y-%m-%d")

        # read df from s3
        s3_key = f"airflow/propnex/{end_date}.parquet.gzip"
        df = read_from_s3(s3_bucket=S3_BUCKET, s3_key=s3_key)
        if df.empty:
            continue

        try:
            cleaned = clean_and_transform(df, augment_data, end_date)
            print(f"Length cleaned for {end_date}: {len(cleaned)}")
        except urllib3.exceptions.NameResolutionError or requests.exceptions.ConnectionError:
            logging.error(f"Connection error for {end_date} -> Sleeping for 2 minutes and retrying...")
            time.sleep(120)

            # revert to previous start_date
            start_date = end_date
            end_date = prev_end
            continue

        push_to_duckdb(DUCKDB_CONN_ID, cleaned)
        time.sleep(60 * 10)


def read_from_s3(s3_bucket, s3_key, **kwargs):
    import pandas as pd
    from io import BytesIO
    from airflow.providers.amazon.aws.operators.s3 import S3Hook

    hook = S3Hook(aws_conn_id='aws_conn')

    buffer = BytesIO()
    try:
        s3_obj = hook.get_key(key=s3_key, bucket_name=s3_bucket)
        s3_obj.download_fileobj(buffer)
    except Exception as err:
        print(err)
        print(f"No file found for {s3_key}")
        return pd.DataFrame()

    df = pd.read_parquet(buffer)
    print(f"length df: {len(df)}")
    return df


def fetch_info(table_name, duckdb_conn_id, target_cols=None, **kwargs):
    from duckdb_provider.hooks.duckdb_hook import DuckDBHook
    import logging
    print(f"Fetching info for {table_name}...")

    duckdb_hook = DuckDBHook.get_hook(duckdb_conn_id)
    conn = duckdb_hook.get_conn()

    df = conn.sql(f"SELECT * FROM {table_name};").df()
    logging.info(df)

    return df[target_cols] if target_cols else df


def clean_and_transform(df, augment_data, date_str: str, **kwargs):
    from lib.transformers.propnex import transform
    import urllib3
    import requests

    try:
        return transform(df, augment_data, date_str)
    except urllib3.exceptions.NameResolutionError or requests.exceptions.ConnectionError as err:
        raise err


def push_to_duckdb(duckdb_conn_id, df, **kwargs):
    from duckdb_provider.hooks.duckdb_hook import DuckDBHook
    from lib.transformers.ninetynineco import insert_df
    from duckdb import DuckDBPyConnection

    duckdb_hook = DuckDBHook.get_hook(duckdb_conn_id)
    conn: DuckDBPyConnection = duckdb_hook.get_conn()
    insert_df(conn, df, INSERT_TBL, CDC_TBL)


task = PythonOperator(
    task_id="process",
    python_callable=process,
    dag=dag
)
