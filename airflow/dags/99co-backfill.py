import os
import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator

DOCKER_IMAGE = "rowentey/fyp-rent-in-sg:99co-scraper-latest"
DOCKER_TARGET_VOLUME = "scraper_data"
DOCKER_INTERNAL_OUTPUT_DIR = "/app/pkg/rental_prices/ninety_nine"
S3_BUCKET = os.environ["S3_BUCKET"]
S3_KEY = "airflow/ninety_nine/{DATE_STR}.parquet.gzip"
AWS_CONN_ID = "aws_conn"
DUCKDB_CONN_ID = "duckdb_conn"
LOCATION_INFO_TBL_NAME = "plan_area_mapping"
TELEGRAM_TOKEN = os.environ['TELEGRAM_TOKEN']
TELEGRAM_CHAT_ID = os.environ['TELEGRAM_CHAT_ID']
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
    "backfill_99co_etl",
    default_args=default_args,
    catchup=False, 
    description="A backfill DAG to scrape data from 99.co, upload to S3 and load to DuckDB",
)

def process(**kwargs):
    augment_data = {}
    for info in INFOS:
        df = fetch_info(f"{info}_info", duckdb_conn_id=DUCKDB_CONN_ID, target_cols=INFOS[info])
        augment_data[info] = df
    
    start_date = "2024-08-18"
    # loop from start date to today with a while loop to access the date
    while start_date <= datetime.datetime.today().strftime("%Y-%m-%d"):
        augment_data["plan_area_mapping"] = fetch_info(LOCATION_INFO_TBL_NAME, duckdb_conn_id=DUCKDB_CONN_ID)
        
        # advance start_date
        end_date = start_date
        start_date = (datetime.datetime.strptime(start_date, "%Y-%m-%d") + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        
        # read df from s3
        s3_key = f"airflow/ninety_nine/{end_date}.parquet.gzip"
        df = read_from_s3(s3_bucket=S3_BUCKET, s3_key=s3_key)
        if df.empty:
            continue
        
        cleaned = clean_and_transform(df, augment_data, end_date)
        print(f"Length cleaned: {len(cleaned)}")
        
        # push to duckdb
        push_to_duckdb(DUCKDB_CONN_ID, cleaned)

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
    

def fetch_info(table_name, duckdb_conn_id, target_cols = None, **kwargs):
    from duckdb_provider.hooks.duckdb_hook import DuckDBHook
    import logging
    print(f"Fetching info for {table_name}...")
    
    duckdb_hook = DuckDBHook.get_hook(duckdb_conn_id)
    conn = duckdb_hook.get_conn()

    df = conn.sql(f"SELECT * FROM {table_name};").df()
    logging.info(df)
    
    return df[target_cols] if target_cols else df
    
def clean_and_transform(df, augment_data, date_str: str, **kwargs):
    from lib.transformers.ninetynineco import transform
    
    return transform(df, augment_data, date_str)

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