import io
import os

import boto3
import pandas as pd


def read_df_from_s3(s3_file_path: str) -> pd.DataFrame:
    aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    bucket_name = os.getenv("S3_BUCKET")
    aws_region = "ap-southeast-1"

    # Create a session with your credentials
    session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_region,
    )

    buffer = io.BytesIO()
    s3 = session.resource("s3")
    object = s3.Object(bucket_name, s3_file_path)
    object.download_fileobj(buffer)
    df = pd.read_parquet(buffer)

    return df


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    df = read_df_from_s3("rental_prices/ninety_nine/2024-05-05.parquet.gzip")
    print(df)
