import hashlib
import json
import os
from datetime import datetime, timedelta
from io import BytesIO

import boto3
import dotenv
import pandas as pd

dotenv.load_dotenv()


def load_uploaded_hashes(hash_file_path: str) -> list[dict]:
    """
    Load previously uploaded hashes from a file.

    Args:
        hash_file_path (str): The path to the file containing the uploaded hashes.

    Returns:
        list[dict]: A list of hashes loaded from the file. If the file doesn't exist, an empty list is returned.
    """
    if os.path.exists(hash_file_path):
        with open(hash_file_path, "r") as file:
            return json.load(file)
    return []


def save_uploaded_hashes(hash_file_path: str, uploaded_hashes: list[dict]) -> None:
    """
    Save uploaded hashes to a file.

    Args:
        hash_file_path (str): The file path where the hashes will be saved.
        uploaded_hashes (list[dict]): A list of dictionaries containing the uploaded hashes.

    Returns:
        None
    """
    with open(hash_file_path, "w") as file:
        json.dump(uploaded_hashes, file)


def calculate_file_hash(file_path: str) -> str:
    """
    Calculate the MD5 hash of a file.

    Args:
        file_path (str): The path to the file.

    Returns:
        str: The MD5 hash of the file.
    """
    md5 = hashlib.md5()
    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            md5.update(chunk)
    return md5.hexdigest()


def parquet(df: pd.DataFrame) -> BytesIO:
    # Code from https://github.com/pandas-dev/pandas/issues/51140
    data: BytesIO = BytesIO()

    # Monkey patch the close method to prevent the file from being closed
    orig_close = data.close
    data.close = lambda: None
    try:
        df.to_parquet(data, compression="gzip", index=False)
    finally:
        data.close = orig_close

    return data


def convert_csv_to_parquet_and_upload(csv_file_path: str, s3_client, bucket_name: str, s3_file_path: str) -> None:
    """
    Convert a CSV file to Parquet format and upload it to S3.

    Args:
        csv_file_path (str): The path to the CSV file.
        s3_client: The S3 client object used for uploading the file.
        bucket_name (str): The name of the S3 bucket.
        s3_file_path (str): The key to use for the uploaded file in S3.

    Returns:
        None
    """
    df = pd.read_csv(csv_file_path)
    parquet_bytes = parquet(df)
    parquet_bytes.seek(0)  # Reset the buffer position to the start
    s3_client.upload_fileobj(parquet_bytes, bucket_name, s3_file_path)
    print(
        f"File uploaded: {csv_file_path} to s3://{bucket_name}/{s3_file_path}")
    parquet_bytes.close()


def upload_files_to_s3(local_directory: str, bucket_name: str) -> None:
    """
    Uploads files from a local directory to an S3 bucket.

    Args:
        local_directory (str): The path to the local directory containing the files to be uploaded.
        bucket_name (str): The name of the S3 bucket to upload the files to.

    Returns:
        None
    """
    # Set your AWS credentials
    aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    aws_region = "ap-southeast-1"

    # Create an S3 client
    s3 = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_region,
    )

    # Load previously uploaded hashes with timestamps
    uploaded_hashes: list = load_uploaded_hashes(HASH_FILE_PATH)

    # Prune old entries based on timestamp
    current_time = datetime.now()
    updated_hashes, outdated_hashes = [], []
    for entry in uploaded_hashes:
        timestamp = datetime.strptime(entry["timestamp"], DATETIME_FORMAT)
        if current_time - timestamp < timedelta(days=MAX_AGE_DAYS):
            updated_hashes.append(entry)
        else:
            outdated_hashes.append(entry)

    updated_hashes_set: set = set(entry["hash"] for entry in updated_hashes)
    outdated_hashes_set: set = set(entry["hash"] for entry in outdated_hashes)

    # List all files in the local directory
    for root, _, files in os.walk(local_directory):
        for file in files:
            local_file_path = os.path.join(root, file)
            # Enforcing the format rental_prices/ninety_nine/*.csv
            s3_file_path = os.path.join(
                root, file).replace(os.path.sep, "/")[2:]

            # handle case where s3_file_path starts with '/' due to os.path.sep between Windows and Linux
            if s3_file_path.startswith("/"):
                s3_file_path = s3_file_path[1:]

            # Calculate the hash of the local file
            file_hash = calculate_file_hash(local_file_path)

            # Check if the file has been uploaded (based on hash)
            if file_hash in updated_hashes_set:
                print(
                    f"Skipping upload for {local_file_path} (already uploaded)")
                continue
            elif file_hash in outdated_hashes_set:
                print(
                    "File is outdated, removing from local directory and skipping upload")
                os.remove(local_file_path)
                continue

            # Upload the Parquet file to S3
            s3_file_path = f"{s3_file_path.replace('.csv', '.parquet')}.gzip"
            convert_csv_to_parquet_and_upload(
                local_file_path, s3, bucket_name, s3_file_path)

            # Add the hash with timestamp to the list of uploaded hashes
            updated_hashes.append(
                {"hash": file_hash, "timestamp": current_time.strftime(DATETIME_FORMAT)})
            updated_hashes_set.add(file_hash)

    # Save updated hashes to the file
    save_uploaded_hashes(HASH_FILE_PATH, updated_hashes)


if __name__ == "__main__":
    HASH_FILE_PATH = ".pkg/logs/s3_uploader/uploaded_hashes.json"
    MAX_AGE_DAYS = 10  # Adjust as needed
    DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

    # Specify the local directory containing CSV files
    local_directory = "./rental_prices/ninety_nine"

    # Specify the S3 bucket name
    bucket_name = os.getenv("S3_BUCKET")

    # Call the function to upload files to S3
    upload_files_to_s3(local_directory, bucket_name)
