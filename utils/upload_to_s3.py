import os
import boto3
import dotenv
import hashlib
import json
import pandas as pd
from datetime import datetime, timedelta

dotenv.load_dotenv()

HASH_FILE = 'uploaded_hashes.json'
MAX_AGE_DAYS = 30  # Adjust as needed


def load_uploaded_hashes():
    # Load previously uploaded hashes from a file
    if os.path.exists(HASH_FILE):
        with open(HASH_FILE, 'r') as file:
            return json.load(file)
    return []


def save_uploaded_hashes(uploaded_hashes):
    # Save uploaded hashes to a file
    with open(HASH_FILE, 'w') as file:
        json.dump(uploaded_hashes, file)


def calculate_file_hash(file_path):
    # Calculate the MD5 hash of the file
    md5 = hashlib.md5()
    with open(file_path, 'rb') as file:
        for chunk in iter(lambda: file.read(4096), b""):
            md5.update(chunk)
    return md5.hexdigest()


def convert_csv_to_parquet(csv_file_path, parquet_file_path):
    # Convert CSV file to Parquet format
    df = pd.read_csv(csv_file_path)
    df.to_parquet(parquet_file_path)


def upload_files_to_s3(local_directory, bucket_name):
    # Set your AWS credentials
    aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    aws_region = 'ap-southeast-1'

    # Create an S3 client
    s3 = boto3.client('s3',
                      aws_access_key_id=aws_access_key_id,
                      aws_secret_access_key=aws_secret_access_key,
                      region_name=aws_region
                      )

    # Load previously uploaded hashes with timestamps
    uploaded_hashes = load_uploaded_hashes()

    # Prune old entries based on timestamp
    current_time = datetime.now()
    updated_hashes = [entry for entry in uploaded_hashes
                      if current_time - entry['timestamp'] < timedelta(days=MAX_AGE_DAYS)]

    # List all files in the local directory
    for root, _, files in os.walk(local_directory):
        for file in files:
            print(file)
            local_file_path = os.path.join(root, file)
            # Enforcing the format rental_prices/ninety_nine/*.csv
            s3_file_path = os.path.join(
                root, file).replace(os.path.sep, '/')[2:]

            # Calculate the hash of the local file
            file_hash = calculate_file_hash(local_file_path)

            # Check if the file has been uploaded (based on hash)
            if file_hash not in {entry['hash'] for entry in updated_hashes}:
                # Convert CSV to Parquet
                parquet_file_path = local_file_path.replace('.csv', '.parquet')
                convert_csv_to_parquet(local_file_path, parquet_file_path)

                # Upload the Parquet file to S3
                # s3.upload_file(parquet_file_path, bucket_name, s3_file_path)
                print(
                    f'File uploaded: {parquet_file_path} to s3://{bucket_name}/{s3_file_path}')

                # Add the hash with timestamp to the list of uploaded hashes
                updated_hashes.append(
                    {'hash': file_hash, 'timestamp': current_time})
            else:
                print(
                    f'Skipping upload for {local_file_path} (already uploaded)')

    # Save updated hashes to the file
    save_uploaded_hashes(updated_hashes)


if __name__ == "__main__":
    # Specify the local directory containing CSV files
    local_directory = '../rental_prices/ninety_nine'

    # Specify the S3 bucket name
    bucket_name = 'fyp-2024-bucket'

    # Call the function to upload files to S3
    upload_files_to_s3(local_directory, bucket_name)
