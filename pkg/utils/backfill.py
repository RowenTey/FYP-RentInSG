import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from io import BytesIO

import boto3
import numpy as np
import pandas as pd
from coordinates import fetch_coordinates


def process_file_coordinates(buffer: BytesIO):
    # Read parquet file into a DataFrame
    df = pd.read_parquet(buffer)

    df["building_name"] = df["property_name"].apply(
        lambda x: x.split(" in ")[-1])
    df["nearest_mrt"] = None
    df["distance_to_nearest_mrt"] = np.nan
    building_names = df["building_name"].unique()

    with ThreadPoolExecutor() as executor:
        print(executor._max_workers)
        future_to_coords = {
            executor.submit(fetch_coordinates, name): name
            for name in building_names}
        for future in as_completed(future_to_coords):
            result = future.result()
            if result is None:
                continue

            building_name, coords = result
            df.loc[df["building_name"] == building_name, "latitude"] = coords[0]
            df.loc[df["building_name"] ==
                   building_name, "longitude"] = coords[1]

    return df


def process_file_distance(buffer: BytesIO, df2: pd.DataFrame):
    df = pd.read_parquet(buffer)
    if "building_name" not in df.columns:
        df["building_name"] = df["property_name"].apply(
            lambda x: x.split(" in ")[-1])

    from find_closest import find_nearest

    df = find_nearest(df, df2)

    print(df[["property_name", "nearest_mrt", "distance_to_nearest_mrt"]])
    print()
    print(df[df["nearest_mrt"].isna()])

    return df


def backfill_coordinates():
    # Set up S3 client
    s3 = boto3.client("s3")

    # Specify S3 bucket and directory
    bucket_name = os.getenv("S3_BUCKET")
    directory_name = "rental_prices/ninety_nine/processed"
    upload_directory_name = "rental_prices/ninety_nine/processed/"

    # Specify date range
    start_date = datetime(2024, 1, 25)
    end_date = datetime(2024, 2, 2)

    # List all files in the S3 directory
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=directory_name)

    # Process CSV files within the specified date range
    for file in response["Contents"]:
        # file_name in the format rental_prices/ninety_nine/2024-01-25.parquet
        file_name = file["Key"]
        if file_name.endswith(".parquet.gzip"):
            # Extract date from the file name
            date_str = file_name.split("/")[-1].split(".")[0]
            file_date = datetime.strptime(date_str, "%Y-%m-%d")

            # Check if the file date is within the specified range
            if start_date <= file_date <= end_date:
                print(f"\nProcessing file: {file_name}")
                buffer = BytesIO()
                s3_resource = boto3.resource("s3")
                object = s3_resource.Object(bucket_name, file_name)
                object.download_fileobj(buffer)
                df = process_file_coordinates(buffer)
                print(df.head(3))

                # Upload DataFrame to S3 as compressed parquet file
                output_file_name = file_name.split("/")[-1]
                output_file_path = f"{upload_directory_name}{output_file_name}"
                df.to_parquet(output_file_path, compression="gzip")
                s3.upload_file(output_file_path, bucket_name, output_file_path)
                print(f"Uploaded file: {output_file_path}")


def backfill_nearest_mrt():
    # Set up S3 client
    s3 = boto3.client("s3")

    # Specify S3 bucket and directory
    bucket_name = os.getenv("S3_BUCKET")
    directory_name = "rental_prices/ninety_nine"
    upload_directory_name = "rental_prices/ninety_nine/processed_v2/"

    # List all files in the S3 directory
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=directory_name)

    from motherduckdb_connector import connect_to_motherduckdb

    db = connect_to_motherduckdb()
    df2 = db.query_df("SELECT * FROM mrt_info")
    print(df2)
    print()
    df2 = df2[["station_name", "latitude", "longitude"]]

    # Specify date range
    start_date = datetime(2024, 2, 3)
    end_date = datetime(2024, 2, 4)

    for file in response["Contents"]:
        # file_name in the format rental_prices/ninety_nine/2024-01-25.parquet
        file_name = file["Key"]
        if file_name.endswith(".parquet.gzip"):
            # Extract date from the file name
            date_str = file_name.split("/")[-1].split(".")[0]
            file_date = datetime.strptime(date_str, "%Y-%m-%d")

            # Check if the file date is within the specified range
            if start_date <= file_date <= end_date:
                print(f"\nProcessing file: {file_name}")
                buffer = BytesIO()
                s3_resource = boto3.resource("s3")
                object = s3_resource.Object(bucket_name, file_name)
                object.download_fileobj(buffer)

                df = process_file_distance(buffer, df2)
                print(df.head(3))

                # Upload DataFrame to S3 as compressed parquet file
                output_file_name = file_name.split("/")[-1]
                output_file_path = f"{upload_directory_name}{output_file_name}"
                print(output_file_path)
                df.to_parquet(output_file_path, compression="gzip")
                s3.upload_file(output_file_path, bucket_name, output_file_path)
                print(f"Uploaded file: {output_file_path}")


if __name__ == "__main__":
    pass
