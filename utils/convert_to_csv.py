import os
import pyarrow.parquet as pq
import pandas as pd


def parquet_to_csv(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".parquet"):
            parquet_file = os.path.join(directory, filename)
            csv_file = os.path.splitext(parquet_file)[0] + ".csv"
            table = pq.read_table(parquet_file)
            df = table.to_pandas()
            df.to_csv(csv_file, index=False)


directory = "/path/to/parquet/files"
parquet_to_csv(directory)
