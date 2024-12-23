import logging
import os

import duckdb
from pandas import DataFrame, concat


class MotherDuckDBConnector:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MotherDuckDBConnector, cls).__new__(cls)
            cls._instance.__init__()
        return cls._instance

    def __init__(self, token_name: str = "MOTHERDUCKDB_TOKEN"):
        self.motherduck_token = os.getenv(token_name)
        self.connection: duckdb.DuckDBPyConnection = None
        self.logger = logging.getLogger(MotherDuckDBConnector.__name__)
        self.cache = {}

    def connect(self):
        self.logger.info("Connecting to MotherDuckDB...")
        try:
            # Connect to your MotherDuck database through 'md:mydatabase' or 'motherduck:mydatabase'
            # If the database doesn't exist, MotherDuck creates it when you connect
            self.connection = duckdb.connect(
                f"md:fyp_rent_in_sg?motherduck_token={self.motherduck_token}")
            self.logger.info("Connected to MotherDuckDB!")
        except Exception as e:
            self.logger.error(f"Error connecting to MotherDuckDB: {str(e)}")

    def create_s3_secret(
            self,
            aws_access_key_id: str,
            aws_secret_access_key: str,
            aws_region: str):
        # Create a secret to set AWS credentials
        self.connection.sql(
            f"CREATE OR REPLACE SECRET (TYPE S3, S3_ACCESS_KEY_ID '{aws_access_key_id}', S3_SECRET_ACCESS_KEY '{aws_secret_access_key}', S3_REGION '{aws_region}')"  # noqa: E501
        )

    def check_connection(self):
        # Run a query to check verify that you are connected
        self.connection.sql("USE fyp_rent_in_sg")
        self.show_tables()

    def create_table_from_s3(
            self,
            table_name: str,
            s3_bucket: str,
            s3_filepath: str):
        # Create table if it doesn't exist and import data from S3
        self.connection.sql(
            f"CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM 's3://{s3_bucket}/{s3_filepath}'"
        )

    def show_tables(self):
        # Show all tables in the database
        self.connection.sql("SHOW TABLES").show()

    def query_df(self, query: str) -> DataFrame:
        # Run a query
        return self.connection.sql(query).df()

    def query_df_in_batch(
            self,
            query: str,
            batch_size: int = 1000) -> DataFrame:
        cursor = self.connection.cursor()
        cursor.execute(query)

        # Fetch rows in batches
        rows = cursor.fetchmany(batch_size)
        cols = [desc[0] for desc in cursor.description]
        df = DataFrame()

        # Process each batch
        i = 0
        while rows:
            print(f"Processing batch... {i}")
            df = concat([df, DataFrame(rows, columns=cols)])

            # Fetch the next batch
            rows = cursor.fetchmany(batch_size)
            i += 1

        cursor.close()
        return df

    def update_table(
            self,
            table_name: str,
            key_col: str,
            updated_cols: list,
            df: DataFrame):
        logging.info(f"Updating table {table_name} with {len(df)} rows...")

        # Update the table with values from a DataFrame
        queries = []
        for _, row in df.iterrows():
            query = f"UPDATE {table_name} SET "
            for column in updated_cols:
                query += f"{column} = '{row[column]}', "
            query = query[:-2]  # Remove the trailing comma and space
            query += f" WHERE {key_col} = '{row[key_col]}'"
            queries.append(query)

        batch_query = ";".join(queries)
        self.connection.sql(batch_query)

    def insert_df(self, table_name: str, df: DataFrame):
        # make sure df matches schema of table_name
        return self.connection.sql(
            f"INSERT INTO {table_name} SELECT * FROM df")

    def fetch_info(self, query, target_cols=None):
        queried_df = self.query_df(query)

        if target_cols:
            return queried_df[target_cols]

        return queried_df

    def close(self):
        # Close the database connection
        if self.connection is None:
            return

        self.logger.info("Closing connection to MotherDuckDB...")
        self.connection.close()

    def __del__(self):
        self.close()


db = MotherDuckDBConnector()
