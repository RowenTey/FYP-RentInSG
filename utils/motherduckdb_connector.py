import os
import logging
import duckdb
from pandas import DataFrame


class MotherDuckDBConnector:
    def __init__(self, token_name: str = 'MOTHERDUCKDB_TOKEN'):
        self.motherduck_token = os.getenv(token_name)
        self.connection: duckdb.DuckDBPyConnection = None
        self.logger = logging.getLogger(MotherDuckDBConnector.__name__)

    def connect(self):
        self.logger.info("Connecting to MotherDuckDB...")
        try:
            # Connect to your MotherDuck database through 'md:mydatabase' or 'motherduck:mydatabase'
            # If the database doesn't exist, MotherDuck creates it when you connect
            self.connection = duckdb.connect(
                f'md:fyp_rent_in_sg?motherduck_token={self.motherduck_token}')
        except Exception as e:
            self.logger.error(f"Error connecting to MotherDuckDB: {str(e)}")

    def create_s3_secret(self, aws_access_key_id: str, aws_secret_access_key: str, aws_region: str):
        # Create a secret to set AWS credentials
        self.connection.sql(
            f"CREATE OR REPLACE SECRET (TYPE S3, S3_ACCESS_KEY_ID '{aws_access_key_id}', S3_SECRET_ACCESS_KEY '{aws_secret_access_key}', S3_REGION '{aws_region}')")

    def check_connection(self):
        # Run a query to check verify that you are connected
        self.connection.sql("USE fyp_rent_in_sg")
        self.show_tables()

    def create_table_from_s3(self, s3_bucket: str, s3_filepath: str):
        # Create table property_listing if it doesn't exist and import data from S3
        self.connection.sql(
            f"CREATE TABLE IF NOT EXISTS property_listing AS SELECT * FROM 's3://{s3_bucket}/{s3_filepath}'")

    def show_tables(self):
        # Show all tables in the database
        self.connection.sql("SHOW TABLES").show()

    def query_df(self, query: str) -> DataFrame:
        # Run a query
        return self.connection.sql(query).df()

    def update_table(self, table_name: str, key_col: str, updated_cols: list, df: DataFrame):
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
        return self.connection.sql(f"INSERT INTO {table_name} SELECT * FROM df")

    def close(self):
        # Close the database connection
        self.logger.info("Closing connection to MotherDuckDB...")
        self.connection.close()

    def __del__(self):
        self.close()


def connect_to_motherduckdb() -> MotherDuckDBConnector:
    db = MotherDuckDBConnector()
    db.connect()
    db.check_connection()
    return db


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    aws_region = 'ap-southeast-1'
    s3_bucket = os.getenv('S3_BUCKET')
    s3_filepath = 'rental_prices/ninety_nine/2024-01-25.parquet.gzip'

    db = connect_to_motherduckdb()
    print("MotherDuckDB connection completed.")
