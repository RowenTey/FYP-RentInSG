import logging
import duckdb
from pandas import DataFrame, concat


class MotherDuckDBConnector:
    def __init__(self, conn: duckdb.DuckDBPyConnection):
        self.connection: duckdb.DuckDBPyConnection = conn
        self.logger = logging.getLogger(MotherDuckDBConnector.__name__)

    def create_s3_secret(self, aws_access_key_id: str, aws_secret_access_key: str, aws_region: str):
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

    def get_table_size(self, table_name: str) -> int:
        # Get the size of a table
        return self.connection.sql(f"SELECT COUNT(*) FROM {table_name};").fetchall()[0][0]

    def query_df(self, query: str) -> DataFrame:
        # Run a query
        print(f"Running query: {query}")
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
            f"INSERT OR IGNORE INTO {table_name} SELECT * FROM df")

    def begin_transaction(self):
        """Start a new transaction."""
        if self.connection:
            self.connection.begin()

    def commit_transaction(self):
        """Commit the current transaction."""
        if self.connection:
            self.connection.commit()

    def rollback_transaction(self):
        """Roll back the current transaction."""
        if self.connection:
            self.connection.rollback()

    def close(self):
        # Close the database connection
        self.logger.info("Closing connection to MotherDuckDB...")
        self.connection.close()


def connect_to_motherduckdb() -> MotherDuckDBConnector:
    db = MotherDuckDBConnector()
    db.connect()
    return db


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    # aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    # aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    # aws_region = "ap-southeast-1"
    # s3_bucket = os.getenv("S3_BUCKET")
    # s3_filepath = "rental_prices/ninety_nine/2024-01-25.parquet.gzip"

    # db = connect_to_motherduckdb()
    # print("MotherDuckDB connection completed.")

    '''
    df = db.query_df_in_batch("""
        SELECT
        *
        FROM property_listing;
    """)
    '''
    # df.to_csv('training_data.csv', index=False)
