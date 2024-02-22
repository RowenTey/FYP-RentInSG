import duckdb
import os
from dotenv import load_dotenv

load_dotenv()

aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
aws_region = 'ap-southeast-1'
s3_bucket = os.getenv('S3_BUCKET')
s3_filepath = 'rental_prices/ninety_nine/2024-01-25.parquet'
motherduck_token = os.getenv('MOTHERDUCKDB_TOKEN')

# connect to your MotherDuck database through 'md:mydatabase' or 'motherduck:mydatabase'
# if the database doesn't exist, MotherDuck creates it when you connect
con = duckdb.connect(f'md:fyp_rent_in_sg?motherduck_token={motherduck_token}')

# create a secret to set AWS credentials
# con.sql(
#     f"CREATE SECRET (TYPE S3, S3_ACCESS_KEY_ID '{aws_access_key_id}', S3_SECRET_ACCESS_KEY '{aws_secret_access_key}', S3_REGION '{aws_region}')")

# run a query to check verify that you are connected
con.sql("USE fyp_rent_in_sg")
con.sql("SHOW TABLES").show()
# con.sql(
#     f"CREATE TABLE property_listing AS SELECT * FROM 's3://{s3_bucket}/{s3_filepath}'")

con.sql("SHOW TABLES").show()
con.sql("SELECT * FROM property_listing LIMIT 25").show()
con.sql("SELECT COUNT(*) FROM property_listing").show()

con.close()
