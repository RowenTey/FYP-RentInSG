import os
import boto3
import dotenv
dotenv.load_dotenv()


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

    # List all files in the local directory
    for root, _, files in os.walk(local_directory):
        for file in files:
            local_file_path = os.path.join(root, file)
            # Enforcing the format rental_prices/ninety_nine/*.csv
            s3_file_path = os.path.join(
                root, file).replace(os.path.sep, '/')[2:]

            # Upload the file to S3
            s3.upload_file(local_file_path, bucket_name, s3_file_path)
            print(
                f'File uploaded: {local_file_path} to s3://{bucket_name}/{s3_file_path}')


if __name__ == "__main__":
    # Specify the local directory containing CSV files
    local_directory = './rental_prices/ninety_nine'

    # Specify the S3 bucket name
    bucket_name = 'fyp-2024-bucket'

    # Call the function to upload files to S3
    upload_files_to_s3(local_directory, bucket_name)
