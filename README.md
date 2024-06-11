# FYP - Singapore Rental Price Analysis Platform 🎓

> This repository contains the code submitted for NTU's Final Year Project

## Execution Flow ➡

1. CRON jobs run daily to scrape data from various sources (only 99.co for now)
2. S3 uploader runs weekly to compress and backup raw data to S3 (Data Warehouse)
3. Transformer runs weekly to read data from S3 and performs data processing and push to MotherDuckDB (Data Sink)
4. Data is then queried from sink to frontend Streamlit application for visualization
5. Data is also periodically queried to perform model training and/or tuning.
6. Trained model is packaged in `pickle` format and copied to Docker image.

## Script Timings

| Name          | UTC Time | SGT Time | Remarks            |
| ------------- | -------- | -------- | ------------------ |
| 99.co scraper | 2am      | 10am     | Takes ~4hrs to run |
| s3_uploader   | 6am      | 2pm      |                    |

## Helpful Code Snippets

- Connecting to EC2 instance

  ```bash
  ssh -i "secrets/AWS.pem" ubuntu@13.250.51.77
  ```

- Transferring files

  ```bash
  sftp -i "secrets/AWS.pem" ubuntu@13.250.51.77
  ```

- Changing Permissions

  ```bash
  chmod 775 /path/to/directory
  ```

- List processes

  ```bash
  ps aux
  ```

- Kill process

  ```bash
  kill <process-id>
  ```
