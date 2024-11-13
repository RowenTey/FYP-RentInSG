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

### View CRON Jobs Log

You can view them at `/var/log/syslog`.

### Docker _"exec: required file not found"_ error

Change the line ending format from CRLF (Windows) to LF (Linux).  
Reference: https://stackoverflow.com/questions/38905135/why-wont-my-docker-entrypoint-sh-execute
