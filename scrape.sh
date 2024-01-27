#!/bin/bash

# Specify the directory path
target_directory="/home/ec2-user/FYP-RentInSG"

# Change to the specified directory
cd "$target_directory" || { echo "Failed to change to directory: $target_directory"; exit 1; }

# Run the Python file (replace "your_python_script.py" with your actual Python file)
venv/bin/python 99co-scraper.py -d >> cron-job.log
