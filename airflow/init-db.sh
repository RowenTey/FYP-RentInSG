#!/bin/bash

echo "Initializing database..."
psql -v ON_ERROR_STOP=1 -U airflow <<-EOSQL
    CREATE DATABASE mlflow;
EOSQL
echo "Database initialized"
