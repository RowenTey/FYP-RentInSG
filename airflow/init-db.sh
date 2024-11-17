#!/bin/bash

echo "Initializing database..."
psql -v ON_ERROR_STOP=1 -U airflow <<-EOSQL
    CREATE DATABASE mlflow;
    CREATE DATABASE optuna;
EOSQL
echo "Database initialized"
