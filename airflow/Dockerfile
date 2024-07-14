FROM apache/airflow:2.9.2-python3.10

COPY requirements.txt /opt/airflow/requirements.txt
RUN pip install -r /opt/airflow/requirements.txt

USER root
ARG AIRFLOW_UID=50000
RUN usermod -u ${AIRFLOW_UID} airflow