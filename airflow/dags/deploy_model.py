from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

MLFLOW_TRACKING_URI = "http://mlflow:5000"
REGISTERED_MODEL_NAME = "rent_in_sg_reg_model"

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
}

dag = DAG(
    'model_deployment',
    default_args=default_args,
    description='A DAG to deploy ML models',
    schedule=None,
    catchup=False
)


def list_models():
    from pprint import pprint
    from mlflow.tracking import MlflowClient

    # Create an MLflow client
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    # List all registered models
    for rm in client.search_registered_models():
        pprint(dict(rm), indent=4)
        
        
def get_latest_model_version():
    from mlflow.tracking import MlflowClient
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    # -1 because models are sorted by version in descending order
    latest_version = dict(client.get_latest_versions(REGISTERED_MODEL_NAME)[-1])
    return latest_version
        

def load_model_from_registry(upstream_task, **kwargs):
    import mlflow
    
    # Load the model from MLflow Model Registry
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    ti = kwargs['ti']
    version_info = ti.xcom_pull(task_ids=upstream_task)
    print(version_info)
    
    model = mlflow.pyfunc.load_model(f"{version_info['source']}")
    return model


def deploy_docker_image(image):
    # Deploy the Docker image (e.g., to a container registry or Kubernetes cluster)
    # This step depends on your specific deployment environment
    pass


def trigger_pipeline():
    # Trigger any additional pipelines or processes
    # This could involve making API calls, running scripts, etc.
    pass


get_latest_model_version_task = PythonOperator(
    task_id='get_latest_model_version',
    python_callable=get_latest_model_version,
    dag=dag,
)

load_model_task = PythonOperator(
    task_id='load_model',
    python_callable=load_model_from_registry,
    op_kwargs={'upstream_task': get_latest_model_version_task.task_id},
    dag=dag,
)

# build_image_task = PythonOperator(
#     task_id='build_docker_image',
#     python_callable=build_docker_image,
#     op_kwargs={'model': '{{ task_instance.xcom_pull(task_ids="load_model") }}'},
#     dag=dag,
# )

# deploy_image_task = PythonOperator(
#     task_id='deploy_docker_image',
#     python_callable=deploy_docker_image,
#     op_kwargs={'image': '{{ task_instance.xcom_pull(task_ids="build_docker_image") }}'},
#     dag=dag,
# )

# trigger_pipeline_task = PythonOperator(
#     task_id='trigger_pipeline',
#     python_callable=trigger_pipeline,
#     dag=dag,
# )

get_latest_model_version_task >> load_model_task

# load_model_task >> build_image_task >> deploy_image_task >> trigger_pipeline_task