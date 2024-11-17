from airflow import DAG
from airflow.datasets import Dataset
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from mlflow_provider.operators.registry import (
    CreateRegisteredModelOperator,
    CreateModelVersionOperator,
    TransitionModelVersionStageOperator,
)
from datetime import datetime
from pandas import DataFrame
import logging
import os

DATASET_URI = "duckdb://fyp_rent_in_sg/property_listing/"
ARTIFACT_BUCKET = os.environ["S3_BUCKET"]
TABLE_NAME = "property_listing"
DUCKDB_CONN_ID = "duckdb_conn"
MLFLOW_CONN_ID = "mlflow_conn"
MLFLOW_TRACKING_URI = "http://mlflow:5000"
EXPERIMENT_NAME = "rent_in_sg"
REGISTERED_MODEL_NAME = "rent_in_sg_reg_model"
RETRAINING_THRESHOLD = 10_000

property_listing_dataset = Dataset(DATASET_URI)

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': True,
}

dag = DAG(
    'train_model',
    default_args=default_args,
    description='A DAG to train a machine learning model on the scraped data',
    start_date=datetime(2024, 8, 18),
    schedule=[property_listing_dataset],
)


def check_and_trigger_retraining(duckdb_conn_id: str, dataset_uri: str, **context):
    from lib.utils.motherduckdb import MotherDuckDBConnector
    from duckdb_provider.hooks.duckdb_hook import DuckDBHook
    from duckdb import DuckDBPyConnection
    from airflow.models import Variable

    duckdb_hook = DuckDBHook.get_hook(duckdb_conn_id)
    conn: DuckDBPyConnection = duckdb_hook.get_conn()
    db = MotherDuckDBConnector(conn)

    # example: s3://fyp-rent-in-sg/property_listing/
    database = dataset_uri.split("duckdb://")[1].split("/")[0]
    table = dataset_uri.split("duckdb://")[1].split("/")[1]
    print(f"Checking database: {database}, table: {table}")

    conn = DuckDBHook.get_hook(duckdb_conn_id).get_conn()
    db = MotherDuckDBConnector(conn)
    current_size = db.get_table_size(table)
    db.close()

    ti = context['ti']
    previous_size = int(Variable.get("previous_property_listing_dataset_size", default_var=1))
    print(f"Current size: {current_size}; Previous size: {previous_size}")

    # if current_size - previous_size <= RETRAINING_THRESHOLD:
    #     Variable.set("previous_property_listing_dataset_size", current_size)
    #     return "retraining_not_triggered"

    Variable.set("previous_property_listing_dataset_size", current_size)
    return "retraining_triggered"


def load_data(duckdb_conn_id: str, table_name: str, **kwargs):
    from lib.utils.motherduckdb import MotherDuckDBConnector
    from duckdb_provider.hooks.duckdb_hook import DuckDBHook
    from duckdb import DuckDBPyConnection

    duckdb_hook = DuckDBHook.get_hook(duckdb_conn_id)
    conn: DuckDBPyConnection = duckdb_hook.get_conn()
    db = MotherDuckDBConnector(conn)

    df = db.query_df(f"SELECT * FROM {table_name}")
    db.close()

    return df


def clean_data(upstream_task: str, **kwargs):
    import pandas as pd
    import numpy as np

    ti = kwargs["ti"]
    df = ti.xcom_pull(task_ids=upstream_task)

    # Perform data cleaning steps
    df["furnishing"] = df["furnishing"].fillna(df["furnishing"].mode()[0])
    df["facing"] = df["facing"].fillna(df["facing"].mode()[0])
    df["floor_level"] = df["floor_level"].fillna(df["floor_level"].mode()[0])
    df["tenure"] = df["tenure"].fillna(df["tenure"].mode()[0])
    df["property_type"] = df["property_type"].replace("Cluster HouseWhole Unit", "Cluster House")
    df["property_type"] = df["property_type"].fillna(df['property_type'].mode()[0])

    valid_built_years = df[df["built_year"] != 9999]["built_year"]
    df["built_year"] = df["built_year"].replace(9999, valid_built_years.median())

    df["distance_to_mrt_in_m"] = df["distance_to_mrt_in_m"].replace(np.inf, df["distance_to_mrt_in_m"].median())
    df["has_pool"] = df["has_pool"].replace(pd.NA, False)
    df["has_gym"] = df["has_gym"].replace(pd.NA, False)

    return df


def create_or_get_experiment(experiment_name: str, artifact_bucket: str, mlflow_conn_id: str, **context):
    """
    Create a new MLFlow experiment with a specified name if it doesn't exist,
    or retrieve the existing experiment's ID.
    """
    from mlflow_provider.hooks.client import MLflowClientHook

    mlflow_hook = MLflowClientHook(mlflow_conn_id=mlflow_conn_id)

    # Check if the experiment already exists
    existing_experiments = mlflow_hook.run(
        endpoint="api/2.0/mlflow/experiments/search",
        headers={"Content-Type": "application/json"},
        request_params={"max_results": 5}
    )
    print(existing_experiments.json())

    resp = existing_experiments.json()
    for exp in resp.get("experiments", []):
        if exp["name"] == experiment_name:
            print(f"Experiment '{experiment_name}' already exists with ID {exp['experiment_id']}")
            return exp["experiment_id"]

    # Create a new experiment if it doesn't exist
    new_experiment_information = mlflow_hook.run(
        endpoint="api/2.0/mlflow/experiments/create",
        request_params={
            "name": experiment_name,
            "artifact_location": f"s3://{artifact_bucket}/mlflow/{experiment_name.lower()}",
        },
    ).json()

    print(f"Created new experiment '{experiment_name}' with ID {new_experiment_information['experiment_id']}")
    return new_experiment_information["experiment_id"]


def perform_eda(upstream_task: list[str], mlflow_tracking_uri: str, **kwargs):
    import mlflow

    ti = kwargs['ti']
    df = ti.xcom_pull(task_ids=upstream_task[0])
    experiment_id = ti.xcom_pull(task_ids=upstream_task[1])

    numerical_columns = [
        "price",
        "bedroom",
        "bathroom",
        "dimensions",
        "built_year",
        "distance_to_mrt_in_m",
        "distance_to_hawker_in_m",
        "distance_to_supermarket_in_m",
        "distance_to_sch_in_m",
        "distance_to_mall_in_m"]
    categorical_columns = ["property_type", "furnishing", "floor_level", "district_id", "tenure", "facing"]

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    with mlflow.start_run(
            experiment_id=experiment_id,
            run_name="EDA"):
        # Log descriptive statistics
        mlflow.log_param("numerical_columns", numerical_columns)
        mlflow.log_param("categorical_columns", categorical_columns)
        mlflow.log_metric("num_rows", len(df))
        mlflow.log_metric("num_columns", len(df.columns))
        mlflow.log_metric("num_unique_rental_price", df["price"].nunique())
        mlflow.log_metric("min_rental_price", df["price"].min())
        mlflow.log_metric("max_rental_price", df["price"].max())
        mlflow.log_metric("mean_rental_price", df["price"].mean())
        mlflow.log_metric("median_rental_price", df["price"].median())
        mlflow.log_metric("std_rental_price", df["price"].std())

        # Log correlation with price
        correlations = df[numerical_columns].corr()["price"].sort_values(ascending=False)
        mlflow.log_param("price_correlations", correlations.to_dict())


def prepare_data(upstream_task: str, **kwargs):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from lib.utils.outlier import OutlierHandlerIQR

    ti = kwargs['ti']
    df = ti.xcom_pull(task_ids=upstream_task)

    numerical_columns = [
        "price",
        "bedroom",
        "bathroom",
        "dimensions",
        "built_year",
        "distance_to_mrt_in_m",
        "distance_to_hawker_in_m",
        "distance_to_supermarket_in_m",
        "distance_to_sch_in_m",
        "distance_to_mall_in_m"]
    categorical_columns = ["property_type", "furnishing", "floor_level", "district_id", "tenure", "facing"]

    # drop columns not in numerical and categorical columns
    df = df.drop(columns=[col for col in df.columns if col not in numerical_columns + categorical_columns])
    logging.info(df.columns)

    rental_price = df['price']
    X = df.drop(['price'], axis=1)

    X_train, X_temp, y_train, y_temp = train_test_split(X, rental_price, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    outlier_handler = OutlierHandlerIQR()
    X_train_new, y_train_new = outlier_handler.fit_transform(X_train, y_train)
    X_val_new, y_val_new = outlier_handler.transform(X_val, y_val)
    X_test_new, y_test_new = outlier_handler.transform(X_test, y_test)

    train_df = pd.concat([X_train_new, y_train_new], axis=1)
    val_df = pd.concat([X_val_new, y_val_new], axis=1)
    test_df = pd.concat([X_test_new, y_test_new], axis=1)

    return (train_df, val_df, test_df)


def train_and_evaluate_model(
        experiment_id: str,
        model_class: any,
        model_name: str,
        mlflow_tracking_uri: str,
        train_data: DataFrame,
        val_data: DataFrame):
    import gc
    import mlflow
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    X_train = train_data.drop('price', axis=1)
    y_train = train_data['price']
    X_val = val_data.drop('price', axis=1)
    y_val = val_data['price']

    numerical_columns = [
        "bedroom",
        "bathroom",
        "dimensions",
        "built_year",
        "distance_to_mrt_in_m",
        "distance_to_hawker_in_m",
        "distance_to_supermarket_in_m",
        "distance_to_sch_in_m",
        "distance_to_mall_in_m"]
    categorical_columns = ["property_type", "furnishing", "floor_level", "district_id", "tenure", "facing"]

    column_transformer = ColumnTransformer(
        transformers=[
            ("scaler", StandardScaler(), [col for col in numerical_columns if col != "price"]),
            ("encoder", OneHotEncoder(drop=None, sparse_output=False), categorical_columns)
        ],
        remainder="passthrough"
    )

    pipeline = Pipeline([
        ('preprocessor', column_transformer),
        ('regressor', model_class)
    ])

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    print(f"Training model: {model_name}")
    run_id = None
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=f"{model_name}",
    ) as run:
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)

        mae = mean_absolute_error(y_val, y_pred).round(2)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred)).round(2)
        r2 = r2_score(y_val, y_pred).round(4)

        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)

        run_id = run.info.run_id
        
        gc.collect()
 
    return (mae, rmse, r2, run_id)


def train_models(upstream_task: list[str], mlflow_tracking_uri: str, **kwargs):
    from sklearn.ensemble import (
        RandomForestRegressor,
        AdaBoostRegressor,
    )
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.linear_model import (
        LinearRegression,
        Lasso,
        Ridge
    )
    from xgboost import XGBRegressor
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor

    ti = kwargs['ti']
    data = ti.xcom_pull(task_ids=upstream_task[0])
    experiment_id = ti.xcom_pull(task_ids=upstream_task[1])
    train_data, val_data, _ = data

    models = [
        # (LinearRegression(), "linear_regression"),
        # (Lasso(), "lasso"),
        # (Ridge(), "ridge"),
        (DecisionTreeRegressor(), "decision_tree"),
        (RandomForestRegressor(), "random_forest"),
        (AdaBoostRegressor(), "ada_boost"),
        (XGBRegressor(), "xgboost"),
        (CatBoostRegressor(), "catboost"),
        (LGBMRegressor(), "lightgbm"),
    ]

    best_model = best_run = None
    best_r2 = float("-inf")

    for model_class, model_name in models:
        mae, rmse, r2, run_id = train_and_evaluate_model(
            experiment_id, model_class, model_name, mlflow_tracking_uri, train_data, val_data)
        print(f"Model: {model_name}, MAE: {mae}, RMSE: {rmse}, r2: {r2}")
        if r2 > best_r2:
            best_r2 = r2
            best_run = run_id
            best_model = model_name
    
    run_summary =  "".join(
        "Best regression model for property listing price prediction is "
        f"{best_model} with accuracy of {best_r2}."
    )

    return (best_model, best_run, run_summary)


def create_objective(model_name: str, best_model: any, X_train, y_train):
    def objective(trial):
        import gc
        from numpy import mean
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import mean_squared_error
        from math import sqrt
        if model_name == 'random_forest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 2000, step=200),
                'max_features': trial.suggest_categorical('max_features', ['log2', 'sqrt']),
                'max_depth': trial.suggest_int('max_depth', 10, 110, step=10),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10, step=3),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4, step=1),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
            }
        elif model_name == 'catboost':
            params = {
                "iterations": trial.suggest_categorical("iterations", [1000]),
                "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 1),
                "depth": trial.suggest_int("depth", 6, 10),
                "subsample": trial.suggest_categorical("subsample", [0.05, 0.2, 0.4, 0.6, 0.8, 1.0]),
                "colsample_bylevel": trial.suggest_categorical("colsample_bylevel", [0.05, 0.2, 0.4, 0.6, 0.8, 1.0]),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 100, step=20)
            }
        elif model_name == 'decision_tree':
            params = {
                "max_depth": trial.suggest_categorical("max_depth", [3, 5, 7, 10, 15]),
                "min_samples_split": trial.suggest_categorical("min_samples_split", [2, 5, 10, 20]),
                "min_samples_leaf": trial.suggest_categorical("min_samples_leaf", [1, 3, 5, 10])
            }
        elif model_name == 'lasso_regression' or model_name == 'ridge_regression':
            params = {
                "alpha": trial.suggest_loguniform("alpha", 0.001, 10.0)
            }
        elif model_name == 'hgb':
            params = {
                "max_depth": trial.suggest_categorical("max_depth", [3, 5, 7, 10]),
                "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 1.0),
                "min_samples_leaf": trial.suggest_categorical("min_samples_leaf", [1, 3, 5, 10])
            }
        elif model_name == 'xgb':
            params = {
                "n_estimators": trial.suggest_categorical("n_estimators", [100, 200, 300, 400, 500]),
                "max_depth": trial.suggest_int("max_depth", 3, 5),
                "subsample": trial.suggest_categorical("subsample", [0.8, 0.9, 1.0]),
                "colsample_bytree": trial.suggest_categorical("colsample_bytree", [0.8, 0.9, 1.0]),
                "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.1),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 3),
                "gamma": trial.suggest_categorical("gamma", [0, 0.1, 0.2])
            }
        elif model_name == 'lgbm':
            params = {
                "max_depth": trial.suggest_categorical("max_depth", [3, 5, 7, 10]),
                "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 1.0),
                "n_estimators": trial.suggest_categorical("n_estimators", [50, 100, 200, 500]),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50, step=5)
            }
            
        model = best_model(**params)
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    
        # Convert scores to positive RMSE values
        rmse_scores = [sqrt(-score) for score in scores]
        print(f"Scores: {rmse_scores}")
        avg_rmse = mean(rmse_scores)
        
        # Clean up
        gc.collect()
        
        return avg_rmse
        
    return objective     


def tune_model(upstream_task: list[str], mlflow_tracking_uri: str, **kwargs):
    import mlflow
    import optuna
    import pickle
    import tempfile
    from pathlib import Path
    from numpy import sqrt
    from optuna.samplers import TPESampler
    from sklearn.ensemble import (
        RandomForestRegressor,
        AdaBoostRegressor,
        HistGradientBoostingRegressor
    )
    from sklearn.tree import DecisionTreeRegressor
    from xgboost import XGBRegressor
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score, r2_score
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer

    ti = kwargs['ti']
    data = ti.xcom_pull(task_ids=upstream_task[0])
    experiment_id = ti.xcom_pull(task_ids=upstream_task[1])
    model_name = ti.xcom_pull(task_ids=upstream_task[2])[0]

    train_data, val_data, test_data = data

    X_train = train_data.drop('price', axis=1)
    y_train = train_data['price']
    X_val = val_data.drop('price', axis=1)
    y_val = val_data['price']
    X_test = test_data.drop('price', axis=1)
    y_test = test_data['price']

    models = {
        "decision_tree": DecisionTreeRegressor,
        "random_forest": RandomForestRegressor,
        "ada_boost": AdaBoostRegressor,
        "hist_gradient_boosting": HistGradientBoostingRegressor,
        "xgboost": XGBRegressor,
        "catboost": CatBoostRegressor,
        "lightgbm": LGBMRegressor,
    }

    numerical_columns = [
        "bedroom",
        "bathroom",
        "dimensions",
        "built_year",
        "distance_to_mrt_in_m",
        "distance_to_hawker_in_m",
        "distance_to_supermarket_in_m",
        "distance_to_sch_in_m",
        "distance_to_mall_in_m"]
    categorical_columns = ["property_type", "furnishing", "floor_level", "district_id", "tenure", "facing"]

    column_transformer = ColumnTransformer(
        transformers=[
            ("scaler", StandardScaler(), [col for col in numerical_columns if col != "price"]),
            ("encoder", OneHotEncoder(drop=None, sparse_output=False), categorical_columns)
        ],
        remainder="passthrough"
    )

    X_train = column_transformer.fit_transform(X_train)
    X_val = column_transformer.transform(X_val)
    X_test = column_transformer.transform(X_test)

    run_id = None
    run_summary = ""
    best_model = models[model_name]
    
    storage_url = "postgresql://airflow:airflow@postgres/optuna"
    
    # Start the MLflow run
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name="hyperparameter_tuning"
    ) as run:        
        # Create an Optuna study
        study = optuna.create_study(direction="minimize", sampler=TPESampler(), load_if_exists=True, storage=storage_url)
        
        # Create the objective function with the specified model_name and best_model
        objective = create_objective(model_name, best_model, X_train, y_train)
        
        # Optimize the study
        # study.optimize(objective, n_trials=15, n_jobs=1, gc_after_trial=True)
        study.optimize(objective, n_trials=1, n_jobs=1, gc_after_trial=True)

        # Get the best hyperparameters
        best_params = study.best_params
        print(f"Best hyperparameters: {best_params}")
        print(f"Best value: {study.best_value}")
        
        # Train the model with the best hyperparameters
        best_model_instance = best_model(**best_params)
        best_model_instance.fit(X_train, y_train)
        
        # Predict on the validation set
        y_pred = best_model_instance.predict(X_val)
        
        # Infer signature for model logging
        signature = mlflow.models.infer_signature(X_val, y_pred)

        mae = mean_absolute_error(y_val, y_pred).round(2)
        rmse = sqrt(mean_squared_error(y_val, y_pred)).round(2)
        evs = explained_variance_score(y_val, y_pred).round(2)
        r2 = r2_score(y_val, y_pred).round(2)
        
        # Log the results
        mlflow.log_params(study.best_params)
        
        mlflow.log_metric('ht_val_mae', mae)
        mlflow.log_metric('ht_val_rmse', rmse)
        mlflow.log_metric('ht_val_explained_variance_score', evs)
        mlflow.log_metric('ht_val_r2_score', r2)
        
        y_pred = best_model_instance.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred).round(2)
        rmse = sqrt(mean_squared_error(y_test, y_pred)).round(2)
        evs = explained_variance_score(y_test, y_pred).round(2)
        r2 = r2_score(y_test, y_pred).round(2)
        
        mlflow.log_metric('ht_test_mae', mae)
        mlflow.log_metric('ht_test_rmse', rmse)
        mlflow.log_metric('ht_test_explained_variance_score', evs)
        mlflow.log_metric('ht_test_r2_score', r2)
        
        run_summary = f"Model {model_name} was tuned to an accuracy of {r2}."
        logging.info(run_summary)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir, "column_transformer.pkl")
            path.write_bytes(pickle.dumps(column_transformer))
            mlflow.log_artifact(local_path=path.absolute(), artifact_path="column_transformer")
            logging.info(f"Logged artifact: {path.absolute()}")
        
        if model_name == 'xgboost':
            mlflow.xgboost.log_model(best_model_instance, artifact_path="model", signature=signature)
        elif model_name == 'lightgbm':
            mlflow.lightgbm.log_model(best_model_instance, artifact_path="model", signature=signature)
        elif model_name == 'catboost':
            mlflow.catboost.log_model(best_model_instance, artifact_path="model", signature=signature)
        else:
            mlflow.sklearn.log_model(best_model_instance, artifact_path="model", signature=signature)

        run_id = run.info.run_id
        
    return (run_id, run_summary, model_name, r2)


def check_if_model_already_registered(
        mlflow_conn_id: str,
        model_name: str,
        create_registered_model_task_id: str,
        model_already_registered_task_id: str,
        **kwargs):
    "Get information about existing registered MLFlow models."
    from mlflow_provider.hooks.client import MLflowClientHook

    mlflow_hook = MLflowClientHook(mlflow_conn_id=mlflow_conn_id, method="GET")
    get_reg_model_response = mlflow_hook.run(
        endpoint="api/2.0/mlflow/registered-models/get",
        request_params={"name": model_name},
    ).json()

    if "error_code" in get_reg_model_response:
        if get_reg_model_response["error_code"] != "RESOURCE_DOES_NOT_EXIST":
            raise ValueError(
                f"Error when checking if model is registered: {get_reg_model_response['error_code']}"
            )
        reg_model_exists = False
    else:
        reg_model_exists = True

    return model_already_registered_task_id if reg_model_exists else create_registered_model_task_id


check_and_trigger_retraining_task = BranchPythonOperator(
    task_id="check_and_trigger_retraining",
    python_callable=check_and_trigger_retraining,
    op_kwargs={
        'duckdb_conn_id': DUCKDB_CONN_ID,
        'dataset_uri': DATASET_URI
    },
    dag=dag,
)

retraining_not_triggered_task = EmptyOperator(task_id="retraining_not_triggered")
retraining_triggered_task = EmptyOperator(task_id="retraining_triggered")
retraining_task = [retraining_not_triggered_task, retraining_triggered_task]

load_data_task = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    op_kwargs={'duckdb_conn_id': DUCKDB_CONN_ID, 'table_name': TABLE_NAME},
    dag=dag,
)

clean_data_task = PythonOperator(
    task_id='clean_data',
    python_callable=clean_data,
    op_kwargs={'upstream_task': load_data_task.task_id},
    dag=dag,
)

create_experiment_task = PythonOperator(
    task_id='create_experiment',
    python_callable=create_or_get_experiment,
    op_kwargs={
        'experiment_name': EXPERIMENT_NAME,
        'artifact_bucket': ARTIFACT_BUCKET,
        'mlflow_conn_id': MLFLOW_CONN_ID
    },
    dag=dag,
)

perform_eda_task = PythonOperator(
    task_id='perform_eda',
    python_callable=perform_eda,
    op_kwargs={
        'upstream_task': [clean_data_task.task_id, create_experiment_task.task_id],
        'mlflow_tracking_uri': MLFLOW_TRACKING_URI
    },
    dag=dag,
)

prepare_data_task = PythonOperator(
    task_id='prepare_data',
    python_callable=prepare_data,
    op_kwargs={
        'upstream_task': clean_data_task.task_id
    },
    dag=dag,
)

train_models_task = PythonOperator(
    task_id='train_models',
    python_callable=train_models,
    op_kwargs={
        'upstream_task': [prepare_data_task.task_id, create_experiment_task.task_id],
        'mlflow_tracking_uri': MLFLOW_TRACKING_URI
    },
    dag=dag,
)

tune_model_task = PythonOperator(
    task_id='tune_model',
    python_callable=tune_model,
    op_kwargs={
        'upstream_task': [prepare_data_task.task_id, create_experiment_task.task_id, train_models_task.task_id],
        'mlflow_tracking_uri': MLFLOW_TRACKING_URI
    },
    dag=dag,
)

create_registered_model_task = CreateRegisteredModelOperator(
    mlflow_conn_id=MLFLOW_CONN_ID,
    task_id="create_registered_model",
    name=REGISTERED_MODEL_NAME,
    tags=[
        {"key": "model_type", "value": "regression"},
        {"key": "data", "value": "property_listing"},
    ],
)

model_already_registered_task = EmptyOperator(task_id="model_already_registered")

register_model_task = [model_already_registered_task, create_registered_model_task]

check_if_model_already_registered_task = BranchPythonOperator(
    task_id="check_if_model_already_registered",
    python_callable=check_if_model_already_registered,
    op_kwargs={
        'mlflow_conn_id': MLFLOW_CONN_ID,
        'model_name': REGISTERED_MODEL_NAME,
        'create_registered_model_task_id': create_registered_model_task.task_id,
        'model_already_registered_task_id': model_already_registered_task.task_id
    },
    dag=dag,
)

create_model_version_task = CreateModelVersionOperator(
    task_id="create_model_version",
    name=REGISTERED_MODEL_NAME,
    mlflow_conn_id=MLFLOW_CONN_ID,
    source="s3://"
    + ARTIFACT_BUCKET
    + "/mlflow"
    + f"/{EXPERIMENT_NAME}/"
    + "{{ ti.xcom_pull(task_ids='tune_model')[0] }}"
    + "/artifacts/model",
    run_id="{{ ti.xcom_pull(task_ids='tune_model')[0] }}",
    description="{{ ti.xcom_pull(task_ids='tune_model')[1] }}",
    tags=[
        {"key": "model_name", "value": "{{ ti.xcom_pull(task_ids='tune_model')[2] }}"}, 
        {"key": "r2_score", "value": "{{ ti.xcom_pull(task_ids='tune_model')[3] }}"},
        {
            "key": "column_transformer_source", 
            "value": 
                f"s3://{ARTIFACT_BUCKET}/mlflow/{EXPERIMENT_NAME}/" + 
                "{{ ti.xcom_pull(task_ids='tune_model')[0] }}" + 
                "/artifacts/column_transformer/column_transformer.pkl"
        },
    ],
    trigger_rule="one_success",
)

transition_model_task = TransitionModelVersionStageOperator(
    task_id="transition_model",
    name=REGISTERED_MODEL_NAME,
    mlflow_conn_id=MLFLOW_CONN_ID,
    version="{{ ti.xcom_pull(task_ids='create_model_version')['model_version']['version'] }}",
    stage="Staging",
    archive_existing_versions=True,
)


check_and_trigger_retraining_task >> retraining_task

retraining_triggered_task >> load_data_task
retraining_triggered_task >> create_experiment_task

load_data_task >> clean_data_task

clean_data_task >> perform_eda_task
clean_data_task >> prepare_data_task

create_experiment_task >> perform_eda_task
create_experiment_task >> train_models_task

prepare_data_task >> train_models_task

train_models_task >> tune_model_task

tune_model_task >> check_if_model_already_registered_task

check_if_model_already_registered_task >> register_model_task

register_model_task >> create_model_version_task >> transition_model_task
