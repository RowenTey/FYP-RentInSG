import os
import shap
import json
import time
import pickle
import mlflow
import logging
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
from utils.constants import DEFAULT_VALUES, DISTRICTS
from utils.motherduckdb import MotherDuckDBConnector
from utils.distance_utils import find_nearest_single

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "https://ks-8000.leejacksonz.com/")
REGISTERED_MODEL_NAME = os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "rent_in_sg_reg_model")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

enrichment = {
    "distance_to_mrt_in_m": (
        "SELECT * FROM mrt_info",
        ["station_name", "latitude", "longitude"],
    ),
    "distance_to_mall_in_m": (
        "SELECT * FROM mall_info",
        ["name", "latitude", "longitude"],
    ),
    "distance_to_sch_in_m": (
        "SELECT * FROM primary_school_info",
        ["name", "latitude", "longitude"],
    ),
    "distance_to_hawker_in_m": (
        "SELECT * FROM hawker_centre_info",
        ["name", "latitude", "longitude"],
    ),
    "distance_to_supermarket_in_m": (
        "SELECT * FROM supermarket_info",
        ["name", "latitude", "longitude"],
    ),
}


class MLflowModel:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MLflowModel, cls).__new__(cls)
            cls._instance.__init__()
        return cls._instance

    def __init__(self):
        self.logger = logging.getLogger(MLflowModel.__name__)
        self.version = self.model = self.column_transformer = self.db = None
        self.geocoder: Nominatim = None
        self.explainer: shap.TreeExplainer = None
        self.transformed_feature_names: list = None

    def initialize(self, db: MotherDuckDBConnector):
        self.version = self.fetch_latest_version()
        self.model = self.get_model(REGISTERED_MODEL_NAME, self.version["source"])
        self.column_transformer = self.get_column_transformer(
            self.version["tags"]["column_transformer_source"])

        self.db = db
        self.geocoder = Nominatim(user_agent="sg_rental_price_dashboard")

        # self.explainer = shap.TreeExplainer(self.model)
        self.explainer = shap.TreeExplainer(self.model)
        # Get the feature names after transformation
        self.transformed_feature_names = self.column_transformer.get_feature_names_out()

    def fetch_latest_version(self) -> dict:
        client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        # -1 because models are sorted by version in descending order
        latest_version = dict(client.get_latest_versions(REGISTERED_MODEL_NAME)[-1])
        self.logger.info(latest_version)
        return latest_version

    def get_model(self, model_name, source):
        if model_name == 'xgboost':
            return mlflow.xgboost.load_model(source)
        elif model_name == 'lightgbm':
            return mlflow.lightgbm.load_model(source)
        elif model_name == 'catboost':
            return mlflow.catboost.load_model(source)
        else:
            return mlflow.sklearn.load_model(source)

    def get_column_transformer(self, source):
        column_transformer = None
        dst_path = mlflow.artifacts.download_artifacts(source)
        with open(dst_path, "rb") as file:
            column_transformer = pickle.load(file)
        self.logger.info("Column transformer loaded!")
        return column_transformer

    def update_feature_names(
        self,
        explanation: shap.Explanation,
        feature_names: list
    ) -> shap.Explanation:
        # Create a mapping from transformed feature names to original feature names
        feature_map = {tf: self.map_transformed_feature_to_original(
            tf, feature_names) for tf in explanation.feature_names}

        # Update feature names in the explanation object
        explanation.feature_names = [feature_map.get(
            f, f) for f in explanation.feature_names]

        return explanation

    def map_transformed_feature_to_original(
            self, transformed_feature, original_features):
        for original in original_features:
            if original in transformed_feature:
                return original
        return transformed_feature

    def transform_form_data(self, form_data):
        for key in form_data:
            # set default values if form data is empty
            if not form_data[key] and key in DEFAULT_VALUES:
                form_data[key] = DEFAULT_VALUES[key]

            # convert to proper format
            if key == "tenure":
                form_data[key] = form_data[key].lower()
            elif key == "district_id":
                form_data[key] = DISTRICTS[form_data[key]]

        return form_data

    def generate_shap_explanation(self, shap_values, feature_names):
        # Get the feature importances
        feature_importance = np.abs(shap_values).mean(0)

        # Sort features by importance
        sorted_idx = feature_importance.argsort()
        sorted_features = self.transformed_feature_names[sorted_idx]

        # Get top 5 most important features
        top_features = sorted_features[-5:]
        top_values = shap_values[0][sorted_idx][-5:]

        explanation = "The rental price predicted is based on several factors. Here are the top 5 most influential features:\n\n"  # noqa: E501
        for feature, value in zip(reversed(top_features), reversed(top_values)):
            # Map back to original feature if possible
            logging.info(feature, feature_names)
            original_feature = self.map_transformed_feature_to_original(
                feature, feature_names)

            if value > 0:
                direction = "increased"
            else:
                direction = "decreased"

            explanation += f"- {original_feature}: This feature {direction} the predicted rental price (relative impact: {abs(value):.4f}).\n"  # noqa: E501

        explanation += "\nThese values highlight the relative impact of each feature on the prediction compared to an average property."  # noqa: E501
        return explanation

    def add_distance_info(self, validated_form_data: dict, fill_default: bool = False) -> dict:
        # set the key in form data so default values are filled in later
        if fill_default:
            for key in enrichment.keys():
                validated_form_data[key] = None

            with open("static/district_coords.json", "r") as f:
                coords = json.load(f)

            validated_form_data["latitude"] = coords[
                validated_form_data["district_id"]][0]
            validated_form_data["longitude"] = coords[
                validated_form_data["district_id"]][1]

            logging.info(
                f"Added default distance values: ({validated_form_data['latitude']}, {validated_form_data['longitude']})")

            return validated_form_data

        location = self.geocoder.geocode(f"{validated_form_data['address']}, Singapore")
        validated_form_data["latitude"] = location.latitude
        validated_form_data["longitude"] = location.longitude

        for key, val in enrichment.items():
            df = self.db.fetch_info(val[0], val[1])
            validated_form_data[key] = find_nearest_single(
                {
                    "latitude": validated_form_data["latitude"],
                    "longitude": validated_form_data["longitude"],
                },
                df,
            )

        return validated_form_data

    async def predict(self, form_data: dict) -> dict:
        if form_data["address"] is not None and form_data["address"] != "":
            logging.info("Fetching distance info...")
            form_data = self.add_distance_info(form_data)
        else:
            logging.info("Address is empty, using default distance values...")
            form_data = self.add_distance_info(form_data, True)

        logging.info("Transforming form data...")
        validated_form_data = self.transform_form_data(form_data)
        print(validated_form_data)

        logging.info("Creating input DataFrame...")
        input_df = pd.DataFrame(validated_form_data, index=[0])
        input_df = input_df.drop(
            columns=["address", "latitude", "longitude"], axis=1)
        logging.info(f"\n{input_df}\n")

        logging.info("Transforming input DataFrame...")
        # Apply the same transformations to the form data as done during training
        transformed_data = self.column_transformer.transform(input_df)

        logging.info("Making predictions...")
        # Make predictions using the model
        prediction, *_ = self.model.predict(transformed_data)

        logging.info("Generating description...")
        # Compute SHAP values
        shap_values = self.explainer.shap_values(transformed_data)
        prediction_desc = self.generate_shap_explanation(
            shap_values,
            input_df.columns)

        # Create a custom Explanation object
        logging.info("Generating SHAP object...")
        explained_shap_values = self.explainer(transformed_data)
        explanation_obj = shap.Explanation(values=explained_shap_values.values,
                                           base_values=explained_shap_values.base_values,
                                           data=transformed_data,
                                           feature_names=self.transformed_feature_names)
        explanation_obj = self.update_feature_names(explanation_obj, input_df.columns)

        result = {
            "prediction": prediction,
            "description": prediction_desc,
            "shap_values": explanation_obj.values.tolist() if isinstance(
                explanation_obj.values,
                np.ndarray) else explanation_obj.values,
            "shap_base_values": explanation_obj.base_values.tolist() if isinstance(
                explanation_obj.base_values,
                np.ndarray) else explanation_obj.base_values,
            "shap_data": explanation_obj.data.tolist() if isinstance(
                explanation_obj.data,
                np.ndarray) else explanation_obj.data,
            "shap_feature_names": explanation_obj.feature_names,
            "coordinates": (
                validated_form_data["latitude"],
                validated_form_data["longitude"])}

        return result

    async def predict_stream(self, form_data: dict) -> float:
        async def progress_generator(form_data: dict):
            yield json.dumps({"progress": 0, "message": "Starting prediction process..."})

            if form_data["address"] is not None and form_data["address"] != "":
                yield json.dumps({"progress": 10, "message": "Fetching distance info..."})
                form_data = self.add_distance_info(form_data)
            else:
                yield json.dumps({"progress": 10, "message": "Address is empty, using default distance values..."})
                form_data = self.add_distance_info(form_data, True)

            yield json.dumps({"progress": 20, "message": "Transforming form data..."})
            validated_form_data = self.transform_form_data(form_data)

            yield json.dumps({"progress": 30, "message": "Creating input DataFrame..."})
            input_df = pd.DataFrame(validated_form_data, index=[0])
            input_df = input_df.drop(columns=["address", "latitude", "longitude"], axis=1)

            yield json.dumps({"progress": 50, "message": "Transforming input DataFrame..."})
            transformed_data = self.column_transformer.transform(input_df)

            yield json.dumps({"progress": 60, "message": "Making predictions..."})
            prediction, *_ = self.model.predict(transformed_data)

            yield json.dumps({"progress": 70, "message": "Generating description..."})
            # Compute SHAP values
            shap_values = self.explainer.shap_values(transformed_data)
            prediction_desc = self.generate_shap_explanation(
                shap_values,
                input_df.columns)

            yield json.dumps({"progress": 80, "message": "Generating SHAP object..."})
            explained_shap_values = self.explainer(transformed_data)
            explanation_obj = shap.Explanation(values=explained_shap_values.values,
                                               base_values=explained_shap_values.base_values,
                                               data=transformed_data,
                                               feature_names=self.transformed_feature_names)
            explanation_obj = self.update_feature_names(explanation_obj, input_df.columns)

            yield json.dumps({"progress": 90, "message": "Finalising result..."})
            result = {
                "prediction": prediction,
                "description": prediction_desc,
                "shap_values": explanation_obj.values.tolist() if isinstance(
                    explanation_obj.values,
                    np.ndarray) else explanation_obj.values,
                "shap_base_values": explanation_obj.base_values.tolist() if isinstance(
                    explanation_obj.base_values,
                    np.ndarray) else explanation_obj.base_values,
                "shap_data": explanation_obj.data.tolist() if isinstance(
                    explanation_obj.data,
                    np.ndarray) else explanation_obj.data,
                "shap_feature_names": explanation_obj.feature_names,
                "coordinates": (
                    validated_form_data["latitude"],
                    validated_form_data["longitude"])}

            yield json.dumps({"progress": 100, "message": "Prediction complete", "result": result})
        return progress_generator(form_data)


model = MLflowModel()
