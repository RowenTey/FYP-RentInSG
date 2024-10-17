import time
import shap
import pickle

import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import streamlit as st
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from components.constants import *
from components.find_closest import find_nearest_single

st.set_page_config(
    "Singapore Rental Price Analysis | Home",
    page_icon="🏠",
    layout="wide",
)

st.title("Singapore Rental Price Prediction Home")
st.text("This page predicts the rental price of a property in Singapore 📈")
st.info(
    """
    Fill in the form below with specific parameteres to predict monthly rental price of a property in Singapore.
    __(\\* indicates required fields)__
"""
)

if "model" not in st.session_state:
    st.session_state["model"] = None

if "column_transformer" not in st.session_state:
    st.session_state["column_transformer"] = None


MLFLOW_TRACKING_URI = "https://ks-8000.leejacksonz.com/"
REGISTERED_MODEL_NAME = "rent_in_sg_reg_model"
listings_df = pd.DataFrame()


@st.cache_data
def fetch_latest_version() -> dict:
    import mlflow
    client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    # -1 because models are sorted by version in descending order
    latest_version = dict(client.get_latest_versions(REGISTERED_MODEL_NAME)[-1])
    print(latest_version)
    return latest_version


@st.cache_resource
def get_model(model_name, source):
    import mlflow
    if model_name == 'xgboost':
        return mlflow.xgboost.load_model(source)
    elif model_name == 'lightgbm':
        return mlflow.lightgbm.load_model(source)
    elif model_name == 'catboost':
        return mlflow.catboost.load_model(source)
    else:
        return mlflow.sklearn.load_model(source)


@st.cache_resource
def get_column_transformer(source):
    import mlflow
    column_transformer = None
    dst_path = mlflow.artifacts.download_artifacts(source)
    with open(dst_path, "rb") as file:
        column_transformer = pickle.load(file)
    return column_transformer


@st.cache_resource
def init_motherduckdb_conn():
    from components.motherduckdb_connector import connect_to_motherduckdb
    return connect_to_motherduckdb()


@st.cache_data
def fetch_info(_db, query, target_cols=None):
    queried_df = _db.query_df(query)

    if target_cols:
        return queried_df[target_cols]

    return queried_df


@st.cache_data
def load_local_data(file_path):
    return pd.read_csv(file_path)


@st.fragment(run_every="2h")
def fetch_listings_df():
    if "db" not in st.session_state:
        return

    global listings_df
    db = st.session_state["db"]
    listings_df = db.query_df("SELECT * FROM property_listing")


def update_feature_names(
        explanation: shap.Explanation,
        feature_names: list) -> shap.Explanation:
    # Create a mapping from transformed feature names to original feature names
    feature_map = {tf: map_transformed_feature_to_original(
        tf, feature_names) for tf in explanation.feature_names}

    # Update feature names in the explanation object
    explanation.feature_names = [feature_map.get(
        f, f) for f in explanation.feature_names]

    return explanation


def transform_form_data(form_data):
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


def fetch_listings_within_radius(lat, lon, radius_km, listings_df):
    listings_df["distance"] = listings_df.apply(lambda row: geodesic(
        (lat, lon), (row["latitude"], row["longitude"])).km, axis=1)
    nearby_listings = listings_df[listings_df["distance"] <= radius_km]
    return nearby_listings


def plot_listings_on_map(listings, user_location):
    # enforce latitude and longitude colummns to be numeric
    listings["latitude"] = pd.to_numeric(listings["latitude"])
    listings["longitude"] = pd.to_numeric(listings["longitude"])

    # convert user_location tuple to float
    user_location = [float(user_location[0]), float(user_location[1])]

    fig = px.scatter_mapbox(
        listings,
        lat="latitude",
        lon="longitude",
        hover_name="property_name",
        hover_data={"price": True, "distance": True},
        color_discrete_sequence=["blue"],
        zoom=12,
        height=600
    )
    user_location_df = pd.DataFrame(
        [{"latitude": user_location[0],
          "longitude": user_location[1],
          "type": "User Location"}])
    fig.add_trace(px.scatter_mapbox(
        user_location_df,
        lat="latitude",
        lon="longitude",
        hover_name="type",
        color_discrete_sequence=["red"],
    ).data[0])
    fig.update_layout(mapbox_style="open-street-map")

    st.subheader("Nearby Listings")
    st.plotly_chart(fig)


def add_distance_info(validated_form_data: dict, fill_default: bool = False) -> dict:
    import json

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

        st.write(
            f"Added default distance values: ({validated_form_data['latitude']}, {validated_form_data['longitude']})")

        return validated_form_data

    if "db" not in st.session_state:
        return validated_form_data

    db = st.session_state["db"]
    location = geocoder.geocode(f"{validated_form_data['address']}, Singapore")
    validated_form_data["latitude"] = location.latitude
    validated_form_data["longitude"] = location.longitude

    for key, val in enrichment.items():
        df2 = fetch_info(db, val[0], val[1])
        validated_form_data[key] = find_nearest_single(
            {
                "latitude": validated_form_data["latitude"],
                "longitude": validated_form_data["longitude"],
            },
            df2,
        )

    return validated_form_data


def generate_shap_explanation(shap_values, feature_names, column_transformer):
    # Get feature names after transformation
    transformed_feature_names = column_transformer.get_feature_names_out()

    # Get the feature importances
    feature_importance = np.abs(shap_values).mean(0)

    # Sort features by importance
    sorted_idx = feature_importance.argsort()
    sorted_features = transformed_feature_names[sorted_idx]

    # Get top 5 most important features
    top_features = sorted_features[-5:]
    top_values = shap_values[0][sorted_idx][-5:]

    explanation = "The rental price predicted is based on several factors. Here are the top 5 most influential features:\n\n"  # noqa: E501

    for feature, value in zip(reversed(top_features), reversed(top_values)):
        # Map back to original feature if possible
        print(feature, feature_names)
        original_feature = map_transformed_feature_to_original(
            feature, feature_names)

        if value > 0:
            direction = "increased"
        else:
            direction = "decreased"

        explanation += f"- {original_feature}: This feature {direction} the predicted rental price (relative impact: {abs(value):.4f}).\n"  # noqa: E501

    explanation += "\nThese values highlight the relative impact of each feature on the prediction compared to an average property."  # noqa: E501

    return explanation


def map_transformed_feature_to_original(
        transformed_feature, original_features):
    for original in original_features:
        if original in transformed_feature:
            return original
    return transformed_feature


def process_form_data(model, column_transformer, form_data) -> float:
    if form_data["address"] is not None and form_data["address"] != "":
        st.write("Fetching distance info...")
        form_data = add_distance_info(form_data)
    else:
        st.write("Address is empty, using default distance values...")
        form_data = add_distance_info(form_data, True)

    st.write("Transforming form data...")
    time.sleep(1)
    validated_form_data = transform_form_data(form_data)

    st.write("Creating input DataFrame...")
    time.sleep(1)
    input_df = pd.DataFrame(validated_form_data, index=[0])
    input_df = input_df.drop(
        columns=["address", "latitude", "longitude"], axis=1)
    print(input_df)

    st.write("Transforming input DataFrame...")
    time.sleep(1)

    # Apply the same transformations to the form data as done during training
    transformed_data = column_transformer.transform(input_df)

    # Compute SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(transformed_data)

    # Get the feature names after transformation
    transformed_feature_names = column_transformer.get_feature_names_out()

    # Create the explainer
    explainer = shap.TreeExplainer(model)

    # Compute SHAP values
    shap_values = explainer(transformed_data)

    # Create a custom Explanation object
    explanation_obj = shap.Explanation(values=shap_values.values,
                                       base_values=shap_values.base_values,
                                       data=transformed_data,
                                       feature_names=transformed_feature_names)

    st.write("Making predictions...")
    time.sleep(1)
    # Make predictions using the model
    prediction, *_ = model.predict(transformed_data)

    st.write("Generating description and plot...")
    prediction_desc = generate_shap_explanation(
        explainer.shap_values(transformed_data),
        input_df.columns,
        column_transformer)
    explanation_obj = update_feature_names(explanation_obj, input_df.columns)

    return prediction, explanation_obj, prediction_desc, (
        validated_form_data["latitude"], validated_form_data["longitude"])


@st.fragment
def get_form_data():
    with st.container(border=True):
        st.subheader("Parameters Selection")

        form_col1, form_col2 = st.columns(2)

        bedroom_select_box = form_col1.number_input(
            label="Number of bedrooms?*",
            placeholder="Enter the number of bedrooms",
            min_value=1,
            value=None,
        )

        bathroom_select_box = form_col2.number_input(
            label="Number of bathrooms?*",
            placeholder="Enter the number of bathrooms",
            min_value=1,
            value=None,
        )

        dimensions_input_box = st.number_input(
            label="Dimensions (sqft)?*",
            placeholder="Enter the dimensions",
            min_value=0,
            value=None,
        )

        district_id_select_box = st.selectbox(
            label="District ID?*",
            options=list(DISTRICTS.keys()),
            placeholder="Select a district ID",
            help="Select the district ID of the property",
            index=None,
        )

        property_type_select_box = st.selectbox(
            label="Property Type?*",
            options=PROPERTY_TYPES,
            placeholder="Select a property type",
            help="Select the property type of the property",
            index=None,
        )

        address_input_box = st.text_input(
            label="Address?", placeholder="Enter the address", value=None)

        built_year_input_box = st.number_input(
            label="Built Year?",
            placeholder="Enter the built year",
            min_value=1950,
            max_value=2024,
            value=None,
        )

        furnishing_select_box = st.selectbox(
            label="Furnishing?",
            options=FURNISHING,
            placeholder="Select a furnishing type",
            help="Select the furnishing type of the property",
            index=None,
        )

        facing_select_box = st.selectbox(
            label="Facing?",
            options=FACING,
            placeholder="Select a facing direction",
            help="Select the facing direction of the property",
            index=None,
        )

        floor_level_select_box = st.selectbox(
            label="Floor Level?",
            options=FLOOR_LEVEL,
            placeholder="Select a floor level",
            help="Select the floor level of the property",
            index=None,
        )

        tenure_select_box = st.selectbox(
            label="Tenure?",
            options=TENURE,
            placeholder="Select a tenure type",
            help="Select the tenure type of the property",
            index=None,
        )

        col3, col4 = st.columns(2)

        has_gym_checkbox = col3.checkbox(
            label="Has Gym?",
        )

        has_pool_checkbox = col4.checkbox(
            label="Has Pool?",
        )

        is_whole_unit_checkbox = st.checkbox(
            label="Is Whole Unit?",
        )

        can_submit = (
            bedroom_select_box
            and bathroom_select_box
            and dimensions_input_box
            and district_id_select_box
            and property_type_select_box
        )
        if st.button("Submit", type="primary", disabled=not can_submit):
            st.session_state.form_data = {
                "bedroom": bedroom_select_box,
                "bathroom": bathroom_select_box,
                "dimensions": dimensions_input_box,
                "address": address_input_box,
                "built_year": built_year_input_box,
                "district_id": district_id_select_box,
                "property_type": property_type_select_box,
                "furnishing": furnishing_select_box,
                "floor_level": floor_level_select_box,
                "facing": facing_select_box,
                "tenure": tenure_select_box,
                "has_gym": has_gym_checkbox,
                "has_pool": has_pool_checkbox,
                "is_whole_unit": is_whole_unit_checkbox,
            }
            st.rerun()


def plot_shap_summary_and_waterfall(explanation: shap.Explanation):
    st.subheader("Plots")

    # Create two columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("SHAP Summary Plot")
        # Create a new figure for the summary plot
        fig_summary, ax_summary = plt.subplots(figsize=(10, 8))

        # Plot the SHAP summary plot
        shap.summary_plot(explanation.values, explanation.data,
                          feature_names=explanation.feature_names,
                          plot_type="bar", show=False)

        plt.tight_layout()
        st.pyplot(fig_summary, bbox_inches='tight')

    with col2:
        st.subheader("SHAP Waterfall Plot")
        # Create a new figure for the waterfall plot
        fig_waterfall, ax_waterfall = plt.subplots(figsize=(10, 8))

        # Plot the SHAP waterfall plot
        shap.plots.waterfall(explanation[0], show=False)

        plt.tight_layout()
        st.pyplot(fig_waterfall, bbox_inches='tight')


with st.spinner("Loading page data..."):
    def init_message_generator(start):
        yield "Initialized! Took "
        yield f"{time.time() - start:.2f}"
        yield " seconds."

    start = time.time()
    st.session_state["db"] = init_motherduckdb_conn()
    fetch_listings_df()

    version = fetch_latest_version()

    st.session_state["column_transformer"] = get_column_transformer(version["tags"]["column_transformer_source"])
    st.session_state["model"] = get_model(version["tags"]["model_name"], version["source"])
    geocoder = Nominatim(user_agent="sg_rental_price_dashboard")

    generator = init_message_generator(start)
    st.write(''.join(generator))

    get_form_data()


if "form_data" in st.session_state:
    with st.status("Predicting rental price...", expanded=True) as status:
        result = st.session_state.pop("form_data")
        prediction, explanation_obj, desc, user_coords = process_form_data(
            st.session_state["model"], st.session_state["column_transformer"], result)
        status.update(label="Prediction Completed",
                      state="complete", expanded=False)

    st.success(f"Predicted Rental Price: SGD **{prediction:.2f}/month**")
    st.toast("Prediction Completed", icon="🎉")

    # Plot SHAP values
    plot_shap_summary_and_waterfall(explanation_obj)
    st.write(desc)

    # Load local listings data
    # listings_df = load_local_data("static/training_data_v3_cleaned.csv")

    # Fetch and plot listings within a radius
    radius_km = 5
    nearby_listings = fetch_listings_within_radius(
        user_coords[0], user_coords[1], radius_km, listings_df)
    plot_listings_on_map(nearby_listings, user_coords)

    # st.write(user_coords)
    # st.write(nearby_listings[["property_name", "district", "distance"]])
