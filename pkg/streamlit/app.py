import streamlit as st
import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import pickle

from components.constants import *
from components.motherduckdb_connector import connect_to_motherduckdb
from components.coordinates import fetch_coordinates
from components.find_closest import find_nearest_single

st.set_page_config(
    "Singapore Rental Price Dashboard",
    page_icon="🏠",
)

st.title("Singapore Rental Price Dashboard")
st.text("This dashboard predicts the rental price of a property in Singapore 📈")
st.info("""
    Fill in the form below with specific parameteres to predict monthly rental price of a property in Singapore.
    __(\* indicates required fields)__
""")

# Load column transformer from pickle file to encode categorical columns and scale numerical columns
with open("static/column_transformer.pkl", "rb") as file:
    column_transformer = pickle.load(file)

# Load catboost model from pickle file
with open("static/catboost.pkl", "rb") as file:
    model = pickle.load(file)

states = {
    "USA": ["", "California", "Washington", "New Jersey"],
    "Canada": ["", "Quebec", "Ontario", "British Columbia"],
    "Germany": ["", "Brandenberg", "Hesse", "Bavaria"]
}


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

    form_data["latitude"] = None
    form_data["longitude"] = None
    return form_data


@st.cache_data
def fetch_info(_db, query, target_cols):
    df2 = _db.query_df(query)
    return df2[target_cols]


def add_distance_info(validated_form_data, fill_default=False) -> dict:
    enrichment = {
        "distance_to_mrt_in_m": ("SELECT * FROM mrt_info", ['station_name', 'latitude', 'longitude']),
        "distance_to_mall_in_m": ("SELECT * FROM mall_info", ['name', 'latitude', 'longitude']),
        "distance_to_sch_in_m": ("SELECT * FROM primary_school_info", ['name', 'latitude', 'longitude']),
        "distance_to_hawker_in_m": ("SELECT * FROM hawker_centre_info", ['name', 'latitude', 'longitude']),
        "distance_to_supermarket_in_m": ("SELECT * FROM supermarket_info", ['name', 'latitude', 'longitude']),
    }
    # set the key in form data so default values are filled in later
    if fill_default:
        for key in enrichment.keys():
            validated_form_data[key] = None
        return validated_form_data

    db = connect_to_motherduckdb()
    _, (validated_form_data["latitude"], validated_form_data["longitude"]) =  \
        fetch_coordinates(validated_form_data["address"])

    for key, val in enrichment.items():
        df2 = fetch_info(db, val[0], val[1])
        validated_form_data[key] = find_nearest_single(
            {"latitude": validated_form_data["latitude"], "longitude": validated_form_data["longitude"]}, df2)

    return validated_form_data


def process_form_data(
    model,
    column_transformer,
    form_data
) -> float:
    # st.json(form_data)

    if form_data["address"] is not None and form_data["address"] != "":
        st.info("Fetching distance info...")
        form_data = add_distance_info(form_data)
    else:
        st.info("Address is empty, using default distance values...")
        form_data = add_distance_info(form_data, True)

    validated_form_data = transform_form_data(form_data)

    input_df = pd.DataFrame(validated_form_data, index=[0])
    input_df = input_df.drop(
        columns=["address", "latitude", "longitude"], axis=1)
    print(input_df)

    # Apply the same transformations to the form data as done during training
    transformed_data = column_transformer.transform(input_df)

    # Make predictions using the model
    prediction, *_ = model.predict(transformed_data)

    return prediction


@st.experimental_fragment
def get_form_data():
    with st.container(border=True):
        st.subheader("Parameters Selection")

        form_col1, form_col2 = st.columns(2)

        bedroom_select_box = form_col1.number_input(
            label="Number of bedrooms?*",
            placeholder="Enter the number of bedrooms",
            min_value=1,
            value=None
        )

        bathroom_select_box = form_col2.number_input(
            label="Number of bathrooms?*",
            placeholder="Enter the number of bathrooms",
            min_value=1,
            value=None
        )

        dimensions_input_box = st.number_input(
            label="Dimensions (sqft)?*",
            placeholder="Enter the dimensions",
            min_value=0,
            value=None
        )

        district_id_select_box = st.selectbox(
            label="District ID?*",
            options=list(DISTRICTS.keys()),
            placeholder="Select a district ID",
            help="Select the district ID of the property",
            index=None
        )

        property_type_select_box = st.selectbox(
            label="Property Type?*",
            options=PROPERTY_TYPES,
            placeholder="Select a property type",
            help="Select the property type of the property",
            index=None
        )

        address_input_box = st.text_input(
            label="Address?",
            placeholder="Enter the address",
            value=None
        )

        built_year_input_box = st.number_input(
            label="Built Year?",
            placeholder="Enter the built year",
            min_value=1950,
            max_value=2024,
            value=None
        )

        furnishing_select_box = st.selectbox(
            label="Furnishing?",
            options=FURNISHING,
            placeholder="Select a furnishing type",
            help="Select the furnishing type of the property",
            index=None
        )

        facing_select_box = st.selectbox(
            label="Facing?",
            options=FACING,
            placeholder="Select a facing direction",
            help="Select the facing direction of the property",
            index=None
        )

        floor_level_select_box = st.selectbox(
            label="Floor Level?",
            options=FLOOR_LEVEL,
            placeholder="Select a floor level",
            help="Select the floor level of the property",
            index=None
        )

        tenure_select_box = st.selectbox(
            label="Tenure?",
            options=TENURE,
            placeholder="Select a tenure type",
            help="Select the tenure type of the property",
            index=None
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

        can_submit = bedroom_select_box and \
            bathroom_select_box and \
            dimensions_input_box and \
            district_id_select_box and \
            property_type_select_box
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


get_form_data()

if "form_data" in st.session_state:
    with st.status("Predicting rental price...", expanded=True) as status:
        result = st.session_state.pop("form_data")
        prediction = process_form_data(model, column_transformer, result)
        status.update(label="Prediction Completed",
                      state="complete", expanded=False)
    # with st.spinner("Predicting Rental Price..."):
    #     time.sleep(1.5)

    st.success(f"Predicted Rental Price: SGD **{prediction:.2f}/month**")
    st.toast("Prediction Completed", icon="🎉")
