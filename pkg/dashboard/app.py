import os
import io
import json
import time
import shap
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from datetime import datetime
from geopy.distance import geodesic
from components.constants import *
from utils.outliers import remove_outliers

st.set_page_config(
    "RentInSG",
    page_icon="🏠",
    layout="wide",
)

st.title("Singapore Rental Price Prediction Home")
st.text("This page predicts the rental price of a property in Singapore 💲")


BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000/")
TWO_HOURS_IN_SECONDS = 2 * 60 * 60
listings_df = pd.DataFrame()
explanation_obj = None


@st.fragment(run_every="2h")
def fetch_listings_df():
    global listings_df

    if "listings_df" in st.session_state and \
            "last_updated" in st.session_state and \
            (time.time() - st.session_state["last_updated"]) < TWO_HOURS_IN_SECONDS:
        listings_df = st.session_state["listings_df"]
        return

    # URL of FastAPI endpoint
    url = f"{BACKEND_URL}/data/"

    try:
        # Make a GET request to the endpoint
        response = requests.get(url, params={"table": "property_listing"})

        # Check if the request was successful
        if response.status_code == 200:
            # Read the CSV content into a DataFrame
            csv_content = io.StringIO(response.text)
            listings_df = pd.read_csv(csv_content)

            listings_df = remove_outliers(listings_df, "price")

            st.session_state["listings_df"] = listings_df
            st.session_state["last_updated"] = time.time()
            st.toast("Data fetched successfully!", icon='🚀')
        else:
            st.toast(f"Failed to fetch data. Status code: {response.status_code}", icon='😭')
            listings_df = None
    except Exception as e:
        print(e)
        st.toast(f"An error occurred while fetching data: {str(e)}", icon='😭')
        listings_df = None


@st.cache_data
def fetch_listings_within_radius(lat, lon, radius_km, listings_df):
    listings_df["distance"] = listings_df.apply(lambda row: geodesic(
        (lat, lon), (row["latitude"], row["longitude"])).km, axis=1)
    nearby_listings = listings_df[listings_df["distance"] <= radius_km]
    return nearby_listings


@st.fragment
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
        custom_data=["property_name", "price", "source", "url"],
        zoom=12,
        height=600,
        center={"lat": user_location[0], "lon": user_location[1]},
    )

    user_location_df = pd.DataFrame(
        [{"latitude": user_location[0],
          "longitude": user_location[1],
          "type": "User Location", "marker_size": 3}])
    fig.add_trace(px.scatter_mapbox(
        user_location_df,
        lat="latitude",
        lon="longitude",
        hover_name="type",
        color_discrete_sequence=["red"],
        size="marker_size"
    ).data[0])

    fig.update_layout(mapbox_style="open-street-map")

    def on_select(event):
        if len(event["selection"]["points"]) == 0 or "customdata" not in event["selection"]["points"][0]:
            return
        prop_name, price, source, url, _ = event["selection"]["points"][0]["customdata"]

        with st.container(border=True):
            st.subheader("Listing Details 🧾")
            st.write(f"Listing Title: {prop_name}")
            st.write(f"Price: ${price}/month")
            st.write(f"Source: {source}")
            st.write(f"URL: {url}")

    st.subheader("Nearby Listings")
    event = st.plotly_chart(fig, on_select='rerun')
    on_select(event)


@st.fragment
def get_form_data():
    st.info(
        """
        Fill in the form below with specific parameteres to predict monthly rental price of a property in Singapore.
        __(\\* indicates required fields)__
    """
    )
    formatted_time = datetime.fromtimestamp(st.session_state['last_updated']).strftime('%I:%M%p %A %dth %b %Y')
    st.write(f"Last updated: {formatted_time}")

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

        with st.expander("Expand for optional fields"):
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


def predict_and_explain(form_data):
    response = requests.post(f"{BACKEND_URL}/inference", json=form_data)
    if response.status_code == 200:
        return response.json()

    st.error(f"Error: {response.status_code} - {response.text}")
    return None


def predict_and_explain_stream(form_data: dict):
    try:
        response = requests.post(f"{BACKEND_URL}/inference/stream", json=form_data, stream=True)

        if response.status_code == 200:
            result = {}
            buffer = ""
            for chunk in response.iter_content(chunk_size=1, decode_unicode=True):
                if not chunk:
                    continue

                buffer += chunk
                try:
                    update = json.loads(buffer)
                    # Reset buffer after successful parse
                    buffer = ""

                    if 'result' in update:
                        result = update['result']

                    if 'progress' in update:
                        # Update the progress bar
                        st.text(update['message'])
                        st.progress(update['progress'] / 100)
                except json.JSONDecodeError:
                    # If we can't parse the JSON yet, continue to the next chunk
                    continue
            return result
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            return None
    except requests.RequestException as e:
        st.error(f"Request failed: {str(e)}")
        return None


def plot_shap_summary_and_waterfall(explanation: shap.Explanation):
    st.subheader("Plots")

    # Create two columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("SHAP Summary Plot")
        # Calculate mean absolute SHAP values for each feature
        mean_abs_shap_values = np.abs(explanation.values).mean(axis=0)

        # Get the indices of the top 10 features with the highest mean absolute SHAP values
        top_10_indices = np.argsort(mean_abs_shap_values)[-10:][::-1]

        # Filter the data to only include the top 10 features
        top_10_values = explanation.values[:, top_10_indices]
        top_10_data = explanation.data[:, top_10_indices]
        top_10_feature_names = [explanation.feature_names[i] for i in top_10_indices]

        # Create a new figure for the summary plot
        fig_summary, _ = plt.subplots(figsize=(10, 6))

        # Plot the SHAP summary plot for the top 10 features
        shap.summary_plot(
            top_10_values,
            top_10_data,
            feature_names=top_10_feature_names,
            plot_type="bar",
        )

        plt.tight_layout()
        st.pyplot(fig_summary, bbox_inches='tight')

    with col2:
        st.subheader("SHAP Waterfall Plot")
        # Create a new figure for the waterfall plot
        fig_waterfall, _ = plt.subplots(figsize=(10, 6))

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
    fetch_listings_df()
    st.toast("".join(init_message_generator(start)), icon="⚡")

    if "listings_df" not in st.session_state:
        st.error("Could not fetch data, please try again later...")
    else:
        get_form_data()


if "form_data" in st.session_state:
    with st.status("Predicting rental price...", expanded=True) as status:
        response = predict_and_explain_stream(form_data=st.session_state.pop("form_data"))
        # print(json.dumps(response, indent=2))
        status.update(label="Prediction Completed",
                      state="complete", expanded=False)

    st.success(f"Predicted Rental Price: SGD **{response['prediction']:.2f}/month**")
    st.toast("Prediction Completed", icon="🎉")

    explanation_obj = shap.Explanation(
        values=np.array(response['shap_values']),
        base_values=np.array(response['shap_base_values']),
        data=np.array(response['shap_data']),
        feature_names=response['shap_feature_names']
    )

    # Plot SHAP values
    plot_shap_summary_and_waterfall(explanation_obj)
    st.write(response['description'])

    # Fetch and plot listings within a radius
    radius_km = 2.5
    user_coords = response['coordinates']
    nearby_listings = fetch_listings_within_radius(
        user_coords[0], user_coords[1], radius_km, listings_df)
    plot_listings_on_map(nearby_listings, user_coords)

    # st.write(user_coords)
    # st.write(nearby_listings[["property_name", "district", "distance"]])
