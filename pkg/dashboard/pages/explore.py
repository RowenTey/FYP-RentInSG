import time

import streamlit as st
from utils.outliers import remove_outliers
from pygwalker.api.streamlit import StreamlitRenderer

st.set_page_config(
    "RentInSG | Explore",
    page_icon="🏠",
    layout="wide",
)

st.title("🔍 Dataset Exploration")
st.text("This page allows you to explore the dataset used to train the model.")


@st.cache_resource
def get_pyg_renderer() -> "StreamlitRenderer":
    df = st.session_state["listings_df"]
    df = df[df['price'] >= 100]
    df = df[df['property_type'].isna() == False]
    df = remove_outliers(df, 'price')
    df = remove_outliers(df, 'price_per_sqft')
    df = remove_outliers(df, 'dimensions')
    return StreamlitRenderer(
        df,
        spec="static/gw_config.json",
        spec_io_mode="rw",
        kernel_computation=True
    )


with st.spinner("Loading data..."):
    renderer = get_pyg_renderer()
    renderer.explorer(default_tab="data")
