import time

import pandas as pd
import streamlit as st
from pygwalker.api.streamlit import StreamlitRenderer

st.set_page_config(
    page_title="Rental Price Prediction",
    page_icon="🏠",
    layout="wide",
)

st.title("🔍 Dataset Exploration")
st.text("This page allows you to explore the dataset used to train the model.")


@st.cache_resource
def get_pyg_renderer() -> "StreamlitRenderer":
    df = pd.read_csv("static/training_data_v3_cleaned.csv")
    return StreamlitRenderer(
        df,
        spec="static/gw_config.json",
        spec_io_mode="rw")


with st.spinner("Loading data..."):
    time.sleep(1)
    renderer = get_pyg_renderer()
    renderer.explorer(default_tab="data")
