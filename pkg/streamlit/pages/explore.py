import time

import pandas as pd
import streamlit as st
from pygwalker.api.streamlit import StreamlitRenderer

st.set_page_config(
    "Singapore Rental Price Analysis | Explore",
    page_icon="🏠",
    layout="wide",
)

st.title("🔍 Dataset Exploration")
st.text("This page allows you to explore the dataset used to train the model.")


@st.cache_resource
def get_pyg_renderer() -> "StreamlitRenderer":
    df = st.session_state["listings_df"]
    df = df[df['price'] >= 100]
    return StreamlitRenderer(
        df,
        spec="static/gw_config.json",
        spec_io_mode="rw",
        kernel_computation=True
    )


with st.spinner("Loading data..."):
    time.sleep(1)
    renderer = get_pyg_renderer()
    renderer.explorer(default_tab="data")
