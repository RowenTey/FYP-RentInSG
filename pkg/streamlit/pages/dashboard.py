import streamlit as st
import plotly.express as px
import pandas as pd
import plotly.figure_factory as ff
from utils.outliers import remove_outliers

st.set_page_config(
    "Singapore Rental Price Analysis | Dashboard",
    page_icon="🏠",
    layout="wide",
)

st.title("📈 Singapore Rental Price Analysis Dashboard")
st.text("This dashboard provides a comprehensive overview of the Singapore rental market based on the dataset used to train the model.")


@st.cache_data
def load_data():
    df = (st.session_state["listings_df"]).copy()
    df = df[df['price'] >= 100]
    df['scraped_on'] = pd.to_datetime(df['scraped_on'])
    df = remove_outliers(df, 'price')
    df = remove_outliers(df, 'dimensions')
    return df


df = load_data()

# Key Statistics
st.subheader("Key Statistics")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total Properties", len(df))
with col2:
    st.metric("Average Price", f"${df['price'].mean():,.2f}")
with col3:
    st.metric("Median Price", f"${df['price'].median():,.2f}")
with col4:
    st.metric("Most Common Property Type", df['property_type'].mode().values[0])
with col5:
    # Price per Square Foot
    avg_price_per_sqft = df['price_per_sqft'].mean()
    st.metric("Average Price per Square Foot", f"${avg_price_per_sqft:,.2f} per sqft")

# Price Range (Min and Max)
min_price = df['price'].min()
max_price = df['price'].max()
col1, col2 = st.columns(2)
with col1:
    st.metric("Lowest Price", f"${min_price:,.2f}")
with col2:
    st.metric("Highest Price", f"${max_price:,.2f}")

# Property Distribution by District
st.subheader("Property Distribution by District")
district_counts = df['district'].value_counts().reset_index()
district_counts.columns = ['district', 'count']
fig_distribution = px.pie(district_counts, values='count', names='district')
fig_distribution.update_traces(textposition='inside', textinfo='percent+label')
st.plotly_chart(fig_distribution, use_container_width=True)


# Average Price per District
st.subheader("Average Price per District")
avg_price = df.groupby('district')['price'].mean().reset_index()
fig_avg_price = px.bar(avg_price, x='district', y='price')
fig_avg_price.update_layout(xaxis_title='District', yaxis_title='Price')
st.plotly_chart(fig_avg_price, use_container_width=True)

# Average Price by Bedroom Count
st.subheader("Average Price by Bedroom Count")
bedroom_avg = df.groupby('bedroom')['price'].mean().reset_index()
fig_bedroom_price = px.bar(bedroom_avg, x='bedroom', y='price')
fig_bedroom_price.update_layout(xaxis_title='Number of Bedrooms', yaxis_title='Average Price')
st.plotly_chart(fig_bedroom_price, use_container_width=True)

# Price Distribution (Histogram)
st.subheader("Price Distribution")
fig_price_distribution = px.histogram(df, x='price', nbins=50)
fig_price_distribution.update_layout(xaxis_title='Price', yaxis_title='Number of Properties')
st.plotly_chart(fig_price_distribution, use_container_width=True)

# Price Trends Over Time
st.subheader("Price Trends Over Time")
df['scraped_on'] = pd.to_datetime(df['scraped_on'])
price_trends = df.groupby('scraped_on')['price'].mean().reset_index()
fig_price_trends = px.line(price_trends, x='scraped_on', y='price', title='Average Price Over Time')
fig_price_trends.update_layout(xaxis_title='Date', yaxis_title='Average Price')
st.plotly_chart(fig_price_trends, use_container_width=True)


# Price Correlation Heatmap
st.subheader("Price Correlation Heatmap")
corr_matrix = df[['price', 'bedroom', 'dimensions']].corr()
fig_corr = ff.create_annotated_heatmap(z=corr_matrix.to_numpy(),
                                       x=corr_matrix.columns.tolist(),
                                       y=corr_matrix.columns.tolist(),
                                       colorscale='Viridis')
st.plotly_chart(fig_corr, use_container_width=True)

# Scatter Plot of Price vs. Property Size
st.subheader("Price vs. Property Size")
fig_price_size = px.scatter(df, x='dimensions', y='price', trendline='ols', title='Price vs. Property Size')
fig_price_size.update_layout(xaxis_title='Size (sqft)', yaxis_title='Price')
st.plotly_chart(fig_price_size, use_container_width=True)

# Scatter Map of Property Locations
st.subheader("Scatter Map: Rental Property Locations with Price")

fig_scatter_map = px.scatter_mapbox(
    df,
    lat='latitude',    # Assuming you have latitude and longitude columns
    lon='longitude',
    color='price',
    size='price',      # Optionally, size can represent price as well
    hover_name='property_type',
    hover_data=['price', 'bedroom', 'dimensions'],
    color_continuous_scale=px.colors.cyclical.IceFire,
    mapbox_style="open-street-map",
    title="Rental Property Locations with Price",
    zoom=10
)
st.plotly_chart(fig_scatter_map, use_container_width=True)
