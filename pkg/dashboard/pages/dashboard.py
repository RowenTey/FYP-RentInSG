import streamlit as st
import plotly.express as px
import pandas as pd
import plotly.figure_factory as ff
from utils.outliers import remove_outliers

st.set_page_config(
    "RentInSG | Dashboard",
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
    df = remove_outliers(df, 'price_per_sqft')
    df = remove_outliers(df, 'dimensions')
    return df


df = load_data()

# Sidebar Filters
st.sidebar.header("Filter Properties")

# Price Filter
price_min, price_max = st.sidebar.slider("Price Range ($)", min_value=int(df['price'].min()), max_value=int(
    df['price'].max()), value=(int(df['price'].min()), int(df['price'].max())))
df = df[(df['price'] >= price_min) & (df['price'] <= price_max)]

# Bedroom Filter
bedroom_options = sorted(df['bedroom'].unique())
bedrooms = st.sidebar.multiselect("Number of Bedrooms", options=bedroom_options, default=bedroom_options)
df = df[df['bedroom'].isin(bedrooms)]

# District Filter
district_options = sorted(df['district'].unique())
districts = st.sidebar.multiselect("District", options=district_options, default=district_options)
df = df[df['district'].isin(districts)]

# Property Type Filter
property_type_options = sorted(df['property_type'].unique())
property_types = st.sidebar.multiselect("Property Type", options=property_type_options, default=property_type_options)
df = df[df['property_type'].isin(property_types)]

if len(df) == 0:
    st.error("No properties found with the selected filters. Please adjust the filters.")
    st.stop()

# Key Statistics
st.subheader("Key Statistics")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Properties", len(df))
with col2:
    st.metric("Most Common Property Type", df['property_type'].mode().values[0])
with col3:
    st.metric("Median Price", f"${df['price'].median():,.2f}")
with col4:
    st.metric("Average Price", f"${df['price'].mean():,.2f}")

# Price Range (Min and Max)
min_price = df['price'].min()
max_price = df['price'].max()
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Lowest Price", f"${min_price:,.2f}")
with col2:
    st.metric("Highest Price", f"${max_price:,.2f}")
with col3:
    avg_price_per_sqft = df['price_per_sqft'].mean()
    st.metric("Average Price/Sqft", f"${avg_price_per_sqft:,.2f} per sqft")


col1, col2 = st.columns(2)
with col1:
    # Property Distribution by District
    st.subheader("Property Distribution by District")
    district_counts = df['district'].value_counts().reset_index()
    district_counts.columns = ['district', 'count']
    fig_distribution = px.pie(district_counts, values='count', names='district')
    fig_distribution.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_distribution, use_container_width=True)
with col2:
    # Price Distribution (Histogram)
    st.subheader("Price Distribution")
    fig_price_distribution = px.histogram(df, x='price', nbins=50)
    fig_price_distribution.update_layout(xaxis_title='Price', yaxis_title='Number of Properties')
    st.plotly_chart(fig_price_distribution, use_container_width=True)


col1, col2 = st.columns(2)
with col1:
    # Average Price by Bedroom Count
    st.subheader("Average Price by Bedroom Count")
    bedroom_avg = df.groupby('bedroom')['price'].mean().reset_index()
    fig_bedroom_price = px.bar(bedroom_avg, x='bedroom', y='price', color='price', color_continuous_scale='sunset')
    fig_bedroom_price.update_layout(xaxis_title='Number of Bedrooms', yaxis_title='Average Price')
    st.plotly_chart(fig_bedroom_price, use_container_width=True)
with col2:
    # Average Price/Sqft per District
    st.subheader("Average Price/Sqft by District")
    avg_price = df.groupby('district')['price_per_sqft'].mean().reset_index()
    avg_price_sorted = avg_price.sort_values(by='price_per_sqft', ascending=False)
    fig_avg_price = px.bar(
        avg_price_sorted,
        x='district',
        y='price_per_sqft',
        color='price_per_sqft',
        color_continuous_scale='sunset'
    )
    fig_avg_price.update_layout(xaxis_title='District', yaxis_title='Price')
    st.plotly_chart(fig_avg_price, use_container_width=True)


# Price Trends Over Time
st.subheader("Average Price Over Time")
df['scraped_on'] = pd.to_datetime(df['scraped_on'])
price_trends = df.groupby('scraped_on')['price'].mean().reset_index()
fig_price_trends = px.line(price_trends, x='scraped_on', y='price')
fig_price_trends.update_layout(xaxis_title='Date', yaxis_title='Average Price')
st.plotly_chart(fig_price_trends, use_container_width=True)


col1, col2 = st.columns(2)
with col1:
    # Price Correlation Heatmap
    st.subheader("Price Correlation Heatmap")
    corr_matrix = df[['price', 'bedroom', 'dimensions']].corr()
    fig_corr = ff.create_annotated_heatmap(z=corr_matrix.to_numpy(),
                                           x=corr_matrix.columns.tolist(),
                                           y=corr_matrix.columns.tolist(),
                                           colorscale='sunset')
    st.plotly_chart(fig_corr, use_container_width=True)
with col2:
    # Scatter Plot of Price vs. Property Size
    st.subheader("Price vs. Property Size")
    fig_price_size = px.scatter(df, x='dimensions', y='price', trendline='ols',
                                color='price', color_continuous_scale='sunset')
    fig_price_size.update_layout(xaxis_title='Size (sqft)', yaxis_title='Price')
    st.plotly_chart(fig_price_size, use_container_width=True)

# Scatter Map of Property Locations
st.subheader("Rental Property Locations with Price")
fig_scatter_map = px.scatter_mapbox(
    df,
    lat='latitude',
    lon='longitude',
    color='price',
    size='price',
    hover_name='property_type',
    hover_data=['price', 'price_per_sqft', 'bedroom', 'dimensions'],
    color_continuous_scale='sunset',
    mapbox_style="open-street-map",
    zoom=11,
    height=900
)
st.plotly_chart(fig_scatter_map, use_container_width=True)
