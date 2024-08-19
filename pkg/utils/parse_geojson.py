import json

import geopandas as gpd
import pandas as pd
from bs4 import BeautifulSoup
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from utils.location_constants import PLAN_AREA_MAPPING


def parse_plan_area_geojson(file_path):
    # for parsing
    # SingaporeResidentsbySubzoneAgeGroupandSexJun2018Gender.geojson

    # Load the GeoJSON file
    with open(file_path, "r") as f:
        data = json.load(f)

    # Initialize lists to store parsed data
    geometry_data = []

    # Iterate through features in the GeoJSON data
    for feature in data["features"]:
        # Parse properties
        properties = feature["properties"]
        description_html = properties["Description"]

        # Parse HTML description to extract SUBZONE_N
        soup = BeautifulSoup(description_html, "html.parser")
        subzone = soup.find("th", string="SUBZONE_N").find_next("td").text
        plan_area = soup.find("th", string="PLN_AREA_N").find_next("td").text

        # Parse geometry
        geometry = feature["geometry"]
        coordinates = geometry["coordinates"][0]

        # further flatten if necessary
        if isinstance(coordinates[0][0], list):
            coordinates = [c for sublist in coordinates for c in sublist]

        # Remove z-coordinate (0.0) if present
        coordinates = [(x, y) for x, y, *_ in coordinates]

        # Create Polygon
        polygon = Polygon(coordinates)

        # Add parsed properties to the list
        geo_data = {
            "subzone": subzone,
            "plan_area": plan_area,
            "polygon": polygon}
        geometry_data.append(geo_data)

    # Create pandas DataFrames
    geometry_df = gpd.GeoDataFrame(geometry_data, geometry="polygon")
    geometry_df.crs = "EPSG:4326"

    return geometry_df


def parse_hawker_centre_geojson(file_path):
    # for parsing HawkerCentresGEOJSON.geojson

    # Load the GeoJSON file
    with open(file_path, "r") as f:
        data = json.load(f)

    # Initialize lists to store parsed data
    geometry_data = []

    # Iterate through features in the GeoJSON data
    for feature in data["features"]:
        # Parse properties
        properties = feature["properties"]
        description_html = properties["Description"]

        # Parse HTML description to extract SUBZONE_N
        soup = BeautifulSoup(description_html, "html.parser")
        name = soup.find("th", string="NAME").find_next("td").text
        building_name = soup.find(
            "th", string="ADDRESSBUILDINGNAME").find_next("td").text
        street_name = soup.find(
            "th", string="ADDRESSSTREETNAME").find_next("td").text
        postal_code = soup.find(
            "th", string="ADDRESSPOSTALCODE").find_next("td").text

        # Parse geometry
        geometry = feature["geometry"]
        coordinates = geometry["coordinates"]

        # Remove z-coordinate (0.0) if present
        x, y, _ = coordinates

        # Add parsed properties to the list
        geo_data = {
            "name": name,
            "building_name": building_name,
            "street_name": street_name,
            "postal_code": postal_code,
            "longitude": x,
            "latitude": y,
        }
        geometry_data.append(geo_data)

    df = pd.DataFrame(geometry_data)
    return df


def parse_supermarket_geojson(file_path):
    # for parsing SupermarketsGEOJSON.geojson

    # Load the GeoJSON file
    with open(file_path, "r") as f:
        data = json.load(f)

    # Initialize lists to store parsed data
    geometry_data = []

    # Iterate through features in the GeoJSON data
    for feature in data["features"]:
        # Parse properties
        properties = feature["properties"]
        description_html = properties["Description"]

        # Parse HTML description to extract SUBZONE_N
        soup = BeautifulSoup(description_html, "html.parser")
        name = soup.find("th", string="LIC_NAME").find_next("td").text
        street_name = soup.find("th", string="STR_NAME").find_next("td").text
        postal_code = soup.find("th", string="POSTCODE").find_next("td").text

        # Parse geometry
        geometry = feature["geometry"]
        coordinates = geometry["coordinates"]

        # Remove z-coordinate (0.0) if present
        x, y, _ = coordinates

        # Add parsed properties to the list
        geo_data = {
            "name": name,
            "street_name": street_name,
            "postal_code": postal_code,
            "longitude": x,
            "latitude": y,
        }
        geometry_data.append(geo_data)

    df = pd.DataFrame(geometry_data)
    return df


def process_plan_area_geometries(df, group_column, geometry_column):
    # Check if polygons are valid
    df["is_valid"] = df[geometry_column].apply(lambda x: x.is_valid)

    # Attempt to fix invalid polygons
    df[geometry_column] = df[geometry_column].apply(
        lambda x: x.buffer(0) if not x.is_valid else x)

    # Perform unary_union
    return df.groupby(group_column)[geometry_column].apply(
        unary_union).reset_index()


def get_district(lat, long, gdf: gpd.GeoDataFrame):
    # swap long and lat
    point = Point(long, lat)
    plan_area = gdf.loc[gdf.contains(point), "plan_area"].squeeze()

    if isinstance(plan_area, pd.Series):
        if plan_area.empty:
            return ""
        plan_area = plan_area.mode()

    return PLAN_AREA_MAPPING[plan_area]


if __name__ == "__main__":
    # Example to load from file
    # file_path = r'C:\Users\kaise\OneDrive\Desktop\Codebase\FYP\data\SingaporeResidentsbySubzoneAgeGroupandSexJun2018Gender.geojson' # noqa: E501
    # geometry_df = parse_geojson(file_path)
    # geometry_df = process_geometries(geometry_df, 'plan_area', 'polygon') # noqa: E501

    # Example to load from DB
    # logging.basicConfig(level=logging.INFO)
    # db = connect_to_motherduckdb()
    # df = db.query_df("SELECT * FROM plan_area_mapping")
    # print(df)

    # df['polygon'] = df['polygon'].apply(wkt.loads)
    # gdf = gpd.GeoDataFrame(df, geometry='polygon')
    # print(get_district(1.27708340707631, 103.849181343548, gdf))

    # Hawker Centre
    # geometry_df = parse_hawker_centre_geojson(
    #     r'C:\Users\kaise\OneDrive\Desktop\Codebase\FYP\data\HawkerCentresGEOJSON.geojson')
    # geometry_df["hawker_id"] = range(1, len(geometry_df) + 1)

    # geometry_df = geometry_df[["hawker_id", "name", "building_name",
    #                            "street_name", "postal_code", "longitude", "latitude"]]

    # print(geometry_df)
    # print(geometry_df.columns)
    # print(geometry_df.dtypes)

    # from motherduckdb_connector import connect_to_motherduckdb
    # conn = connect_to_motherduckdb()
    # conn.insert_df("hawker_centre_info", geometry_df)

    geometry_df = parse_supermarket_geojson(
        r"C:\Users\kaise\OneDrive\Desktop\Codebase\FYP\data\SupermarketsGEOJSON.geojson"
    )
    geometry_df["supermarket_id"] = range(1, len(geometry_df) + 1)

    geometry_df = geometry_df[
        [
            "supermarket_id",
            "name",
            "street_name",
            "postal_code",
            "longitude",
            "latitude",
        ]
    ]

    print(geometry_df)
    print(geometry_df.columns)
    print(geometry_df.dtypes)

    from motherduckdb_connector import connect_to_motherduckdb

    conn = connect_to_motherduckdb()
    conn.insert_df("supermarket_info", geometry_df)
    conn.close()
    pass
