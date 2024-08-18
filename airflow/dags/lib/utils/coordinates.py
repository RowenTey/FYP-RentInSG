import logging
import numpy as np
import pandas as pd
from typing import Tuple


def fetch_coordinates(location_name: str) -> Tuple[str, Tuple[float, float]]:
    """
    Fetches the coordinates of the specified location using the OneMap API.

    Args:
        location_name (str): The name of the location to fetch coordinates for.

    Returns:
        Tuple[str, Tuple[float, float]]: A tuple containing the location name and a tuple of latitude and longitude.

    If coordinates are found for the location, returns the location name along with the latitude and longitude.
    If no coordinates are found, returns the location name with NaN values for latitude and longitude.
    """
    import requests

    url = "https://www.onemap.gov.sg/api/common/elastic/search"
    params = {
        "searchVal": location_name,
        "returnGeom": "Y",
        "getAddrDetails": "Y",
        "pageNum": 1,
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if data["found"] > 0:
            return location_name, (
                data["results"][0]["LATITUDE"],
                data["results"][0]["LONGITUDE"],
            )

    logging.info(f"No results found for location: {location_name}")
    return location_name, (np.nan, np.nan)


def find_nearest(
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        target_landmark: str,
        distance_to_target_landmark: str,
        is_inference: bool = False) -> pd.DataFrame:
    """
    A function that finds the nearest locations from the 2nd table to the 1st address based on geodesic distance calculations.
    Taken from https://medium.com/@michael.wy.ong/web-scrape-geospatial-data-analyse-singapores-property-price-part-i-276caba320b

    Parameters:
        df1: pd.DataFrame - The first DataFrame containing the addresses.
        df2: pd.DataFrame - The second DataFrame containing location coordinates.
        target_landmark: str - The target landmark column name to store the nearest landmark.
        distance_to_target_landmark: str - The target column name to store the distance to the landmark.
        is_inference: bool - Flag indicating if it's an inference operation.

    Returns:
        pd.DataFrame - The updated DataFrame with information on the nearest landmarks and distances.
    """
    from geopy.distance import geodesic

    if not is_inference:
        assert "building_name" in df1.columns, "building_name column not found in df1"

    building_names = df1["building_name"].unique()
    for building_name in building_names:
        try:
            prop_loc = (
                float(df1.loc[df1["building_name"] == building_name, "latitude"].unique()[0]),
                float(df1.loc[df1["building_name"] == building_name, "longitude"].unique()[0]),
            )
        except Exception as e:
            print(e)
            print(f"Skipping {building_name} because it has no coordinates")
            continue

        landmark_info = ["", "", float("inf")]
        for idx, eachloc in enumerate(df2.iloc[:, 0]):
            landmark_loc = (float(df2.iloc[idx, 1]), float(df2.iloc[idx, 2]))
            distance = geodesic(prop_loc, landmark_loc).m  # convert to m

            if distance < landmark_info[2]:
                landmark_info[0] = df2.iloc[idx, 0]
                landmark_info[1] = eachloc
                landmark_info[2] = round(distance, 3)

        df1.loc[df1["building_name"] == building_name, target_landmark] = landmark_info[0]
        df1.loc[df1["building_name"] == building_name, distance_to_target_landmark] = landmark_info[2]

    return df1
