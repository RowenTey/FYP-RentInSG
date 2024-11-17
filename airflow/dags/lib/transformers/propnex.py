import re
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from duckdb import DuckDBPyConnection
from typing import Tuple, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from lib.constants.db_constants import *
from lib.constants.location_constants import *
from lib.utils.coordinates import fetch_coordinates, find_nearest
from lib.utils.motherduckdb import MotherDuckDBConnector
from lib.model.property_listing import PropertyListing

# Type definitions
DataFrame = pd.DataFrame

MRT_INFO: Optional[DataFrame] = None
HAWKER_INFO: Optional[DataFrame] = None
SUPERMARKET_INFO: Optional[DataFrame] = None
PRIMARY_SCHOOL_INFO: Optional[DataFrame] = None
MALL_INFO: Optional[DataFrame] = None

REVERSE_DISTRICTS = {v: k for k, v in DISTRICTS.items()}
COL_MAPPER = {
    "price/sqft": "price_per_sqft",
    "distance_to_nearest_mrt": "distance_to_mrt_in_m",
    "distance_to_nearest_hawker": "distance_to_hawker_in_m",
    "distance_to_nearest_sch": "distance_to_sch_in_m",
    "distance_to_nearest_supermarket": "distance_to_supermarket_in_m",
    "distance_to_nearest_mall": "distance_to_mall_in_m",
}


def update_coordinates(df, building_map) -> Tuple[pd.DataFrame, dict]:
    # Locate rows with null lat and long
    df_null_coords = df[(df["latitude"].isnull()) & (df["longitude"].isnull())]

    # Process building names
    building_names = df_null_coords["building_name"].unique()
    building_names_duplicate = [name for name in building_names if name in building_map]
    for building_name in building_names_duplicate:
        df.loc[df["building_name"] == building_name, "latitude"] = building_map[building_name][0]
        df.loc[df["building_name"] == building_name, "longitude"] = building_map[building_name][1]

    building_names_to_fetch = [name for name in building_names if name not in building_map]

    # Process addresses
    addresses = df_null_coords["address"].unique()

    with ThreadPoolExecutor() as executor:
        logging.info(f"Using {executor._max_workers} threads to fetch coordinates")

        future_to_coords = {executor.submit(fetch_coordinates, name): name for name in building_names_to_fetch}
        future_to_coords.update({executor.submit(fetch_coordinates, address): address for address in addresses})

        for future in as_completed(future_to_coords):
            result = future.result()
            if result is None:
                continue

            name, coords = result

            # try to convert coords tuple to float else nan
            try:
                coords = (float(coords[0]), float(coords[1]))
            except ValueError:
                coords = (np.nan, np.nan)

            if name in building_names_to_fetch:
                building_map[name] = coords
                df.loc[df["building_name"] == name, "latitude"] = coords[0]
                df.loc[df["building_name"] == name, "longitude"] = coords[1]
            else:
                df.loc[df["address"] == name, "latitude"] = coords[0]
                df.loc[df["address"] == name, "longitude"] = coords[1]

    return df, building_map


def simplify_lease_type(lease_type: Optional[str]) -> Optional[str]:
    if pd.isnull(lease_type):
        return None
    return "leasehold" if "leasehold" in lease_type.lower() else "freehold"


def extract_num(x: Any) -> Optional[str]:
    if not x:
        return None
    if isinstance(x, float) or isinstance(x, int):
        x = str(x)
    if not isinstance(x, str):
        logging.warning(f"Unexpected data: {x} with type {type(x)}")
    res = re.findall(r"\d[\d,]*", x)
    return res[0] if res else None


def extract_num_price(x: Any) -> str:
    return extract_num(x) or ""


def update_nearest_info(df: DataFrame, info_type: str, augment_df: DataFrame) -> DataFrame:
    df_null = df[df[f"nearest_{info_type}"].isnull()]
    df_null = find_nearest(df_null, augment_df, f"nearest_{info_type}", f"distance_to_nearest_{info_type}")
    df.update(df_null)
    return df


def augment_df_w_add_info(df: DataFrame, augment_data: dict[str, DataFrame]) -> DataFrame:
    logging.info("Augmenting dataframe with additional info...")
    for info_type in ["mrt", "hawker_centre", "supermarket", "primary_school", "mall"]:
        info_type_df = augment_data[info_type]

        if info_type == "hawker_centre":
            info_type = "hawker"
        elif info_type == "primary_school":
            info_type = "sch"

        df[f"nearest_{info_type}"] = None
        df[f"distance_to_nearest_{info_type}"] = float("inf")
        df = update_nearest_info(df, info_type, info_type_df)
    return df


def set_metadata(date: str, df: DataFrame) -> DataFrame:
    logging.info("Setting metadata...")
    df["source"] = "propnex"
    df["scraped_on"] = datetime.strptime(date, "%Y-%m-%d")
    df["last_updated"] = df["scraped_on"]
    return df


def drop_null_coords(df: DataFrame) -> DataFrame:
    logging.info("Dropping null coordinates...")
    null_coord_indexes = df["latitude"].isnull() | df["longitude"].isnull()
    logging.info(f"Length of null coordinates: {null_coord_indexes.sum()}")
    return df.drop(df[null_coord_indexes].index)


def get_building_map(df: DataFrame) -> Dict[str, Tuple[float, float]]:
    logging.info("Getting building map...")
    building_map = {}
    for building_name, group in df.groupby("building_name"):
        for _, row in group.iterrows():
            if pd.notna(row["latitude"]) and pd.notna(row["longitude"]):
                building_map[building_name] = (row["latitude"], row["longitude"])
                break
    return building_map


def transform_categorical_values(df: DataFrame) -> DataFrame:
    logging.info("Transforming categorical values...")
    try:
        df["listing_id"] = df["listing_id"].astype(str)
        df['property_type'] = df['property_type'].astype('category')
        df['furnishing'] = df['furnishing'].fillna('Unfurnished').astype('category')
        df['tenure'] = df['tenure'].apply(simplify_lease_type).astype('category')
        df['floor_level'] = df['floor_level'].astype('category')
        df['district'] = df['district'].astype('category')
    except AttributeError as e:
        logging.error(f"No categorical values on this day: {e}")
    return df


def transform_numerical_values(df: DataFrame) -> DataFrame:
    logging.info("Transforming numerical values...")
    df = df.dropna(subset=['dimensions'])
    try:
        df["price"] = df["price"].apply(extract_num_price).str.replace(",", "").astype(int)
        # at least 1 room
        df["bedroom"] = df["bedroom"].fillna(1).astype(int)
        # assume each room / unit has at least 1 toilet
        df["bathroom"] = df["bathroom"].apply(extract_num).fillna("1").astype(int)
        df["dimensions"] = df["dimensions"].astype(int)
        df["built_year"] = df["built_year"].fillna(9999).astype(int)
        df["price/sqft"] = df["price/sqft"].apply(extract_num).str.replace(",", "").astype(int)
    except TypeError as e:
        logging.error(f"Error in numerical values transformation: {e}")
        raise e
    return df


def get_listing_type(url):
    import time
    import random
    import requests
    from bs4 import BeautifulSoup
    
    logging.info("Extracting listing type...")
    time.sleep(random.randint(1, 10))
    res = requests.get(url)
    soup = BeautifulSoup(res.content, 'html.parser')
    prop_list_box = soup.find_all("div", class_="property-list-box")

    for box in prop_list_box:
        labels = box.find_all("b")
        values = box.find_all("span")

        for col, val in zip(labels, values):
            label = col.get_text(strip=True)
            if label != "Listing Type":
                continue
            listing_type = val.get_text(strip=True)
            is_whole_unit = True if listing_type != "ROOM" else False
            return is_whole_unit
    return True


def extract_facilities(df: DataFrame) -> DataFrame:
    logging.info("Extracting facilities...")
    df["has_pool"] = df["facilities"].apply(lambda x: "pool" in x.lower() if x else None)
    df["has_gym"] = df["facilities"].apply(lambda x: "gym" in x.lower() if x else None)
    return df


def validate_dataframe(df: DataFrame) -> DataFrame:
    """Validate dataframe using Pydantic model."""
    logging.info("Validating dataframe...")
    return pd.DataFrame([PropertyListing(**row).model_dump() for _, row in df.iterrows()])


def process_dataframe(df: DataFrame, augment_data: dict[str, DataFrame], date: str) -> DataFrame:
    df["building_name"] = df["property_name"]
    logging.info(f"Unique building names: {len(df['building_name'].unique())}")
    
    building_map = get_building_map(df)
    df, building_map = update_coordinates(df, building_map)

    df["district_id"] = df["district"].map(REVERSE_DISTRICTS)

    df = drop_null_coords(df)
    df = augment_df_w_add_info(df, augment_data)
    df = transform_categorical_values(df)
    df = transform_numerical_values(df)

    if "is_whole_unit" not in df.columns:
        df['is_whole_unit'] = df['url'].apply(get_listing_type)

    df = extract_facilities(df)

    df["fingerprint"] = df["listing_id"].astype(str) + "-" + df["price"].astype(str)
    df = set_metadata(date, df)

    df.rename(columns=COL_MAPPER, inplace=True)
    return df[PROPERTY_LISTING_COLS]


def print_all_columns(df: DataFrame) -> None:
    for column in df.columns:
        logging.info(f"{column} with type {df[column].dtype}:\n")
        logging.info(f"\n{df[column]}\n")
        print()


def transform(df: DataFrame, augment_data: dict[str, DataFrame], date: str, debug: bool = False) -> DataFrame:
    logging.info(f"Dataframe received with shape {df.shape}")

    if debug:
        print_all_columns(df)

    # Drop duplicates
    logging.info(f"Length of duplicates: {len(df[df.duplicated(subset='listing_id', keep=False)])}")
    df.drop_duplicates(subset="listing_id", keep="first", inplace=True)

    df = process_dataframe(df, augment_data, date)
    df = validate_dataframe(df)

    if debug:
        print_all_columns(df)

    return df

