import re
import logging
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely import wkt
from datetime import datetime
from duckdb import DuckDBPyConnection
from typing import Tuple, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from lib.constants.db_constants import *
from lib.constants.location_constants import *
from lib.utils.coordinates import fetch_coordinates, find_nearest, get_district
from lib.utils.motherduckdb import MotherDuckDBConnector
from lib.model.property_listing import PropertyListing

# Type definitions
DataFrame = pd.DataFrame
GeoDataFrame = gpd.GeoDataFrame

# Global variables (consider making these part of a class instead)
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

# Data transformation functions


def simplify_lease_type(lease_type: Optional[str]) -> Optional[str]:
    if pd.isnull(lease_type):
        return None
    return "leasehold" if "leasehold" in lease_type.lower() else "freehold"


def simplify_property_type(property_type: Optional[str]) -> Optional[str]:
    if pd.isnull(property_type):
        return None

    property_type = property_type.strip()

    if "Condo" in property_type:
        return "Executive Condo" if "Executive" in property_type else "Condo"
    elif "Apartment" in property_type:
        return "Executive Apartment" if "Executive" in property_type else "Apartment"
    elif "HDB" in property_type:
        return "Executive HDB" if "Executive" in property_type else "HDB"
    elif "Walk-up" in property_type:
        return "Walk-up"
    elif "Bungalow" in property_type:
        return "Bungalow"
    elif "Land" in property_type:
        return "Landed"
    elif "Cluster House" in property_type:
        return "Cluster House"

    logging.warning(f"Unrecognized property type: {property_type}")
    return property_type


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


def extract_num_bedroom(x: Any) -> str:
    num = extract_num(x)
    return num if num else "1"


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
    df["source"] = "ninety_nine"
    df["scraped_on"] = datetime.strptime(date, "%Y-%m-%d")
    df["last_updated"] = df["scraped_on"]
    return df


def transform_address(df: DataFrame) -> DataFrame:
    logging.info("Transforming address...")
    df["address"] = df["address"].fillna("")

    try:
        landed_indexes = df["address"].str.contains("Landed House For Rent", case=False)
        df.loc[landed_indexes, "address"] = df.loc[landed_indexes, "building_name"].apply(
            lambda x: [s.strip() for s in re.split(r"\bon\b|\bin\b", x) if s.strip()][-1]
        )

        for_rent_indexes = df["address"].str.contains("For Rent", case=False)
        df.loc[for_rent_indexes, "address"] = df.loc[for_rent_indexes, "building_name"]
    except AttributeError as e:
        logging.error(f"No address on this day: {e}")

    return df


def drop_duplicates(df: DataFrame, geometry_df: GeoDataFrame) -> DataFrame:
    logging.info("Dropping duplicates...")
    df["tmp_district_id"] = df.apply(
        lambda x: get_district(x["latitude"], x["longitude"], geometry_df), axis=1)
    df.drop(df[
        (df.duplicated(subset="listing_id", keep=False)) &
        (df["district_id"] != df["tmp_district_id"])].index, inplace=True)
    df.drop(columns=["tmp_district_id"], inplace=True)

    logging.info(f"Length of real duplicates: {len(df[df.duplicated(subset='listing_id', keep=False)])}")
    df.drop_duplicates(subset="listing_id", keep="first", inplace=True)
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


def transform_gdf(df: DataFrame) -> GeoDataFrame:
    logging.info("Transforming GeoDataFrame...")
    df["polygon"] = df["polygon"].apply(wkt.loads)
    return gpd.GeoDataFrame(df, geometry="polygon")


def transform_categorical_values(df: DataFrame) -> DataFrame:
    logging.info("Transforming categorical values...")
    try:
        df["property_type"] = df["property_type"].apply(simplify_property_type).astype("category")
        df["furnishing"] = df["furnishing"].fillna("Unfurnished").astype("category")
        df["tenure"] = df["tenure"].apply(simplify_lease_type).astype("category")
        df["district"] = df["district"].astype("category")
        df["facing"] = df["facing"].astype("category")
        df["floor_level"] = df["floor_level"].str.replace(r"\s*\(\d+ total\)", "", regex=True).astype("category")
    except AttributeError as e:
        logging.error(f"No categorical values on this day: {e}")
    return df


def transform_numerical_values(df: DataFrame) -> DataFrame:
    logging.info("Transforming numerical values...")
    df = df.dropna(subset=['dimensions'])
    try:
        df["price"] = df["price"].apply(extract_num_price).str.replace(",", "").astype(int)
        df["bedroom"] = df["bedroom"].apply(extract_num_bedroom).astype(int)
        # assume each room / unit has at least 1 toilet
        df["bathroom"] = df["bathroom"].apply(extract_num).fillna("1").astype(int)
        df["dimensions"] = df["dimensions"].apply(extract_num).str.replace(",", "").astype(int)
        df["built_year"] = df["built_year"].fillna(9999).astype(int)
        df["price/sqft"] = df["price/sqft"].apply(extract_num).str.replace(",", "").astype(float)
    except TypeError as e:
        logging.error(f"Error in numerical values transformation: {e}")
        raise e
    return df


def update_room_rental_properties(df):
    logging.info("Updating room rental properties...")
    # TODO: find out why bathroom is still set as 0
    indexes = df.loc[(df["property_name"].str.contains(
        "Room", case=False))].index
    df.loc[indexes, "is_whole_unit"] = False
    df.loc[indexes, "bedroom"] = 1
    # Assume 1 bathroom for room rental
    df.loc[indexes, "bathroom"] = 1
    logging.debug(
        df.loc[indexes, ["property_name", "bedroom", "bathroom", "is_whole_unit"]])

    indexes = df.loc[(df["property_name"].str.contains(
        "Studio", case=False))].index
    df.loc[indexes, "bedroom"] = 1
    return df


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
    df["building_name"] = df["property_name"].apply(lambda x: x.split(" in ")[-1])
    logging.info(f"Unique building names: {len(df['building_name'].unique())}")

    df = transform_address(df)
    building_map = get_building_map(df)
    df, building_map = update_coordinates(df, building_map)

    df["district_id"] = df["district"].map(REVERSE_DISTRICTS)

    geometry_df = transform_gdf(augment_data['plan_area_mapping'])
    df = drop_duplicates(df, geometry_df)
    df = drop_null_coords(df)
    df = augment_df_w_add_info(df, augment_data)
    df = transform_categorical_values(df)
    df = transform_numerical_values(df)

    df["is_whole_unit"] = True
    df = update_room_rental_properties(df)
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

    logging.info(f"Length of duplicates: {len(df[df.duplicated(subset='listing_id', keep=False)])}")

    df = process_dataframe(df, augment_data, date)
    df = validate_dataframe(df)

    if debug:
        print_all_columns(df)

    return df


def change_data_capture(
        db: MotherDuckDBConnector,
        df: DataFrame,
        insert_tbl: str,
        cdc_tbl: str,
        debug: bool = False) -> None:
    logging.info("Calculating change data capture...")
    changed = df[(df["fingerprint"] != df["fingerprint_old"]) & (df["_merge"] == "both")][
        ["listing_id", "fingerprint", "scraped_on", "fingerprint_old", "last_updated_old", "_merge"]
    ]
    logging.info(f"Changed: \n{changed}\n")

    cdc = changed[["listing_id", "fingerprint", "last_updated_old"]]
    cdc["price"] = cdc["fingerprint"].apply(lambda x: int(x.split("-")[1]))
    cdc.rename(columns={"last_updated_old": "timestamp"}, inplace=True)
    cdc = cdc[RENTAL_PRICE_HISTORY_COLS]
    logging.info(f"CDC: \n{cdc}\n")

    changed = changed[["listing_id", "fingerprint", "scraped_on"]]
    changed["price"] = changed["fingerprint"].apply(lambda x: int(x.split("-")[1]))
    changed.rename(columns={"scraped_on": "last_updated"}, inplace=True)
    logging.info(f"Changed: \n{changed}\n")

    if not cdc.empty and not debug:
        db.insert_df(cdc_tbl, cdc)

    COLS_TO_UPDATE = ["price", "fingerprint", "last_updated"]
    if not changed.empty and not debug:
        db.update_table(insert_tbl, "listing_id", COLS_TO_UPDATE, changed)


def insert_df(db_conn: DuckDBPyConnection, df: DataFrame, insert_tbl: str, cdc_tbl: str, debug: bool = False) -> None:
    logging.info("Inserting dataframe...")
    db = MotherDuckDBConnector(db_conn)
    try:
        db.begin_transaction()

        existing = db.query_df(f"SELECT listing_id, fingerprint, last_updated FROM {insert_tbl}")
        df = df.merge(existing, on="listing_id", how="left", indicator=True, suffixes=("", "_old"))

        change_data_capture(db, df, insert_tbl, cdc_tbl, debug)

        new = df[df["_merge"] == "left_only"][PROPERTY_LISTING_COLS]
        logging.info(f"New: \n{new}\n")

        if not new.empty and not debug:
            db.insert_df(insert_tbl, new)

        db.commit_transaction()
        logging.info("Successfully inserted dataframe!")
    except Exception as e:
        db.rollback_transaction()
        logging.error(f"Error in insert_df: {e}")
        raise e
