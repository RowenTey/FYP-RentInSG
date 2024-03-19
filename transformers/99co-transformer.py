import argparse
import os
import sys

sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../')))

import re
import boto3
import logging
import pandas as pd
import geopandas as gpd
from shapely import wkt
from typing import Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers.db_constants import *
from utils.location_constants import *
from utils.find_closest import find_nearest
from utils.coordinates import fetch_coordinates
from utils.parse_geojson import get_district
from utils.read_df_from_s3 import read_df_from_s3
from utils.motherduckdb_connector import MotherDuckDBConnector, connect_to_motherduckdb


def update_coord_w_building_name(df, building_map) -> Tuple[pd.DataFrame, dict]:
    # Locate rows with null lat and long
    df_null_coords = df[(df['latitude'].isnull()) & (df['longitude'].isnull())]
    building_names = df_null_coords['building_name'].unique()

    # Fill from building map first
    building_names_duplicate = [
        name for name in building_names if name in building_map]
    for building_name in building_names_duplicate:
        df.loc[df['building_name'] == building_name,
               'latitude'] = building_map[building_name][0]
        df.loc[df['building_name'] == building_name,
               'longitude'] = building_map[building_name][1]

    building_names_to_fetch = [
        name for name in building_names if name not in building_map]
    with ThreadPoolExecutor() as executor:
        logging.info(
            f"Using {executor._max_workers} threads to fetch coordinates")
        future_to_coords = {executor.submit(
            fetch_coordinates, name): name for name in building_names_to_fetch}
        for future in as_completed(future_to_coords):
            result = future.result()
            if result is None:
                continue

            building_name, coords = result
            building_map[building_name] = coords
            df.loc[df['building_name'] ==
                   building_name, 'latitude'] = coords[0]
            df.loc[df['building_name'] ==
                   building_name, 'longitude'] = coords[1]

    return df, building_map


def update_coord_w_address(df):
    # Locate rows with null lat and long
    df_null_coords = df[(df['latitude'].isnull()) & (df['longitude'].isnull())]
    addresses = df_null_coords['address'].unique()

    with ThreadPoolExecutor() as executor:
        logging.info(
            f"Using {executor._max_workers} threads to fetch coordinates")
        future_to_coords = {executor.submit(
            fetch_coordinates, name): name for name in addresses}
        for future in as_completed(future_to_coords):
            result = future.result()
            if result is None:
                continue

            address, coords = result
            df.loc[df['address'] ==
                   address, 'latitude'] = coords[0]
            df.loc[df['address'] ==
                   address, 'longitude'] = coords[1]

    return df


def fetch_mrt_info(db: MotherDuckDBConnector):
    df2 = db.query_df("SELECT * FROM mrt_info")

    logging.info(f"mrt_info: \n{df2.head()}\n")

    df2 = df2[['station_name', 'latitude', 'longitude']]
    return df2


def fetch_hawker_info(db: MotherDuckDBConnector):
    df2 = db.query_df("SELECT * FROM hawker_centre_info")

    logging.info(f"hawker_centre_info: \n{df2.head()}\n")

    df2 = df2[['name', 'latitude', 'longitude']]
    return df2


def fetch_supermarket_info(db: MotherDuckDBConnector):
    df2 = db.query_df("SELECT * FROM supermarket_info")

    logging.info(f"supermarket_info: \n{df2.head()}\n")

    df2 = df2[['name', 'latitude', 'longitude']]
    return df2


def fetch_primary_school_info(db: MotherDuckDBConnector):
    df2 = db.query_df("SELECT * FROM primary_school_info")

    logging.info(f"primary_school_info: \n{df2.head()}\n")

    df2 = df2[['name', 'latitude', 'longitude']]
    return df2


def fetch_mall_info(db: MotherDuckDBConnector):
    df2 = db.query_df("SELECT * FROM mall_info")

    logging.info(f"mall_info: \n{df2.head()}\n")

    df2 = df2[['name', 'latitude', 'longitude']]
    return df2


def update_mrt(db: MotherDuckDBConnector, df):
    df_null_mrt = df[df["nearest_mrt"].isnull()]

    mrt_info = fetch_mrt_info(db)
    df_null_mrt = find_nearest(
        df_null_mrt, mrt_info, "nearest_mrt", "distance_to_nearest_mrt")
    df.update(df_null_mrt)

    return df


def update_hawker(db: MotherDuckDBConnector, df):
    df_null_hawker = df[df["nearest_hawker"].isnull()]

    hawker_info = fetch_hawker_info(db)
    df_null_hawker = find_nearest(
        df_null_hawker, hawker_info, "nearest_hawker", "distance_to_nearest_hawker")
    df.update(df_null_hawker)

    return df


def update_supermarket(db: MotherDuckDBConnector, df):
    df_null_supermarket = df[df["nearest_supermarket"].isnull()]

    supermarket_info = fetch_supermarket_info(db)
    df_null_supermarket = find_nearest(
        df_null_supermarket, supermarket_info, "nearest_supermarket", "distance_to_nearest_supermarket")
    df.update(df_null_supermarket)

    return df


def update_primary_school(db: MotherDuckDBConnector, df):
    df_null_sch = df[df["nearest_sch"].isnull()]

    sch_info = fetch_primary_school_info(db)
    df_null_sch = find_nearest(
        df_null_sch, sch_info, "nearest_sch", "distance_to_nearest_sch")
    df.update(df_null_sch)

    return df


def update_mall(db: MotherDuckDBConnector, df):
    df_null_mall = df[df["nearest_mall"].isnull()]

    mall_info = fetch_mall_info(db)
    df_null_mall = find_nearest(
        df_null_mall, mall_info, "nearest_mall", "distance_to_nearest_mall")
    df.update(df_null_mall)

    return df


def simplify_lease_type(lease_type):
    if pd.isnull(lease_type):
        return None
    elif 'leasehold' in lease_type:
        return 'leasehold'
    else:
        return 'freehold'


def simplify_property_type(property_type):
    if pd.isnull(property_type):
        return None
    elif 'Condo' in property_type and 'Executive' not in property_type:
        return 'Condo'
    elif 'HDB' in property_type and 'Executive' not in property_type:
        return 'HDB'
    elif 'Apartment' in property_type and 'Executive' not in property_type:
        return 'Apartment'
    elif 'Bungalow' in property_type:
        return 'HDB'
    elif 'Land' in property_type:
        return 'Landed'

    return property_type.strip()


def extract_num_price(x):
    if not x:
        return ""
    res = re.findall(r'\d[\d,]*', x)
    return res[0] if res else ""


def extract_num_bedroom(x):
    if not x:
        return "0"
    res = re.findall(r'\d[\d,]*', x)
    if not res:
        return "1"
    return res[0]


def extract_num(x):
    if not x:
        return None
    res = re.findall(r'\d[\d,]*', x)
    return res[0] if res else None


def update_room_rental_properties(df):
    indexes = df.loc[(df['property_name'].str.contains(
        'Room', case=False))].index
    df.loc[indexes, 'is_whole_unit'] = False
    df.loc[indexes, 'bedroom'] = 1
    df.loc[indexes, 'bathroom'] = 0

    indexes = df.loc[(df['property_name'].str.contains(
        'Studio', case=False))].index
    df.loc[indexes, 'bedroom'] = 1

    return df


def augment_df_w_add_info(db: MotherDuckDBConnector, df):
    df = update_mrt(db, df)

    df["nearest_hawker"] = None
    df["distance_to_nearest_hawker"] = float("inf")
    df = update_hawker(db, df)

    df["nearest_supermarket"] = None
    df["distance_to_nearest_supermarket"] = float("inf")
    df = update_supermarket(db, df)

    df["nearest_sch"] = None
    df["distance_to_nearest_sch"] = float("inf")
    df = update_primary_school(db, df)

    df["nearest_mall"] = None
    df["distance_to_nearest_mall"] = float("inf")
    df = update_mall(db, df)
    return df


def set_metadata(date: str, df):
    df['source'] = 'ninety_nine'
    df['scraped_on'] = datetime.strptime(date, '%Y-%m-%d')
    df['last_updated'] = df['scraped_on']
    return df


def transform_address(df):
    indexes = df.loc[df['address'].str.contains(
        'Landed House For Rent', case=False)].index
    df.loc[indexes, 'address'] = df.loc[indexes, 'building_name'] \
        .apply(lambda x: [s.strip() for s in re.split(r'\bon\b|\bin\b', x) if s.strip()][-1])
    df.loc[indexes, ['address', 'property_name', 'building_name']]

    indexes = df.loc[df['address'].str.contains('For Rent', case=False)].index
    df.loc[indexes, 'address'] = df.loc[indexes, 'building_name']
    df.loc[indexes, ['address', 'property_name', 'building_name']]
    return df


def drop_duplicates(df, geometry_df: gpd.GeoDataFrame) -> pd.DataFrame:
    # Get temp district id to compare with real district id
    df["tmp_district_id"] = df.apply(
        lambda x: get_district(x["latitude"], x["longitude"], geometry_df), axis=1)
    df.drop(df[(df.duplicated(subset='listing_id', keep=False)) & (
        df["district_id"] != df["tmp_district_id"])].index, inplace=True)
    df.drop(columns=["tmp_district_id"], inplace=True)

    logging.info("Length of real duplicates: " +
                 str(len(df[df.duplicated(subset='listing_id', keep=False)])))
    df.drop_duplicates(subset='listing_id', keep='first', inplace=True)
    return df


def get_building_map(df):
    building_map = {}
    for building_name, group in df.groupby('building_name'):
        for _, row in group.iterrows():
            if pd.isna(row['latitude']) or pd.isna(row['longitude']):
                continue

            building_map[building_name] = (row['latitude'], row['longitude'])
            break
    return building_map


def fetch_gdf(db) -> gpd.GeoDataFrame:
    geometry_df = db.query_df("SELECT * FROM plan_area_mapping")
    geometry_df['polygon'] = geometry_df['polygon'].apply(wkt.loads)
    geometry_df = gpd.GeoDataFrame(geometry_df, geometry='polygon')
    return geometry_df


def transform_categorical_values(df) -> pd.DataFrame:
    df['property_type'] = df['property_type'].apply(
        simplify_property_type).astype('category')
    df['furnishing'] = df['furnishing'].fillna(
        'Unfurnished').astype('category')
    df['facing'] = df['facing'].astype('category')
    df['tenure'] = df['tenure'].apply(simplify_lease_type).astype('category')
    df['floor_level'] = df['floor_level'].str.replace(
        r'\s*\(\d+ total\)', '', regex=True).astype('category')
    df['district'] = df['district'].astype('category')
    return df


def extract_facilities(df) -> pd.DataFrame:
    df['has_pool'] = df['facilities'].apply(lambda x: 'pool' in x.lower())
    df['has_gym'] = df['facilities'].apply(lambda x: 'gym' in x.lower())
    return df


def transform_numerical_values(df) -> pd.DataFrame:
    df['price'] = df['price'].apply(
        extract_num_price).str.replace(',', '').astype(int)
    df['bedroom'] = df['bedroom'].apply(
        extract_num_bedroom).astype(int)
    df['bathroom'] = df['bathroom'].apply(
        extract_num).fillna("0").astype(int)
    df['dimensions'] = df['dimensions'].apply(
        extract_num).str.replace(',', '').astype(int)
    df['price/sqft'] = df['price/sqft'].apply(
        extract_num).astype(float)
    df['built_year'] = df['built_year'].fillna(9999).astype(int)
    return df


def insert_df(db, df, debug: bool = False) -> None:
    existing = db.query_df(
        "SELECT listing_id, fingerprint, last_updated FROM property_listing")
    df = df.merge(existing, on="listing_id", how="left",
                  indicator=True, suffixes=('', '_old'))

    change_data_capture(df, db, debug)

    # Insert new property listing
    new = df[df['_merge'] == 'left_only'][PROPERTY_LISTING_COLS]
    logging.info(f"New: \n{new}\n")

    if not new.empty and not debug:
        db.insert_df("property_listing", new)


def change_data_capture(df, db: MotherDuckDBConnector, debug: bool = False) -> None:
    """ 
    changed:  
    1. insert to rental price history
    2. update old one (fingerprint and last_updated)
    """
    changed = df[(df['fingerprint'] != df['fingerprint_old']) & (df['_merge'] == 'both')][[
        'listing_id', 'fingerprint', 'scraped_on', 'fingerprint_old', 'last_updated_old', '_merge']]
    logging.info(f"Changed: \n{changed}\n")

    # Change data capture
    cdc = changed[['listing_id', 'fingerprint', 'last_updated_old']]
    cdc['price'] = cdc['fingerprint'].apply(
        lambda x: int(x.split("-")[1]))
    cdc.rename(columns={'last_updated_old': 'timestamp'}, inplace=True)
    cdc = cdc[RENTAL_PRICE_HISTORY_COLS]
    logging.info(f"CDC: \n{cdc}\n")

    # Update old listings with new fingerprint and last_updated
    changed = changed[['listing_id', 'fingerprint', 'scraped_on']]
    changed['price'] = changed['fingerprint'].apply(
        lambda x: int(x.split("-")[1]))
    changed.rename(columns={'scraped_on': 'last_updated'}, inplace=True)
    logging.info(f"Changed: \n{changed}\n")

    # Insert to rental price history
    if not cdc.empty and not debug:
        db.insert_df("rental_price_history", cdc)

    # Update property listing
    COLS_TO_UPDATE = ["price", "fingerprint", "last_updated"]
    if not changed.empty and not debug:
        db.update_table("property_listing", "listing_id",
                        COLS_TO_UPDATE, changed)


def transform(db: MotherDuckDBConnector, date: str, debug: bool = False):
    df = read_df_from_s3(
        f"rental_prices/ninety_nine/{date}.parquet.gzip")
    logging.info(f"Dataframe downloaded with shape {df.shape}")
    logging.info(
        f"Length of duplicates: {len(df[df.duplicated(subset='listing_id', keep=False)])}")

    # Get building name
    df['building_name'] = df['property_name'].apply(
        lambda x: x.split(" in ")[-1])
    logging.info(f"Unique building names: {len(df['building_name'].unique())}")

    # Fix address column
    df = transform_address(df)

    # Build building map
    building_map = get_building_map(df)

    # Fetch using building names first
    df, building_map = update_coord_w_building_name(df, building_map)

    # Fetch using addresses next
    df = update_coord_w_address(df)

    # Get district id
    df["district_id"] = df["district"].map(REVERSE_DISTRICTS)

    # Fetch geometry df
    geometry_df = fetch_gdf(db)

    # Drop duplicates
    df = drop_duplicates(df, geometry_df)

    # Add in additional info
    df = augment_df_w_add_info(db, df)

    # Transform data - categorical
    df = transform_categorical_values(df)

    # Transform data - numerical
    df = transform_numerical_values(df)

    # Update room rental properties
    df["is_whole_unit"] = True
    df = update_room_rental_properties(df)

    # Facilities
    df = extract_facilities(df)

    # Fingerprint
    df["fingerprint"] = df["listing_id"] + "-" + df["price"].astype(str)
    logging.info(f"Fingerprint: \n{df['fingerprint']}\n")

    # Metadata
    df = set_metadata(date, df)

    # Rename columns
    df.rename(columns=COL_MAPPER, inplace=True)
    df.info()

    # Standardize columns
    df = df[PROPERTY_LISTING_COLS]
    insert_df(db, df, debug)


def get_s3_file_names(bucket_name, prefix):
    s3 = boto3.client('s3')
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    file_names = []
    for obj in response['Contents']:
        file_names.append(obj['Key'])
    return file_names


def delete_s3_file(bucket_name, filename):
    s3 = boto3.client('s3')
    s3.delete_object(Bucket=bucket_name, Key=filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true",
                        help="Enable debug mode")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s:%(name)s:%(filename)s-%(lineno)s [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    REVERSE_DISTRICTS = {v: k for k, v in DISTRICTS.items()}
    COL_MAPPER = {
        'price/sqft': 'price_per_sqft',
        'distance_to_nearest_mrt': 'distance_to_mrt_in_m',
        'distance_to_nearest_hawker': 'distance_to_hawker_in_m',
        'distance_to_nearest_sch': 'distance_to_sch_in_m',
        'distance_to_nearest_supermarket': 'distance_to_supermarket_in_m',
        'distance_to_nearest_mall': 'distance_to_mall_in_m',
    }

    BUCKET_NAME = os.getenv('S3_BUCKET')
    PREFIX = 'rental_prices/ninety_nine/'

    KEEP_FILE_THRESHOLD = 5

    with open('logs/transformer/last_transformed_date.log', 'r') as file:
        LAST_TRANSFORMED_DATE = file.read().strip()

    db = connect_to_motherduckdb()
    try:
        # today = datetime.now().strftime('%Y-%m-%d')
        today = "2024-03-01"
        for filename in get_s3_file_names(bucket_name=BUCKET_NAME, prefix=PREFIX):
            file_date = filename.split('/')[-1].split('.')[0]
            if LAST_TRANSFORMED_DATE < file_date <= today:
                logging.info(F"Transforming {filename}...")
                transform(db, file_date, args.debug)
            elif LAST_TRANSFORMED_DATE >= file_date and \
                    (datetime.strptime(LAST_TRANSFORMED_DATE, '%Y-%m-%d') - datetime.strptime(file_date, '%Y-%m-%d')).days > KEEP_FILE_THRESHOLD:
                logging.info(f"Deleting {filename}...")
                if not debug:
                    delete_s3_file(BUCKET_NAME, filename)
            else:
                logging.info(f"Skipping {filename}...")

        if not args.debug:
            with open('logs/transformer/last_transformed_date.log', 'w') as file:
                file.write(today)
    except Exception as e:
        logging.error(f"{e.__class__.__name__}: {e}")
    finally:
        db.close()
