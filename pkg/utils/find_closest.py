from math import atan2, cos, radians, sin, sqrt

from geopy.distance import geodesic


def haversine(coord1, coord2):
    """
    Depracated, use geodesic instead
    """
    R = 6371  # radius of Earth in kilometers
    lat1, lon1 = map(radians, coord1)
    lat2, lon2 = map(radians, coord2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c * 1000  # convert to meters


def find_nearest(df1, df2, target_landmark, distance_to_target_landmark, is_inference=False):
    """
    Taken from https://medium.com/@michael.wy.ong/web-scrape-geospatial-data-analyse-singapores-property-price-part-i-276caba320b

    This function finds the nearest locations from the 2nd table from the 1st address
    df2 format:
        1st column: any string column
        2nd column: latitude (float)
        3rd column: longitude (float)
    """
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


def find_nearest_single(data, df):
    """ """
    src_loc = (data["latitude"], data["longitude"])
    distance = float("inf")
    for ind, _ in enumerate(df.iloc[:, 0]):
        df_loc = (df.iloc[ind, 1], df.iloc[ind, 2])
        distance = min(geodesic(src_loc, df_loc).m, distance)
    return distance


if __name__ == "__main__":
    # Example usage
    from read_df_from_s3 import read_df_from_s3

    df1 = read_df_from_s3("rental_prices/ninety_nine/2024-05-05.parquet.gzip")

    from motherduckdb_connector import connect_to_motherduckdb

    db = connect_to_motherduckdb()
    df2 = db.query_df("SELECT * FROM mrt_info")
    print(df2)
    print()
    df2 = df2[["station_name", "latitude", "longitude"]]

    df1["building_name"] = df1["property_name"].apply(lambda x: x.split(" in ")[-1])

    df1 = find_nearest(df1, df2, "nearest_mrt", "distance_to_nearest_mrt")
    print(df1[["property_name", "nearest_mrt", "distance_to_nearest_mrt"]])
    print(df1["nearest_mrt"].unique())
    print()
    print(df1[df1["nearest_mrt"].isna()])
