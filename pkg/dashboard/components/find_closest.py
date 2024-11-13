from math import atan2, cos, radians, sin, sqrt
from geopy.distance import geodesic


def find_nearest(
        df1,
        df2,
        target_landmark,
        distance_to_target_landmark,
        is_inference=False):
    """
    Taken from https://medium.com/@michael.wy.ong/web-scrape-geospatial-data-analyse-singapores-property-price-part-i-276caba320b # noqa: E501

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

        df1.loc[df1["building_name"] == building_name,
                target_landmark] = landmark_info[0]
        df1.loc[df1["building_name"] == building_name,
                distance_to_target_landmark] = landmark_info[2]

    return df1


def find_nearest_single(data, df):
    src_loc = (data["latitude"], data["longitude"])
    distance = float("inf")
    for ind, _ in enumerate(df.iloc[:, 0]):
        df_loc = (df.iloc[ind, 1], df.iloc[ind, 2])
        distance = min(geodesic(src_loc, df_loc).m, distance)
    return distance
