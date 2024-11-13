import geopandas as gpd
import pandas as pd
import pyproj
import requests
from bs4 import BeautifulSoup
from shapely.geometry import shape

MRT_LINE_TO_CODE_MAP = {
    "Circle": "CC",
    "Circle Extension": "CE",
    "East West": "EW",
    "North East": "NE",
    "Downtown": "DT",
    "North South": "NS",
    "Bukit Panjang LRT": "BP",
    "Punggol LRT": "PW | PE | PTC",
    "Sengkang LRT": "SE | SW | STC",
    "Changi Airport Branch": "CG",
    "Thomson–East Coast": "TE",
}

GDF_STATION_NAME_MAP = {
    "ONE-NORTH": "one-north",
    "MACPHERSON": "MacPherson",
    "HARBOURFRONT": "HarbourFront",
}

URL_STATION_NAME_MAP = {"Pungol Point": "Punggol Point"}

MISSING_STATIONS = {
    "Springleaf": {
        "station_code": "TE4",
        "color": "Brown",
        "line": "Thomson–East Coast",
    },
    "Lentor": {"station_code": "TE5", "color": "Brown", "line": "Thomson–East Coast"},
    "Mayflower": {
        "station_code": "TE6",
        "color": "Brown",
        "line": "Thomson–East Coast",
    },
    "Bright Hill": {
        "station_code": "TE7",
        "color": "Brown",
        "line": "Thomson–East Coast",
    },
    "Upper Thomson": {
        "station_code": "TE8",
        "color": "Brown",
        "line": "Thomson–East Coast",
    },
    "Caldecott": {
        "station_code": "TE9",
        "color": "Brown",
        "line": "Thomson–East Coast",
    },
    "Stevens": {"station_code": "TE11", "color": "Brown", "line": "Thomson–East Coast"},
    "Napier": {"station_code": "TE12", "color": "Brown", "line": "Thomson–East Coast"},
    "Orchard Boulevard": {
        "station_code": "TE13",
        "color": "Brown",
        "line": "Thomson–East Coast",
    },
    "Orchard": {"station_code": "TE14", "color": "Brown", "line": "Thomson–East Coast"},
    "Great World": {
        "station_code": "TE15",
        "color": "Brown",
        "line": "Thomson–East Coast",
    },
    "Havelock": {
        "station_code": "TE16",
        "color": "Brown",
        "line": "Thomson–East Coast",
    },
    "Outram Park": {
        "station_code": "TE17",
        "color": "Brown",
        "line": "Thomson–East Coast",
    },
    "Maxwell": {"station_code": "TE18", "color": "Brown", "line": "Thomson–East Coast"},
    "Shenton Way": {
        "station_code": "TE19",
        "color": "Brown",
        "line": "Thomson–East Coast",
    },
}


def extract_relevant_code(row, mapping):
    if pd.isna(row["station_code"]):
        return None

    station_codes = row["station_code"].split()
    line_code = mapping.get(row["line"], "")
    if "|" in line_code:
        line_codes = line_code.split(" | ")
        relevant_codes = []
        for line_code in line_codes:
            relevant_codes.extend(
                [code for code in station_codes
                 if code.startswith(line_code)])
        return " ".join(relevant_codes) if relevant_codes else None

    relevant_codes = [code for code in station_codes
                      if code.startswith(line_code)]
    return " ".join(relevant_codes) if relevant_codes else None


def convert_to_shapely(geometry_string: str) -> shape:
    """
    Convert a geometry string to a Shapely Polygon object.

    Args:
        geometry_string (str): The geometry string representing a polygon.

    Returns:
        shapely.geometry.Polygon: The Shapely Polygon object.

    Example:
        >>> convert_to_shapely("POLYGON ((30566.073713729158 30621.214118300006, ...), (...))")
        <shapely.geometry.polygon.Polygon object at 0x7f9e8a1e5a90>
    """
    return shape(
        {
            "type": "Polygon",
            "coordinates": [
                [list(map(float, coord.split())) for coord in part.split(", ")]
                for part in geometry_string[10:-2].split("), (")
            ],
        }
    )


def xy_to_lonlat(x: float, y: float) -> tuple:
    """
    Converts coordinates from a projected coordinate system (EPSG:3414) to a geographic coordinate system (EPSG:4326).

    Args:
        x (float): The x-coordinate in the projected coordinate system.
        y (float): The y-coordinate in the projected coordinate system.

    Returns:
        tuple: A tuple containing the longitude and latitude in the geographic coordinate system.
    """
    p = pyproj.CRS("epsg:3414")
    p_to = pyproj.CRS("epsg:4326")
    transformer = pyproj.Transformer.from_crs(p, p_to, always_xy=True)
    lonlat = transformer.transform(x, y)
    return (lonlat[0], lonlat[1])


def convert_to_4326_WGS84_coordinates(geometry_string) -> tuple:
    """
    Converts a geometry string to WGS84 coordinates (longitude, latitude) in EPSG:4326 projection.

    Args:
        geometry_string (str): The geometry string representing the polygon.

    Returns:
        tuple: A tuple containing the longitude and latitude coordinates in EPSG:4326 projection.
    """
    polygon = convert_to_shapely(geometry_string)
    centroid_point = polygon.centroid
    return xy_to_lonlat(centroid_point.x, centroid_point.y)


def convert_to_camel_case(text: str) -> str:
    """
    Converts a given text to camel case.

    Args:
        text (str): The text to be converted.

    Returns:
        str: The converted text in camel case.
    """
    if text in GDF_STATION_NAME_MAP:
        return GDF_STATION_NAME_MAP[text]

    words = text.split()
    camel_case = [word.capitalize() for word in words]
    return " ".join(camel_case)


def transform_data(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Transform the GeoDataFrame by converting coordinates, renaming columns, and applying camel case to station names.

    Args:
        gdf (gpd.GeoDataFrame): The GeoDataFrame containing the MRT and LRT data.

    Returns:
        gpd.GeoDataFrame: The transformed GeoDataFrame.

    Example:
        >>> transform_data(gdf)
        station_name  longitude   latitude
        0   Jurong East  103.742287  1.333115
        1   Bukit Batok  103.749567  1.349057
        ...
    """
    gdf["longitude"], gdf["latitude"] = zip(
        *gdf["geometry"].apply(lambda x: convert_to_4326_WGS84_coordinates(str(x))))
    gdf = gdf.rename(columns={"STN_NAM_DE": "station_name"})
    gdf["station_name"] = (
        gdf["station_name"].str.replace(
            " MRT STATION",
            "").str.replace(
            " LRT STATION",
            "").apply(convert_to_camel_case))

    return gdf[["station_name", "longitude", "latitude"]]


def scrape_mrt_and_lrt_data() -> pd.DataFrame:
    """
    Scrapes MRT and LRT data from the specified URLs and returns a concatenated DataFrame.

    Returns:
        pd.DataFrame: A concatenated DataFrame containing the scraped MRT and LRT data.

    Example:
        >>> scrape_mrt_and_lrt_data()
            station_code  station_name  color  line
        0   NS1            Jurong East   RED    North South Line
        1   NS2            Bukit Batok   RED    North South Line
        ...
    """
    mrt_columns = ["station_code", "station_name", "color", "line"]
    lrt_columns = ["station_code", "station_name", "line"]

    dfs = []
    for url in URLS:
        resp = requests.get(url)
        soup = BeautifulSoup(resp.content, "html.parser")
        table = soup.find("table")
        rows = table.find_all("tr")

        # Determine if the data is for MRT or LRT based on the number of
        # columns
        headers = mrt_columns if len(rows[0].find_all(
            "th")) == len(mrt_columns) else lrt_columns

        data = []
        for row in rows[1:]:
            cols = row.find_all(["td", "th"])
            row_data = [col.text.strip() for col in cols]
            data.append(row_data)

        # Create a GeoDataFrame from the collected data and columns
        gdf = gpd.GeoDataFrame(data, columns=headers)
        dfs.append(gdf)

    missing_data = (
        pd.DataFrame.from_dict(
            MISSING_STATIONS,
            orient="index").reset_index().rename(
            columns={
                "index": "station_name"}))
    dfs.append(missing_data)

    return pd.concat(dfs, ignore_index=True)


def shp_to_csv(shp_file_path, csv_file_path) -> None:
    """
    Convert Shapefile to CSV by transforming the data and scraping MRT and LRT information.

    Args:
        shp_file_path (str): The path to the Shapefile.
        csv_file_path (str): The path to the CSV file to be created.

    Example:
        >>> shp_to_csv("data\\mrt_location_data\\RapidTransitSystemStation.shp", "data\\mrt_lrt_sg.csv")
        station_name  longitude   latitude   color  line
        0   Jurong East  103.742287  1.333115   NaN    North South Line
        1   Bukit Batok  103.749567  1.349057   NaN    North South Line
        ...
    """
    gdf = gpd.read_file(shp_file_path)
    gdf = transform_data(gdf)
    print(gdf)

    mrt_info = scrape_mrt_and_lrt_data()
    mrt_info["station_name"] = mrt_info["station_name"].apply(
        lambda x: URL_STATION_NAME_MAP[x] if x in URL_STATION_NAME_MAP else x
    )
    print(mrt_info)
    print(mrt_info[mrt_info["line"] == "Sengkang LRT"])

    gdf = gdf.merge(mrt_info, how="left", on="station_name")
    gdf["station_code"] = gdf.apply(
        lambda row: extract_relevant_code(
            row, MRT_LINE_TO_CODE_MAP), axis=1)
    gdf.drop(gdf[gdf["line"].isna()].index, inplace=True)
    gdf.drop_duplicates(subset=["station_name", "line"], inplace=True)
    print(gdf)

    # Write to CSV
    gdf.to_csv(csv_file_path, index=False)


if __name__ == "__main__":
    URLS = [
        "https://mrtmapsingapore.com/mrt-stations-singapore/",
        "https://mrtmapsingapore.com/lrt-stations/",
    ]
    SHP_FILE_PATH = "..\\data\\mrt_location_data\\RapidTransitSystemStation.shp"
    CSV_FILE_PATH = "..\\data\\mrt_lrt_sg.csv"

    shp_to_csv(SHP_FILE_PATH, CSV_FILE_PATH)
    print(
        f"Conversion from Shapefile to CSV completed. CSV file saved at: {CSV_FILE_PATH}")
