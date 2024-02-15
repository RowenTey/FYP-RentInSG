import pyproj
import requests
import pandas as pd
import geopandas as gpd
from bs4 import BeautifulSoup
from shapely.geometry import shape

MRT_LINE_TO_CODE_MAP = {
    "Circle": "CC",
    "Circle Extension": "CE",
    "East West": "EW",
    "North East": "NE",
    "Downtown": "DT",
    "North South": "NS",
    "Punggol LRT": "PW",
    "Sengkang LRT": "SE",
    "Changi Airport Branch": "CG",
}

# Convert geometry string to Shapely geometry object


def convert_to_shapely(geometry_string):
    # example input: "POLYGON ((30566.073713729158 30621.214118300006, ...), (...))"
    return shape({
        "type": "Polygon",
        "coordinates": [
            [list(map(float, coord.split())) for coord in part.split(", ")]
            for part in geometry_string[10:-2].split("), (")
        ]
    })


# Convert [x, y] 3414 (SVY21) format to [lat, lon] 4326 (WGS84) format
def xy_to_lonlat(x, y):
    p = pyproj.CRS("epsg:3414")
    p_to = pyproj.CRS("epsg:4326")
    transformer = pyproj.Transformer.from_crs(p, p_to, always_xy=True)
    lonlat = transformer.transform(x, y)
    return lonlat[0], lonlat[1]


# Convert geometry string to coordinates
def convert_to_4326_WGS84_coordinates(geometry_string):
    polygon = convert_to_shapely(geometry_string)
    centroid_point = polygon.centroid

    lon, lat = xy_to_lonlat(centroid_point.x, centroid_point.y)

    return (lon, lat)


def convert_to_camel_case(text):
    words = text.split()
    camel_case = [word.capitalize() for word in words]
    return " ".join(camel_case)


def transform_data(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf["longitude"], gdf["latitude"] = zip(*gdf["geometry"].apply(
        lambda x: convert_to_4326_WGS84_coordinates(str(x))))
    gdf = gdf.rename(columns={"STN_NAM_DE": "station_name"})
    gdf["station_name"] = gdf["station_name"].str.replace(
        " MRT STATION", "").str.replace(" LRT STATION", "").apply(convert_to_camel_case)

    return gdf[["station_name", "longitude", "latitude"]]


def scrape_mrt_and_lrt_data():
    mrt_columns = ["station_code", "station_name", "color", "line"]
    lrt_columns = ["station_code", "station_name", "line"]

    dfs = []
    for url in URLS:
        resp = requests.get(url)
        soup = BeautifulSoup(resp.content, 'html.parser')
        table = soup.find('table')
        rows = table.find_all('tr')

        # Determine if the data is for MRT or LRT based on the number of columns
        headers = mrt_columns if len(rows[0].find_all(
            'th')) == len(mrt_columns) else lrt_columns

        data = []
        for row in rows[1:]:
            cols = row.find_all(['td', 'th'])
            row_data = [col.text.strip() for col in cols]
            data.append(row_data)

        # Create a GeoDataFrame from the collected data and columns
        gdf = gpd.GeoDataFrame(data, columns=headers)
        dfs.append(gdf)

    return pd.concat(dfs, ignore_index=True)


def shp_to_csv(shp_file, csv_file):
    gdf = gpd.read_file(shp_file)
    gdf = transform_data(gdf)
    print(gdf)

    mrt_info = scrape_mrt_and_lrt_data()
    print(mrt_info)

    gdf = gdf.merge(mrt_info, how="left", on="station_name")
    print(gdf)

    # gdf = gdf[["station_name", "longitude", "latitude", "color"]]

    # Write to CSV
    gdf.to_csv(csv_file, index=False)


if __name__ == "__main__":
    URLS = ["https://mrtmapsingapore.com/mrt-stations-singapore/",
            "https://mrtmapsingapore.com/lrt-stations/"]
    SHP_FILE_PATH = "data\mrt_location_data\RapidTransitSystemStation.shp"
    CSV_FILE_PATH = "data\mrt_lrt_sg.csv"

    shp_to_csv(SHP_FILE_PATH, CSV_FILE_PATH)
    print(
        f"Conversion from Shapefile to CSV completed. CSV file saved at: {CSV_FILE_PATH}")
