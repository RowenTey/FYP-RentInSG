import logging

import numpy as np
import requests


def fetch_coordinates(location_name):
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
