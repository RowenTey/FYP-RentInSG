from components.constants import DISTRICTS
from geopy.geocoders import Nominatim
import json
import time


def get_coordinates(district):
    geolocator = Nominatim(user_agent="test")
    location = geolocator.geocode(f"{district}, Singapore")
    print(location)
    return (location.latitude, location.longitude)


memo = {}
for district in DISTRICTS:
    districts = district[3:].split("/")
    for d in districts:
        coords = get_coordinates(d.strip())
        print(f"{d.strip()}: {coords}")
        memo[d.strip()] = coords
        time.sleep(60)

print(json.dumps(memo, indent=4))
with open("coords.json", "w") as f:
    json.dump(memo, f)
