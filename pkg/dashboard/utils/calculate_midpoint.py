from components.constants import DISTRICTS
import json


def calculate_central_coordinate(area, district_coordinates):
    total_lat = 0
    total_lon = 0
    count = 0

    for district in area:
        if district in district_coordinates:
            lat, lon = district_coordinates[district]
            total_lat += lat
            total_lon += lon
            count += 1

    if count == 0:
        return None

    return (total_lat / count, total_lon / count)


district_coordinates = {}
with open("coords.json", "r") as f:
    district_coordinates = json.load(f)

area_map = {}
for district in DISTRICTS:
    area = [d.strip() for d in district[3:].split("/")]
    coord = calculate_central_coordinate(area, district_coordinates)
    print(area, coord)
    area_map[district] = coord

print(json.dumps(area_map, indent=4))
with open("cleaned_coords.json", "w") as f:
    json.dump(area_map, f)
