DISTRICTS = {
    "D01 Boat Quay / Raffles Place / Marina": "D01",
    "D02 Chinatown / Tanjong Pagar": "D02",
    "D03 Alexandra / Commonwealth": "D03",
    "D04 Harbourfront / Telok Blangah": "D04",
    "D05 Buona Vista / West Coast / Clementi": "D05",
    "D06 City Hall / Clarke Quay": "D06",
    "D07 Beach Road / Bugis / Rochor": "D07",
    "D08 Farrer Park / Serangoon Rd": "D08",
    "D09 Orchard / River Valley": "D09",
    "D10 Tanglin / Holland": "D10",
    "D11 Newton / Novena": "D11",
    "D12 Balestier / Toa Payoh": "D12",
    "D13 Macpherson / Potong Pasir": "D13",
    "D14 Eunos / Geylang / Paya Lebar": "D14",
    "D15 East Coast / Marine Parade": "D15",
    "D16 Bedok / Upper East Coast": "D16",
    "D17 Changi Airport / Changi Village": "D17",
    "D18 Pasir Ris / Tampines": "D18",
    "D19 Hougang / Punggol / Sengkang": "D19",
    "D20 Ang Mo Kio / Bishan / Thomson": "D20",
    "D21 Clementi Park / Upper Bukit Timah": "D21",
    "D22 Boon Lay / Jurong / Tuas": "D22",
    "D23 Bukit Batok / Bukit Panjang / Choa Chu Kang": "D23",
    "D24 Lim Chu Kang / Tengah": "D24",
    "D25 Admiralty / Woodlands": "D25",
    "D26 Mandai / Upper Thomson": "D26",
    "D27 Sembawang / Yishun": "D27",
    "D28 Seletar / Yio Chu Kang": "D28",
}

PROPERTY_TYPES = [
    "Condo",
    "Apartment",
    "HDB",
    "Conservation House",
    "Bungalow",
    "Terraced House",
    "HDB Executive",
    "Shophouse",
    "Corner Terrace",
    "Cluster House",
    "Semi-Detached House",
    "Executive Condo",
    "Landed",
    "Townhouse",
    "Walk-up",
    "Executive HDB",
]

FURNISHING = ["Partial", "Fully", "Unfurnished", "Flexible"]

FACING = [
    "South",
    "North",
    "South East",
    "South West",
    "North East",
    "West",
    "East",
    "North West",
    "North South",
]

FLOOR_LEVEL = ["Mid", "High", "Penthouse", "Low", "Ground", "Top"]

TENURE = ["Leasehold", "Freehold"]

# obtained from median / mode values from the dataset
DEFAULT_VALUES = {
    "bedrooms": 1,
    "bathrooms": 1,
    "dimensions": 0,
    "furnishing": "Fully",
    "facing": "North",
    "floor_level": "High",
    "tenure": "freehold",
    "property_type": "Condo",
    "address": "",
    "built_year": 2008,
    "has_gym": False,
    "has_pool": False,
    "is_whole_unit": False,
    "distance_to_mrt_in_m": 463.000000,
    "distance_to_hawker_in_m": 651.937988,
    "distance_to_supermarket_in_m": 651.937988,
    "distance_to_sch_in_m": 564.494995,
    "distance_to_mall_in_m": 631.247009,
    "latitude": None,
    "longitude": None
}

REQUIRED_FIELDS = ["bedrooms", "bathrooms", "dimensions", "district_id"]
