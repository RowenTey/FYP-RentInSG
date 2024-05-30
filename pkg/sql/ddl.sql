CREATE TABLE fyp_rent_in_sg.plan_area_mapping (
    plan_area_id UTINYINT PRIMARY KEY NOT NULL,
    plan_area VARCHAR NOT NULL,
    polygon VARCHAR NOT NULL
);

CREATE TABLE fyp_rent_in_sg.mrt_info (
    station_code VARCHAR PRIMARY KEY NOT NULL,
    station_name VARCHAR NOT NULL,
    longitude DECIMAL(9,6) NOT NULL,
    latitude DECIMAL(9,6) NOT NULL,
    color VARCHAR,
    line VARCHAR NOT NULL,
);

CREATE TABLE fyp_rent_in_sg.hawker_centre_info (
    hawker_id INT PRIMARY KEY NOT NULL,
    name VARCHAR NOT NULL,
    building_name VARCHAR,
    street_name VARCHAR,
    postal_code VARCHAR,
    longitude DECIMAL(9,6),
    latitude DECIMAL(9,6)
);

CREATE TABLE fyp_rent_in_sg.supermarket_info (
    supermarket_id INT PRIMARY KEY NOT NULL,
    name VARCHAR NOT NULL,
    street_name VARCHAR,
    postal_code VARCHAR,
    longitude DECIMAL(9,6),
    latitude DECIMAL(9,6)
);

CREATE TABLE fyp_rent_in_sg.primary_school_info (
    school_id INT PRIMARY KEY NOT NULL,
    name VARCHAR NOT NULL,
    area VARCHAR,
    longitude DECIMAL(9,6),
    latitude DECIMAL(9,6)
);

CREATE TABLE fyp_rent_in_sg.mall_info (
    mall_id INT PRIMARY KEY NOT NULL,
    name VARCHAR NOT NULL,
    longitude DECIMAL(9,6),
    latitude DECIMAL(9,6)
);

CREATE TABLE fyp_rent_in_sg.property_listing (
    listing_id PRIMARY KEY VARCHAR NOT NULL,
    property_name VARCHAR NOT NULL,
    district VARCHAR NOT NULL,
    price INT NOT NULL,
    bedroom INT NOT NULL,
    bathroom INT NOT NULL,
    dimensions INT NOT NULL,
    address VARCHAR,
    price_per_sqft FLOAT,
    floor_level VARCHAR,
    furnishing VARCHAR,
    facing VARCHAR,
    built_year INT,
    tenure VARCHAR,
    property_type VARCHAR,
    url VARCHAR NOT NULL,
    facilities VARCHAR,
    latitude FLOAT NOT NULL,
    longitude FLOAT NOT NULL,
    building_name VARCHAR,
    nearest_mrt VARCHAR,
    distance_to_mrt_in_m FLOAT,
    district_id VARCHAR,
    nearest_hawker VARCHAR,
    distance_to_hawker_in_m FLOAT,
    nearest_supermarket VARCHAR,
    distance_to_supermarket_in_m FLOAT,
    nearest_sch VARCHAR,
    distance_to_sch_in_m FLOAT,
    nearest_mall VARCHAR,
    distance_to_mall_in_m FLOAT,
    is_whole_unit BOOLEAN,
    has_pool BOOLEAN,
    has_gym BOOLEAN,
    fingerprint VARCHAR NOT NULL,
    source VARCHAR NOT NULL,
    scraped_on DATETIME NOT NULL,
    last_updated DATETIME NOT NULL,
);

CREATE TABLE fyp_rent_in_sg.rental_price_history (  
    listing_id VARCHAR NOT NULL,
    price INT NOT NULL,
    timestamp DATETIME NOT NULL,
    PRIMARY KEY (listing_id, timestamp),
);