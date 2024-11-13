from datetime import datetime
from lib.constants.location_constants import DISTRICTS
from typing import Annotated, Any, Optional, TypeVar
from pydantic import BaseModel, BeforeValidator, field_validator


def coerce_nan_to_none(x: Any) -> Any:
    from math import isnan
    if isinstance(x, float) and isnan(x):
        return None
    return x


T = TypeVar('T')
# Custom type to coerce nan to None
# from https://github.com/pydantic/pydantic/issues/1779#issuecomment-1524062596
NoneOrNan = Annotated[Optional[T], BeforeValidator(coerce_nan_to_none)]


class PropertyListing(BaseModel):
    listing_id: str
    property_name: str
    district: str
    price: int
    bedroom: int
    bathroom: int
    dimensions: int
    address: Optional[str]
    price_per_sqft: Optional[float]
    floor_level: Optional[NoneOrNan[str]]
    furnishing: Optional[NoneOrNan[str]]
    facing: Optional[NoneOrNan[str]]
    built_year: Optional[int]
    tenure: Optional[NoneOrNan[str]]
    property_type: Optional[NoneOrNan[str]]
    url: str
    facilities: Optional[str]  # stringified list
    latitude: float
    longitude: float
    building_name: Optional[str]
    nearest_mrt: Optional[str]
    distance_to_mrt_in_m: Optional[float]
    district_id: Optional[str]
    nearest_hawker: Optional[str]
    distance_to_hawker_in_m: Optional[float]
    nearest_supermarket: Optional[str]
    distance_to_supermarket_in_m: Optional[float]
    nearest_sch: Optional[str]
    distance_to_sch_in_m: Optional[float]
    nearest_mall: Optional[str]
    distance_to_mall_in_m: Optional[float]
    is_whole_unit: Optional[bool]
    has_pool: Optional[bool]
    has_gym: Optional[bool]
    fingerprint: str
    source: str
    scraped_on: datetime
    last_updated: datetime

    @field_validator('district_id')
    def validate_district_id(cls, v):
        if v not in DISTRICTS:
            raise ValueError(f'Invalid district_id: {v}')
        return v

    @field_validator('bathroom')
    def validate_bathroom(cls, v):
        if v <= 0:
            raise ValueError('Number of bathrooms must be greater than 0')
        return v

    @field_validator('dimensions')
    def validate_dimensions(cls, v):
        if v <= 0:
            raise ValueError('Dimensions must be greater than 0')
        return v
