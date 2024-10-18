from pydantic import BaseModel
from typing import Optional


class PredictionFormData(BaseModel):
    bedroom: int
    bathroom: int
    dimensions: int
    district_id: str
    property_type: str
    address: Optional[str]
    built_year: Optional[int]
    furnishing: Optional[str]
    floor_level: Optional[str]
    facing: Optional[str]
    tenure: Optional[str]
    has_gym: Optional[bool]
    has_pool: Optional[bool]
    is_whole_unit: Optional[bool]
