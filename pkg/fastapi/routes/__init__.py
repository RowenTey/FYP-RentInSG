from fastapi import APIRouter

from routes import data, inference

api_router = APIRouter()

api_router.include_router(data.router, prefix="/data", tags=["data"])
api_router.include_router(inference.router, prefix="/inference", tags=["inference"])
