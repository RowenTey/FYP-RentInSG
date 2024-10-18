import uvicorn
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from routes import api_router
from utils.motherduckdb import db
from utils.mlflow_model import model
from cashews import cache
from starlette.middleware.cors import CORSMiddleware


@asynccontextmanager
async def lifespan(_: FastAPI):
    # configure as in-memory cache
    cache.setup("mem://")

    db.connect()
    db.check_connection()

    model.initialize(db)

    yield

    db.close()

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/ping")
def root():
    return "pong"


app.include_router(api_router)

if __name__ == "__main__":
    uvicorn.run(
        app="main:app",
        host="0.0.0.0",
        port=8000,
        log_level=logging.INFO,
    )
