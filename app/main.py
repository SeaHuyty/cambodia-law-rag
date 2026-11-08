from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.database import get_collection
from app.engine.embedding import get_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load heavy resources once at startup instead of on the first request.
    get_model()
    get_collection()
    yield


app = FastAPI(
    title="Khmer Document Retrieval API",
    description="An API for retrieving Khmer documents based on user queries.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def read_health():
    return {"status": "healthy"}
