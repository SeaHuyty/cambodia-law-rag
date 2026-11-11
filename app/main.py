from contextlib import asynccontextmanager

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.database import get_collection, search
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


STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


@app.post("/api/ask")
async def ask(req: AskRequest):
    return {"answer": None, "sources": search(req.question.strip())}


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(STATIC_DIR / "index.html")
