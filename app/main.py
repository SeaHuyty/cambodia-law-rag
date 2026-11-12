from contextlib import asynccontextmanager

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.database import get_collection
from app.api.api import router as api_router
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

app.include_router(api_router)


@app.get("/health")
async def read_health():
    return {"status": "healthy"}


STATIC_DIR = Path(__file__).parent / "static"

app.mount("/static", StaticFiles(directory=STATIC_DIR, html=True), name="static")

@app.get("/", include_in_schema=False)
async def index():
    # Redirect so the page's relative style.css/app.js resolve under /static/.
    return RedirectResponse("/static/")
