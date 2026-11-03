from fastapi import FastAPI

app = FastAPI(
    title="Khmer Document Retrieval API",
    description="An API for retrieving Khmer documents based on user queries.",
    version="1.0.0",
)

@app.get("/health")
async def read_health():
    return {"status": "healthy"}