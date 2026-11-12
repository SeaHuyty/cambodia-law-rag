from fastapi import APIRouter

from app.database import search
from app.services.llm import REFUSAL, generate_answer
from app.models.model import AskRequest

router = APIRouter(prefix="/api")


@router.post("/ask")
async def ask(req: AskRequest):
    question = req.question.strip()
    sources = search(question)
    try:
        answer = await generate_answer(question, sources)
    except Exception:
        # Still return the retrieved articles if Gemini is unavailable or misconfigured.
        answer = None
    return {"answer": answer, "refused": answer == REFUSAL, "sources": sources}
