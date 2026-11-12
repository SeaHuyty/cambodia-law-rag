"""Embed app/data/articles.json into the Chroma collection (rebuilds it from scratch).

Run from project root: venv/bin/python -m app.engine.scripts.index_articles
"""
import json
from pathlib import Path

from app.database import add_chunks, get_collection, reset_collection

ARTICLES = Path(__file__).parent.parent.parent / "data" / "articles.json"


def main() -> None:
    articles = json.loads(ARTICLES.read_text(encoding="utf-8"))
    chunks = [
        {
            "id": a["id"],
            "text": a["text"],
            "metadata": {
                "article_number": a["article_number"],
                "article_title": a.get("article_title") or "",
                "chapter_number": a.get("chapter_number") or 0,
                "chapter_title": a.get("chapter_title") or "",
                "is_amended": bool(a.get("is_amended")),
            },
        }
        for a in articles
    ]
    reset_collection()
    n = add_chunks(chunks)
    print(f"indexed {n} articles; collection count = {get_collection().count()}")


if __name__ == "__main__":
    main()
