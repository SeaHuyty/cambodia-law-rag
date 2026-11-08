"""ChromaDB vector store for legal chunks (cosine similarity)."""
from functools import lru_cache
from pathlib import Path

import chromadb
import numpy as np
import yaml

from app.engine.embedding import embed_query, embed_texts

_ENGINE_DIR = Path(__file__).parent / "engine"
_PROJECT_ROOT = Path(__file__).parent.parent

with open(_ENGINE_DIR / "configs" / "network_configs.yml", encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)["vector_store"]

PERSIST_DIR = _PROJECT_ROOT / _cfg["persist_dir"]
COLLECTION_NAME = _cfg["collection_name"]
DEFAULT_TOP_K = _cfg["top_k"]


@lru_cache(maxsize=1)
def get_collection() -> chromadb.Collection:
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(PERSIST_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def add_chunks(chunks: list[dict]) -> int:
    """Embed and upsert chunks.

    Each chunk: {"id": str, "text": str, "metadata": {"article": ..., "chapter": ..., "page": ...}}.
    Re-adding an existing id overwrites it, so indexing is idempotent.
    """
    if not chunks:
        return 0
    vectors: np.ndarray = embed_texts([c["text"] for c in chunks])
    get_collection().upsert(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        embeddings=vectors.tolist(),
        metadatas=[c.get("metadata") or None for c in chunks],
    )
    return len(chunks)


def search(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """Return top-k chunks by cosine similarity: [{id, text, metadata, score}]."""
    collection = get_collection()
    if collection.count() == 0:
        return []
    res = collection.query(
        query_embeddings=[embed_query(query).tolist()],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    return [
        {
            "id": id_,
            "text": doc,
            "metadata": meta or {},
            "score": 1.0 - dist,  # chroma cosine distance = 1 - similarity
        }
        for id_, doc, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        )
    ]


def reset_collection() -> None:
    """Drop and recreate the collection (for re-indexing from scratch)."""
    get_collection.cache_clear()
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(PERSIST_DIR))
    client.delete_collection(COLLECTION_NAME)
    get_collection()
