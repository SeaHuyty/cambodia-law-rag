"""Vector stores for legal chunks (cosine similarity)."""
import json
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

INDEX_DIR = _PROJECT_ROOT / _cfg["index_dir"]
PERSIST_DIR = _PROJECT_ROOT / _cfg["persist_dir"]
COLLECTION_NAME = _cfg["collection_name"]
DEFAULT_TOP_K = _cfg["top_k"]


# ---- In-memory index (active) ----

@lru_cache(maxsize=1)
def load_index() -> tuple[np.ndarray, list[dict]]:
    vectors = np.load(INDEX_DIR / "vectors.npy")
    chunks = json.loads((INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
    if len(vectors) != len(chunks):
        raise ValueError(f"{INDEX_DIR}: {len(vectors)} vectors but {len(chunks)} chunks")
    return vectors, chunks


def search(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """Top-k chunks by cosine similarity: [{id, text, metadata, score}].

    Stored and query vectors are both L2-normalized, so cosine similarity is a dot product.
    """
    vectors, chunks = load_index()
    scores = vectors @ embed_query(query)
    top = np.argsort(-scores)[:top_k]
    return [
        {
            "id": chunks[i]["id"],
            "text": chunks[i]["text"],
            "metadata": chunks[i].get("metadata") or {},
            "score": float(scores[i]),
        }
        for i in top
    ]


# ---- ChromaDB (not used by the server right now) ----

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


def search_chroma(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """Same output as search(), but queries the Chroma collection."""
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
