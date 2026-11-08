"""Dense embedding of Khmer legal text with BAAI/bge-m3 (1024-dim)."""
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

CONFIG_PATH = Path(__file__).parent / "configs" / "network_configs.yml"

with open(CONFIG_PATH, encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)["embedding"]

EMBEDDING_MODEL_NAME = _cfg["model_name"]
EMBEDDING_DIM = _cfg["dim"]
EMBEDDING_BATCH_SIZE = _cfg["batch_size"]
EMBEDDING_MAX_LENGTH = _cfg["max_length"]


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Load the model once per process."""
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    model.max_seq_length = EMBEDDING_MAX_LENGTH
    return model


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed documents/chunks. Returns an (N, 1024) L2-normalized float32 array.

    Vectors are normalized, so cosine similarity equals the dot product.
    """
    if not texts:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)
    vectors = get_model().encode(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > EMBEDDING_BATCH_SIZE,
        convert_to_numpy=True,
    )
    return vectors.astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    """Embed a single user question. Returns a (1024,) normalized vector."""
    return embed_texts([query])[0]


def cosine_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """Cosine similarity between one query (D,) and many docs (N, D)."""
    q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    d = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-12)
    return d @ q
