"""Shared helpers for the model-comparison notebooks in this folder.

Every notebook does the same thing with a different model, so all the logic lives here:
    chunks = build_chunks(load_articles())
    result = run_experiment("bge_m3", "BAAI/bge-m3", chunks, questions)

Embeddings are cached on disk inside each model's own folder:
    <name>/vectors.npy   (N, dim) float32, L2-normalised, row i <-> chunks.json[i]
    <name>/chunks.json   [{id, text, embed_text, metadata}, ...]
    <name>/info.json     model, dim, timings, chunking version, content hash
(<name> is the folder next to this file, e.g. bge-m3/; results.csv stays here, shared.)
"""
import csv
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ARTICLES_PATH = HERE.parent.parent / "data" / "articles.json"
QUESTIONS_PATH = HERE / "eval_questions.json"
RESULTS_PATH = HERE / "results.csv"

# Bump when build_chunks() changes, so stale caches are not reused.
CHUNKING_VERSION = "v1-article-maxchars1500"
MAX_CHARS = 1500

_KH = str.maketrans("0123456789", "០១២៣៤៥៦៧៨៩")


# --------------------------------------------------------------------- data
def load_articles() -> list[dict]:
    return json.loads(ARTICLES_PATH.read_text(encoding="utf-8"))


def load_questions() -> list[dict]:
    return json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))["questions"]


def _split_long(text: str, max_chars: int) -> list[str]:
    """Greedy-pack lines into pieces <= max_chars; split oversize lines at '។'."""
    pieces: list[str] = []
    for line in (l for l in text.splitlines() if l.strip()):
        if len(line) <= max_chars:
            pieces.append(line)
        else:
            sentences = [s + "។" for s in line.split("។") if s.strip()]
            pieces.extend(sentences or [line])
    parts, cur = [], ""
    for p in pieces:
        if cur and len(cur) + len(p) + 1 > max_chars:
            parts.append(cur)
            cur = p
        else:
            cur = f"{cur}\n{p}" if cur else p
    if cur:
        parts.append(cur)
    return parts


def build_chunks(articles: list[dict], max_chars: int = MAX_CHARS) -> list[dict]:
    """One chunk per article; long articles are split into several chunks.

    - `text`        : raw text, shown to the user as the source.
    - `embed_text`  : header (chapter + article + title) + text, what gets embedded.
    """
    chunks = []
    for a in articles:
        num = str(a["article_number"]).translate(_KH)
        amended = " ថ្មី" if a["is_amended"] else ""
        header = (
            f"ជំពូកទី{str(a['chapter_number']).translate(_KH)} {a['chapter_title']} · "
            f"មាត្រា{num}{amended} {a['article_title']}"
        )
        parts = _split_long(a["text"], max_chars) if len(a["text"]) > max_chars else [a["text"]]
        for k, part in enumerate(parts):
            chunks.append({
                "id": a["id"] if len(parts) == 1 else f"{a['id']}#{k + 1}",
                "text": part,
                "embed_text": f"{header}\n{part}",
                "metadata": {
                    "article_id": a["id"],
                    "article_number": a["article_number"],
                    "is_amended": a["is_amended"],
                    "article_title": a["article_title"],
                    "chapter_number": a["chapter_number"],
                    "chapter_title": a["chapter_title"],
                    "part": k + 1,
                    "parts": len(parts),
                },
            })
    return chunks


# ---------------------------------------------------------------- embedding
def _content_hash(chunks: list[dict]) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update(c["id"].encode())
        h.update(c["embed_text"].encode())
    return h.hexdigest()[:16]


def _load_model(model_name: str, max_length: int):
    from sentence_transformers import SentenceTransformer  # heavy import, keep lazy

    model = SentenceTransformer(model_name)
    model.max_seq_length = max_length
    return model


def embed_chunks(
    name: str,
    model_name: str,
    chunks: list[dict],
    *,
    passage_prefix: str = "",
    max_length: int = 1024,
    batch_size: int = 16,
    force: bool = False,
) -> tuple[np.ndarray, dict]:
    """Embed chunks (or load the cache). Returns (vectors, info)."""
    out = HERE / name
    vec_path, info_path = out / "vectors.npy", out / "info.json"
    chash = _content_hash(chunks) + passage_prefix

    if not force and vec_path.exists() and info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        if info.get("chunking_version") == CHUNKING_VERSION and info.get("content_hash") == chash:
            vectors = np.load(vec_path)
            if len(vectors) == len(chunks):
                print(f"[cache] loaded {vectors.shape} from {vec_path}")
                return vectors, info

    model = _load_model(model_name, max_length)
    t0 = time.time()
    vectors = model.encode(
        [passage_prefix + c["embed_text"] for c in chunks],
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    elapsed = time.time() - t0

    out.mkdir(parents=True, exist_ok=True)
    np.save(vec_path, vectors)
    (out / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    info = {
        "name": name,
        "model_name": model_name,
        "dim": int(vectors.shape[1]),
        "n_chunks": int(vectors.shape[0]),
        "max_length": max_length,
        "passage_prefix": passage_prefix,
        "chunking_version": CHUNKING_VERSION,
        "content_hash": chash,
        "embed_seconds": round(elapsed, 2),
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"[saved] {vectors.shape} in {elapsed:.1f}s -> {out}")
    return vectors, info


def embed_queries(
    model_name: str, questions: list[dict], *, query_prefix: str = "", max_length: int = 1024
) -> np.ndarray:
    model = _load_model(model_name, max_length)
    return model.encode(
        [query_prefix + q["question"] for q in questions],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)


# --------------------------------------------------------------- evaluation
def evaluate(
    vectors: np.ndarray,
    chunks: list[dict],
    query_vectors: np.ndarray,
    questions: list[dict],
    ks: tuple[int, ...] = (1, 3, 5),
) -> dict:
    """Article-level Recall@K and MRR.

    Chunks are ranked by cosine similarity (dot product, vectors are normalised), then
    collapsed to unique article ids in rank order. A question counts as a hit at K if any
    relevant article is within the top K articles.
    """
    sims = query_vectors @ vectors.T  # (Q, N)
    article_of = [c["metadata"]["article_id"] for c in chunks]

    hits = {k: 0 for k in ks}
    rr_sum = 0.0
    details = []
    for qi, q in enumerate(questions):
        ranked, seen = [], set()
        for idx in np.argsort(-sims[qi]):
            aid = article_of[idx]
            if aid not in seen:
                seen.add(aid)
                ranked.append(aid)
        relevant = set(q["relevant_ids"])
        rank = next((r for r, aid in enumerate(ranked, 1) if aid in relevant), None)
        for k in ks:
            hits[k] += rank is not None and rank <= k
        rr_sum += 1.0 / rank if rank else 0.0
        details.append({"id": q["id"], "question": q["question"], "relevant": q["relevant_ids"],
                        "rank": rank, "top5": ranked[:5]})

    n = len(questions)
    metrics = {f"recall@{k}": round(hits[k] / n, 4) for k in ks}
    metrics["mrr"] = round(rr_sum / n, 4)
    return {"metrics": metrics, "details": details}


def append_result(info: dict, metrics: dict) -> None:
    """Upsert one row per experiment name into results.csv."""
    row = {
        "name": info["name"],
        "model_name": info["model_name"],
        "dim": info["dim"],
        "n_chunks": info["n_chunks"],
        **metrics,
        "embed_seconds": info["embed_seconds"],
        "chunking_version": info["chunking_version"],
        "date": datetime.now().isoformat(timespec="seconds"),
    }
    rows = []
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, encoding="utf-8", newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["name"] != row["name"]]
    rows.append({k: str(v) for k, v in row.items()})
    with open(RESULTS_PATH, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerows(rows)


def run_experiment(
    name: str,
    model_name: str,
    chunks: list[dict],
    questions: list[dict],
    *,
    query_prefix: str = "",
    passage_prefix: str = "",
    max_length: int = 1024,
    batch_size: int = 16,
    force: bool = False,
) -> dict:
    vectors, info = embed_chunks(
        name, model_name, chunks,
        passage_prefix=passage_prefix, max_length=max_length, batch_size=batch_size, force=force,
    )
    qv = embed_queries(model_name, questions, query_prefix=query_prefix, max_length=max_length)
    result = evaluate(vectors, chunks, qv, questions)
    append_result(info, result["metrics"])
    result["info"] = info
    return result
