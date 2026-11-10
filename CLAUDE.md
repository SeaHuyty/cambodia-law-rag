# CLAUDE.md

Khmer road-traffic-law Q&A: retrieve relevant law articles (RAG), then answer in plain Khmer. Pipeline is in `doc/methodology.md`.

## Layout
- `app/main.py`, `app/database.py` (ChromaDB), `app/engine/embedding.py`: FastAPI serving + embedding.
- `app/engine/configs/network_configs.yml`: serving model name/dim/max_length, Chroma settings.
- `app/data/chapter_*.md`: source text. `app/data/articles.json`: parsed articles (generated).
- `app/engine/scripts/prepare_dataset.py`: markdown -> `articles.json`.
- `app/engine/model/`: model-comparison notebooks. **Read `doc/guideline.md` before touching anything there.**

## Commands
Use the venv's python explicitly (its `pip`/script shebangs are stale):
- Install: `venv/bin/python -m pip install -r requirements.txt`
- Regenerate articles: `venv/bin/python -m app.engine.scripts.prepare_dataset` (from project root)
- Serve: `venv/bin/python -m uvicorn app.main:app`
- Notebooks: kernel `Python (khmer-doc)`; run on CPU (`DEVICE = "cpu"`), MPS runs out of memory.

## Rules
- Text is Khmer: keep UTF-8, don't "normalize" or transliterate it; article numbers use Khmer digits in source, ints in JSON.
- Chunking, `eval_questions.json` and `articles.json` are shared across all model experiments. Changing them invalidates `results.csv`; bump `CHUNKING_VERSION` and re-run every model.
- Don't commit generated embeddings (`vectors.npy`, `chunks.json`, `info.json`), `data/chroma`, `.env`, or `venv/`.
- Notebooks: clear outputs/tracebacks before committing.
- Answers must come only from retrieved law text and cite the article (មាត្រា); never invent law.
