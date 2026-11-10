# Model comparison guideline

Goal: compare embedding models on the **same chunks and same questions**, then pick one for serving.

## Rules (so results stay comparable)
- **Don't change** `common.py` chunking, `eval_questions.json`, or `articles.json` for one model. If you must, bump `CHUNKING_VERSION` and re-run **every** model.
- Compare models only when `chunking_version` matches in `results.csv`.
- Primary metric: **MRR**, then Recall@3 / Recall@5.
- Use `DEVICE = "cpu"`. MPS runs out of memory (~3.4 GB cap) on bge-m3-sized models.
- Don't commit `vectors.npy`, `chunks.json`, `info.json` (gitignored). Commit the notebook and `results.csv`.
- Check the "Misses" table: if many models miss the same question, fix the label in `eval_questions.json` (tell the team, re-run all).

## Suggested models to try
| model | prefix | note |
|---|---|---|
| `intfloat/multilingual-e5-large` | `query: ` / `passage: ` (required) | strong multilingual retrieval, same size as bge-m3 |
| `Alibaba-NLP/gte-multilingual-base` | none | smaller/faster, 8k context; needs `trust_remote_code=True` |
| `sentence-transformers/LaBSE` | none | Khmer-supported, but built for translation matching, expect weaker; a useful low baseline |

Also worth trying: `intfloat/multilingual-e5-base` (cheaper) if the large one is too slow.

## Fine-tuning (try after the off-the-shelf baselines)
Fine-tune the best candidate (e.g. multilingual-e5, or bge-m3) on Khmer (question, article) pairs. Record it as its own row/folder (e.g. `multilingual-e5-large-ft`) next to the untuned one.
- **Never train on `eval_questions.json`.** Write a separate train set (`train_questions.json`, same format); otherwise the scores are meaningless.
- Keep the same chunks and eval set, so the fine-tuned row stays comparable.
- Tiny data (~100 pairs) overfits easily: use a low learning rate, 1-3 epochs, in-batch negatives (`MultipleNegativesRankingLoss`), and check Recall doesn't drop on the eval set.
- Fine-tuning needs more memory than inference; use a smaller model (e5-base) or a GPU/Colab.