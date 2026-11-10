"""Parse app/data/*.md into a structured JSON (one record per article).

Run from the project root:
    python -m app.engine.scripts.prepare_dataset

Output: app/data/articles.json  -> reviewed by hand, then chunked/embedded.
"""
import json
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
OUT_PATH = DATA_DIR / "articles.json"

KH_DIGITS = str.maketrans("០១២៣៤៥៦៧៨៩", "0123456789")

CHAPTER_RE = re.compile(r"^[#*\s]*ជំពូកទី\s*([០-៩]+)[#*\s]*$")
ARTICLE_RE = re.compile(r"^\**\s*មាត្រា\s*([០-៩]+)\s*(?:\.{2,}\s*(ថ្មី))?\s*\**$")
CLAUSE_RE = re.compile(r"^\s*(\d+)\.\s+")
OVERVIEW_TITLE_RE = re.compile(r"^មាត្រា\s*([០-៩]+)\s*៖\s*(.+)$")


def to_int(kh: str) -> int:
    return int(kh.translate(KH_DIGITS))


def clean_heading(line: str) -> str:
    return line.strip().strip("#* ").strip()


def load_article_titles() -> dict[int, str]:
    """Article titles only exist in overview.md (e.g. 'មាត្រា១ ៖ គោលបំណង')."""
    titles = {}
    for line in (DATA_DIR / "overview.md").read_text(encoding="utf-8").splitlines():
        m = OVERVIEW_TITLE_RE.match(line.strip())
        if m:
            titles[to_int(m.group(1))] = m.group(2).strip()
    return titles


def split_clauses(body: str) -> list[str]:
    """Split on top-level numbered clauses ('1. ...'); [] if the article has none."""
    clauses, current = [], None
    for line in body.splitlines():
        if CLAUSE_RE.match(line):
            if current is not None:
                clauses.append("\n".join(current).strip())
            current = [line]
        elif current is not None:
            current.append(line)
    if current is not None:
        clauses.append("\n".join(current).strip())
    return clauses if len(clauses) > 1 else []


def parse_chapter_file(path: Path, titles: dict[int, str]) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    chapter_no, chapter_title = None, None
    articles: list[dict] = []
    current: dict | None = None
    body: list[str] = []

    def flush():
        nonlocal current, body
        if current is not None:
            text = "\n".join(body).strip()
            current["text"] = text
            current["clauses"] = split_clauses(text)
            articles.append(current)
        current, body = None, []

    i = 0
    while i < len(lines):
        line = lines[i]
        if m := CHAPTER_RE.match(line):
            flush()
            chapter_no = to_int(m.group(1))
            # chapter title = next non-empty line
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            chapter_title = clean_heading(lines[j])
            i = j + 1
            continue
        if m := ARTICLE_RE.match(line.strip()):
            flush()
            number, is_new = to_int(m.group(1)), bool(m.group(2))
            current = {
                "id": f"article-{number}" + ("-new" if is_new else ""),
                "article_number": number,
                "is_amended": is_new,  # 'ថ្មី' = replaced/new version of the article
                "article_title": titles.get(number),
                "chapter_number": chapter_no,
                "chapter_title": chapter_title,
                "source_file": path.name,
            }
        elif current is not None:
            body.append(line)
        i += 1
    flush()
    return articles


def main() -> None:
    titles = load_article_titles()
    files = sorted(DATA_DIR.glob("chapter_*.md"), key=lambda p: int(p.stem.split("_")[1]))
    articles = [a for f in files for a in parse_chapter_file(f, titles)]

    ids = [a["id"] for a in articles]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise SystemExit(f"Duplicate ids: {sorted(dupes)}")

    OUT_PATH.write_text(json.dumps(articles, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(articles)} articles from {len(files)} chapters -> {OUT_PATH}")


if __name__ == "__main__":
    main()
