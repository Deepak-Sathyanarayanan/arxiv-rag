import arxiv
from pathlib import Path

TOPIC = "large language models"
MAX_RESULTS = 50
OUT_DIR = Path("data/pdfs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

search = arxiv.Search(
    query=TOPIC,
    max_results=MAX_RESULTS,
    sort_by=arxiv.SortCriterion.SubmittedDate
)

for result in search.results():
    safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in result.title)[:120]
    pdf_path = OUT_DIR / f"{safe_title}.pdf"
    if not pdf_path.exists():
        print(f"Downloading: {result.title}")
        result.download_pdf(dirpath=str(OUT_DIR), filename=pdf_path.name)
