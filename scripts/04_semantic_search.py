
from pathlib import Path
import csv
import time
from sentence_transformers import SentenceTransformer, util

ROOT = Path(__file__).resolve().parents[1]
DOC_DIR = ROOT / "data" / "documents"
QUERY_FILE = ROOT / "data" / "queries.csv"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

files = sorted(DOC_DIR.glob("*.txt"))
if not files:
    raise FileNotFoundError(f"No .txt documents found in {DOC_DIR}")

documents = [
    (path.name, path.read_text(encoding="utf-8").strip())
    for path in files
]
documents = [(name, text) for name, text in documents if text]

with QUERY_FILE.open(encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    if not reader.fieldnames or "query" not in reader.fieldnames:
        raise ValueError("queries.csv must have a column named 'query'")
    queries = [row["query"].strip() for row in reader if row["query"].strip()]

if not queries:
    raise ValueError("No queries found in queries.csv")

print(f"Documents: {len(documents)} | Queries: {len(queries)}")
start = time.perf_counter()
model = SentenceTransformer(MODEL_NAME, device="cpu")
load_time = time.perf_counter() - start

start = time.perf_counter()
doc_embeddings = model.encode(
    [text for _, text in documents],
    normalize_embeddings=True,
    convert_to_tensor=True,
    show_progress_bar=True,
)
embedding_time = time.perf_counter() - start

rows = []
search_times = []

for query in queries:
    start = time.perf_counter()
    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
        convert_to_tensor=True,
    )
    scores = util.cos_sim(query_embedding, doc_embeddings)[0]
    top = scores.topk(k=min(3, len(documents)))
    elapsed = time.perf_counter() - start
    search_times.append(elapsed)

    for rank, index in enumerate(top.indices.tolist(), start=1):
        rows.append({
            "query": query,
            "rank": rank,
            "document": documents[index][0],
            "similarity": round(scores[index].item(), 4),
            "search_seconds": round(elapsed, 5),
        })

output = RESULTS / "semantic_search_results.csv"
with output.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "query", "rank", "document", "similarity", "search_seconds"
    ])
    writer.writeheader()
    writer.writerows(rows)

print(f"Model load time: {load_time:.3f} seconds")
print(f"Document embedding time: {embedding_time:.3f} seconds")
print(f"Average query search time: {sum(search_times)/len(search_times):.5f} seconds")
print(f"Saved top-three results to: {output}")
