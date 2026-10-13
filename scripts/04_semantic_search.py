
from pathlib import Path
import csv
import time
from sentence_transformers import SentenceTransformer, util

# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DOC_DIR = ROOT / "data" / "documents"
QUERY_FILE = ROOT / "data" / "queries.csv"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EXPECTED_DOCUMENT_COUNT = 18
TOP_K = 3


# --------------------------------------------------
# 2. Load and validate documents
# --------------------------------------------------

files = sorted(DOC_DIR.glob("*.txt"))

documents = [
    (
        path.stem.split("_", 1)[0],
        path.stem,
        path.read_text(encoding="utf-8").strip(),
    )
    for path in files
]

documents = [
    (doc_id, filename, text)
    for doc_id, filename, text in documents
    if text
]

if len(documents) != EXPECTED_DOCUMENT_COUNT:
    raise ValueError(
        f"Expected {EXPECTED_DOCUMENT_COUNT} non-empty documents, "
        f"but loaded {len(documents)} from {DOC_DIR}"
    )

document_ids = [doc_id for doc_id, _, _ in documents]

if len(document_ids) != len(set(document_ids)):
    raise ValueError(
        "Duplicate document IDs found. Check document filenames."
    )


# --------------------------------------------------
# 3. Load and validate queries
# --------------------------------------------------

with QUERY_FILE.open(
    encoding="utf-8-sig", newline=""
) as f:
    reader = csv.DictReader(f)

    required_columns = {
        "query_id",
        "question",
        "expected_doc_ids",
    }

    if (
        not reader.fieldnames
        or not required_columns.issubset(reader.fieldnames)
    ):
        raise ValueError(
            "queries.csv must contain these columns: "
            "query_id, question, expected_doc_ids. "
            f"Found: {reader.fieldnames}"
        )

    queries = []

    for row in reader:
        query_id = (row.get("query_id") or "").strip()
        question = (row.get("question") or "").strip()
        expected_raw = (
            row.get("expected_doc_ids") or ""
        ).strip()

        if not query_id or not question:
            continue

        expected_ids = {
            doc_id.strip()
            for doc_id in expected_raw.split(";")
            if doc_id.strip()
        }

        if not expected_ids:
            raise ValueError(
                f"No expected document IDs for {query_id}"
            )

        unknown_ids = expected_ids - set(document_ids)

        if unknown_ids:
            raise ValueError(
                f"{query_id} contains expected IDs not found "
                f"in the documents: {sorted(unknown_ids)}"
            )

        queries.append({
            "query_id": query_id,
            "question": question,
            "expected_doc_ids": expected_raw,
            "expected_ids": expected_ids,
        })

if not queries:
    raise ValueError("No valid queries found in queries.csv")

query_ids = [item["query_id"] for item in queries]

if len(query_ids) != len(set(query_ids)):
    raise ValueError("Duplicate query IDs found in queries.csv")


# --------------------------------------------------
# 4. Load model and embed all documents once
# --------------------------------------------------

print(f"Documents loaded: {len(documents)}")
print(f"Queries loaded: {len(queries)}")

start = time.perf_counter()
model = SentenceTransformer(MODEL_NAME, device="cpu")
load_time = time.perf_counter() - start

start = time.perf_counter()

doc_embeddings = model.encode(
    [text for _, _, text in documents],
    normalize_embeddings=True,
    convert_to_tensor=True,
    show_progress_bar=True,
)

embedding_time = time.perf_counter() - start


# --------------------------------------------------
# 5. Search each query against all documents
# --------------------------------------------------

result_rows = []
search_times = []

for item in queries:
    start = time.perf_counter()

    query_embedding = model.encode(
        item["question"],
        normalize_embeddings=True,
        convert_to_tensor=True,
    )

    scores = util.cos_sim(
        query_embedding, doc_embeddings
    )[0]

    top = scores.topk(
        k=min(TOP_K, len(documents))
    )

    elapsed = time.perf_counter() - start
    search_times.append(elapsed)

    for rank, index in enumerate(
        top.indices.tolist(), start=1
    ):
        doc_id, filename, _ = documents[index]

        result_rows.append({
            "query_id": item["query_id"],
            "question": item["question"],
            "expected_doc_ids": item["expected_doc_ids"],
            "rank": rank,
            "document_id": doc_id,
            "document": filename,
            "similarity": round(
                scores[index].item(), 4
            ),
            "search_seconds": round(elapsed, 5),
        })


# --------------------------------------------------
# 6. Save ranked search results
# --------------------------------------------------

results_file = RESULTS / "semantic_search_results.csv"

result_fields = [
    "query_id",
    "question",
    "expected_doc_ids",
    "rank",
    "document_id",
    "document",
    "similarity",
    "search_seconds",
]

with results_file.open(
    "w", encoding="utf-8", newline=""
) as f:
    writer = csv.DictWriter(
        f, fieldnames=result_fields
    )
    writer.writeheader()
    writer.writerows(result_rows)


# --------------------------------------------------
# 7. Calculate Recall@3 and MRR automatically
# --------------------------------------------------

metrics_rows = []

for item in queries:
    query_id = item["query_id"]
    expected_ids = item["expected_ids"]

    ranked = [
        row for row in result_rows
        if row["query_id"] == query_id
    ]

    ranked.sort(key=lambda row: row["rank"])

    retrieved_ids = {
        row["document_id"] for row in ranked
    }

    found_count = len(expected_ids & retrieved_ids)

    # Recall@3 = relevant expected documents retrieved
    # divided by the total number of expected documents.
    recall_at_3 = found_count / len(expected_ids)

    # Reciprocal rank = 1 / rank of the first relevant result.
    reciprocal_rank = 0.0

    for row in ranked:
        if row["document_id"] in expected_ids:
            reciprocal_rank = 1.0 / row["rank"]
            break

    metrics_rows.append({
        "query_id": query_id,
        "question": item["question"],
        "expected_doc_ids": item["expected_doc_ids"],
        "expected_count": len(expected_ids),
        "found_in_top3": found_count,
        "recall_at_3": round(recall_at_3, 4),
        "reciprocal_rank": round(
            reciprocal_rank, 4
        ),
    })


# --------------------------------------------------
# 8. Save per-query metrics
# --------------------------------------------------

metrics_file = RESULTS / "search_metrics.csv"

metric_fields = [
    "query_id",
    "question",
    "expected_doc_ids",
    "expected_count",
    "found_in_top3",
    "recall_at_3",
    "reciprocal_rank",
]

with metrics_file.open(
    "w", encoding="utf-8", newline=""
) as f:
    writer = csv.DictWriter(
        f, fieldnames=metric_fields
    )
    writer.writeheader()
    writer.writerows(metrics_rows)


# --------------------------------------------------
# 9. Calculate and save overall averages
# --------------------------------------------------

mean_recall = sum(
    row["recall_at_3"] for row in metrics_rows
) / len(metrics_rows)

mrr = sum(
    row["reciprocal_rank"] for row in metrics_rows
) / len(metrics_rows)

summary_file = RESULTS / "search_metrics_summary.txt"

with summary_file.open("w", encoding="utf-8") as f:
    f.write(f"Documents searched: {len(documents)}\n")
    f.write(f"Queries evaluated: {len(queries)}\n")
    f.write(f"Top K: {TOP_K}\n")
    f.write(f"Model: {MODEL_NAME}\n")
    f.write(f"Model load time: {load_time:.3f} seconds\n")
    f.write(
        f"Document embedding time: "
        f"{embedding_time:.3f} seconds\n"
    )
    f.write(
        f"Average query search time: "
        f"{sum(search_times) / len(search_times):.5f} seconds\n"
    )
    f.write(f"Recall@3: {mean_recall:.4f}\n")
    f.write(f"MRR: {mrr:.4f}\n")


# --------------------------------------------------
# 10. Print run summary
# --------------------------------------------------

print("\n========== SEARCH SUMMARY ==========")
print(f"Documents searched: {len(documents)}")
print(f"Queries evaluated: {len(queries)}")
print(f"Result rows saved: {len(result_rows)}")
print(f"Recall@3: {mean_recall:.4f}")
print(f"MRR: {mrr:.4f}")
print(f"Results: {results_file}")
print(f"Per-query metrics: {metrics_file}")
print(f"Summary: {summary_file}")
print("Search and evaluation completed successfully.")
