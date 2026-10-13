
from pathlib import Path
from collections import defaultdict
import csv
import json
import re
import threading
import time

import psutil
from transformers import pipeline


# ============================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ============================================================

# Find the project root whether this script is in the root
# folder or inside a scripts/ folder.
SCRIPT_DIR = Path(__file__).resolve().parent

if (SCRIPT_DIR / "data" / "ground_truth.csv").exists():
    ROOT = SCRIPT_DIR
elif (SCRIPT_DIR.parent / "data" / "ground_truth.csv").exists():
    ROOT = SCRIPT_DIR.parent
else:
    raise FileNotFoundError(
        "Cannot find data/ground_truth.csv. "
        "Place this script in the project root or scripts/."
    )

DOC_DIR = ROOT / "data" / "documents"
GROUND_TRUTH_FILE = ROOT / "data" / "ground_truth.csv"
RESULTS_DIR = ROOT / "results"

# The only output file created by this script.
REPORT_FILE = RESULTS_DIR / "ner_evaluation_report.json"

MODEL_NAME = "dslim/bert-base-NER"

ENTITY_TYPES = [
    "persons",
    "organisations",
    "locations",
    "emails",
    "phones",
]

ENTITY_MAP = {
    "PER": "persons",
    "PERSON": "persons",
    "ORG": "organisations",
    "LOC": "locations",
    "GPE": "locations",
}


# ============================================================
# 2. REGEX FOR EMAILS AND PHONE NUMBERS
# ============================================================

EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+"
    r"@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+\b"
)

PHONE_RE = re.compile(
    r"(?<!\w)(?:\+\d{1,3}[\s.-]?)?"
    r"(?:\(?\d{2,5}\)?[\s.-]?)?"
    r"\d{3,5}[\s.-]?\d{4}(?!\w)"
)


# ============================================================
# 3. NORMALIZATION AND GROUND-TRUTH HELPERS
# ============================================================

def normalize(value, entity_type):
    value = " ".join(value.split()).strip()

    if entity_type == "phones":
        # Ignore formatting when comparing phone numbers.
        return re.sub(r"\D", "", value)

    # Names, organisations, locations and emails are
    # compared without case sensitivity.
    return value.casefold()


def split_entities(value):
    # Ground-truth entities are separated by semicolons.
    return [
        item.strip()
        for item in (value or "").split(";")
        if item.strip()
    ]


def unique_entities(values, entity_type):
    """Deduplicate using the matching rule for this type."""
    unique = {}

    for value in values:
        value = " ".join(value.split()).strip()

        if not value:
            continue

        key = normalize(value, entity_type)

        if key:
            unique.setdefault(key, value)

    return unique


def load_csv(path):
    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        return list(csv.DictReader(file))


# ============================================================
# 4. MAIN EXPERIMENT
# ============================================================

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    document_paths = sorted(DOC_DIR.glob("*.txt"))

    if len(document_paths) != 18:
        raise ValueError(
            f"Expected 18 text documents, found {len(document_paths)}."
        )

    ground_rows = load_csv(GROUND_TRUTH_FILE)

    if len(ground_rows) != 18:
        raise ValueError(
            f"Expected 18 ground-truth rows, found {len(ground_rows)}."
        )

    # Validate ground-truth document IDs and filenames.
    ground_ids = [row["doc_id"] for row in ground_rows]

    if len(set(ground_ids)) != 18:
        raise ValueError("Ground-truth doc_id values must be unique.")

    expected_ids = set(ground_ids)

    document_by_id = {}

    for path in document_paths:
        doc_id = path.stem.split("_", 1)[0]

        if doc_id in document_by_id:
            raise ValueError(f"Duplicate document ID: {doc_id}")

        document_by_id[doc_id] = path

    if set(document_by_id) != expected_ids:
        raise ValueError(
            "Document IDs do not match ground_truth.csv.\n"
            f"Missing documents: {sorted(expected_ids - set(document_by_id))}\n"
            f"Unexpected documents: {sorted(set(document_by_id) - expected_ids)}"
        )

    # Prepare ground truth.
    expected_by_doc = {}

    for row in ground_rows:
        doc_id = row["doc_id"]
        expected_by_doc[doc_id] = {}

        for entity_type in ENTITY_TYPES:
            expected_by_doc[doc_id][entity_type] = unique_entities(
                split_entities(row.get(entity_type, "")),
                entity_type,
            )

    # Start measuring memory before model loading.
    process = psutil.Process()
    peak_rss = [process.memory_info().rss]
    stop_sampling = threading.Event()

    def sample_memory():
        while not stop_sampling.is_set():
            try:
                peak_rss[0] = max(
                    peak_rss[0],
                    process.memory_info().rss,
                )
            except psutil.Error:
                pass

            stop_sampling.wait(0.1)

    sampler = threading.Thread(
        target=sample_memory,
        daemon=True,
    )
    sampler.start()

    all_predictions = []
    per_document = []
    model_load_seconds = None

    try:
        # Required aggregation strategy: first.
        model_start = time.perf_counter()

        ner = pipeline(
            "token-classification",
            model=MODEL_NAME,
            aggregation_strategy="first",
            device=-1,
        )

        model_load_seconds = time.perf_counter() - model_start

        # Process all 18 documents.
        for doc_id in sorted(expected_ids):
            path = document_by_id[doc_id]
            text = path.read_text(encoding="utf-8")
            start = time.perf_counter()

            # Keep unique predictions for this document.
            document_predictions = {}

            def add_prediction(entity_type, entity, source):
                entity = " ".join(entity.split()).strip()

                if not entity:
                    return

                key = (
                    entity_type,
                    normalize(entity, entity_type),
                )

                if key not in document_predictions:
                    document_predictions[key] = {
                        "doc_id": doc_id,
                        "filename": path.name,
                        "entity_type": entity_type,
                        "entity": entity,
                        "source": source,
                    }

            # BERT predictions.
            for item in ner(text):
                label = item.get("entity_group", "")
                entity_type = ENTITY_MAP.get(label)

                if entity_type:
                    add_prediction(
                        entity_type,
                        item.get("word", ""),
                        "bert",
                    )

            # Email predictions.
            for match in EMAIL_RE.finditer(text):
                add_prediction(
                    "emails",
                    match.group(),
                    "regex",
                )

            # Phone predictions.
            for match in PHONE_RE.finditer(text):
                add_prediction(
                    "phones",
                    match.group(),
                    "regex",
                )

            elapsed = time.perf_counter() - start

            all_predictions.extend(
                document_predictions.values()
            )

            per_document.append({
                "doc_id": doc_id,
                "filename": path.name,
                "processing_seconds": round(elapsed, 6),
                "prediction_count": len(document_predictions),
            })

            print(
                f"Processed {path.name}: {elapsed:.3f} seconds"
            )

        peak_rss[0] = max(
            peak_rss[0],
            process.memory_info().rss,
        )

    finally:
        stop_sampling.set()
        sampler.join(timeout=2)

    peak_memory_mb = round(
        peak_rss[0] / (1024 ** 2),
        2,
    )

    # ========================================================
    # 5. SCORE EACH ENTITY TYPE
    # ========================================================

    predicted_by_doc = defaultdict(
        lambda: defaultdict(list)
    )

    for prediction in all_predictions:
        predicted_by_doc[prediction["doc_id"]][
            prediction["entity_type"]
        ].append(prediction["entity"])

    metrics_by_type = {}
    misses = []
    false_positives = []

    for entity_type in ENTITY_TYPES:
        correct_total = 0
        predicted_total = 0
        expected_total = 0

        for doc_id in sorted(expected_ids):
            expected = expected_by_doc[doc_id][entity_type]

            predicted = unique_entities(
                predicted_by_doc[doc_id][entity_type],
                entity_type,
            )

            expected_keys = set(expected)
            predicted_keys = set(predicted)

            correct_keys = expected_keys & predicted_keys
            missed_keys = expected_keys - predicted_keys
            false_positive_keys = predicted_keys - expected_keys

            correct_total += len(correct_keys)
            predicted_total += len(predicted_keys)
            expected_total += len(expected_keys)

            for key in sorted(missed_keys):
                misses.append({
                    "doc_id": doc_id,
                    "filename": document_by_id[doc_id].name,
                    "entity_type": entity_type,
                    "entity": expected[key],
                })

            for key in sorted(false_positive_keys):
                false_positives.append({
                    "doc_id": doc_id,
                    "filename": document_by_id[doc_id].name,
                    "entity_type": entity_type,
                    "entity": predicted[key],
                })

        precision = (
            correct_total / predicted_total
            if predicted_total else None
        )

        recall = (
            correct_total / expected_total
            if expected_total else None
        )

        metrics_by_type[entity_type] = {
            "correct": correct_total,
            "predicted": predicted_total,
            "found": correct_total,
            "expected": expected_total,
            "precision": (
                round(precision, 4)
                if precision is not None else None
            ),
            "recall": (
                round(recall, 4)
                if recall is not None else None
            ),
        }

    # ========================================================
    # 6. BUILD ONE COMPLETE RESULTS FILE
    # ========================================================

    total_processing_seconds = sum(
        row["processing_seconds"]
        for row in per_document
    )

    report = {
        "experiment": "NER evaluation on all 18 documents",
        "model": MODEL_NAME,
        "aggregation_strategy": "first",
        "documents_processed": len(per_document),

        "matching_rule": {
            "persons_organisations_locations": (
                "Case-insensitive exact match after whitespace "
                "normalization."
            ),
            "emails": (
                "Case-insensitive exact match after whitespace "
                "normalization."
            ),
            "phones": (
                "Exact match after removing non-digit characters."
            ),
            "partial_names": (
                "Not counted as a match: 'Priya' does not match "
                "'Priya Raman'."
            ),
            "deduplication": (
                "Each normalized entity is counted once per document "
                "and entity type."
            ),
            "reason": (
                "Exact normalized matching is consistent and "
                "reproducible. Partial names could incorrectly "
                "count an incomplete extraction as correct."
            ),
        },

        "metrics_by_type": metrics_by_type,

        # Every unique prediction, including correct and incorrect ones.
        "predictions": all_predictions,

        # Every expected entity not found.
        "misses": misses,

        # Every predicted entity absent from the ground truth.
        "false_positives": false_positives,

        "performance": {
            "per_document": per_document,
            "total_document_processing_seconds": round(
                total_processing_seconds, 4
            ),
            "model_load_seconds": round(
                model_load_seconds, 4
            ),
            "peak_process_memory_mb": peak_memory_mb,
            "memory_note": (
                "Approximate peak resident memory (RSS) for this "
                "Python process, sampled every 0.1 seconds. It "
                "includes model loading and document processing."
            ),
            "timing_note": (
                "Per-document time includes BERT and regex extraction, "
                "but excludes model loading."
            ),
        },
    }

    REPORT_FILE.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # ========================================================
    # 7. DISPLAY A SUMMARY
    # ========================================================

    print("\nMETRICS BY ENTITY TYPE")
    print("-" * 82)

    print(
        f"{'Type':<16}{'Correct':>9}{'Predicted':>11}"
        f"{'Found':>9}{'Expected':>11}"
        f"{'Precision':>12}{'Recall':>10}"
    )

    for entity_type, values in metrics_by_type.items():
        print(
            f"{entity_type:<16}"
            f"{values['correct']:>9}"
            f"{values['predicted']:>11}"
            f"{values['found']:>9}"
            f"{values['expected']:>11}"
            f"{str(values['precision']):>12}"
            f"{str(values['recall']):>10}"
        )

    print(f"\nMisses: {len(misses)}")
    for item in misses:
        print(
            f"MISS | {item['doc_id']} | "
            f"{item['entity_type']} | {item['entity']}"
        )

    print(f"\nFalse positives: {len(false_positives)}")
    for item in false_positives:
        print(
            f"FALSE POSITIVE | {item['doc_id']} | "
            f"{item['entity_type']} | {item['entity']}"
        )

    print(f"\nDocuments processed: {len(per_document)}")
    print(f"Total processing time: {total_processing_seconds:.3f} seconds")
    print(f"Model load time: {model_load_seconds:.3f} seconds")
    print(f"Peak process memory: {peak_memory_mb} MB")
    print(f"\nAll results saved to:\n{REPORT_FILE}")


if __name__ == "__main__":
    main()
