
from pathlib import Path
from collections import defaultdict
import argparse
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
    r"\d{3,5}[\s.-]?\d{4,5}(?!\w)"
)


# ============================================================
# 3. NORMALIZATION AND GROUND-TRUTH HELPERS
# ============================================================

def normalize(value, entity_type):
    value = " ".join(value.split()).strip()

    if entity_type == "phones":
        return re.sub(r"\D", "", value)

    return value.casefold()


def split_entities(value):
    return [
        item.strip()
        for item in (value or "").split(";")
        if item.strip()
    ]


def unique_entities(values, entity_type):
    """Return unique entities indexed by their normalized value."""
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


def prepare_ground_truth(rows):
    """Convert CSV rows into the format used by the scorer."""
    expected_by_doc = {}

    for row in rows:
        doc_id = row["doc_id"]

        if doc_id in expected_by_doc:
            raise ValueError(f"Duplicate ground-truth doc_id: {doc_id}")

        expected_by_doc[doc_id] = {}

        for entity_type in ENTITY_TYPES:
            expected_by_doc[doc_id][entity_type] = unique_entities(
                split_entities(row.get(entity_type, "")),
                entity_type,
            )

    return expected_by_doc


def predictions_from_ground_truth(expected_by_doc):
    """Create perfect predictions for controlled testing."""
    predictions = defaultdict(lambda: defaultdict(list))

    for doc_id, entities_by_type in expected_by_doc.items():
        for entity_type, entities in entities_by_type.items():
            predictions[doc_id][entity_type] = list(entities.values())

    return predictions


# ============================================================
# 4. REUSABLE SCORER
# ============================================================

def score_predictions(
    expected_by_doc,
    predicted_by_doc,
    doc_ids=None,
    filename_by_id=None,
):
    """
    Calculate exact-match precision and recall.

    Matching uses the same normalization and deduplication
    rules as the original NER experiment.
    """

    if doc_ids is None:
        doc_ids = sorted(expected_by_doc)

    filename_by_id = filename_by_id or {}

    metrics_by_type = {}
    misses = []
    false_positives = []

    for entity_type in ENTITY_TYPES:
        correct_total = 0
        predicted_total = 0
        expected_total = 0

        for doc_id in sorted(doc_ids):
            expected = expected_by_doc[doc_id][entity_type]

            predicted = unique_entities(
                predicted_by_doc.get(doc_id, {}).get(
                    entity_type, []
                ),
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
                    "filename": filename_by_id.get(doc_id),
                    "entity_type": entity_type,
                    "entity": expected[key],
                })

            for key in sorted(false_positive_keys):
                false_positives.append({
                    "doc_id": doc_id,
                    "filename": filename_by_id.get(doc_id),
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

    return metrics_by_type, misses, false_positives


# ============================================================
# 5. STEP 3: AUTOMATED SCORER SELF-TESTS
# ============================================================

def run_self_tests():
    """
    Test the real scoring function without loading BERT.

    All modifications to ground truth are made in memory.
    The source CSV is never changed.
    """

    print("\nNER SCORER SELF-TESTS")
    print("=" * 65)

    results = []

    def check(name, condition, details=""):
        passed = bool(condition)
        results.append(passed)

        status = "PASS" if passed else "FAIL"
        print(f"{status} - {name}")

        if details:
            print(f"       {details}")

        return passed

    rows = load_csv(GROUND_TRUTH_FILE)
    original_expected = prepare_ground_truth(rows)
    doc_ids = sorted(original_expected)

    # --------------------------------------------------------
    # TEST 1: Empty predictions must give zero recall.
    # --------------------------------------------------------

    empty_predictions = defaultdict(lambda: defaultdict(list))

    empty_metrics, _, _ = score_predictions(
        original_expected,
        empty_predictions,
        doc_ids,
    )

    types_with_expected = [
        entity_type
        for entity_type in ENTITY_TYPES
        if sum(
            len(original_expected[doc_id][entity_type])
            for doc_id in doc_ids
        ) > 0
    ]

    empty_pass = (
        bool(types_with_expected)
        and all(
            empty_metrics[entity_type]["recall"] == 0.0
            for entity_type in types_with_expected
        )
    )

    check(
        "Empty predictions produce zero recall",
        empty_pass,
        "Recall is 0.00 for entity types with expected entities.",
    )

    # --------------------------------------------------------
    # TEST 2: Ground truth used as predictions must score 1.00.
    # --------------------------------------------------------

    perfect_predictions = predictions_from_ground_truth(
        original_expected
    )

    perfect_metrics, _, _ = score_predictions(
        original_expected,
        perfect_predictions,
        doc_ids,
    )

    perfect_pass = bool(types_with_expected) and all(
        perfect_metrics[entity_type]["precision"] == 1.0
        and perfect_metrics[entity_type]["recall"] == 1.0
        for entity_type in types_with_expected
    )

    check(
        "Ground-truth predictions produce perfect scores",
        perfect_pass,
        "Precision and recall are 1.00 wherever ground truth exists.",
    )

    # --------------------------------------------------------
    # TEST 3: Remove one expected phone from a temporary copy.
    # Keep predictions unchanged.
    # --------------------------------------------------------

    expected_copy = {
        doc_id: {
            entity_type: dict(entities)
            for entity_type, entities in types.items()
        }
        for doc_id, types in original_expected.items()
    }

    phone_doc_id = next(
        (
            doc_id
            for doc_id in doc_ids
            if expected_copy[doc_id]["phones"]
        ),
        None,
    )

    if phone_doc_id is None:
        # If the real dataset has no phones, use a tiny controlled
        # fixture so this test can still verify phone scoring.
        phone_test_expected = {
            "test_doc": {
                entity_type: {}
                for entity_type in ENTITY_TYPES
            }
        }

        phone_test_expected["test_doc"]["phones"] = unique_entities(
            ["+1 212 555 0101", "+1 212 555 0102"],
            "phones",
        )

        phone_test_predictions = predictions_from_ground_truth(
            phone_test_expected
        )

        phone_doc_id = "test_doc"
        phone_test_ids = ["test_doc"]
        phone_test_label = "controlled phone fixture"
    else:
        phone_test_expected = expected_copy
        phone_test_predictions = predictions_from_ground_truth(
            original_expected
        )
        phone_test_ids = doc_ids
        phone_test_label = "project ground truth"

    baseline_metrics, _, baseline_fp = score_predictions(
        phone_test_expected,
        phone_test_predictions,
        phone_test_ids,
    )

    original_phone_metrics = baseline_metrics["phones"]
    original_phone_count = original_phone_metrics["expected"]

    # Find one expected phone and remove only that entity.
    removed_key = next(
        iter(phone_test_expected[phone_doc_id]["phones"])
    )

    removed_phone = phone_test_expected[phone_doc_id][
        "phones"
    ].pop(removed_key)

    changed_metrics, changed_misses, changed_fp = score_predictions(
        phone_test_expected,
        phone_test_predictions,
        phone_test_ids,
    )

    changed_phone_metrics = changed_metrics["phones"]

    # The prediction for the removed phone still exists.
    # It is now a false positive, not a correct match.
    expected_precision = (
        (original_phone_count - 1) / original_phone_count
    )

    precision_changed_correctly = (
        changed_phone_metrics["correct"]
        == original_phone_metrics["correct"] - 1
        and changed_phone_metrics["expected"]
        == original_phone_metrics["expected"] - 1
        and changed_phone_metrics["predicted"]
        == original_phone_metrics["predicted"]
        and changed_phone_metrics["precision"]
        == round(expected_precision, 4)
    )

    removed_phone_key = normalize(removed_phone, "phones")

    removed_phone_is_false_positive = any(
        item["entity_type"] == "phones"
        and normalize(item["entity"], "phones") == removed_phone_key
        for item in changed_fp
    )

    # Other entity-type metrics must remain identical.
    other_types_unchanged = all(
        changed_metrics[entity_type]
        == baseline_metrics[entity_type]
        for entity_type in ENTITY_TYPES
        if entity_type != "phones"
    )

    phone_test_pass = (
        precision_changed_correctly
        and removed_phone_is_false_positive
        and other_types_unchanged
    )

    old_precision = original_phone_metrics["precision"]
    new_precision = changed_phone_metrics["precision"]

    check(
        "Removing an expected phone changes phone metrics correctly",
        phone_test_pass,
        (
            f"Source: {phone_test_label}; removed {removed_phone!r}. "
            f"Phone precision: {old_precision} -> {new_precision}; "
            f"phone recall: {original_phone_metrics['recall']} -> "
            f"{changed_phone_metrics['recall']}. "
            "The unchanged prediction is now a false positive."
        ),
    )

    # --------------------------------------------------------
    # TEST 4: Phone pattern extracts every real document format,
    # including numbers followed by a full stop.
    # Must fail with the legacy pattern and pass with the new one.
    # --------------------------------------------------------

    legacy_phone_re = re.compile(
        r"(?<!\w)(?:\+\d{1,3}[\s.-]?)?"
        r"(?:\(?\d{2,5}\)?[\s.-]?)?"
        r"\d{3,5}[\s.-]?\d{4}(?!\w)"
    )

    real_phone_cases = [
        (
            "Questions go to arjun.mehta@luminasystems.example or call +91 98450 11223.",
            "+91 98450 11223",
            "Indian mobile (5+5) followed by a full stop (m01)",
        ),
        (
            "Phone: +91 98765 43210",
            "+91 98765 43210",
            "Indian mobile (5+5) standard format (cv01)",
        ),
        (
            "Contact: Thomas Becker, thomas.becker@meridianbank.example, +91 22 5550 7788.",
            "+91 22 5550 7788",
            "Indian landline (2-digit STD + 4+4) with trailing full stop (m03)",
        ),
        (
            "Questions: hr@luminasystems.example or +91 80 4000 1200.",
            "+91 80 4000 1200",
            "Indian landline (Bangalore 80 + 4+4) with trailing full stop (p01)",
        ),
        (
            "Booking contact: bookings@palmcove-resort.example, +91 832 555 0147.",
            "+91 832 555 0147",
            "Indian landline (3-digit STD + 3+4) with trailing full stop (m06)",
        ),
        (
            "Guest speaker: Laura Bennett, San Francisco. Email laura.bennett@speakers.example, phone +1 415 555 0132.",
            "+1 415 555 0132",
            "US format (+1 3+3+4) with trailing full stop (m06)",
        ),
    ]

    current_matches = [
        [m.group() for m in PHONE_RE.finditer(text)]
        for text, _, _ in real_phone_cases
    ]
    expected_matches = [[expected] for _, expected, _ in real_phone_cases]
    current_passed = current_matches == expected_matches

    legacy_matches = [
        [m.group() for m in legacy_phone_re.finditer(text)]
        for text, _, _ in real_phone_cases
    ]
    legacy_failed = legacy_matches != expected_matches

    check(
        "Phone regex extracts all document formats (including trailing full stop)",
        current_passed and legacy_failed,
        (
            f"Extracted {len([m for m in current_matches if m])}/{len(real_phone_cases)} "
            "formats correctly with new pattern. Legacy pattern fails on 5+5 numbers "
            f"(extracted only {len([m for m in legacy_matches if m])}/{len(real_phone_cases)})."
        ),
    )

    # --------------------------------------------------------
    # TEST SUMMARY
    # --------------------------------------------------------

    passed_count = sum(results)
    total_count = len(results)

    print("=" * 65)
    print(f"{passed_count}/{total_count} tests passed")

    if passed_count != total_count:
        print("SELF-TEST RESULT: FAIL")
        return 1

    print("SELF-TEST RESULT: PASS")
    return 0


# ============================================================
# 6. MAIN NER EXPERIMENT
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

    ground_ids = [row["doc_id"] for row in ground_rows]

    if len(set(ground_ids)) != 18:
        raise ValueError("Ground-truth doc_id values must be unique.")

    expected_ids = set(ground_ids)
    expected_by_doc = prepare_ground_truth(ground_rows)

    document_by_id = {}

    for path in document_paths:
        doc_id = path.stem.split("_", 1)[0]

        if doc_id in document_by_id:
            raise ValueError(f"Duplicate document ID: {doc_id}")

        document_by_id[doc_id] = path

    if set(document_by_id) != expected_ids:
        raise ValueError(
            "Document IDs do not match ground_truth.csv.\n"
            f"Missing documents: "
            f"{sorted(expected_ids - set(document_by_id))}\n"
            f"Unexpected documents: "
            f"{sorted(set(document_by_id) - expected_ids)}"
        )

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
        model_start = time.perf_counter()

        ner = pipeline(
            "token-classification",
            model=MODEL_NAME,
            aggregation_strategy="first",
            device=-1,
        )

        model_load_seconds = time.perf_counter() - model_start

        for doc_id in sorted(expected_ids):
            path = document_by_id[doc_id]
            text = path.read_text(encoding="utf-8")
            start = time.perf_counter()

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

            # BERT entity extraction.
            for item in ner(text):
                label = item.get("entity_group", "")
                entity_type = ENTITY_MAP.get(label)

                if entity_type:
                    add_prediction(
                        entity_type,
                        item.get("word", ""),
                        "bert",
                    )

            # Regex entity extraction.
            for match in EMAIL_RE.finditer(text):
                add_prediction("emails", match.group(), "regex")

            for match in PHONE_RE.finditer(text):
                add_prediction("phones", match.group(), "regex")

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

            print(f"Processed {path.name}: {elapsed:.3f} seconds")

        peak_rss[0] = max(
            peak_rss[0],
            process.memory_info().rss,
        )

    finally:
        stop_sampling.set()
        sampler.join(timeout=2)

    peak_memory_mb = round(peak_rss[0] / (1024 ** 2), 2)

    # Convert model output to the scorer's input format.
    predicted_by_doc = defaultdict(lambda: defaultdict(list))

    for prediction in all_predictions:
        predicted_by_doc[prediction["doc_id"]][
            prediction["entity_type"]
        ].append(prediction["entity"])

    filename_by_id = {
        doc_id: path.name
        for doc_id, path in document_by_id.items()
    }

    # Use the same reusable scorer tested by --self-test.
    metrics_by_type, misses, false_positives = score_predictions(
        expected_by_doc,
        predicted_by_doc,
        sorted(expected_ids),
        filename_by_id,
    )

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
        },

        "metrics_by_type": metrics_by_type,
        "predictions": all_predictions,
        "misses": misses,
        "false_positives": false_positives,

        "performance": {
            "per_document": per_document,
            "total_document_processing_seconds": round(
                total_processing_seconds, 4
            ),
            "model_load_seconds": round(model_load_seconds, 4),
            "peak_process_memory_mb": peak_memory_mb,
            "memory_note": (
                "Approximate peak resident memory (RSS) for this "
                "Python process, sampled every 0.1 seconds."
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


# ============================================================
# 7. COMMAND-LINE ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NER extraction, evaluation and scorer self-tests."
    )

    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run scorer tests without loading BERT.",
    )

    args = parser.parse_args()

    if args.self_test:
        raise SystemExit(run_self_tests())
    else:
        main()
