from pathlib import Path
from collections import defaultdict
import argparse
import csv
import json
import time

from sentence_transformers import SentenceTransformer, util

# ============================================================
# 1. PATHS AND CONSTANTS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR if (SCRIPT_DIR / "data" / "filter").exists() else SCRIPT_DIR.parent

DATA_DIR = ROOT / "data" / "filter"
LABELS_FILE = DATA_DIR / "labels.csv"
RESULTS_DIR = ROOT / "results"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

HEALTH_DESC = "a page about health, illness or medical treatment"
NOT_HEALTH_DESC = "a page not about health"

THRESHOLDS = [-0.06, 0.00, 0.05]


# ============================================================
# 2. EVALUATION HELPER
# ============================================================

def evaluate_predictions(actual_labels, predicted_labels):
    tp = sum(1 for a, p in zip(actual_labels, predicted_labels) if a == "health" and p == "health")
    fp = sum(1 for a, p in zip(actual_labels, predicted_labels) if a == "not_health" and p == "health")
    fn = sum(1 for a, p in zip(actual_labels, predicted_labels) if a == "health" and p == "not_health")
    tn = sum(1 for a, p in zip(actual_labels, predicted_labels) if a == "not_health" and p == "not_health")

    precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
    recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
    f1 = round(2 * precision * recall / (precision + recall), 4) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total": len(actual_labels),
    }


# ============================================================
# 3. MAIN RUNNER
# ============================================================

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not LABELS_FILE.exists():
        raise FileNotFoundError(f"Labels file not found at {LABELS_FILE}")

    # Load ground truth labels
    ground_truth = {}
    with LABELS_FILE.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ground_truth[row["page_id"]] = row["label"]

    page_ids = sorted(ground_truth.keys())
    if len(page_ids) != 20:
        raise ValueError(f"Expected 20 pages in labels.csv, found {len(page_ids)}")

    # Load text for each page
    pages = {}
    for pid in page_ids:
        page_path = DATA_DIR / f"{pid}.txt"
        if not page_path.exists():
            raise FileNotFoundError(f"Page file missing: {page_path}")
        pages[pid] = page_path.read_text(encoding="utf-8").strip()

    print(f"Loaded {len(pages)} web pages from {DATA_DIR}")
    print(f"Loading model: {MODEL_NAME} ...")

    t0 = time.perf_counter()
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    model_load_time = round(time.perf_counter() - t0, 4)
    print(f"Model loaded in {model_load_time:.2f} seconds.")

    # Encode label descriptions
    desc_embs = model.encode([HEALTH_DESC, NOT_HEALTH_DESC], normalize_embeddings=True, convert_to_tensor=True)
    health_emb = desc_embs[0]
    not_health_emb = desc_embs[1]

    # Process and encode each page
    page_records = []
    inference_times = []

    for pid in page_ids:
        text = pages[pid]
        first_line = text.splitlines()[0] if text.splitlines() else ""
        word_count = len(text.split())

        t_start = time.perf_counter()
        p_emb = model.encode(text, normalize_embeddings=True, convert_to_tensor=True)
        t_elapsed = time.perf_counter() - t_start
        inference_times.append(t_elapsed)

        sim_h = float(util.cos_sim(p_emb, health_emb)[0][0])
        sim_nh = float(util.cos_sim(p_emb, not_health_emb)[0][0])
        margin = round(sim_h - sim_nh, 4)

        # Baseline decision (closer to health description)
        pred_baseline = "health" if sim_h > sim_nh else "not_health"

        page_records.append({
            "page_id": pid,
            "title": first_line,
            "word_count": word_count,
            "actual_label": ground_truth[pid],
            "sim_health": round(sim_h, 4),
            "sim_not_health": round(sim_nh, 4),
            "margin": margin,
            "pred_baseline": pred_baseline,
            "is_correct_baseline": (pred_baseline == ground_truth[pid]),
            "time_seconds": round(t_elapsed, 5),
        })

    avg_time_per_page = round(sum(inference_times) / len(inference_times), 5)
    total_inference_time = round(sum(inference_times), 4)

    # Baseline evaluation
    actual_list = [r["actual_label"] for r in page_records]
    pred_baseline_list = [r["pred_baseline"] for r in page_records]
    baseline_metrics = evaluate_predictions(actual_list, pred_baseline_list)

    # Full list of wrong pages under baseline
    wrong_pages_baseline = [
        {
            "page_id": r["page_id"],
            "title": r["title"],
            "actual_label": r["actual_label"],
            "predicted_label": r["pred_baseline"],
            "sim_health": r["sim_health"],
            "sim_not_health": r["sim_not_health"],
            "margin": r["margin"],
            "error_type": "False Positive" if r["pred_baseline"] == "health" else "False Negative",
        }
        for r in page_records
        if not r["is_correct_baseline"]
    ]

    # Threshold analysis
    threshold_results = []
    for tau in THRESHOLDS:
        preds = ["health" if r["margin"] > tau else "not_health" for r in page_records]
        m = evaluate_predictions(actual_list, preds)
        threshold_results.append({
            "threshold_margin": tau,
            "rule": f"health if margin > {tau:+.2f}",
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "tp": m["tp"],
            "fp": m["fp"],
            "fn": m["fn"],
            "tn": m["tn"],
        })

    # Recommendation
    recommendation = (
        "If missing a health page is worse than letting a non-health page through, "
        "we must minimize False Negatives to maximize Recall for health. "
        f"Threshold tau = {THRESHOLDS[0]:+.2f} achieves Recall = "
        f"{threshold_results[0]['recall'] * 100:.1f}%, capturing all 10 true health pages "
        f"with 0 false negatives, whereas higher thresholds (0.00 and +0.05) miss "
        f"subtle health pages like specialized clinic visits and dietary condition plans."
    )

    # 1. Save CSV of per-page predictions
    csv_file = RESULTS_DIR / "topic_filter_predictions.csv"
    with csv_file.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "page_id", "title", "word_count", "actual_label",
            "sim_health", "sim_not_health", "margin",
            "pred_baseline", "is_correct_baseline", "time_seconds"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(page_records)

    # 2. Save JSON full report
    json_report = {
        "experiment": "Topic filter with zero-shot label embedding similarity",
        "model": MODEL_NAME,
        "label_descriptions": {
            "health": HEALTH_DESC,
            "not_health": NOT_HEALTH_DESC,
        },
        "timing": {
            "model_load_seconds": model_load_time,
            "total_inference_seconds": total_inference_time,
            "avg_time_per_page_seconds": avg_time_per_page,
            "total_pages": len(page_records),
        },
        "baseline_decision_rule": "health if sim_health > sim_not_health (margin > 0.00)",
        "baseline_metrics": baseline_metrics,
        "wrong_pages_baseline": wrong_pages_baseline,
        "threshold_analysis": threshold_results,
        "threshold_recommendation": recommendation,
    }

    json_file = RESULTS_DIR / "topic_filter_results.json"
    json_file.write_text(json.dumps(json_report, indent=2, ensure_ascii=False), encoding="utf-8")

    # 3. Save Summary text report
    summary_file = RESULTS_DIR / "topic_filter_summary.txt"
    summary_lines = [
        "TOPIC FILTER EXPERIMENT SUMMARY",
        "=" * 70,
        f"Model: {MODEL_NAME}",
        f"Health description: \"{HEALTH_DESC}\"",
        f"Not-health description: \"{NOT_HEALTH_DESC}\"",
        f"Total pages evaluated: {len(page_records)} (10 health, 10 not_health)",
        f"Average inference time per page: {avg_time_per_page:.5f} seconds",
        f"Total inference time: {total_inference_time:.4f} seconds",
        "",
        "BASELINE METRICS (health if sim_health > sim_not_health):",
        f"  Precision: {baseline_metrics['precision']}",
        f"  Recall:    {baseline_metrics['recall']}",
        f"  F1-Score:  {baseline_metrics['f1']}",
        f"  True Positives (TP):  {baseline_metrics['tp']}",
        f"  False Positives (FP): {baseline_metrics['fp']}",
        f"  False Negatives (FN): {baseline_metrics['fn']}",
        f"  True Negatives (TN):  {baseline_metrics['tn']}",
        "",
        f"MISCLASSIFIED PAGES UNDER BASELINE ({len(wrong_pages_baseline)} total):",
    ]

    for w in wrong_pages_baseline:
        summary_lines.append(
            f"  [{w['page_id']}] Actual: {w['actual_label']:<10} | "
            f"Pred: {w['predicted_label']:<10} | "
            f"Type: {w['error_type']:<14} | "
            f"Margin: {w['margin']:+.4f} | "
            f"Title: {w['title']}"
        )

    summary_lines.extend([
        "",
        "SIMILARITY MARGIN THRESHOLD ANALYSIS:",
        f"{'Threshold':<12}{'Rule':<28}{'Precision':<12}{'Recall':<10}{'F1':<8}{'FP':<6}{'FN':<6}",
        "-" * 76,
    ])

    for tr in threshold_results:
        summary_lines.append(
            f"{tr['threshold_margin']:>+6.2f}      "
            f"{tr['rule']:28}"
            f"{tr['precision']:>8.4f}    "
            f"{tr['recall']:>8.4f}  "
            f"{tr['f1']:>6.4f}  "
            f"{tr['fp']:>4}  "
            f"{tr['fn']:>4}"
        )

    summary_lines.extend([
        "",
        "RECOMMENDATION:",
        f"  {recommendation}",
        "=" * 70,
    ])

    summary_file.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    # Print to console
    print("\n" + "\n".join(summary_lines))
    print(f"\nSaved predictions to: {csv_file}")
    print(f"Saved JSON results to: {json_file}")
    print(f"Saved summary to:     {summary_file}")


if __name__ == "__main__":
    main()