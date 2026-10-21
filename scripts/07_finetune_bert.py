"""
Phase 2: Fast Stratified 80/20 Train-Val Fine-Tuning Pipeline
Trains BERT on balanced medical passages + hard negatives in ~3-4 minutes on CPU,
validates on held-out test URLs, and exports the production model.
"""

import os
import sys
import json
import csv
import re
import random
from pathlib import Path
from html.parser import HTMLParser
from typing import List, Dict, Tuple

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split

torch.set_num_threads(4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
MODELS_DIR = REPO_ROOT / "models"
MODEL_OUTPUT_DIR = MODELS_DIR / "bert_medical_classifier"
CACHE_FILE = DATA_DIR / "scraped_pages_cache.json"
DEFAULT_INPUT_CSV = DATA_DIR / "classified_reference_2124.csv"

BASE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 16
LEARNING_RATE = 2.5e-5
EPOCHS = 3
CHUNK_SIZE = 280
CHUNK_STRIDE = 70


# ============================================================
# 1. TEXT CHUNKER
# ============================================================

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, stride: int = CHUNK_STRIDE) -> List[str]:
    words = text.split()
    if len(words) <= chunk_size:
        return [text] if text else []
    chunks = []
    for i in range(0, len(words), chunk_size - stride):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks


# ============================================================
# 2. DATASET
# ============================================================

class WebPassageDataset(Dataset):
    def __init__(self, passages: List[str], labels: List[int], tokenizer, max_length: int = 256):
        self.passages = passages
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.passages)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.passages[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ============================================================
# 3. TRAINING & PREDICTION HELPERS
# ============================================================

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch_idx, batch in enumerate(loader, 1):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def score_page(model, tokenizer, text: str, device) -> Tuple[int, float]:
    if not text.strip():
        return 0, 0.0
    chunks = chunk_text(text, chunk_size=256, stride=64)
    if not chunks:
        return 0, 0.0
    encodings = tokenizer(
        chunks,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt"
    ).to(device)
    outputs = model(**encodings)
    probs = torch.softmax(outputs.logits, dim=1)[:, 1]
    max_prob = float(torch.max(probs).cpu().item())
    label = 1 if max_prob >= 0.50 else 0
    return label, round(max_prob, 4)


def compute_metrics(y_true, y_pred, set_name="Test Set"):
    tp = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 0)
    tn = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 0)

    accuracy = (tp + tn) / max(len(y_true), 1)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print("\n" + "=" * 65)
    print(f"EVALUATION RESULTS: {set_name.upper()}")
    print("=" * 65)
    print(f"Total Evaluated:        {len(y_true)}")
    print(f"True Positives (TP):    {tp}  (True clinical medical captured)")
    print(f"False Positives (FP):   {fp}  (Non-medical misclassified)")
    print(f"False Negatives (FN):   {fn}  (Clinical medical missed)")
    print(f"True Negatives (TN):    {tn}  (Non-medical correctly dismissed)")
    print("-" * 65)
    print(f"Agreement Rate:         {accuracy * 100:.2f}% ({tp + tn} / {len(y_true)})")
    print(f"Precision:              {precision * 100:.2f}%")
    print(f"Recall (Sensitivity):   {recall * 100:.2f}% ({tp} of {tp + fn})")
    print(f"F1-Score:               {f1:.4f}")
    print("=" * 65)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


# ============================================================
# 4. MAIN RUNNER
# ============================================================

def run_finetune_80_20(input_csv: Path):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(f"LOADING DATASET: {input_csv}")
    print("=" * 65)

    rows = []
    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "url_id": r["url-id"],
                "url": r["url"],
                "label": 1 if r["label"].strip().lower() == "medical" else 0,
                "disagreement_type": r.get("disagreement_type", "-")
            })

    total_medical = sum(r["label"] for r in rows)
    print(f"Total Rows:              {len(rows)}")
    print(f"Ground Truth Medical:    {total_medical} ({total_medical / len(rows) * 100:.2f}%)")
    print(f"Ground Truth Non-Medical:{len(rows) - total_medical}")

    # Load Scraped Pages from Local Cache
    if not CACHE_FILE.exists():
        print(f"Error: Cache file not found at {CACHE_FILE}")
        return
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        cache = json.load(f)
    print(f"[Cache] Successfully loaded {len(cache)} cached pages.")

    # Stratified 80/20 Train-Test Split
    y_labels = [r["label"] for r in rows]
    train_indices, test_indices = train_test_split(
        range(len(rows)),
        test_size=0.20,
        stratify=y_labels,
        random_state=42
    )

    train_rows = [rows[i] for i in train_indices]
    test_rows = [rows[i] for i in test_indices]

    train_med = sum(r["label"] for r in train_rows)
    test_med = sum(r["label"] for r in test_rows)
    print(f"\n[Split] 80% Training Set: {len(train_rows)} URLs ({train_med} Medical, {len(train_rows)-train_med} Non-Medical)")
    print(f"[Split] 20% Held-Out Test Set: {len(test_rows)} URLs ({test_med} Medical, {len(test_rows)-test_med} Non-Medical)")

    # Build Balanced Training Passages (Fast CPU Training)
    train_passages = []
    train_labels = []
    random.seed(42)

    for r in train_rows:
        text = cache.get(r["url_id"], {}).get("text", "")
        if not text:
            continue
        if r["label"] == 1:
            # Positive medical pages: full sliding-window chunking
            for ch in chunk_text(text, CHUNK_SIZE, CHUNK_STRIDE):
                train_passages.append(ch)
                train_labels.append(1)
        elif "False Positive" in r["disagreement_type"]:
            # Hard negatives from Phase 1 (pet care, brain games, cooking)
            for ch in chunk_text(text, CHUNK_SIZE, CHUNK_STRIDE):
                train_passages.append(ch)
                train_labels.append(0)

    # Subsample regular non-medical to maintain balanced training
    reg_non_med = [r for r in train_rows if r["label"] == 0 and "False Positive" not in r["disagreement_type"]]
    sampled_reg = random.sample(reg_non_med, min(180, len(reg_non_med)))
    for r in sampled_reg:
        text = cache.get(r["url_id"], {}).get("text", "")
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_STRIDE)
        if chunks:
            train_passages.append(chunks[0])
            train_labels.append(0)

    pos_count = sum(train_labels)
    neg_count = len(train_labels) - pos_count
    print(f"\n[Augmentation] Fast Balanced Training Passages: {len(train_passages)}")
    print(f"  - Positive Medical Chunks: {pos_count}")
    print(f"  - Hard Negatives + General Negatives: {neg_count}")

    # Load Model & Tokenizer
    print(f"\n[Model] Initializing {BASE_MODEL_NAME} for sequence classification...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL_NAME, num_labels=2).to(device)

    # Weighted Cross-Entropy Loss (penalize missing medical 6x)
    class_weights = torch.tensor([1.0, 6.0], dtype=torch.float, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    train_dataset = WebPassageDataset(train_passages, train_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"\n[Training] Starting fast 3-epoch fine-tuning (~2-3 minutes on CPU)...")
    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"  Epoch {epoch}/{EPOCHS} Complete - Loss: {loss:.4f}", flush=True)

    # 1. Evaluate on 20% Held-Out Test Set (Zero Leakage)
    print(f"\n[Evaluation] Scoring 20% Held-Out Test Set ({len(test_rows)} URLs)...")
    model.eval()
    test_preds = []
    test_scores = []
    for r in test_rows:
        text = cache.get(r["url_id"], {}).get("text", "")
        pred_label, score = score_page(model, tokenizer, text, device)
        test_preds.append(pred_label)
        test_scores.append(score)

    test_y = [r["label"] for r in test_rows]
    compute_metrics(test_y, test_preds, set_name="20% Held-Out Test Set (Unseen)")

    # 2. Evaluate across all 2,124 URLs to produce full comparison
    print(f"\n[Full Evaluation] Scoring all 2,124 URLs for updated deliverable...")
    all_preds = []
    all_scores = []
    for r in rows:
        text = cache.get(r["url_id"], {}).get("text", "")
        pred_label, score = score_page(model, tokenizer, text, device)
        all_preds.append(pred_label)
        all_scores.append(score)

    all_y = [r["label"] for r in rows]
    compute_metrics(all_y, all_preds, set_name="Full 2,124 URLs Combined")

    # Save output comparison CSV
    out_csv = RESULTS_DIR / "finetuned_bert_predictions_2124.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "url_id", "url", "split", "ground_truth", "fine_tuned_label", "confidence_score", "status"
        ])
        for idx, (r, p, s) in enumerate(zip(rows, all_preds, all_scores)):
            split_tag = "TEST_20" if idx in test_indices else "TRAIN_80"
            gt_str = "medical" if r["label"] == 1 else "non_medical"
            p_str = "medical" if p == 1 else "non_medical"
            status = "MATCH" if gt_str == p_str else ("FP_EXTRA" if p == 1 else "FN_MISSED")
            writer.writerow([r["url_id"], r["url"], split_tag, gt_str, p_str, f"{s:.4f}", status])
    print(f"\n[Deliverable] Full results saved to: {out_csv}")

    # Save Production Model
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
    print(f"[Model Saved] Fine-tuned model saved to: {MODEL_OUTPUT_DIR}")


if __name__ == "__main__":
    csv_file = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT_CSV
    run_finetune_80_20(csv_file)

