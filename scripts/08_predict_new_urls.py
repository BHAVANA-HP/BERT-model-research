"""
Production Inference Script: Predict Medical vs Non-Medical on New Unseen URLs.
Can process any new batch of 2,000+ URLs from an Excel (.xlsx) or CSV file.
"""

import os
import sys
import argparse
import asyncio
import csv
import re
from pathlib import Path
from html.parser import HTMLParser
from typing import List, Dict, Tuple

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import httpx
import openpyxl

torch.set_num_threads(4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = REPO_ROOT / "models" / "bert_medical_classifier"
FALLBACK_BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CONCURRENCY = 30
TIMEOUT_SECONDS = 7.0
CHUNK_SIZE = 256
CHUNK_STRIDE = 64


# ============================================================
# 1. HTML PARSER & ASYNC SCRAPER
# ============================================================

class CleanTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.texts = []
        self.ignore = False
        self.ignore_tags = {
            "script", "style", "noscript", "header", "footer", "nav", "svg",
            "aside", "select", "option", "iframe", "button"
        }

    def handle_starttag(self, tag, attrs):
        if tag.lower() in self.ignore_tags:
            self.ignore = True

    def handle_endtag(self, tag):
        if tag.lower() in self.ignore_tags:
            self.ignore = False

    def handle_data(self, data):
        if not self.ignore:
            cleaned = data.strip()
            if cleaned:
                self.texts.append(cleaned)

    def get_text(self):
        return " ".join(self.texts)


def extract_clean_text(html_content: str) -> str:
    try:
        extractor = CleanTextExtractor()
        extractor.feed(html_content)
        return extractor.get_text()
    except Exception:
        clean = re.sub(r"<(script|style|nav|footer)[^>]*>.*?</\1>", " ", html_content, flags=re.DOTALL | re.IGNORECASE)
        clean = re.sub(r"<[^>]+>", " ", clean)
        return " ".join(clean.split())


async def fetch_page(client: httpx.AsyncClient, semaphore: asyncio.Semaphore, url_id: str, url: str) -> Tuple[str, str, str]:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    fetch_url = url if url.startswith("http") else f"https://{url}"
    async with semaphore:
        try:
            response = await client.get(fetch_url, headers=headers, timeout=TIMEOUT_SECONDS, follow_redirects=True)
            if response.status_code == 200:
                clean_text = extract_clean_text(response.text)
                return url_id, clean_text, "success"
            return url_id, "", f"http_{response.status_code}"
        except httpx.TimeoutException:
            return url_id, "", "timeout"
        except Exception as e:
            return url_id, "", f"error_{type(e).__name__}"


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
# 2. INFERENCE ENGINE
# ============================================================

@torch.no_grad()
def score_page(model, tokenizer, text: str) -> Tuple[str, float]:
    if not text.strip():
        return "non_medical", 0.0

    chunks = chunk_text(text, chunk_size=CHUNK_SIZE, stride=CHUNK_STRIDE)
    if not chunks:
        return "non_medical", 0.0

    # Batch encode passages
    encodings = tokenizer(
        chunks,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt"
    ).to(device)

    outputs = model(**encodings)
    probs = torch.softmax(outputs.logits, dim=1)[:, 1]  # Medical class probability
    max_medical_prob = float(torch.max(probs).cpu().item())
    label = "medical" if max_medical_prob >= 0.50 else "non_medical"
    return label, round(max_medical_prob, 4)


# ============================================================
# 3. INPUT FILE LOADER
# ============================================================

def load_urls_from_file(file_path: Path) -> List[Dict]:
    items = []
    if file_path.suffix.lower() == ".xlsx":
        wb = openpyxl.load_workbook(file_path, data_only=True)
        sheet = wb.active
        headers = [str(cell.value).strip().lower() if cell.value is not None else "" for cell in sheet[1]]
        url_col_idx = None
        id_col_idx = None
        for idx, h in enumerate(headers):
            if "url" in h and "id" not in h:
                url_col_idx = idx + 1
            if "id" in h:
                id_col_idx = idx + 1
        if url_col_idx is None:
            url_col_idx = 1

        for row_idx in range(2, sheet.max_row + 1):
            url_val = sheet.cell(row=row_idx, column=url_col_idx).value
            if not url_val:
                continue
            url_str = str(url_val).strip()
            url_id = str(sheet.cell(row=row_idx, column=id_col_idx).value) if id_col_idx else f"URL_{row_idx-1:04d}"
            items.append({"url_id": url_id, "url": url_str})
    else:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            url_col = None
            id_col = None
            for col in reader.fieldnames:
                if "url" in col.lower() and "id" not in col.lower():
                    url_col = col
                if "id" in col.lower():
                    id_col = col
            if not url_col:
                url_col = reader.fieldnames[0]

            for idx, row in enumerate(reader, 1):
                url_val = row.get(url_col, "").strip()
                if not url_val:
                    continue
                url_id = row.get(id_col, f"URL_{idx:04d}").strip()
                items.append({"url_id": url_id, "url": url_val})
    return items


# ============================================================
# 4. MAIN PIPELINE
# ============================================================

async def run_prediction_pipeline(input_file: Path, output_file: Path, model_dir: Path):
    print("=" * 65)
    print("PRODUCTION BERT PREDICTOR ON NEW URLS")
    print("=" * 65)
    print(f"Input file:  {input_file}")
    print(f"Output file: {output_file}")
    print(f"Model path:  {model_dir}")

    items = load_urls_from_file(input_file)
    print(f"Loaded {len(items)} URLs from input file.")
    if not items:
        print("Error: No URLs found to process.")
        return

    # Load Model
    if model_dir.exists():
        print(f"\n[Model] Loading fine-tuned classifier from: {model_dir}")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    else:
        print(f"\n[Warning] Fine-tuned model not found at {model_dir}. Using base model: {FALLBACK_BASE_MODEL}")
        tokenizer = AutoTokenizer.from_pretrained(FALLBACK_BASE_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(FALLBACK_BASE_MODEL, num_labels=2).to(device)
    model.eval()

    # Async Scrape
    print(f"\n[Scraper] Fetching {len(items)} URLs (concurrency={CONCURRENCY})...")
    semaphore = asyncio.Semaphore(CONCURRENCY)
    async with httpx.AsyncClient(verify=False) as client:
        tasks = [fetch_page(client, semaphore, item["url_id"], item["url"]) for item in items]
        scraped_results = await asyncio.gather(*tasks)

    scraped_map = {res[0]: (res[1], res[2]) for res in scraped_results}

    # Predict
    print(f"\n[Inference] Running neural prediction over extracted pages...")
    results = []
    medical_count = 0
    for idx, item in enumerate(items, 1):
        uid = item["url_id"]
        url = item["url"]
        text, status = scraped_map.get(uid, ("", "unknown"))
        word_count = len(text.split())

        pred_label, conf = score_page(model, tokenizer, text)
        if pred_label == "medical":
            medical_count += 1

        results.append({
            "url_id": uid,
            "url": url,
            "word_count": word_count,
            "predicted_label": pred_label,
            "confidence_score": conf,
            "status": status
        })

        if idx % 200 == 0 or idx == len(items):
            print(f"  Processed {idx}/{len(items)} URLs (Medical calls so far: {medical_count})")

    # Save output Excel & CSV
    output_csv = output_file.with_suffix(".csv")
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["url_id", "url", "word_count", "predicted_label", "confidence_score", "status"])
        writer.writeheader()
        writer.writerows(results)

    output_xlsx = output_file.with_suffix(".xlsx")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "BERT Predictions"
    headers = ["url-id", "url", "word_count", "predicted_label", "confidence_score", "status"]
    ws.append(headers)
    for r in results:
        ws.append([r["url_id"], r["url"], r["word_count"], r["predicted_label"], r["confidence_score"], r["status"]])
    wb.save(output_xlsx)

    print("\n" + "=" * 65)
    print("PREDICTION COMPLETE")
    print("=" * 65)
    print(f"Total Processed: {len(results)}")
    print(f"Medical:         {medical_count} ({medical_count / len(results) * 100:.2f}%)")
    print(f"Non-Medical:     {len(results) - medical_count}")
    print(f"CSV Saved to:    {output_csv}")
    print(f"Excel Saved to:  {output_xlsx}")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Predict Medical vs Non-Medical on new URLs using fine-tuned BERT")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Path to input .xlsx or .csv containing URLs")
    parser.add_argument("--output", "-o", type=Path, default=Path("results/new_urls_predictions"), help="Output path (without extension)")
    parser.add_argument("--model", "-m", type=Path, default=DEFAULT_MODEL_DIR, help="Path to fine-tuned model directory")
    args = parser.parse_args()

    asyncio.run(run_prediction_pipeline(args.input, args.output, args.model))


if __name__ == "__main__":
    main()

