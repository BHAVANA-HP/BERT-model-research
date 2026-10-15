import os
﻿import asyncio
import csv
import math
import re
import sys
import time
import zipfile
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path
import httpx
import openpyxl

DOWNLOADS_DIR = Path.home() / "Downloads"

def find_input_file() -> Path:
    if DOWNLOADS_DIR.exists():
        matches = list(DOWNLOADS_DIR.glob("*.xlsx"))
        if matches:
            return matches[0]
    return Path("data/urls_input.xlsx")

INPUT_XLSX = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    Path(os.environ["INPUT_XLSX"]) if "INPUT_XLSX" in os.environ else find_input_file()
)
CHECKPOINT_CSV = Path(os.environ.get("CHECKPOINT_CSV", DOWNLOADS_DIR / "checkpoint_urls.csv" if DOWNLOADS_DIR.exists() else "checkpoint_urls.csv"))
OUTPUT_XLSX = Path(os.environ.get("OUTPUT_XLSX", DOWNLOADS_DIR / "classified_reference.xlsx" if DOWNLOADS_DIR.exists() else "classified_reference.xlsx"))
OUTPUT_CSV = Path(os.environ.get("OUTPUT_CSV", DOWNLOADS_DIR / "classified_reference.csv" if DOWNLOADS_DIR.exists() else "classified_reference.csv"))

CONCURRENCY = 20
TIMEOUT_SECONDS = 8.0

# ============================================================
# 1. COMPREHENSIVE CLINICAL ONTOLOGY & LEXICON
# ============================================================

LEXICON = {
    "pathologies": [
        "cancer", "carcinoma", "sarcoma", "melanoma", "leukemia", "lymphoma", "tumor", "neoplasm",
        "metastasis", "diabetes", "diabetic", "hypertension", "hypertensive", "asthma", "asthmatic",
        "arthritis", "arthritic", "copd", "pneumonia", "bronchitis", "meningitis", "sepsis", "septic",
        "infection", "stroke", "aneurysm", "tuberculosis", "hepatitis", "cirrhosis", "pancreatitis",
        "nephritis", "colitis", "celiac", "anemia", "anemic", "alzheimer", "dementia", "epilepsy",
        "neuropathy", "ischemia", "ischemic", "infarction", "atherosclerosis", "thrombosis", "arrhythmia",
        "osteoporosis", "psoriasis", "lupus", "sclerosis", "fibrosis", "glaucoma", "cataract",
        "cardiomyopathy", "embolism", "endocarditis", "dermatitis", "appendicitis", "gastritis"
    ],
    "symptoms": [
        "symptom", "fever", "febrile", "headache", "migraine", "nausea", "vomiting", "fatigue",
        "vertigo", "dyspnea", "jaundice", "edema", "hemorrhage", "cyanosis", "syncope", "inflammation",
        "inflammatory", "chills", "photophobia", "tachycardia", "bradycardia", "ataxia", "paresthesia",
        "diarrhea", "insomnia", "malaise", "pruritus", "erythema", "hypothermia", "hyperthermia"
    ],
    "pharmacology": [
        "prescription", "prescribed", "medication", "antibiotic", "antihistamine", "analgesic",
        "corticosteroid", "insulin", "amoxicillin", "penicillin", "paracetamol", "acetaminophen",
        "ibuprofen", "prednisone", "metformin", "statin", "lisinopril", "chemotherapy", "pharmacotherapy",
        "dosage", "contraindication", "antidepressant", "antihypertensive", "immunosuppressive",
        "bronchodilator", "vaccine", "vaccination", "immunization", "omperazole", "atorvastatin"
    ],
    "procedures": [
        "biopsy", "endoscopy", "colonoscopy", "laparoscopy", "echocardiogram", "electrocardiogram",
        "ecg", "ekg", "mri", "ct scan", "ultrasound", "mammogram", "dialysis", "catheterization",
        "intubation", "surgery", "surgical", "resection", "angioplasty", "stent", "mastectomy",
        "amputation", "transfusion", "chemotherapy", "radiation therapy", "endoscopic"
    ],
    "clinical_setting": [
        "physician", "cardiologist", "oncologist", "neurologist", "pediatrician", "gastroenterologist",
        "psychiatrist", "surgeon", "dermatologist", "endocrinologist", "rheumatologist", "nephrologist",
        "inpatient", "outpatient", "triage", "icu", "pathology", "prognosis", "clinic", "hospital",
        "patient", "clinical", "intensive care", "palliative"
    ]
}

COMPILED_LEXICON = {
    cat: [re.compile(r"\b" + re.escape(w) + r"(?:s|es|ed|ing|ic)?\b", re.IGNORECASE) for w in words]
    for cat, words in LEXICON.items()
}


# ============================================================
# 2. FAST HTML TEXT EXTRACTOR
# ============================================================

class TextExtractor(HTMLParser):
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
        extractor = TextExtractor()
        extractor.feed(html_content)
        return extractor.get_text()
    except Exception:
        # Fallback regex tag stripping
        clean = re.sub(r"<(script|style|nav|footer)[^>]*>.*?</\1>", " ", html_content, flags=re.DOTALL | re.IGNORECASE)
        clean = re.sub(r"<[^>]+>", " ", clean)
        return " ".join(clean.split())


# ============================================================
# 3. MEDICAL SCORING (ZERO-ENCODER)
# ============================================================

def score_page_text(text: str):
    words = text.split()
    total_words = len(words)

    if total_words < 15:
        return {
            "score": 0.0,
            "label": "non_medical",
            "word_count": total_words,
            "evidence": "",
        }

    matched_counts = {}
    categories_hit = set()
    sample_text = " ".join(words[:2500])  # Scan up to first 2500 words for speed

    for cat, patterns in COMPILED_LEXICON.items():
        for pat in patterns:
            found = pat.findall(sample_text)
            if found:
                lemma = pat.pattern.replace(r"\b", "").replace(r"(?:s|es|ed|ing|ic)?", "")
                matched_counts[lemma] = len(found)
                categories_hit.add(cat)

    if not matched_counts:
        return {
            "score": 0.0,
            "label": "non_medical",
            "word_count": total_words,
            "evidence": "",
        }

    num_unique = len(matched_counts)
    total_matches = sum(matched_counts.values())
    num_cats = len(categories_hit)

    # Diversity-weighted density normalized by log(word_count)
    density = (num_unique * 1.5 + num_cats * 1.8 + min(total_matches, 30) * 0.2) / math.log(total_words + 10)

    # Sigmoid calibration: center at 2.4, slope 1.4
    raw_score = 1.0 / (1.0 + math.exp(-1.4 * (density - 2.4)))
    score = round(min(max(raw_score, 0.0), 1.0), 4)
    label = "medical" if score >= 0.50 else "non_medical"

    # Top evidence terms
    top_terms = sorted(matched_counts.items(), key=lambda x: x[1], reverse=True)[:6]
    evidence_str = "; ".join([f"{k}({v})" for k, v in top_terms])

    return {
        "score": score,
        "label": label,
        "word_count": total_words,
        "evidence": evidence_str,
    }


# ============================================================
# 4. EXCEL INGESTION
# ============================================================

def load_urls_from_xlsx(path: Path):
    with zipfile.ZipFile(path, "r") as z:
        shared_strings = []
        if "xl/sharedStrings.xml" in z.namelist():
            tree = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in tree.findall("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}si"):
                t = si.find("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t")
                shared_strings.append(t.text if t is not None else "")

        sheet_data = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
        urls = []
        for row in sheet_data.findall(".//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}row"):
            cells = row.findall("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}c")
            if not cells:
                continue
            c = cells[0]
            v = c.find("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}v")
            val = v.text if v is not None else ""
            if c.attrib.get("t") == "s" and val:
                val = shared_strings[int(val)]
            if val and val != "page_url" and val.startswith("http"):
                urls.append(val.strip())
    return urls


# ============================================================
# 5. ASYNC CONCURRENT SCRAPING & SCORING
# ============================================================

async def process_single_url(client: httpx.AsyncClient, semaphore: asyncio.Semaphore, url_id: str, url: str):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    async with semaphore:
        try:
            response = await client.get(url, headers=headers, timeout=TIMEOUT_SECONDS, follow_redirects=True)
            if response.status_code == 200:
                html = response.text
                clean_text = extract_clean_text(html)
                scored = score_page_text(clean_text)
                return {
                    "url-id": url_id,
                    "url": url,
                    "word_count": scored["word_count"],
                    "label": scored["label"],
                    "confidence_score": scored["score"],
                    "evidence_terms": scored["evidence"],
                    "status": "success",
                }
            else:
                return {
                    "url-id": url_id,
                    "url": url,
                    "word_count": 0,
                    "label": "non_medical",
                    "confidence_score": 0.0,
                    "evidence_terms": "",
                    "status": f"http_{response.status_code}",
                }
        except httpx.TimeoutException:
            return {
                "url-id": url_id,
                "url": url,
                "word_count": 0,
                "label": "non_medical",
                "confidence_score": 0.0,
                "evidence_terms": "",
                "status": "timeout",
            }
        except Exception as e:
            err_name = type(e).__name__
            return {
                "url-id": url_id,
                "url": url,
                "word_count": 0,
                "label": "non_medical",
                "confidence_score": 0.0,
                "evidence_terms": "",
                "status": f"error_{err_name}",
            }


async def main():
    print(f"Loading URLs from {INPUT_XLSX} ...")
    raw_urls = load_urls_from_xlsx(INPUT_XLSX)
    print(f"Found {len(raw_urls)} URLs to evaluate.")

    # Check existing checkpoint
    completed = {}
    if CHECKPOINT_CSV.exists():
        with CHECKPOINT_CSV.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed[row["url"]] = row
        print(f"Loaded {len(completed)} previously completed rows from checkpoint.")

    # Prepare queue
    tasks_to_run = []
    all_results = []
    
    for idx, url in enumerate(raw_urls, start=1):
        url_id = f"URL_{idx:04d}"
        if url in completed:
            all_results.append(completed[url])
        else:
            tasks_to_run.append((url_id, url))

    print(f"URLs remaining to process: {len(tasks_to_run)}")

    if tasks_to_run:
        semaphore = asyncio.Semaphore(CONCURRENCY)
        limits = httpx.Limits(max_keepalive_connections=CONCURRENCY, max_connections=CONCURRENCY * 2)

        fieldnames = ["url-id", "url", "word_count", "label", "confidence_score", "evidence_terms", "status"]
        checkpoint_mode = "a" if CHECKPOINT_CSV.exists() and CHECKPOINT_CSV.stat().st_size > 0 else "w"
        cp_file = CHECKPOINT_CSV.open(checkpoint_mode, newline="", encoding="utf-8")
        cp_writer = csv.DictWriter(cp_file, fieldnames=fieldnames)
        if checkpoint_mode == "w":
            cp_writer.writeheader()
            cp_file.flush()

        t_start = time.perf_counter()
        processed_count = 0
        total_remaining = len(tasks_to_run)

        async with httpx.AsyncClient(limits=limits, verify=False) as client:
            batch_size = 50
            for i in range(0, total_remaining, batch_size):
                batch = tasks_to_run[i:i + batch_size]
                batch_tasks = [process_single_url(client, semaphore, uid, u) for uid, u in batch]
                batch_results = await asyncio.gather(*batch_tasks)

                for res in batch_results:
                    all_results.append(res)
                    cp_writer.writerow(res)

                cp_file.flush()
                processed_count += len(batch_results)
                elapsed = time.perf_counter() - t_start
                rate = processed_count / elapsed if elapsed > 0 else 0
                remaining_sec = (total_remaining - processed_count) / rate if rate > 0 else 0

                med_count = sum(1 for r in all_results if r["label"] == "medical")
                print(
                    f"[{processed_count}/{total_remaining}] ({processed_count/total_remaining*100:.1f}%) | "
                    f"Rate: {rate:.1f} urls/s | ETA: {remaining_sec:.0f}s | "
                    f"Cumulative Medical: {med_count} ({med_count/len(all_results)*100:.1f}%)"
                )

        cp_file.close()

    print("\nSorting results by original URL ID order...")
    all_results.sort(key=lambda x: x["url-id"])

    # Export to CSV
    print(f"Writing final CSV to {OUTPUT_CSV} ...")
    fieldnames = ["url-id", "url", "word_count", "label", "confidence_score", "evidence_terms", "status"]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    # Export to Excel (.xlsx) using openpyxl
    print(f"Writing final Excel to {OUTPUT_XLSX} ...")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Classified Reference"
    ws.append(fieldnames)

    for row in all_results:
        ws.append([
            row["url-id"],
            row["url"],
            int(row["word_count"]),
            row["label"],
            float(row["confidence_score"]),
            row["evidence_terms"],
            row["status"],
        ])

    wb.save(OUTPUT_XLSX)

    total_medical = sum(1 for r in all_results if r["label"] == "medical")
    total_non_medical = len(all_results) - total_medical
    print("\n" + "=" * 60)
    print("CLASSIFICATION COMPLETE")
    print("=" * 60)
    print(f"Total URLs processed:   {len(all_results)}")
    print(f"Medical pages:          {total_medical} ({total_medical/len(all_results)*100:.1f}%)")
    print(f"Non-medical pages:      {total_non_medical} ({total_non_medical/len(all_results)*100:.1f}%)")
    print(f"Final Excel saved at:   {OUTPUT_XLSX}")
    print(f"Final CSV saved at:     {OUTPUT_CSV}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())