# Setup and Usage Guide

This guide walks through reproducing all experimental results from a fresh clone of the repository.

---

## 1. Prerequisites

- **Python:** Python 3.12 (Python 3.10+ supported).
- **Git:** Installed and available on your system path.
- **Hardware:** Standard CPU laptop (GPU is not required; all models run on CPU).
- **Internet:** Required during initial setup to download pretrained Hugging Face models.

---

## 2. Step 1: Clone the Repository

Clone the repository and navigate into the project root directory:

```powershell
git clone https://github.com/BHAVANA-HP/BERT-model-research.git
cd BERT-model-research
```

---

## 3. Step 2: Create and Activate Virtual Environment

Create an isolated virtual environment and activate it:

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

*(If script execution is disabled in PowerShell, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

**On Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 4. Step 3: Install Required Dependencies

Upgrade `pip` and install the pinned dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Verify your installed packages and CPU PyTorch build:
```powershell
python scripts/record_versions.py
```
This generates `results/versions.txt` and confirms package integrity.

---

## 5. Step 4: Run the Complete Automated Pipeline

Run the entire research pipeline with a single command:

```powershell
python scripts/run_all.py
```

`scripts/run_all.py` executes the four core research stages sequentially and halts immediately on the first failure:
1. **Stage 1 (Self-Tests):** Validates scoring logic, edge cases, and phone regex patterns (`scripts/02_ner.py --self-test`).
2. **Stage 2 (Semantic Search):** Encodes 18 documents, queries 12 questions, and evaluates Recall@3 and MRR (`scripts/04_semantic_search.py`).
3. **Stage 3 (NER Scoring):** Runs `dslim/bert-base-NER` and regex extraction across all 18 documents, scoring predictions against ground truth (`scripts/02_ner.py`).
4. **Stage 4 (Topic Filter):** Evaluates zero-shot health topic filtering across 20 web pages with margin threshold analysis (`scripts/05_topic_filter.py`).

---

## 6. Step 5: (Optional) Run Individual Experiments

You can also run any experiment individually:

- **Masked Language Modeling:**
  ```powershell
  python scripts/01_fill_mask.py
  ```
  *Output:* `results/fill_mask_results.txt`

- **Sentiment Classification:**
  ```powershell
  python scripts/03_sentiment.py
  ```
  *Output:* `results/sentiment_results.txt`

- **Sentence Similarity:**
  ```powershell
  python scripts/sentence_similarity.py
  ```
  *Output:* `results/sentence_similarity_results.txt`

---

## 7. Step 6: Verify Generated Results

All outputs are saved to the `results/` directory:

| Experiment / Benchmark | Primary Output Files |
|---|---|
| **Semantic Search** | `results/search_metrics_summary.txt`<br>`results/search_metrics.csv`<br>`results/semantic_search_results.csv` |
| **Named Entity Recognition** | `results/ner_evaluation_report.json` |
| **Topic Filter** | `results/topic_filter_summary.txt`<br>`results/topic_filter_results.json`<br>`results/topic_filter_predictions.csv` |
| **Environment Check** | `results/environment.txt`<br>`results/dependency-check.txt`<br>`results/versions.txt` |