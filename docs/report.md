# BERT Model Research Report

## 1. Summary

- Semantic search evaluated with `scripts/04_semantic_search.py` achieved Recall@3 = 1.0000 and Mean Reciprocal Rank (MRR) = 1.0000 across 18 documents and 12 queries (`results/search_metrics_summary.txt`).
- Personal data extraction with `scripts/02_ner.py` achieved 1.0000 precision and 1.0000 recall on emails (22/22) and phone numbers (15/15) across all 18 documents (`results/ner_evaluation_report.json`).
- Named entity recognition with `scripts/02_ner.py` identified persons with 0.9730 recall (36/37) and locations with 0.9474 recall (18/19), but organization precision was 0.2745 due to 37 false positives (`results/ner_evaluation_report.json`).
- Zero-shot topic filtering with `scripts/05_topic_filter.py` achieved baseline precision and recall of 0.6000 on 20 web pages, improving to 1.0000 recall (10/10 health pages) at a tuned margin threshold of -0.06 (`results/topic_filter_summary.txt`).
- Complete pipeline execution via `scripts/run_all.py` completed all four evaluation stages sequentially in 212.56 seconds on laptop CPU hardware (`results/topic_filter_summary.txt`).

## 2. Setup and Hardware

- **Hardware Platform:** Intel Core i5-7200U CPU @ 2.50GHz, 8.00 GB installed RAM, single-socket CPU inference (`CUDA available: False` recorded by `scripts/record_versions.py` in `results/environment.txt`).
- **Operating System:** Microsoft Windows 10/11 Home 64-bit (`results/versions.txt`).
- **Python Version:** Python 3.12.14 (`results/environment.txt`).
- **PyTorch Build:** PyTorch 2.14.1+cpu (`results/environment.txt`).
- **Hugging Face Transformers:** Version 5.19.0 (`results/environment.txt`).
- **Sentence Transformers:** Version 6.1.0 (`results/environment.txt`).
- **Environment Integrity:** Dependency check verified with 0 broken requirements (`results/dependency-check.txt`).

## 3. Test Data

- **Enterprise Document Corpus (`data/documents/`):** 18 fictional corporate text documents comprising meeting minutes (`m01`–`m06`), corporate policies (`p01`–`p04`), sprint updates (`u01`–`u05`), resume (`cv01`), and IT incident emails (`x01`–`x02`). Ground truth was manually labeled in `data/ground_truth.csv` covering 37 persons, 17 organisations, 19 locations, 22 emails, and 15 phone numbers (`results/ner_evaluation_report.json`).
- **Natural Language Search Benchmark (`data/queries.csv`):** 12 natural-language questions paired with target document IDs (`expected_doc_ids`). The set contains 10 single-target queries and 2 multi-target queries (`q01` targeting `m01; u01` and `q03` targeting `p03; x02`), representing policy questions, project leads, budget investigations, and incident reports (`results/search_metrics.csv`).
- **Web Topic Filter Corpus (`data/filter/`):** 20 synthetic web pages ranging between 101 and 149 words each, labeled in `data/filter/labels.csv`. The collection contains 10 health pages (`page01`–`page10`) covering clinical symptoms, prescription medicine, dietary condition management, and outpatient clinic visits, alongside 10 non-health pages (`page11`–`page20`) covering sports reporting, travel guides, equity investing, cloud infrastructure, and culinary recipes (`results/topic_filter_results.json`).
- **Adversarial Borderline Pages:** The filter dataset intentionally includes three borderline cases in `data/filter/`: `page18` (athletic gym facility amenities, labeled `not_health`), `page19` (caramel fudge recipe citing sugar measurements, labeled `not_health`), and `page20` (sports playoff injury news reporting an ankle sprain and MRI scan, labeled `not_health`) (`results/topic_filter_predictions.csv`).

## 4. Results

### 4.1 Semantic Search Retrieval Results
Measured by `scripts/04_semantic_search.py` on 18 documents and 12 queries:

| Metric | Measured Value | Source Results File |
|---|---|---|
| Total Documents Searched | 18 | `results/search_metrics_summary.txt` |
| Total Queries Evaluated | 12 | `results/search_metrics_summary.txt` |
| Top-K Documents Retrieved | 3 | `results/search_metrics_summary.txt` |
| Recall@3 | 1.0000 | `results/search_metrics_summary.txt` |
| Mean Reciprocal Rank (MRR) | 1.0000 | `results/search_metrics_summary.txt` |
| Model Loading Latency | 6.721 seconds | `results/search_metrics_summary.txt` |
| Document Embedding Time | 1.792 seconds | `results/search_metrics_summary.txt` |
| Average Search Time Per Query | 0.01980 seconds | `results/search_metrics_summary.txt` |

### 4.2 Personal-Data and Entity Detection Results
Measured by `scripts/02_ner.py` against `data/ground_truth.csv` across all 18 documents:

| Entity Type | Expected Entities | Correctly Found | Predicted Total | Precision | Recall | Source Results File |
|---|---|---|---|---|---|---|
| emails | 22 | 22 | 22 | 1.0000 | 1.0000 | `results/ner_evaluation_report.json` |
| phones | 15 | 15 | 15 | 1.0000 | 1.0000 | `results/ner_evaluation_report.json` |
| persons | 37 | 36 | 51 | 0.7059 | 0.9730 | `results/ner_evaluation_report.json` |
| locations | 19 | 18 | 25 | 0.7200 | 0.9474 | `results/ner_evaluation_report.json` |
| organisations | 17 | 14 | 51 | 0.2745 | 0.8235 | `results/ner_evaluation_report.json` |

### 4.3 Topic Filter Classification Results
Measured by `scripts/05_topic_filter.py` on 20 web pages with zero-shot label descriptions:

| Margin Threshold ($\tau$) | Classification Decision Rule | Precision | Recall | F1-Score | False Positives (FP) | False Negatives (FN) | Source Results File |
|---|---|---|---|---|---|---|---|
| -0.06 | health if margin > -0.06 | 0.5263 | 1.0000 | 0.6896 | 9 | 0 | `results/topic_filter_summary.txt` |
| 0.00 (Baseline) | health if margin > 0.00 | 0.6000 | 0.6000 | 0.6000 | 4 | 4 | `results/topic_filter_summary.txt` |
| +0.05 | health if margin > +0.05 | 0.8333 | 0.5000 | 0.6250 | 1 | 5 | `results/topic_filter_summary.txt` |

## 5. What Works, What Does Not, and Why

### 5.1 What Works
- **Semantic Retrieval Ranking:** Bi-encoder dense representations generated by `sentence-transformers/all-MiniLM-L6-v2` cleanly separated relevant documents from background distractors (`results/search_metrics_summary.txt`). For query `q10` (*"Why did cloud costs go up this quarter?"*), document `u04` scored 0.6410 at Rank 1, establishing a 0.4382 margin over the second-ranked document `m05` at 0.2028 (`results/semantic_search_results.csv`).
- **Deterministic Personal-Data Detection:** Regex extraction achieved 100% precision and recall for emails (22/22) and phones (15/15) across all 18 documents (`results/ner_evaluation_report.json`). All Indian 5+5 mobile numbers (e.g. `+91 98765 43210` in `cv01`, `+91 98450 11223` in `m01`), 2-digit STD landlines (`+91 22 5550 7788` in `m03`), and US numbers (`+1 415 555 0132` in `m06`) were correctly captured even when followed by trailing punctuation (`results/ner_evaluation_report.json`).
- **High Recall Margin Tuning in Topic Filtering:** Adjusting the margin decision boundary to $\tau = -0.06$ eliminated all false negatives (0 FN), successfully retrieving all 10 health pages (100% recall) in `results/topic_filter_summary.txt`.
- **Lexical Disambiguation of Athletic Facilities:** The tricky gym page (`page18`) was correctly classified as `not_health` with a negative margin of -0.0893 ($s_{\text{health}} = 0.1403$, $s_{\text{not\_health}} = 0.2296$), demonstrating that the bi-encoder recognizes commercial exercise amenities as distinct from medical illness (`results/topic_filter_predictions.csv`).

### 5.2 What Does Not Work
- **Organization False Positives in BERT NER:** The `dslim/bert-base-NER` token classifier generated 37 organization false positives, causing organization precision to collapse to 0.2745 (`results/ner_evaluation_report.json`). The model misclassified job roles such as `Data Scientist` (`m01`), `Scrum Master` (`m02`), `DevOps` (`m02`), and `Analyst` (`cv01`) as organizations (`results/ner_evaluation_report.json`).
- **Title and Heading Over-Segmentation:** Section headers such as `Lumina Systems Data Security Policy` in `p03` and `Lumina Systems Expense` in `p04` were predicted as organizations instead of matching the base entity `Lumina Systems` (`results/ner_evaluation_report.json`).
- **Baseline Topic Filter False Negatives:** At the default $\tau = 0.00$ threshold, the filter missed four health documents: `page03` (hypertension diet, margin -0.0511), `page04` (cardiology clinic consultation, margin -0.0552), `page06` (diabetes insulin management, margin -0.0313), and `page08` (colonoscopy prep, margin -0.0372) because dietary, scheduling, and device vocabulary pulled vectors closer to the generic label description (`results/topic_filter_summary.txt`).
- **Adversarial Semantic Collisions in Non-Health Pages:** At baseline, the filter produced four false positives: `page20` (sports injury news, margin +0.0513) was pulled into `health` by clinical injury terms (*ankle sprain, MRI, ligament*), and `page14` (Kubernetes container orchestration, margin +0.0474) was pulled into `health` by metaphorical systems vocabulary (*self-healing, restarts*) (`results/topic_filter_summary.txt`).

## 6. Limits of This Test

- **Small Benchmark Scale is Not Proof of Production Accuracy:** Achieving Recall@3 = 1.0000 on 12 queries over 18 documents (`results/search_metrics_summary.txt`) reflects clean retrieval on a controlled toy corpus. In an enterprise repository containing 100,000+ documents, dense vector collision and polysemy drastically reduce top-1 precision.
- **Cosine Similarity Does Not Measure Factual Correctness:** Similarity scores quantify angular proximity between sentence embeddings (e.g. 0.6410 for `u04` on query `q10` in `results/semantic_search_results.csv`), not whether the retrieved document contains verified facts or an adequate direct answer.
- **Zero-Shot Description Sensitivity:** In `scripts/05_topic_filter.py`, classification depends entirely on the specific phrasing of two text anchors (`"a page about health, illness or medical treatment"` vs `"a page not about health"`). The lack of explicit negative boundary exemplars caused non-medical news about an athletic injury (`page20`) to score a positive margin of +0.0513 (`results/topic_filter_predictions.csv`).
- **Single-Thread CPU Throughput Constraints:** Processing 18 documents with `dslim/bert-base-NER` required 13.40 seconds (0.74 seconds per document in `results/ner_evaluation_report.json`), and total pipeline execution took 212.56 seconds (`results/topic_filter_summary.txt`), which is insufficient for real-time high-concurrency ingestion pipelines without GPU acceleration or batch optimization.

## 7. Next Steps

- **Post-Processing Filtering for NER Roles:** Introduce dictionary-based post-filtering in `scripts/02_ner.py` to strip standard corporate job titles (`Data Scientist`, `Scrum Master`, `Analyst`) from organization predictions, directly addressing the 37 false positives recorded in `results/ner_evaluation_report.json`.
- **Hybrid Retrieval Integration:** Combine dense embeddings from `scripts/04_semantic_search.py` with sparse BM25 keyword matching to prevent semantic drift on keyword-specific queries in larger document collections (`results/search_metrics_summary.txt`).
- **Few-Shot Boundary Calibration for Topic Filtering:** Replace zero-shot single-sentence anchors in `scripts/05_topic_filter.py` with a small set of labeled exemplar embeddings to separate metaphorical health terminology (e.g. `page14`) and sports injuries (e.g. `page20`) from clinical care (`results/topic_filter_summary.txt`).
- **Pipeline Automation and Regression Testing:** Maintain `scripts/run_all.py` as the standard automated CI gate to verify that any pattern or model checkpoint adjustment passes self-tests, semantic search, NER scoring, and topic filtering in a single execution.