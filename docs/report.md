# BERT Model Research Report

## 1. Introduction

- Study BERT, its architecture, variants, and practical applications.
- Explore five NLP tasks using pretrained Hugging Face models.
- Understand when to use BERT-based models for different problems.

## 2. Evolution of Language Models

**RNN → LSTM/GRU → Transformer → BERT / GPT**

- **RNN:** Processes tokens sequentially; struggles with long-range dependencies and parallelization.
- **LSTM/GRU:** Improve the handling of long-term dependencies but still process sequences recurrently.
- **Transformer:** Uses attention mechanisms to capture relationships between tokens and process sequences more efficiently.
- **BERT and GPT:** Different Transformer architectures designed for different objectives.

## 3. Transformer and Attention

- **Self-attention:** Determines how much each token should attend to other tokens.
- **Multi-head attention:** Learns different types of token relationships in parallel.
- **Positional information:** Helps the model understand token order.
- **Encoder:** Builds contextual representations of input text.
- **Decoder:** Generates output tokens autoregressively in GPT-style models.

**Key paper:** *Attention Is All You Need* — Google Research, 2017.

Article: https://arxiv.org/abs/1706.03762

## 4. BERT vs GPT

| Feature | BERT | GPT |
|---|---|---|
| Architecture | Encoder-only Transformer | Decoder-only Transformer |
| Context | Uses left and right context | Uses previous tokens to predict the next token |
| Main objective | Understand text representations | Generate text |
| Pretraining | Masked Language Modeling (MLM) | Causal Language Modeling |
| Best suited for | Classification, NER, embeddings and language understanding | Text generation, continuation and conversational applications |

**Simplified flow**

- **BERT:** Input text → Bidirectional context → Contextual representations → Task-specific output.
- **GPT:** Prompt → Predict next token → Append token → Repeat.

## 5. BERT Architecture

- **Input:** Text is tokenized into model-compatible tokens.
- **Embeddings:** Token embeddings + positional embeddings + token-type embeddings (where applicable).
- **Encoder layers:** Self-attention + feed-forward networks.
- **Output:** Contextual representation of each token.
- **Task-specific head:** Converts representations into predictions, such as token labels or classification scores.

**Original BERT pretraining**
- Masked Language Modeling (MLM).
- Next Sentence Prediction (NSP).

## 6. BERT Family and Related Models

- **BERT:** General-purpose bidirectional language understanding.
- **DistilBERT:** Smaller, faster model distilled from BERT.
- **RoBERTa:** Improved BERT-style pretraining approach.
- **ModernBERT:** A newer encoder model designed for modern NLP workloads.
- **Sentence Transformers:** Produces sentence or document embeddings optimized for similarity and retrieval tasks.

**Important:** Sentence Transformers is not simply a newer BERT architecture. It is a framework and set of models for generating useful embeddings.

## 7. Experiments and Evaluation

| Experiment | Model | Purpose |
|---|---|---|
| Fill-mask | google-bert/bert-base-uncased | Predict masked tokens |
| Named Entity Recognition | dslim/bert-base-NER | Identify entities such as people, organizations and locations |
| Sentiment Analysis | distilbert/distilbert-base-uncased-finetuned-sst-2-english | Classify text as positive or negative |
| Semantic Search | sentence-transformers/all-MiniLM-L6-v2 | Retrieve documents relevant to a query |
| Sentence Similarity | sentence-transformers/all-MiniLM-L6-v2 | Compare the semantic similarity of two sentences |

### 7.1 Masked Language Modeling
Using google-bert/bert-base-uncased, the model was tested on sentences with masked tokens. For the input:
Employees must protect confidential [MASK].
The top candidate predicted was information with a score of 0.8444, showing the bidirectional ability of the model to synthesize both preceding and following context.

### 7.2 Named Entity Recognition and Phone Pattern Fix

The NER pipeline combines dslim/bert-base-NER for entities (persons, organisations, locations) with targeted regular expressions for structured patterns (emails and phone numbers).

#### Quantitative Evaluation Across 18 Documents

| Entity Type | Correct | Predicted | Found | Expected | Precision | Recall |
|---|---|---|---|---|---|---|
| persons | 36 | 51 | 36 | 37 | 0.7059 | 0.9730 |
| organisations | 14 | 51 | 14 | 17 | 0.2745 | 0.8235 |
| locations | 18 | 25 | 18 | 19 | 0.7200 | 0.9474 |
| emails | 22 | 22 | 22 | 22 | 1.0000 | 1.0000 |
| phones | 15 | 15 | 15 | 15 | 1.0000 | 1.0000 |

#### Phone Pattern Fix and Detected Phone Numbers

Initially, the regex pattern required exactly 4 terminal digits. This missed all phone numbers formatted as Indian 5+5 mobile numbers (such as +91 98765 43210), resulting in a phone recall of only **0.5333 (8/15)**.

The pattern was updated to allow 4 or 5 digits at the end (PHONE_RE = re.compile(r'(?<!\w)(?:\+\d{1,3}[ -]?)?(?:\(?\d{2,5}\)?[- ]?)?\d{3,5}[ -]?\d{4,5}(?!\w)')).
This change improved phone recall to **1.0000 (15/15, 100%)** with **1.0000 precision** (zero misses and zero false positives). The updated pattern successfully detects all 15 expected phone numbers across the documents:

1. **cv01** (cv01_aditi_sharma.txt): +91 98765 43210
2. **m01** (m01_meeting_2026-09-02_atlas_kickoff.txt): +91 98450 11223
3. **m02** (m02_meeting_2026-09-09_sprint3_planning.txt): +91 91234 56780
4. **m03** (m03_meeting_2026-09-14_meridian_bank_call.txt): +91 22 5550 7788
5. **m04** (m04_meeting_2026-09-16_hiring_sync.txt): +91 99001 77889
6. **m06** (m06_meeting_2026-09-18_leadership_offsite.txt): +91 832 555 0147
7. **m06** (m06_meeting_2026-09-18_leadership_offsite.txt): +1 415 555 0132
8. **p01** (p01_policy_leave.txt): +91 80 4000 1200
9. **p02** (p02_policy_remote_work.txt): +91 80 4000 1250
10. **p03** (p03_policy_data_security.txt): +91 80 4000 1299
11. **p04** (p04_policy_expenses.txt): +91 80 4000 1230
12. **u02** (u02_update_orion_mobile_app.txt): +91 98480 33210
13. **u03** (u03_update_orchid_data_migration.txt): +91 98400 55671
14. **u05** (u05_update_build_day_hackathon.txt): +91 90080 44556
15. **x02** (x02_incident_report_phishing.txt): +91 80 4000 1299

An automated self-test suite verified that the old pattern fails on the 5+5 formats, while the updated pattern passes all 6 distinct format variations (including numbers followed by trailing periods, commas, and parentheses).

### 7.3 Sentiment Analysis
Tested with distilbert-base-uncased-finetuned-sst-2-english. The model performs binary classification (POSITIVE/NEGATIVE). While accurate for strongly valenced customer comments or clear progress updates, mixed or neutral operational statements are forced into one of the binary categories.

### 7.4 Semantic Search Evaluation and Score Analysis

Semantic search was implemented using sentence-transformers/all-MiniLM-L6-v2 across all 18 documents and evaluated using 12 query-document pairs in data/queries.csv.

#### Retrieval Performance
- **Recall@3:** 1.0000 (100% of expected documents appeared in the top-3 results for every query)
- **Mean Reciprocal Rank (MRR):** 1.0000 (Target documents were returned at Rank 1 for all single-target queries)

#### Reading of the Search Scores
Cosine similarity scores produced by ll-MiniLM-L6-v2 represent geometric alignment between normalized 384-dimensional dense vectors:

1. **Relative proximity, not probabilities:** A cosine similarity of 0.6410 does not represent a 64% likelihood of correctness; it indicates angle closeness in semantic representation space.
2. **Top-score range variation:** Top-1 scores ranged from **0.3546** (Query q06: 'Where is the leadership offsite and who is attending?') to **0.6410** (Query q10: 'Why did cloud costs go up this quarter?'). Even with a lower absolute score (0.3546), query q06 retrieved the exact correct target document (m06), demonstrating that ranking separation between candidates is what drives retrieval success, not an arbitrary absolute threshold.
3. **Separation from distractors:** For specific queries with direct semantic matches, there is a prominent score margin between Rank 1 and lower ranks. For query q10, Rank 1 scored 0.6410, whereas Rank 2 scored 0.2028 (a margin of 0.4382). Irrelevant documents generally scored below 0.20-0.25.
4. **Static threshold limitation:** Because scores vary according to query brevity and terminology overlap, applying a rigid cutoff threshold across different queries would either reject valid matches (if set too high, e.g. >0.40) or admit noisy false positives (if set too low).

#### The u04 Document Explanation

Query q10 asks: 'Why did cloud costs go up this quarter?'

The retrieval system ranked **u04** (u04_update_q3_cloud_costs.txt) at **Rank 1 with a similarity score of 0.6410**, which was the highest similarity score in the entire benchmark.

**Why u04 scored so high:**
- **Direct semantic alignment:** Document u04 is titled 'Project Update: Q3 Cloud Cost Review' and directly answers the query's causal inquiry. It states:
  - 'Total cloud spend for Q3 was Rs 18.4 lakh, up 12% from Q2.'
  - 'The increase came from the Atlas staging servers and larger storage for the Orchid migration.'
  - 'About 20% of compute instances ran idle outside working hours.'
- **Vocabulary and concept overlap:** The bi-encoder model maps key query concepts ('cloud costs', 'go up', 'this quarter') directly to the document's core content ('cloud spend', 'up 12%', 'increase', 'Q3').
- **Distractor distance:** The next best candidates were m05 (0.2028) and m01 (0.1856). While both mention project names like *Atlas*, neither discusses cloud infrastructure costs or quarterly budgeting. This large margin of **0.4382** ensured u04 was retrieved cleanly with zero ambiguity.

### 7.5 Sentence Similarity
Tested using sentence-transformers/all-MiniLM-L6-v2. Paraphrased statements ('The employee can work from home' vs 'Staff are allowed to work remotely') produced high cosine similarity (0.87), while contradictory sentences produced lower scores and unrelated statements approached zero, illustrating the model's ability to capture semantic intent beyond surface tokens.

## 8. Strengths and Limitations

**Strengths**
- Pretrained models reduce training effort.
- Effective for many language-understanding tasks.
- Can be fine-tuned for specific datasets.
- Hugging Face makes models accessible through a common API.

**Limitations**
- Results depend on training data and the selected checkpoint.
- Predictions can be incorrect or misleading.
- Some models are domain-specific.
- Similarity scores are not automatically probabilities or measures of factual correctness.
- Evaluation requires suitable test data and task-specific metrics.

## 9. Model Licences

| Model | Licence |
|---|---|
| `google-bert/bert-base-uncased` | Apache-2.0 |
| `dslim/bert-base-NER` | MIT |
| `distilbert/distilbert-base-uncased-finetuned-sst-2-english` | Apache-2.0 |
| `sentence-transformers/all-MiniLM-L6-v2` | Apache-2.0 |

Licences refer to these specific checkpoints. Always verify the model card and licence before redistributing or using a model commercially.