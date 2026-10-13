# Research Log

This document records the experiments performed, observations made, implementation decisions, and remaining work during the BERT and Transformer-based NLP research project.

## 1. Research Setup

- **Dataset:** 18 fictional text documents.
- **Approach:** Inference using pretrained models.
- **Fine-tuning:** No model fine-tuning performed.
- **Environment:** Python 3.12, Transformers, PyTorch, and Sentence Transformers.
- **Hardware:** Intel i5-7200U, 8 GB RAM, CPU only.

## 2. Experiment Log

### Experiment 1: Masked Language Modeling

- **Model:** `google-bert/bert-base-uncased`
- **Script:** `scripts/01_fill_mask.py`
- **Objective:** Investigate how BERT predicts a masked token using its surrounding context.

**Observation:** For the input `Employees must protect confidential [MASK].`, the model ranked `information` first, with a score of approximately 0.8444.

**Interpretation:** The model uses context to assign scores to candidate tokens. This score is not a measure of overall model accuracy.

### Experiment 2: Named Entity Recognition

- **Model:** `dslim/bert-base-NER`
- **Script:** `scripts/02_ner.py`
- **Objective:** Identify entities such as organizations, locations, and people in text.

**Observation:** The model identified plausible entities, including `Lumina Systems` and `Bengaluru`. Some entities were split into separate predictions, and some labels appeared questionable.

**Interpretation:** Pretrained NER models can identify useful entities, but their predictions may not match the requirements or terminology of a particular document collection.

**Next step:** Create manually annotated expected entities and evaluate predictions using precision, recall, and F1-score.

### Experiment 3: Sentiment Classification

- **Model:** `distilbert/distilbert-base-uncased-finetuned-sst-2-english`
- **Script:** `scripts/03_sentiment.py`
- **Objective:** Explore how a pretrained model classifies text sentiment.

**Observation:** The selected checkpoint is designed for binary positive/negative sentiment classification.

**Interpretation:** The model's output should be interpreted within its training task. It does not provide a dedicated neutral class, so neutral or mixed business messages may be misclassified.

### Experiment 4: Semantic Search

- **Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Script:** `scripts/04_semantic_search.py`
- **Objective:** Retrieve documents relevant to a natural-language query.

**Observation:** The model converts text into 384-dimensional embeddings, which can be compared to rank documents by semantic similarity.

**Interpretation:** Semantic search can retrieve documents based on meaning rather than exact keyword matches. However, a high similarity score does not guarantee that a document contains the correct answer.

## 3. Evaluation and Benchmarking

Quantitative evaluation and hardware benchmarking must be based on actual script outputs.

The planned evaluation includes:

- Retrieval: Recall@3 and MRR using expected document IDs.
- Named entity recognition: Precision, recall, and F1-score against manually annotated entities.
- Performance: Model loading time, inference time, and memory usage where measured.

Benchmark comparisons should account for differences in task complexity and input size. Results from different tasks are not automatically directly comparable.

## 4. Key Learnings

- BERT uses contextual token embeddings and bidirectional self-attention.
- Different pretrained checkpoints are designed for different tasks.
- Sentence embeddings are useful for similarity-based retrieval and clustering.
- Model predictions and similarity scores need interpretation; they are not sufficient evidence of correctness.
- A small fictional dataset is useful for experimentation but cannot establish general model performance.

## 5. Current Limitations

- The dataset contains only 18 fictional documents.
- Some observations are qualitative rather than quantitatively evaluated.
- NER evaluation requires manually annotated expected entities.
- Retrieval metrics and benchmark results must be verified from actual runs.
- No fine-tuning or production-scale testing has been performed.
