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

## 7. Experiments

| Experiment | Model | Purpose |
|---|---|---|
| Fill-mask | `google-bert/bert-base-uncased` | Predict masked tokens |
| Named Entity Recognition | `dslim/bert-base-NER` | Identify entities such as people, organizations and locations |
| Sentiment Analysis | `distilbert/distilbert-base-uncased-finetuned-sst-2-english` | Classify text as positive or negative |
| Semantic Search | `sentence-transformers/all-MiniLM-L6-v2` | Retrieve documents relevant to a query |
| Sentence Similarity | `sentence-transformers/all-MiniLM-L6-v2` | Compare the semantic similarity of two sentences |

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