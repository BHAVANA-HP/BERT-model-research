# BERT and Transformer-Based NLP Research

An exploratory study of Transformer architectures and pretrained NLP models through hands-on experiments.

## Overview

This project investigates the evolution from recurrent neural networks to Transformers, the architecture behind BERT, and the practical application of pretrained models to different NLP tasks.

The work combines architectural study with experiments on a small collection of 18 fictional documents. The objective is to understand model behavior, inspect predictions, and identify limitations that require further evaluation.

## Research Questions

- Why did Transformers become an important alternative to recurrent architectures?
- How do self-attention, BERT's encoder architecture, and GPT's decoder architecture differ?
- How do pretrained models support different NLP tasks?
- What can be learned from inspecting their outputs, limitations, and evaluation requirements?

Detailed analysis will be documented in [`docs/report.md`](docs/report.md).

## Experiments

| Task | Model |
|---|---|
| Masked token prediction | `google-bert/bert-base-uncased` |
| Named entity recognition | `dslim/bert-base-NER` |
| Sentiment classification | `distilbert/distilbert-base-uncased-finetuned-sst-2-english` |
| Semantic search | `sentence-transformers/all-MiniLM-L6-v2` |
| Document clustering | MiniLM embeddings, if the clustering experiment is included |

## Repository Structure

- `docs/report.md` — Research background, analysis, and conclusions
- `docs/setup.md` — Installation and execution instructions
- `docs/research-log.md` — Experiment history, observations, and next steps
- `data/` — Input documents and test cases
- `scripts/` — Experiment implementations
- `results/` — Actual experiment outputs and evaluation results

## Method

Pretrained models are used for inference. This project does not fine-tune model weights.

Experiments are inspected qualitatively, with task-specific quantitative evaluation and hardware benchmarking to be documented when verified.

## Limitations

The document collection is small and fictional. Initial observations do not establish general model accuracy or production readiness.

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [Sentence Transformers](https://www.sbert.net/)

