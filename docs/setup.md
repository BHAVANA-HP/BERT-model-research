# Setup and Usage

## 1. Prerequisites

- Python 3.12 recommended
- Git
- Internet connection for downloading pretrained models

Experiments can run on a CPU; a GPU is not required.

## 2. Clone the Repository

```powershell
git clone https://github.com/BHAVANA-HP/BERT-model-research.git
cd BERT-model-research
```

## 3. Create a Virtual Environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 4. Install Dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Run the Experiments

Run each script from the repository root:

```powershell
python scripts/01_fill_mask.py
python scripts/02_ner.py
python scripts/03_sentiment.py
python scripts/04_semantic_search.py
```

Add the sentence-similarity command here once its script filename is confirmed.

The first run may download pretrained models from Hugging Face. Execution time depends on the model, hardware, and network speed.

## 6. Data and Results

- `data/` contains the fictional text documents used in experiments.
- `scripts/` contains the experiment implementations.
- `results/` contains generated outputs, where configured.

Check each script for its input paths and output behavior.

## 7. Notes

- The experiments use pretrained models for inference; model fine-tuning is not performed.
- CPU execution may be slower than GPU execution.
- Model predictions and similarity scores should not be interpreted as proof of correctness.