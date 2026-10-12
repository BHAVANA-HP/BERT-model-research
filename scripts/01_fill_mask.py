
from pathlib import Path
from transformers import pipeline
import time

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

MODEL = "google-bert/bert-base-uncased"

sentences = [
    "The team will submit the [MASK] by Friday.",
    "Employees must protect confidential [MASK].",
    "The project meeting is scheduled for [MASK].",
]

print("Loading fill-mask model...")
start = time.perf_counter()

fill_mask = pipeline(
    "fill-mask",
    model=MODEL,
    device=-1
)

load_seconds = time.perf_counter() - start
lines = [f"Model: {MODEL}", f"Load time: {load_seconds:.2f} seconds"]

for sentence in sentences:
    start = time.perf_counter()
    predictions = fill_mask(sentence, top_k=5)
    elapsed = time.perf_counter() - start

    lines.append(f"\nInput: {sentence}")
    lines.append(f"Inference time: {elapsed:.3f} seconds")

    for result in predictions:
        lines.append(
            f"  {result['token_str'].strip()}: "
            f"{result['score']:.4f}"
        )

output = "\n".join(lines)
(RESULTS / "fill_mask_results.txt").write_text(
    output, encoding="utf-8"
)

print(output)
print("\nSaved to results\\fill_mask_results.txt")
