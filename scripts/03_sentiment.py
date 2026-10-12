
from pathlib import Path
from transformers import pipeline
import time

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

MODEL = (
    "distilbert/"
    "distilbert-base-uncased-finetuned-sst-2-english"
)

examples = [
    "The team delivered the project ahead of schedule.",
    "The application keeps crashing and the service is terrible.",
    "The meeting is scheduled for Tuesday afternoon but everyone is on leave.",
    "The new release improved performance and reduced errors.",
]

print("Loading sentiment model...")
start = time.perf_counter()

classifier = pipeline(
    "text-classification",
    model=MODEL,
    device=-1
)

load_seconds = time.perf_counter() - start
lines = [
    f"Model: {MODEL}",
    f"Load time: {load_seconds:.2f} seconds"
]

for text in examples:
    start = time.perf_counter()
    result = classifier(text, truncation=True)[0]
    elapsed = time.perf_counter() - start

    lines.extend([
        f"\nText: {text}",
        f"Prediction: {result['label']}",
        f"Score: {result['score']:.4f}",
        f"Inference time: {elapsed:.3f} seconds"
    ])

output = "\n".join(lines)
(RESULTS / "sentiment_results.txt").write_text(
    output, encoding="utf-8"
)

print(output)
print("\nSaved to results\\sentiment_results.txt")
