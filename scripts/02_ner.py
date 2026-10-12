
from pathlib import Path
from transformers import pipeline
import re
import time

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

MODEL = "dslim/bert-base-NER"
DOCUMENT = (
    ROOT / "data" / "documents"
    / "m01_meeting_2026-09-02_atlas_kickoff.txt"
)

if not DOCUMENT.exists():
    raise FileNotFoundError(f"Document not found: {DOCUMENT}")

text = DOCUMENT.read_text(encoding="utf-8")

print("Loading NER model...")
start = time.perf_counter()

ner = pipeline(
    "token-classification",
    model=MODEL,
    aggregation_strategy="simple",
    device=-1
)

load_seconds = time.perf_counter() - start

start = time.perf_counter()
entities = ner(text)
inference_seconds = time.perf_counter() - start

lines = [
    f"Model: {MODEL}",
    f"Document: {DOCUMENT.name}",
    f"Load time: {load_seconds:.2f} seconds",
    f"Inference time: {inference_seconds:.3f} seconds",
    "\nNamed entities:"
]

for entity in entities:
    lines.append(
        f"  {entity['word']} | {entity['entity_group']} | "
        f"score={entity['score']:.4f}"
    )

# NER may miss email addresses and phone numbers.
# Check these separately in the original text.
emails = sorted(set(re.findall(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    text
)))

phones = sorted(set(re.findall(
    r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)",
    text
)))

lines.append("\nEmail addresses found by regex:")
lines.extend(f"  {item}" for item in emails)
lines.append("\nPhone-like strings found by regex:")
lines.extend(f"  {item}" for item in phones)

output = "\n".join(lines)
(RESULTS / "ner_results.txt").write_text(
    output, encoding="utf-8"
)

print(output)
print("\nSaved to results\\ner_results.txt")
