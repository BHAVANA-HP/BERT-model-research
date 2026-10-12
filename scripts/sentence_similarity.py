
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
import time

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

pairs = [
    (
        "The employee can work from home.",
        "Staff are allowed to work remotely.",
    ),
    (
        "The project deadline has been delayed.",
        "The team will deliver the project earlier than planned.",
    ),
    (
        "The company requires employees to protect confidential data.",
        "Sensitive company information must be kept secure.",
    ),
    (
        "The meeting is scheduled for Monday.",
        "A laptop was delivered to the IT department.",
    ),
]

print("Loading model...")
start = time.perf_counter()
model = SentenceTransformer(MODEL_NAME, device="cpu")
load_time = time.perf_counter() - start

sentences = [s for pair in pairs for s in pair]

start = time.perf_counter()
embeddings = model.encode(
    sentences,
    normalize_embeddings=True,
    convert_to_tensor=True,
)
inference_time = time.perf_counter() - start

lines = [
    f"Model: {MODEL_NAME}",
    f"Model load time: {load_time:.3f} seconds",
    f"Total embedding time: {inference_time:.3f} seconds",
    "",
]

for i, (first, second) in enumerate(pairs):
    score = util.cos_sim(
        embeddings[2 * i],
        embeddings[2 * i + 1],
    ).item()

    lines.extend([
        f"Sentence A: {first}",
        f"Sentence B: {second}",
        f"Cosine similarity: {score:.4f}",
        "",
    ])

output = RESULTS / "sentence_similarity_results.txt"
output.write_text("\n".join(lines), encoding="utf-8")

print("\n".join(lines))
print(f"\nSaved: {output}")
