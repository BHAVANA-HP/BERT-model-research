import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PIPELINE = [
    {
        "name": "NER Scorer Self-Tests",
        "command": [sys.executable, str(ROOT / "scripts" / "02_ner.py"), "--self-test"],
        "description": "Verifies scoring logic, edge cases, and phone regex patterns",
    },
    {
        "name": "Semantic Search",
        "command": [sys.executable, str(ROOT / "scripts" / "04_semantic_search.py")],
        "description": "Runs document embedding and query evaluation against ground truth",
    },
    {
        "name": "NER Scoring and Entity Extraction",
        "command": [sys.executable, str(ROOT / "scripts" / "02_ner.py")],
        "description": "Runs dslim/bert-base-NER and regex extraction across all 18 documents",
    },
    {
        "name": "Topic Filter Evaluation",
        "command": [sys.executable, str(ROOT / "scripts" / "05_topic_filter.py")],
        "description": "Evaluates zero-shot health topic filtering across 20 web pages",
    },
]


def main():
    print("=" * 70)
    print("BERT MODEL RESEARCH: COMPLETE EVALUATION PIPELINE")
    print("=" * 70)
    print(f"Working Directory: {ROOT}")
    print(f"Python Interpreter: {sys.executable}\n")

    overall_start = time.perf_counter()

    for idx, stage in enumerate(PIPELINE, start=1):
        print(f"[{idx}/{len(PIPELINE)}] RUNNING: {stage['name']}")
        print(f"      Purpose: {stage['description']}")
        print("-" * 70)

        t_stage = time.perf_counter()
        result = subprocess.run(
            stage["command"],
            cwd=str(ROOT),
        )
        elapsed = time.perf_counter() - t_stage

        if result.returncode != 0:
            print("\n" + "!" * 70)
            print(f"PIPELINE FAILURE at stage: {stage['name']}")
            print(f"Exit code: {result.returncode}")
            print(f"Pipeline stopped immediately on first failure.")
            print("!" * 70)
            sys.exit(result.returncode)

        print("-" * 70)
        print(f"PASSED: {stage['name']} ({elapsed:.2f}s)\n")

    total_time = time.perf_counter() - overall_start
    print("=" * 70)
    print(f"ALL {len(PIPELINE)} PIPELINE STAGES PASSED SUCCESSFULLY")
    print(f"Total Pipeline Runtime: {total_time:.2f} seconds")
    print("All results updated in results/")
    print("=" * 70)


if __name__ == "__main__":
    main()