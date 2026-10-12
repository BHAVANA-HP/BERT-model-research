"""Record the Python and library versions used in this project.

Run from the repo root with the virtual environment activated:
    python scripts/record_versions.py
Output: results/versions.txt
"""
import platform
import sys
from datetime import date
from pathlib import Path
from importlib.metadata import version

packages = ["pip", "torch", "transformers", "sentence-transformers"]

lines = [
    f"Date: {date.today().isoformat()}",
    f"OS: {platform.platform()}",
    f"Python: {sys.version.split()[0]}",
]
for pkg in packages:
    lines.append(f"{pkg}: {version(pkg)}")

import torch
lines.append(f"CUDA available: {torch.cuda.is_available()}")

out = Path("results")
out.mkdir(exist_ok=True)
(out / "versions.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
print("\nSaved to results/versions.txt")
