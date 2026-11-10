"""Stream Hindi text from Hugging Face into raw/<lang>.txt.
usage: python scripts/download.py [lang] [max_docs]
Default source: Wikipedia (hi ~160k articles). Bigger: set SOURCE=sangraha (AI4Bharat, needs HF login).
"""
import os, sys
from datasets import load_dataset
from tqdm import tqdm

lang = sys.argv[1] if len(sys.argv) > 1 else "hi"
max_docs = int(sys.argv[2]) if len(sys.argv) > 2 else None
src = os.environ.get("SOURCE", "wiki")

if src == "sangraha":
    iso3 = {"hi": "hin", "mr": "mar", "ta": "tam", "te": "tel", "bn": "ben"}[lang]
    ds = load_dataset("ai4bharat/sangraha", data_dir=f"verified/{iso3}", split="train", streaming=True)
else:
    ds = load_dataset("wikimedia/wikipedia", f"20231101.{lang}", split="train", streaming=True)

os.makedirs("raw", exist_ok=True)
with open(f"raw/{lang}.txt", "w", encoding="utf-8") as out:
    for i, row in enumerate(tqdm(ds, total=max_docs)):
        if max_docs and i >= max_docs:
            break
        out.write(row["text"].replace("\r", "") + "\n\n")
