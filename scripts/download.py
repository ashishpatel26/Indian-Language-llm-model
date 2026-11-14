"""Get Hindi text from Hugging Face into raw/<lang>.txt (wiki) or raw/<lang>_sangraha.txt.
usage: python scripts/download.py [lang] [max_docs]
       SOURCE=sangraha FILES=16 python scripts/download.py hi
Wikipedia is streamed. Sangraha (AI4Bharat verified, ~375 MB parquet per file, 100 files for hi)
is fetched file by file with hf_hub_download, which retries and resumes on flaky connections.
Re-running skips files already in the HF cache.
"""
import os, sys
from tqdm import tqdm

lang = sys.argv[1] if len(sys.argv) > 1 else "hi"
max_docs = int(sys.argv[2]) if len(sys.argv) > 2 else None
src = os.environ.get("SOURCE", "wiki")
os.makedirs("raw", exist_ok=True)

if src == "sangraha":
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    iso3 = {"hi": "hin", "mr": "mar", "ta": "tam", "te": "tel", "bn": "ben"}[lang]
    n_files = int(os.environ.get("FILES", 16))
    docs = 0
    with open(f"raw/{lang}_sangraha.txt", "w", encoding="utf-8") as out:
        for k in range(n_files):
            name = f"verified/{iso3}/data-{k}.parquet"
            for attempt in range(10):  # the hub drops connections often; hf_hub_download resumes
                try:
                    path = hf_hub_download("ai4bharat/sangraha", name, repo_type="dataset")
                    break
                except Exception as e:
                    print(f"{name}: retry {attempt} ({type(e).__name__})")
            else:
                sys.exit(f"could not download {name}")
            for batch in tqdm(pq.ParquetFile(path).iter_batches(columns=["text"], batch_size=2000),
                              desc=f"file {k + 1}/{n_files}"):
                for t in batch.column(0).to_pylist():
                    out.write(t.replace("\r", "") + "\n\n"); docs += 1
    print(f"wrote {docs:,} docs to raw/{lang}_sangraha.txt")
else:
    from datasets import load_dataset
    ds = load_dataset("wikimedia/wikipedia", f"20231101.{lang}", split="train", streaming=True)
    with open(f"raw/{lang}.txt", "w", encoding="utf-8") as out:
        for i, row in enumerate(tqdm(ds, total=max_docs)):
            if max_docs and i >= max_docs:
                break
            out.write(row["text"].replace("\r", "") + "\n\n")
