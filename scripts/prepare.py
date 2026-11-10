"""Tokenize + pack into flat uint16 streams. Every VAL_EVERY-th line -> val.
usage: python scripts/prepare.py clean/hi.txt hi        (appends to data/train.bin, data/val.bin)
"""
import os, sys
import numpy as np
import sentencepiece as spm
from tqdm import tqdm

src, lang = sys.argv[1], sys.argv[2]
VAL_EVERY = 200
sp = spm.SentencePieceProcessor(model_file="tokenizer/spm.model")
assert sp.get_piece_size() <= 65535, "vocab too big for uint16"
tag, eos = [sp.piece_to_id(f"<{lang}>")], sp.eos_id()
os.makedirs("data", exist_ok=True)

outs = {k: open(f"data/{k}.bin", "ab") for k in ("train", "val")}
bufs = {"train": [], "val": []}
counts = {"train": 0, "val": 0}

def flush(k):
    np.asarray(bufs[k], dtype=np.uint16).tofile(outs[k]); counts[k] += len(bufs[k]); bufs[k] = []

with open(src, encoding="utf-8") as f:
    lines = (l.strip() for l in f)
    for i, line in enumerate(tqdm(lines)):
        k = "val" if i % VAL_EVERY == 0 else "train"
        bufs[k].extend(tag + sp.encode(line) + [eos])
        if len(bufs[k]) > 5_000_000:
            flush(k)
for k in bufs:
    flush(k)
print({k: f"{v/1e6:.1f}M tokens" for k, v in counts.items()})
