"""Build the Hindi SFT set for Angad from mashriram/AI4Bharat-Indic-Languages-and-Cultures.
The Hindi-belt state configs repeat the same indic/conv rows, so we pull them all and dedup;
the per-state 'cult' splits add region-specific Q&A. Output: data/sft_{train,val}.jsonl
usage: uv run python scripts/sft_data.py
"""
import hashlib, json, os, random, re, unicodedata
from datasets import load_dataset

REPO = "mashriram/AI4Bharat-Indic-Languages-and-Cultures"
# Hindi-speaking states only. Bihar skipped: its indic split is romanised and cult is Maithili.
STATES = ["Uttar_Pradesh", "Madhya_Pradesh", "Rajasthan", "Haryana", "Delhi",
          "Uttarakhand", "Himachal_Pradesh", "Chhattisgarh", "Jharkhand"]
SPLITS = ["indic", "conv", "cult"]
MARATHI = re.compile(r"आहे|आहेत|च्या|ल्या|नाही ")  # Devanagari but not Hindi


def deva_ratio(s):
    letters = [c for c in s if c.isalpha()]
    return sum("ऀ" <= c <= "ॿ" for c in letters) / max(1, len(letters))


def ok(q, a):
    return (len(q) >= 5 and len(a) >= 2 and deva_ratio(q) >= 0.8 and deva_ratio(a) >= 0.8
            and not MARATHI.search(q + a))


rows, seen, stats = [], set(), {}
for st in STATES:
    for sp in SPLITS:
        try:
            ds = load_dataset(REPO, st, split=sp)
        except Exception as e:  # missing split for a state
            print(f"skip {st}/{sp}: {type(e).__name__}")
            continue
        kept = 0
        for r in ds:
            q = unicodedata.normalize("NFC", (r["instruction"] + "\n" + (r.get("input") or "")).strip())
            a = unicodedata.normalize("NFC", (r["response"] or "").strip())
            h = hashlib.md5((q + "\x00" + a).encode()).hexdigest()
            if h in seen or not ok(q, a):
                continue
            seen.add(h); rows.append({"q": q, "a": a, "src": f"{st}/{sp}"}); kept += 1
        stats[f"{st}/{sp}"] = (len(ds), kept)
        print(f"{st}/{sp}: {len(ds)} rows, {kept} new")

random.Random(0).shuffle(rows)
n_val = max(200, len(rows) // 100)
os.makedirs("data", exist_ok=True)
for name, part in (("val", rows[:n_val]), ("train", rows[n_val:])):
    with open(f"data/sft_{name}.jsonl", "w", encoding="utf-8") as f:
        for r in part:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"total unique Hindi pairs {len(rows)} -> train {len(rows) - n_val}, val {n_val}")
assert len(rows) > 1000, "too little data; check dataset access"
