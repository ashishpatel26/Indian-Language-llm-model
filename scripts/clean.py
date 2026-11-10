"""Line-level cleaning: NFC, URL strip, script filter, exact dedup.
usage: python scripts/clean.py hi raw/hi.txt clean/hi.txt
"""
import sys, re, hashlib, unicodedata

SCRIPT_RANGES = {
    "hi": (0x0900, 0x097F), "mr": (0x0900, 0x097F), "ne": (0x0900, 0x097F),
    "bn": (0x0980, 0x09FF), "pa": (0x0A00, 0x0A7F), "gu": (0x0A80, 0x0AFF),
    "or": (0x0B00, 0x0B7F), "ta": (0x0B80, 0x0BFF), "te": (0x0C00, 0x0C7F),
    "kn": (0x0C80, 0x0CFF), "ml": (0x0D00, 0x0D7F),
}
URL = re.compile(r"https?://\S+|www\.\S+")


def script_ratio(s, lo, hi):
    letters = [c for c in s if c.isalpha()]
    return sum(lo <= ord(c) <= hi for c in letters) / max(len(letters), 1)


def main(lang, src, dst):
    lo, hi = SCRIPT_RANGES[lang]
    seen, kept, total = set(), 0, 0  # ponytail: in-memory exact dedup; datatrove MinHash past ~50M lines
    with open(src, encoding="utf-8", errors="ignore") as f, open(dst, "w", encoding="utf-8") as out:
        for line in f:
            total += 1
            s = unicodedata.normalize("NFC", URL.sub("", line)).strip()
            if len(s) < 20 or script_ratio(s, lo, hi) < 0.8:
                continue
            h = hashlib.md5(s.encode()).digest()
            if h in seen:
                continue
            seen.add(h)
            out.write(s + "\n")
            kept += 1
    print(f"kept {kept:,}/{total:,} lines")


if __name__ == "__main__":
    assert script_ratio("नमस्ते दुनिया", 0x0900, 0x097F) == 1.0
    assert script_ratio("hello", 0x0900, 0x097F) == 0.0
    main(*sys.argv[1:4])
