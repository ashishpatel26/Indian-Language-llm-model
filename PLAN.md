# Plan: Training an Indian-Language LLM From Scratch on a 6 GB GPU

## 0. Short answer

**Yes, you can train a Hindi LLM, or a multilingual Indian one, from scratch on a 6 GB GPU.** The ceiling is roughly **30–120M parameters**. A model that size learns fluent grammar, spelling, and sentence structure. It does **not** learn reliable facts or reasoning, which need billions of parameters.

| Goal                                              | Feasible on 6 GB?    | Path                                                                  |
| ------------------------------------------------- | -------------------- | --------------------------------------------------------------------- |
| Fluent Hindi text generator (story or news style) | Yes                  | This plan (scratch, ~50–110M)                                        |
| Multilingual Indic generator (4–8 languages)     | Yes                  | This plan, Phase 6                                                    |
| Chat assistant that knows facts                   | No, not from scratch | Fine-tune an existing model (Sarvam-1, Gemma, Llama, Qwen) with QLoRA |

---

## 1. Fixes to the pasted guides (read this first)

The two pasted answers have the right idea but contain bugs that will waste days of training:

| # | Problem in the pasted code                                                                                                                              | Fix                                                                                                                                                     |
| - | ------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1 | Each line is **truncated** to 512 tokens and padded. Most lines are short, so roughly 70% of compute goes to padding.                             | **Pack** all text into one continuous token stream and cut it into fixed 512-token blocks.                                                        |
| 2 | `pad_token = eos_token`, combined with the collator, **masks every EOS** from the loss, so the model never learns when to stop.                 | Use a dedicated `<pad>` token, or none at all, because packed blocks need no padding.                                                                 |
| 3 | **Byte-level BPE** on Indic scripts: each Devanagari or Tamil character is 3 UTF-8 bytes, which inflates token counts (low fertility efficiency). | Train a **SentencePiece Unigram** tokenizer with `character_coverage=0.9995` and `byte_fallback=True`.                                        |
| 4 | `AutoTokenizer.from_pretrained("./shared-indian-tokenizer")` on a raw `tokenizer.json` fails because there is no config.                            | Wrap the tokenizer in `LlamaTokenizerFast` / `PreTrainedTokenizerFast` and call `save_pretrained` once.                                           |
| 5 | `fp16=True` on scratch training is prone to loss spikes and NaNs.                                                                                     | Use `bf16=True` on RTX 30xx/40xx. Use fp16 only on GTX 16xx/20xx.                                                                                     |
| 6 | `num_train_epochs=3` on a small corpus overfits.                                                                                                      | Budget by **tokens**: about 20 tokens per parameter (Chinchilla). For a 100M model that is about 2B tokens. Prefer one pass over more data.       |
| 7 | "Balance languages equally"                                                                                                                             | Use **temperature sampling** (α≈0.3) instead. Equal mixing wastes the rich languages and overfits the small ones.                               |
| 8 | Vocab of 60k with a tiny model                                                                                                                          | The embedding matrix takes most of the parameters. Use **32k for Hindi-only** and **64k for multilingual**, with **tied embeddings**. |

---

## 2. Hardware and stack

- GPU: 6 GB (RTX 3050, 3060 Laptop, 4050, or 2060). Check bf16 support with `torch.cuda.is_bf16_supported()`.
- RAM: 16 GB minimum. Disk: 50–100 GB for corpora and tokenized shards.
- OS: Windows works. WSL2 is smoother for `bitsandbytes` and `flash-attn`, but neither is required.

```bash
python -m venv .venv && .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install transformers datasets accelerate sentencepiece numpy tqdm
pip install indic-nlp-library datatrove   # cleaning (optional but recommended)
pip install bitsandbytes                  # 8-bit AdamW (optional)
```

---

## 3. Phases

```
Phase 1  Data collection        ->  raw/*.txt
Phase 2  Cleaning + dedup       ->  clean/<lang>.txt
Phase 3  Tokenizer training     ->  tokenizer/
Phase 4  Tokenize + pack        ->  data/train.bin, data/val.bin
Phase 5  Pretrain (Hindi)       ->  ckpt/
Phase 6  Multilingual extension ->  same pipeline, more languages
Phase 7  Evaluate + generate
Phase 8  (Optional) Instruction-tune small chat model
```

### Phase 1: Data (the single most important factor)

| Source                           | Languages     | Notes                                          |
| -------------------------------- | ------------- | ---------------------------------------------- |
| **AI4Bharat IndicCorp v2** | 22 Indic + en | Best clean news corpus; Hindi about 20B tokens |
| **Sangraha** (AI4Bharat)   | 22 Indic      | Verified web + PDF + synthetic                 |
| **CulturaX / mC4 / OSCAR** | many          | Web text that is noisy and must be cleaned     |
| **Wikipedia dumps**        | all           | Small but high quality                         |
| **FineWeb-2**              | many          | Pre-filtered web                               |

Target for a **Hindi-only** run: **1–3 GB of clean text** (about 0.5–1.5B tokens). A 6 GB GPU cannot use much more than that in a reasonable time.

### Phase 2: Cleaning

1. **Unicode NFC normalize** and Indic normalize (`indicnlp.normalize`), which fixes nukta and ZWJ variants.
2. **Language-ID filter.** Keep only lines where at least 80% of characters fall in the target script block (Devanagari is U+0900–U+097F).
3. **Drop junk:** lines under 20 characters, lines that are mostly digits or URLs, and boilerplate ("Click here", cookie banners).
4. **Deduplicate.** Exact dedup by hash at minimum. Use MinHash (`datatrove`) if the corpus is large.
5. **Hold out 0.5%** as validation.

`scripts/clean.py`:

```python
import sys, re, hashlib, unicodedata

SCRIPT_RANGES = {  # target-script Unicode blocks
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
    seen = set()  # ponytail: in-memory exact dedup; switch to datatrove MinHash past ~50M lines
    with open(src, encoding="utf-8", errors="ignore") as f, open(dst, "w", encoding="utf-8") as out:
        for line in f:
            s = unicodedata.normalize("NFC", URL.sub("", line)).strip()
            if len(s) < 20 or script_ratio(s, lo, hi) < 0.8:
                continue
            h = hashlib.md5(s.encode()).digest()
            if h in seen:
                continue
            seen.add(h)
            out.write(s + "\n")

if __name__ == "__main__":
    assert script_ratio("नमस्ते दुनिया", 0x0900, 0x097F) == 1.0
    assert script_ratio("hello", 0x0900, 0x097F) == 0.0
    main(*sys.argv[1:4])   # python clean.py hi raw/hi.txt clean/hi.txt
```

### Phase 3: Tokenizer (SentencePiece Unigram)

`scripts/train_tokenizer.py`:

```python
import sys, sentencepiece as spm
from transformers import LlamaTokenizerFast

files = sys.argv[1].split(",")          # "clean/hi.txt" or "clean/hi.txt,clean/ta.txt,..."
vocab = int(sys.argv[2])                # 32000 Hindi-only, 64000 multilingual
LANG_TAGS = ["<hi>", "<mr>", "<bn>", "<ta>", "<te>", "<gu>", "<kn>", "<ml>", "<pa>", "<or>"]

spm.SentencePieceTrainer.train(
    input=",".join(files),
    model_prefix="tokenizer/spm",
    model_type="unigram",
    vocab_size=vocab,
    character_coverage=0.9995,          # keep rare conjuncts / matras
    byte_fallback=True,                 # never <unk>
    input_sentence_size=10_000_000,     # sample, keeps RAM bounded
    shuffle_input_sentence=True,
    normalization_rule_name="nfkc",
    split_digits=True,
    pad_id=0, unk_id=1, bos_id=2, eos_id=3,
    user_defined_symbols=LANG_TAGS,
)

tok = LlamaTokenizerFast(vocab_file="tokenizer/spm.model", legacy=False,
                         add_bos_token=True, add_eos_token=False, pad_token="<pad>")
tok.save_pretrained("tokenizer")
ids = tok("भारत एक विशाल देश है।")["input_ids"]
print(len(ids), tok.convert_ids_to_tokens(ids))
```

Sanity check: a Hindi sentence should average **about 1.3–1.8 tokens per word**. If it is above 3, coverage or vocab is too small.

> **Multilingual vocab balance:** if Hindi has 10× more data than Tamil, build the tokenizer input from an equal-size **sample** per language (for example 2M lines each). Without that, Tamil gets fragmented into tiny pieces.

### Phase 4: Tokenize and pack into a binary stream

The model reads one flat array of token ids. This approach needs no padding and makes loading instant (via memmap).

`scripts/prepare.py`:

```python
import sys, numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm

tok = AutoTokenizer.from_pretrained("tokenizer")
src, dst, tag = sys.argv[1], sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else "")
tag_ids = tok.convert_tokens_to_ids([tag]) if tag else []
eos = tok.eos_token_id

buf, out = [], open(dst, "ab")             # append mode: run once per language into same file
with open(src, encoding="utf-8") as f:
    for line in tqdm(f):
        buf.extend(tag_ids + tok.encode(line.strip(), add_special_tokens=False) + [eos])
        if len(buf) > 5_000_000:
            np.array(buf, dtype=np.uint32 if len(tok) > 65535 else np.uint16).tofile(out); buf = []
np.array(buf, dtype=np.uint32 if len(tok) > 65535 else np.uint16).tofile(out)
```

```bash
python scripts/prepare.py clean/hi.train.txt data/train.bin "<hi>"
python scripts/prepare.py clean/hi.val.txt   data/val.bin   "<hi>"
```

> Note: `uint16` holds up to 65,535 ids, so a 64k vocab still fits. Keep vocab ≤ 65,535 to stay at 2 bytes per token.

### Phase 5: Model and training

**Architecture (Llama-style, fits 6 GB):**

| Config                    | Small (fast iteration) | **Main (recommended)**            |
| ------------------------- | ---------------------- | --------------------------------------- |
| vocab                     | 32k                    | 32k (Hindi) / 64k (multi)               |
| hidden                    | 512                    | 768                                     |
| layers                    | 8                      | 12                                      |
| heads / kv-heads          | 8 / 8                  | 12 / 4 (GQA)                            |
| FFN (SwiGLU)              | 1408                   | 2048                                    |
| context                   | 512                    | 1024                                    |
| tied embeddings           | yes                    | yes                                     |
| **params**          | **~42M**         | **~110M (Hindi) / ~135M (multi)** |
| tokens to train           | ~0.8B                  | ~2B                                     |
| est. time (RTX 3060 6 GB) | ~10–15 h              | ~3–5 days                              |

Start with **Small** to debug the whole pipeline end to end, then move to Main.

`scripts/train.py` is a plain PyTorch loop, which is simpler and faster than `Trainer` for packed memmap data:

```python
import math, os, time, numpy as np, torch
from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer

# ---------------- config ----------------
SIZE        = os.environ.get("SIZE", "small")
CTX         = 512 if SIZE == "small" else 1024
MICRO_BS    = 8 if SIZE == "small" else 4       # lower if OOM
GRAD_ACC    = 16 if SIZE == "small" else 32     # ~65k-131k tokens / step
LR, MIN_LR  = 6e-4, 6e-5
WARMUP      = 1000
MAX_STEPS   = int(os.environ.get("MAX_STEPS", 15000))
EVAL_EVERY  = 500
DTYPE       = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
dev         = "cuda"

tok = AutoTokenizer.from_pretrained("tokenizer")
cfg = LlamaConfig(
    vocab_size=len(tok),
    hidden_size=512 if SIZE == "small" else 768,
    intermediate_size=1408 if SIZE == "small" else 2048,
    num_hidden_layers=8 if SIZE == "small" else 12,
    num_attention_heads=8 if SIZE == "small" else 12,
    num_key_value_heads=8 if SIZE == "small" else 4,
    max_position_embeddings=CTX,
    rope_theta=10000.0,
    tie_word_embeddings=True,
    bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
    attn_implementation="sdpa",          # PyTorch flash/mem-efficient attention, no extra install
)
model = LlamaForCausalLM(cfg).to(dev)
model.gradient_checkpointing_enable()    # remove if VRAM allows -> ~30% faster
print(f"params: {model.num_parameters()/1e6:.1f}M")

# ---------------- data ----------------
dt = np.uint32 if len(tok) > 65535 else np.uint16
train = np.memmap("data/train.bin", dtype=dt, mode="r")
val   = np.memmap("data/val.bin",   dtype=dt, mode="r")

def batch(split):
    d = train if split == "train" else val
    ix = np.random.randint(0, len(d) - CTX - 1, MICRO_BS)
    x = torch.from_numpy(np.stack([d[i:i+CTX].astype(np.int64) for i in ix]))
    return x.pin_memory().to(dev, non_blocking=True)

# ---------------- optim ----------------
decay   = [p for n, p in model.named_parameters() if p.dim() >= 2]
nodecay = [p for n, p in model.named_parameters() if p.dim() < 2]
groups  = [{"params": decay, "weight_decay": 0.1}, {"params": nodecay, "weight_decay": 0.0}]
try:
    import bitsandbytes as bnb
    opt = bnb.optim.AdamW8bit(groups, lr=LR, betas=(0.9, 0.95))
except ImportError:
    opt = torch.optim.AdamW(groups, lr=LR, betas=(0.9, 0.95), fused=True)
scaler = torch.amp.GradScaler(enabled=(DTYPE == torch.float16))

def lr_at(step):
    if step < WARMUP: return LR * step / WARMUP
    p = (step - WARMUP) / max(1, MAX_STEPS - WARMUP)
    return MIN_LR + 0.5 * (LR - MIN_LR) * (1 + math.cos(math.pi * min(p, 1.0)))

@torch.no_grad()
def evaluate(n=50):
    model.eval(); losses = []
    for _ in range(n):
        x = batch("val")
        with torch.autocast("cuda", dtype=DTYPE):
            losses.append(model(input_ids=x, labels=x).loss.item())
    model.train(); return sum(losses) / n

# ---------------- resume ----------------
os.makedirs("ckpt", exist_ok=True)
step = 0
if os.path.exists("ckpt/state.pt"):
    s = torch.load("ckpt/state.pt", map_location=dev)
    model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); step = s["step"]
    print("resumed at", step)

# ---------------- loop ----------------
model.train(); t0 = time.time()
while step < MAX_STEPS:
    for g in opt.param_groups: g["lr"] = lr_at(step)
    for _ in range(GRAD_ACC):
        x = batch("train")
        with torch.autocast("cuda", dtype=DTYPE):
            loss = model(input_ids=x, labels=x).loss / GRAD_ACC
        scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
    step += 1

    if step % 50 == 0:
        tps = 50 * GRAD_ACC * MICRO_BS * CTX / (time.time() - t0); t0 = time.time()
        print(f"step {step} loss {loss.item()*GRAD_ACC:.3f} lr {lr_at(step):.2e} tok/s {tps:,.0f} "
              f"vram {torch.cuda.max_memory_allocated()/1e9:.2f}GB")
    if step % EVAL_EVERY == 0:
        vl = evaluate()
        print(f"== val loss {vl:.3f}  ppl {math.exp(vl):.1f}")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, "ckpt/state.pt")
        model.save_pretrained("ckpt/hf"); tok.save_pretrained("ckpt/hf")
```

```bash
set SIZE=small && set MAX_STEPS=12000 && python scripts/train.py
```

**If you hit an OOM error, try these in order:** halve `MICRO_BS` and double `GRAD_ACC` → keep gradient checkpointing on → `CTX=512` → drop layers from 12 to 10.

**Healthy loss curve (Hindi, 32k vocab):** about 10.4 at start, below 5 within 1k steps, and roughly **3.0–3.5 val loss** at the end. Val loss rising while train loss falls means overfitting, so you need more data rather than more steps.

### Phase 6: Multilingual (Hindi + other Indian languages)

The same pipeline works with three changes:

1. **Language grouping.** Start with one script family, then add others:
   - Step A: Devanagari (**hi, mr, ne, sa**), which share a script and so share tokens.
   - Step B: add Dravidian (**ta, te, kn, ml**) and Eastern (**bn, or, as**).
   - A ~135M model handles **4–8 languages** well. Beyond that, each language degrades.
2. **Temperature sampling** for the mix. Language *i* gets probability `p_i ∝ n_i^α` with `α = 0.3`:

```python
import numpy as np
tokens = {"hi": 1.5e9, "mr": 0.4e9, "ta": 0.5e9, "te": 0.4e9, "bn": 0.6e9}   # available tokens
alpha  = 0.3
w = {k: v**alpha for k, v in tokens.items()}; s = sum(w.values())
budget = 2.0e9
for k in tokens:
    take = budget * w[k] / s
    print(f"{k}: {take/1e9:.2f}B tokens ({take/tokens[k]:.1f} epochs)")
```

   Write each language into `train.bin` according to its share, repeating small languages (at most about 4 epochs) and truncating large ones. Then **shuffle at the document level** before packing, or interleave chunks, so batches mix languages.
3. **Language tags.** `prepare.py` already prefixes every document with `<hi>`, `<ta>`, and so on. At inference time, start the prompt with the tag to choose the output language.

Tokenizer: retrain at **vocab 64k** on an **equal sample per language** (Phase 3 note).

### Phase 7: Evaluate and generate

`scripts/generate.py`:

```python
import sys, torch
from transformers import AutoTokenizer, LlamaForCausalLM
tok = AutoTokenizer.from_pretrained("ckpt/hf")
m = LlamaForCausalLM.from_pretrained("ckpt/hf", torch_dtype=torch.bfloat16).cuda().eval()
prompt = sys.argv[1] if len(sys.argv) > 1 else "<hi>भारत की राजधानी"
x = tok(prompt, return_tensors="pt").to("cuda")
y = m.generate(**x, max_new_tokens=120, do_sample=True, temperature=0.8, top_p=0.95,
               repetition_penalty=1.15, eos_token_id=tok.eos_token_id)
print(tok.decode(y[0], skip_special_tokens=True))
```

Metrics to track:

- **Val perplexity per language** (separate `val_<lang>.bin` files).
- **Tokenizer fertility** (tokens per word) per language.
- Manual review of 20 samples per language for grammar and script mixing.
- Optional: IndicGLUE or IndicXTREME classification after a small fine-tune.

### Phase 8 (optional): Make it chat

1. Instruction data: **Aya dataset (Hindi subset)**, **IndicInstruct**, Alpaca-Hindi.
2. Format: `<hi><user>…</user><assistant>…</assistant></s>`. Add these as special tokens before pretraining so their embeddings already exist.
3. SFT for 1–3 epochs at LR 2e-5 using the same `train.py`, with the loss masked to assistant tokens.
4. Expect simple conversational replies, not factual accuracy.

---

## 4. Timeline

| Week | Work                                                                                                                   |
| ---- | ---------------------------------------------------------------------------------------------------------------------- |
| 1    | Install the stack. Download IndicCorp Hindi plus Wikipedia. Run `clean.py`.                                          |
| 2    | Train the tokenizer, check fertility, run `prepare.py`. Run a **Small** model for 1–2k steps as a smoke test. |
| 3    | Full **Small** run (~12k steps), then evaluate and tune LR and batch size.                                       |
| 4–5 | **Main** 110M Hindi run (~2B tokens).                                                                            |
| 6+   | Multilingual: retrain the tokenizer with a 64k vocab, add Devanagari languages, then Dravidian.                        |

## 5. Project layout

```
Indian-Language-llm-model/
├── PLAN.md
├── raw/            # downloaded corpora (gitignored)
├── clean/          # cleaned per-language text
├── tokenizer/      # spm.model + HF tokenizer files
├── data/           # train.bin, val.bin
├── ckpt/           # state.pt + hf/ export
└── scripts/
    ├── clean.py
    ├── train_tokenizer.py
    ├── prepare.py
    ├── train.py
    └── generate.py
```

## 6. Realistic expectations

- **Will do:** fluent, grammatical Hindi (or Tamil, Telugu, and so on), correct matras and conjuncts, a news or story style, and language switching via tags.
- **Won't do:** answer factual questions reliably, reason, or follow complex instructions.
- **To get a useful assistant:** keep this scratch model as the learning project, and separately run **QLoRA fine-tuning of a 1–3B open model** (for example Sarvam-1 2B or Gemma 2B). That run also fits on 6 GB in 4-bit.
