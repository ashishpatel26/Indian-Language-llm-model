# Indian-Language LLM from Scratch

Train a small **Llama-style Hindi language model from scratch on a single 6 GB consumer GPU**, then extend it to more Indian languages.

No pretrained weights and no `Trainer` abstraction. The pipeline is plain PyTorch plus a SentencePiece tokenizer, small enough to read in one sitting.

| | |
|---|---|
| **Languages** | Hindi now. Marathi, Nepali, Bengali, Tamil, Telugu, Gujarati, Kannada, Malayalam, Punjabi and Odia are planned (tags already in the tokenizer) |
| **Architecture** | Llama (RoPE, SwiGLU, RMSNorm, GQA, tied embeddings) via `transformers.LlamaForCausalLM` |
| **Sizes** | `tiny` (smoke test) · `small` ~42M · `main` ~110M |
| **Tokenizer** | SentencePiece Unigram, 32k vocab, byte fallback |
| **Hardware** | Tested target: RTX 3060 Laptop 6 GB, bf16 |

> **What to expect:** a model this size learns fluent, grammatical Hindi in a news or encyclopedic style. It will **not** know facts reliably or reason. For a useful assistant, QLoRA fine-tune a 1–3B open model instead. See [PLAN.md](PLAN.md#6-realistic-expectations).

---

## Pipeline

```
download.py ──► raw/hi.txt
clean.py    ──► clean/hi.txt          NFC, URL strip, script filter, exact dedup
train_tokenizer.py ──► tokenizer/spm.model
prepare.py  ──► data/train.bin, data/val.bin   packed uint16 token stream
train.py    ──► ckpt/state.pt, ckpt/hf/
generate.py      sample text
push_to_hub.py   publish to Hugging Face
```

## Quickstart

Requires Python 3.12, [uv](https://docs.astral.sh/uv/) and an NVIDIA GPU with CUDA 12.x drivers.

```bash
git clone https://github.com/ashishpatel26/Indian-Language-llm-model.git
cd Indian-Language-llm-model
uv sync                      # installs torch cu126 + deps
uv sync --extra opt          # optional: bitsandbytes 8-bit AdamW
```

### 1. Download data

```bash
uv run python scripts/download.py hi                 # Hindi Wikipedia (~160k articles)
SOURCE=sangraha uv run python scripts/download.py hi # larger: AI4Bharat Sangraha (needs `hf auth login`)
```

### 2. Clean

```bash
uv run python scripts/clean.py hi raw/hi.txt clean/hi.txt
```

This keeps lines that are at least 20 characters long and at least 80% target script (Devanagari for `hi`). It also NFC-normalizes text, strips URLs and removes exact duplicates.

### 3. Train the tokenizer

```bash
uv run python scripts/train_tokenizer.py clean/hi.txt 32000
```

The script prints **fertility** (tokens per word). Healthy Hindi is about 1.3–1.8. Above 3 means vocab or coverage is too small.

### 4. Tokenize and pack

```bash
uv run python scripts/prepare.py clean/hi.txt hi
```

Every document becomes `<hi> … </s>`, and the result is appended to one flat token stream. Every 200th line goes to validation. Packing means **no padding**, and training reads the stream through `np.memmap`.

### 5. Train

```bash
# bash
SIZE=small MAX_STEPS=12000 uv run python scripts/train.py
```

```powershell
# PowerShell
$env:SIZE="small"; $env:MAX_STEPS="12000"; uv run python scripts/train.py
```

| Env var | Default | Meaning |
|---|---|---|
| `SIZE` | `small` | `tiny`, `small` or `main` |
| `MAX_STEPS` | `12000` | optimizer steps (cosine LR, warmup = min(1000, 10%)) |
| `MICRO_BS` / `GRAD_ACC` | per size | micro-batch size and gradient accumulation |
| `OUT` | `ckpt` | checkpoint directory |

Training **resumes automatically** from `ckpt/state.pt`. Run `tiny` for a few hundred steps first to check the whole pipeline end to end.

**Out of memory?** Halve `MICRO_BS` and double `GRAD_ACC` to keep the same effective batch.

### 6. Generate

```bash
uv run python scripts/generate.py "भारत की राजधानी"   # one prompt
uv run python scripts/generate.py                   # interactive
```

### 7. Publish (optional)

```bash
uv run hf auth login
uv run python scripts/push_to_hub.py <user>/<repo> [ckpt] [--private]
```

This uploads the weights, the SentencePiece model and an auto-generated model card.

### GPU speed check

```bash
uv run python scripts/bench.py 0 8   # gradient checkpointing off, batch 8
```

A healthy RTX 3060 Laptop reaches more than 15k tokens/s on the `small` config.

---

## Model configs

| | tiny | small | main |
|---|---|---|---|
| hidden | 256 | 512 | 768 |
| layers | 4 | 8 | 12 |
| heads / kv-heads | 4 / 4 | 8 / 8 | 12 / 4 (GQA) |
| FFN (SwiGLU) | 704 | 1408 | 2048 |
| context | 256 | 512 | 1024 |
| params | — | **~42M** | **~110M** |
| token budget (~20 tok/param) | — | ~0.8B | ~2B |

**Healthy loss curve (Hindi, 32k vocab):** about 10.4 at step 0, below 5 within roughly 1k steps, and 3.0–3.5 val loss at the end. If val loss rises while train loss keeps falling, you need more data, not more steps.

## Design choices

- **Packed token stream rather than padded lines**, so compute isn't wasted on padding and the model sees real EOS tokens.
- **SentencePiece Unigram rather than byte-level BPE.** Indic characters are 3 UTF-8 bytes each, so byte-level BPE inflates token counts. `byte_fallback=True` means the tokenizer never emits `<unk>`.
- **bf16 rather than fp16**, which avoids loss spikes in scratch training. fp16 with GradScaler is used only as a fallback on GPUs without bf16.
- **Token-budgeted training rather than epochs.** One pass over more data beats several passes over less.
- **Language tags** (`<hi>`, `<ta>`, …) are reserved in the vocab, so adding languages later only needs the same pipeline, plus temperature sampling (α≈0.3) for the data mix.

See [PLAN.md](PLAN.md) for the full rationale, the multilingual roadmap and instruction-tuning notes.

## Project layout

```
├── PLAN.md                 full design doc and roadmap
├── pyproject.toml          uv project (torch cu126)
└── scripts/
    ├── download.py         HF datasets → raw/<lang>.txt
    ├── clean.py            normalize, filter, dedup
    ├── train_tokenizer.py  SentencePiece Unigram
    ├── prepare.py          tokenize + pack → data/*.bin
    ├── train.py            pretraining loop
    ├── generate.py         sampling
    ├── bench.py            GPU throughput check
    └── push_to_hub.py      publish to Hugging Face
```

`raw/`, `clean/`, `data/`, `tokenizer/` and `ckpt*/` are generated locally and gitignored.

## Status

- [x] Hindi Wikipedia downloaded and cleaned
- [x] 32k SentencePiece tokenizer trained
- [x] ~56M Hindi tokens packed
- [ ] `small` pretraining run
- [ ] Larger corpus (Sangraha / IndicCorp) + `main` run
- [ ] Per-language eval (val perplexity, fertility)
- [ ] Multilingual extension (Devanagari, then Dravidian)
- [ ] Optional chat SFT

## Data sources

[Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) · [AI4Bharat Sangraha](https://huggingface.co/datasets/ai4bharat/sangraha) · [IndicCorp v2](https://github.com/AI4Bharat/IndicBERT). Check each dataset's license before you redistribute a trained model.
