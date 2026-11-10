"""Upload trained model + tokenizer to Hugging Face Hub.
Run `uv run hf auth login` first (paste a WRITE token).
usage: uv run python scripts/push_to_hub.py <username>/<repo-name> [ckpt] [--private]
"""
import shutil, sys
from huggingface_hub import HfApi

repo, ckpt = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "ckpt")
private = "--private" in sys.argv

# Bundle tokenizer next to weights so the repo is self-contained.
for f in ("spm.model", "spm.vocab"):
    shutil.copy(f"tokenizer/{f}", f"{ckpt}/hf/{f}")

card = f"""---
language: hi
license: apache-2.0
tags: [hindi, llama, from-scratch, causal-lm]
---
# {repo.split('/')[-1]}

Small Llama-style Hindi language model trained **from scratch** (random init) on a single RTX 3060 6 GB.
Generates fluent Hindi text; not instruction-tuned, not factual.

```python
import torch, sentencepiece as spm
from huggingface_hub import hf_hub_download
from transformers import LlamaForCausalLM
m = LlamaForCausalLM.from_pretrained("{repo}")
sp = spm.SentencePieceProcessor(model_file=hf_hub_download("{repo}", "spm.model"))
x = torch.tensor([sp.encode("<hi>भारत की राजधानी")])
y = m.generate(x, max_new_tokens=100, do_sample=True, top_p=0.95, temperature=0.8)
print(sp.decode(y[0].tolist()))
```
"""
open(f"{ckpt}/hf/README.md", "w", encoding="utf-8").write(card)

api = HfApi()
api.create_repo(repo, private=private, exist_ok=True)
api.upload_folder(folder_path=f"{ckpt}/hf", repo_id=repo, commit_message="Upload Hindi from-scratch model")
print(f"https://huggingface.co/{repo}")
