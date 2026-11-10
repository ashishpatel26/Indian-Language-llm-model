"""Generate Hindi text from a trained checkpoint.
usage: uv run python scripts/generate.py "भारत की राजधानी" [ckpt]
       uv run python scripts/generate.py            (interactive)
"""
import sys, torch
import sentencepiece as spm
from transformers import LlamaForCausalLM

ckpt = sys.argv[2] if len(sys.argv) > 2 else "ckpt"
sp = spm.SentencePieceProcessor(model_file="tokenizer/spm.model")
model = LlamaForCausalLM.from_pretrained(f"{ckpt}/hf", dtype=torch.bfloat16).cuda().eval()
model.config.use_cache = True

def gen(prompt, lang="hi", n=150):
    x = torch.tensor([sp.encode(f"<{lang}>" + prompt)], device="cuda")
    y = model.generate(x, max_new_tokens=n, do_sample=True, temperature=0.8, top_p=0.95,
                       repetition_penalty=1.15, eos_token_id=sp.eos_id(), pad_token_id=sp.pad_id())
    return sp.decode(y[0].tolist())

if len(sys.argv) > 1 and sys.argv[1]:
    print(gen(sys.argv[1]))
else:
    while (p := input("\nप्रॉम्प्ट> ").strip()):
        print(gen(p))
