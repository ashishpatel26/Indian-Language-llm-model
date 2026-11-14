"""Pretrain a Llama-style model from scratch on data/train.bin.
env: SIZE=small|main  MAX_STEPS=12000  MICRO_BS / GRAD_ACC to override  OUT=ckpt  DATA=data
usage: uv run python scripts/train.py
"""
import math, os, time
import numpy as np, torch
import sentencepiece as spm
from transformers import LlamaConfig, LlamaForCausalLM

SIZES = {  #        hidden ffn  layers heads kv  ctx  micro_bs grad_acc
    "tiny":  dict(h=256, ffn=704,  L=4,  nh=4,  kv=4, ctx=256,  bs=16, acc=4),   # smoke test only
    "small": dict(h=512, ffn=1408, L=8,  nh=8,  kv=8, ctx=512,  bs=8,  acc=16),  # ~42M
    "main":  dict(h=768, ffn=2048, L=12, nh=12, kv=4, ctx=1024, bs=4,  acc=32),  # ~110M
}
S = SIZES[os.environ.get("SIZE", "small")]
CTX = S["ctx"]
MICRO_BS = int(os.environ.get("MICRO_BS", S["bs"]))
GRAD_ACC = int(os.environ.get("GRAD_ACC", S["acc"]))
MAX_STEPS = int(os.environ.get("MAX_STEPS", 12000))
LR, MIN_LR, WARMUP = 6e-4, 6e-5, min(1000, MAX_STEPS // 10)
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", 500))
OUT = os.environ.get("OUT", "ckpt")
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True

sp = spm.SentencePieceProcessor(model_file="tokenizer/spm.model")
cfg = LlamaConfig(
    vocab_size=sp.get_piece_size(), hidden_size=S["h"], intermediate_size=S["ffn"],
    num_hidden_layers=S["L"], num_attention_heads=S["nh"], num_key_value_heads=S["kv"],
    max_position_embeddings=CTX, tie_word_embeddings=True,
    bos_token_id=sp.bos_id(), eos_token_id=sp.eos_id(), pad_token_id=sp.pad_id(),
)
cfg._attn_implementation = "sdpa"
model = LlamaForCausalLM(cfg).to(dev)
if os.environ.get("CKPT", "1") == "1":
    model.gradient_checkpointing_enable()
model.config.use_cache = False
print(f"params {model.num_parameters()/1e6:.1f}M | tokens/step {MICRO_BS*GRAD_ACC*CTX:,}")

DATA = os.environ.get("DATA", "data")
train = np.memmap(f"{DATA}/train.bin", dtype=np.uint16, mode="r")
val = np.memmap(f"{DATA}/val.bin", dtype=np.uint16, mode="r")
print(f"train {len(train)/1e6:.1f}M tok | val {len(val)/1e6:.1f}M tok | "
      f"epochs at end ~{MAX_STEPS*MICRO_BS*GRAD_ACC*CTX/len(train):.2f}")

def batch(d):
    ix = np.random.randint(0, len(d) - CTX - 1, MICRO_BS)
    x = torch.from_numpy(np.stack([d[i:i + CTX].astype(np.int64) for i in ix]))
    return x.pin_memory().to(dev, non_blocking=True)

params = [p for p in model.parameters() if p.requires_grad]
groups = [{"params": [p for p in params if p.dim() >= 2], "weight_decay": 0.1},
          {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0}]
opt = torch.optim.AdamW(groups, lr=LR, betas=(0.9, 0.95), fused=True)
scaler = torch.amp.GradScaler(enabled=(DTYPE == torch.float16))

def lr_at(step):
    if step < WARMUP:
        return LR * (step + 1) / WARMUP
    p = min((step - WARMUP) / max(1, MAX_STEPS - WARMUP), 1.0)
    return MIN_LR + 0.5 * (LR - MIN_LR) * (1 + math.cos(math.pi * p))

@torch.no_grad()
def evaluate(n=40):
    model.eval(); tot = 0.0
    for _ in range(n):
        x = batch(val)
        with torch.autocast("cuda", dtype=DTYPE):
            tot += model(input_ids=x, labels=x).loss.item()
    model.train(); return tot / n

@torch.no_grad()
def sample(prompt="<hi>भारत"):
    model.eval(); model.config.use_cache = True
    x = torch.tensor([sp.encode(prompt)], device=dev)
    with torch.autocast("cuda", dtype=DTYPE):
        y = model.generate(x, max_new_tokens=40, do_sample=True, top_p=0.9, temperature=0.8,
                           eos_token_id=sp.eos_id(), pad_token_id=sp.pad_id())
    model.config.use_cache = False; model.train()
    return sp.decode(y[0].tolist())

os.makedirs(OUT, exist_ok=True)
state_path, step = f"{OUT}/state.pt", 0
if os.path.exists(state_path):
    s = torch.load(state_path, map_location=dev)
    model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); step = s["step"]
    print("resumed at step", step)

def save():
    # write to temp then atomic rename: a kill mid-save never corrupts the last good checkpoint
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, state_path + ".tmp")
    os.replace(state_path + ".tmp", state_path)
    model.save_pretrained(f"{OUT}/hf")
    print(f"saved {OUT}/ at step {step}")

model.train(); t0 = time.time()
try:
    while step < MAX_STEPS:
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        for _ in range(GRAD_ACC):
            x = batch(train)
            with torch.autocast("cuda", dtype=DTYPE):
                loss = model(input_ids=x, labels=x).loss / GRAD_ACC
            scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        step += 1
        if step % 20 == 0:
            dt = time.time() - t0; t0 = time.time()
            tps = 20 * GRAD_ACC * MICRO_BS * CTX / dt
            eta_h = (MAX_STEPS - step) * dt / 20 / 3600
            print(f"step {step} loss {loss.item()*GRAD_ACC:.3f} lr {lr_at(step):.1e} "
                  f"tok/s {tps:,.0f} vram {torch.cuda.max_memory_allocated()/1e9:.2f}GB eta {eta_h:.1f}h", flush=True)
        if step % EVAL_EVERY == 0 or step == MAX_STEPS:
            vl = evaluate()
            print(f"== step {step} val loss {vl:.3f} ppl {math.exp(vl):.1f}\n   sample: {sample()}", flush=True)
            save()
except KeyboardInterrupt:
    print("\nCtrl+C: saving checkpoint, do NOT press again...")
    save()
