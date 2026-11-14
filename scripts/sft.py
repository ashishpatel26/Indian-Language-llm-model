"""Supervised fine-tuning (instruction tuning) of a base checkpoint -> Angad.
Loss only on answer tokens. Format: <hi>प्रश्न: {q}\nउत्तर: {a}</s>
env: BASE=ckpt_small OUT=ckpt_angad EPOCHS=2 LR=1e-4 MICRO_BS=8 GRAD_ACC=4 EVAL_EVERY=300
usage: uv run python scripts/sft.py
"""
import json, math, os, random, time
import torch
import sentencepiece as spm
from transformers import LlamaForCausalLM

BASE = os.environ.get("BASE", "ckpt_small")
OUT = os.environ.get("OUT", "ckpt_angad")
EPOCHS = float(os.environ.get("EPOCHS", 2))
LR = float(os.environ.get("LR", 1e-4))
MICRO_BS = int(os.environ.get("MICRO_BS", 8))
GRAD_ACC = int(os.environ.get("GRAD_ACC", 4))
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", 300))
PROMPT = "<hi>प्रश्न: {q}\nउत्तर: "
TEST_QS = ["भारत की राजधानी क्या है?", "स्वस्थ रहने के तीन उपाय बताइए।", "होली का त्योहार क्यों मनाया जाता है?"]
dev = "cuda"

sp = spm.SentencePieceProcessor(model_file="tokenizer/spm.model")
model = LlamaForCausalLM.from_pretrained(f"{BASE}/hf").to(dev)
CTX = model.config.max_position_embeddings


def encode(path):
    out, dropped = [], 0
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        p, a = sp.encode(PROMPT.format(q=r["q"])), sp.encode(r["a"]) + [sp.eos_id()]
        if len(p) > CTX // 2:
            dropped += 1; continue
        a = a[:CTX - len(p)]  # long answers are cut (and lose </s>, so the model never learns to stop mid-answer)
        out.append((p + a, [-100] * len(p) + a))
    print(f"{path}: {len(out)} examples, {dropped} too long")
    return out


train, val = encode("data/sft_train.jsonl"), encode("data/sft_val.jsonl")


def collate(items):
    n = max(len(x) for x, _ in items)
    ids = torch.full((len(items), n), sp.pad_id()); lab = torch.full((len(items), n), -100)
    for i, (x, y) in enumerate(items):
        ids[i, :len(x)] = torch.tensor(x); lab[i, :len(y)] = torch.tensor(y)
    return ids.to(dev), lab.to(dev), (ids != sp.pad_id()).long().to(dev)


def batches(data, bs, shuffle):
    idx = list(range(len(data)))
    if shuffle:  # bucket by length so padding stays small
        random.shuffle(idx)
        idx = [i for k in range(0, len(idx), bs * 50)
               for i in sorted(idx[k:k + bs * 50], key=lambda i: len(data[i][0]))]
        chunks = [idx[k:k + bs] for k in range(0, len(idx), bs)]; random.shuffle(chunks)
    else:
        chunks = [idx[k:k + bs] for k in range(0, len(idx), bs)]
    for c in chunks:
        yield collate([data[i] for i in c])


steps_per_epoch = len(train) // (MICRO_BS * GRAD_ACC)
MAX_STEPS = int(steps_per_epoch * EPOCHS)
WARMUP = max(1, MAX_STEPS // 30)
opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.0, fused=True)


def lr_at(s):
    if s < WARMUP:
        return LR * (s + 1) / WARMUP
    return LR * 0.1 + 0.45 * LR * (1 + math.cos(math.pi * min(1, (s - WARMUP) / max(1, MAX_STEPS - WARMUP))))


@torch.no_grad()
def evaluate():
    model.eval(); tot = n = 0
    for ids, lab, att in batches(val, 16, False):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tot += model(input_ids=ids, attention_mask=att, labels=lab).loss.item(); n += 1
    model.train(); return tot / max(1, n)


@torch.no_grad()
def answer(q):
    model.eval()
    x = torch.tensor([sp.encode(PROMPT.format(q=q))], device=dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = model.generate(x, max_new_tokens=80, do_sample=True, temperature=0.7, top_p=0.9,
                           repetition_penalty=1.15, eos_token_id=sp.eos_id(), pad_token_id=sp.pad_id())
    model.train(); return sp.decode(y[0, x.shape[1]:].tolist())


os.makedirs(OUT, exist_ok=True)
state_path, step = f"{OUT}/state.pt", 0
if os.path.exists(state_path):
    s = torch.load(state_path, map_location=dev)
    model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); step = s["step"]
    print("resumed at step", step)


def save():
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, state_path + ".tmp")
    os.replace(state_path + ".tmp", state_path)
    model.save_pretrained(f"{OUT}/hf")
    print(f"saved {OUT}/ at step {step}", flush=True)


print(f"base {BASE} | train {len(train)} | steps {MAX_STEPS} ({EPOCHS} epochs, {MICRO_BS * GRAD_ACC}/step)")
print(f"== start val loss {evaluate():.3f}", flush=True)
model.train(); t0 = time.time(); tok = 0
random.seed(step)
stream = iter(())
try:
    while step < MAX_STEPS:
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        for _ in range(GRAD_ACC):
            try:
                ids, lab, att = next(stream)
            except StopIteration:
                stream = batches(train, MICRO_BS, True); ids, lab, att = next(stream)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=ids, attention_mask=att, labels=lab).loss / GRAD_ACC
            loss.backward(); tok += int(att.sum())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True); step += 1
        if step % 20 == 0:
            dt = time.time() - t0; t0 = time.time()
            print(f"step {step}/{MAX_STEPS} loss {loss.item() * GRAD_ACC:.3f} lr {lr_at(step):.1e} "
                  f"tok/s {tok / dt:,.0f} eta {(MAX_STEPS - step) * dt / 20 / 3600:.2f}h", flush=True); tok = 0
        if step % EVAL_EVERY == 0 or step == MAX_STEPS:
            print(f"== step {step} val loss {evaluate():.3f}", flush=True)
            for q in TEST_QS:
                print(f"   Q: {q}\n   A: {answer(q)}", flush=True)
            save()
except KeyboardInterrupt:
    print("\nCtrl+C: saving checkpoint, do NOT press again...")
    save()
