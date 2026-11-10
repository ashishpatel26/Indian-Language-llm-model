"""GPU speed check. usage: uv run python scripts/bench.py 0 8   (ckpt off, batch 8). Healthy RTX 3060 laptop: >15k tok/s."""
import time, torch
from transformers import LlamaConfig, LlamaForCausalLM
import sys
ck = sys.argv[1] == "1"; bs = int(sys.argv[2]); ctx = 512
cfg = LlamaConfig(vocab_size=32000, hidden_size=512, intermediate_size=1408, num_hidden_layers=8,
                  num_attention_heads=8, num_key_value_heads=8, max_position_embeddings=ctx, tie_word_embeddings=True)
cfg._attn_implementation = "sdpa"
m = LlamaForCausalLM(cfg).cuda(); m.config.use_cache=False
if ck: m.gradient_checkpointing_enable()
opt = torch.optim.AdamW(m.parameters(), lr=1e-4, fused=True)
x = torch.randint(0, 32000, (bs, ctx), device="cuda")
for i in range(25):
    if i == 5: torch.cuda.synchronize(); t = time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        l = m(input_ids=x, labels=x).loss
    l.backward(); opt.step(); opt.zero_grad()
torch.cuda.synchronize()
print(f"ckpt={ck} bs={bs} tok/s {20*bs*ctx/(time.time()-t):,.0f} vram {torch.cuda.max_memory_allocated()/1e9:.2f}GB attn={m.config._attn_implementation}")
