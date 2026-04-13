"""Benchmark: BigBaseline (frank) vs RibosomeTiny (olares) on wikitext-103 val."""
import torch
import torch.nn.functional as F
import numpy as np
import time, os, sys, json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp2_lighter import BigBaseline, RibosomeTiny
from train_native import get_wikitext_loader
from transformers import AutoTokenizer

device = torch.device("cuda")
print(f"GPU: {torch.cuda.get_device_name()}")

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
V = len(tokenizer)

val_loader = get_wikitext_loader(tokenizer, 256, 8, "validation", "wikitext-103-raw-v1")

# ---- BigBaseline (frank) ----
big = BigBaseline(vocab_size=V, hidden_size=512, n_heads=8,
                  n_layers=12, max_seq_len=256).to(device)
ckpt_big = torch.load("weights_frank/best.pt", map_location=device, weights_only=False)
big.load_state_dict(ckpt_big["model"])
print(f"\nBigBaseline: step={ckpt_big['step']}, val_loss={ckpt_big['val_loss']:.4f}, params={ckpt_big['params']:,}")

big.eval()
big_losses = []
t0 = time.time()
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        loss, logits = big(input_ids, labels)
        big_losses.append(loss.item())
big_time = time.time() - t0
big_ce = np.mean(big_losses)
big_ppl = np.exp(big_ce)
del big; torch.cuda.empty_cache()

# ---- RibosomeTiny (olares) ----
tiny = RibosomeTiny(vocab_size=V, hidden_size=512, n_heads=8,
                    embed_layers=2, upper_layers=4,
                    max_seq_len=256, n_chunks=16).to(device)
ckpt_tiny = torch.load("weights_olares/best.pt", map_location=device, weights_only=False)
tiny.load_state_dict(ckpt_tiny["model"])
print(f"RibosomeTiny: step={ckpt_tiny['step']}, val_loss={ckpt_tiny['val_loss']:.4f}, params={ckpt_tiny['params']:,}")

tiny.eval()
tiny_losses = []
importance_stats = []
t0 = time.time()
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        loss, logits, importance = tiny(input_ids, labels)
        tiny_losses.append(loss.item())
        imp = importance.detach().cpu().numpy()
        importance_stats.append({
            "mean": float(np.mean(imp)),
            "std": float(np.std(imp)),
            "min": float(np.min(imp)),
            "max": float(np.max(imp)),
        })
tiny_time = time.time() - t0
tiny_ce = np.mean(tiny_losses)
tiny_ppl = np.exp(tiny_ce)
del tiny; torch.cuda.empty_cache()

# ---- Results ----
print(f"\n{'='*65}")
print(f"  EXP2 HEAD-TO-HEAD: BigBaseline vs RibosomeTiny")
print(f"{'='*65}")
print(f"  {'':30s} {'Big 12L':>12s}  {'Ribo Tiny':>12s}")
print(f"  {'-'*56}")
print(f"  {'Machine':30s} {'frank':>12s}  {'olares':>12s}")
print(f"  {'GPU':30s} {'GTX 1070':>12s}  {'5090 Laptop':>12s}")
print(f"  {'Params':30s} {ckpt_big['params']:>12,}  {ckpt_tiny['params']:>12,}")
print(f"  {'Layers':30s} {'12':>12s}  {'2+4':>12s}")
print(f"  {'Tokens processed':30s} {'256 raw':>12s}  {'16 meta':>12s}")
print(f"  {'Best step':30s} {ckpt_big['step']:>12,}  {ckpt_tiny['step']:>12,}")
print(f"  {'-'*56}")
print(f"  {'Val CE':30s} {big_ce:>12.4f}  {tiny_ce:>12.4f}")
print(f"  {'Val PPL':30s} {big_ppl:>12.2f}  {tiny_ppl:>12.2f}")
print(f"  {'PPL ratio':30s} {'':>12s}  {big_ppl/tiny_ppl:>11.1f}x")
print(f"  {'-'*56}")
imp_mean = np.mean([s['mean'] for s in importance_stats])
imp_std = np.mean([s['std'] for s in importance_stats])
print(f"  {'Importance mean':30s} {'n/a':>12s}  {imp_mean:>12.4f}")
print(f"  {'Importance std':30s} {'n/a':>12s}  {imp_std:>12.4f}")
print(f"{'='*65}")

results = {
    "big": {"params": ckpt_big["params"], "step": ckpt_big["step"],
            "val_ce": float(big_ce), "val_ppl": float(big_ppl)},
    "tiny": {"params": ckpt_tiny["params"], "step": ckpt_tiny["step"],
             "val_ce": float(tiny_ce), "val_ppl": float(tiny_ppl),
             "importance_mean": imp_mean, "importance_std": imp_std},
}
with open("benchmark_exp2_comparison.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to benchmark_exp2_comparison.json")
