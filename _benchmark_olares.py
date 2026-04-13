"""Quick benchmark: load olares RibosomeTiny weights, eval on wikitext-103 val."""
import torch
import torch.nn.functional as F
import numpy as np
import time, os, sys, json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp2_lighter import RibosomeTiny
from train_native import get_wikitext_loader
from transformers import AutoTokenizer

device = torch.device("cuda")
print(f"GPU: {torch.cuda.get_device_name()}")

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
V = len(tokenizer)

# Load model
model = RibosomeTiny(vocab_size=V, hidden_size=512, n_heads=8,
                     embed_layers=2, upper_layers=4,
                     max_seq_len=256, n_chunks=16).to(device)

ckpt = torch.load("weights_olares/best.pt", map_location=device, weights_only=False)
model.load_state_dict(ckpt["model"])
print(f"Loaded best.pt: step={ckpt['step']}, val_loss={ckpt['val_loss']:.4f}, params={ckpt['params']:,}")

# Eval on wikitext-103 val
val_loader = get_wikitext_loader(tokenizer, 256, 8, "validation", "wikitext-103-raw-v1")

model.eval()
losses = []
importance_stats = []
t0 = time.time()

with torch.no_grad():
    for i, batch in enumerate(val_loader):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        loss, logits, importance = model(input_ids, labels)
        losses.append(loss.item())

        # Collect importance score statistics
        imp = importance.detach().cpu().numpy()
        importance_stats.append({
            "mean": float(np.mean(imp)),
            "std": float(np.std(imp)),
            "min": float(np.min(imp)),
            "max": float(np.max(imp)),
            "entropy": float(-np.mean(imp * np.log(imp + 1e-8) + (1-imp) * np.log(1-imp + 1e-8)))
        })

elapsed = time.time() - t0
val_ce = np.mean(losses)
val_ppl = np.exp(val_ce)

print(f"\n{'='*60}")
print(f"OLARES BENCHMARK — RibosomeTiny (49M params)")
print(f"{'='*60}")
print(f"  Wikitext-103 val CE:  {val_ce:.4f}")
print(f"  Wikitext-103 val PPL: {val_ppl:.2f}")
print(f"  Batches evaluated:    {len(losses)}")
print(f"  Time:                 {elapsed:.1f}s")
print(f"\n  Importance scores:")
print(f"    mean:    {np.mean([s['mean'] for s in importance_stats]):.4f}")
print(f"    std:     {np.mean([s['std'] for s in importance_stats]):.4f}")
print(f"    min:     {np.min([s['min'] for s in importance_stats]):.4f}")
print(f"    max:     {np.max([s['max'] for s in importance_stats]):.4f}")
print(f"    entropy: {np.mean([s['entropy'] for s in importance_stats]):.4f}")
print(f"{'='*60}")

results = {
    "model": "RibosomeTiny",
    "checkpoint": "weights_olares/best.pt",
    "step": ckpt["step"],
    "params": ckpt["params"],
    "val_ce": float(val_ce),
    "val_ppl": float(val_ppl),
    "importance": {
        "mean": float(np.mean([s['mean'] for s in importance_stats])),
        "std": float(np.mean([s['std'] for s in importance_stats])),
        "entropy": float(np.mean([s['entropy'] for s in importance_stats])),
    },
    "eval_batches": len(losses),
    "eval_time_s": round(elapsed, 1),
}
with open("benchmark_olares.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to benchmark_olares.json")
