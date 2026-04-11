"""Diagnostic: check if ribosome val CE is real or an artifact."""
import torch
import numpy as np
import sys, os
os.environ['HF_HOME'] = '/var/hf_cache'
sys.path.insert(0, '/var/ribosome-cascade')

from native_arch_v1 import RibosomeCascadeNative
from transformers import AutoTokenizer
from datasets import load_dataset

device = torch.device('cuda')
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

model = RibosomeCascadeNative(
    vocab_size=len(tokenizer), hidden_size=768, n_heads=12,
    lower_layers=4, upper_layers=4, cascade_layers=2,
    max_seq_len=256, max_chunks=64
).to(device)

ckpt = torch.load('/var/ribosome-cascade/native_wt103/best.pt',
                   map_location=device, weights_only=False)
model.load_state_dict(ckpt['model'])
model.eval()
print(f"Loaded step={ckpt['step']}, reported val={ckpt['val_loss']:.4f}")

# Load val data
ds = load_dataset('wikitext', 'wikitext-103-raw-v1', split='validation')
ds = ds.filter(lambda x: len(x['text'].strip()) > 20)
all_ids = []
for ex in ds:
    all_ids.extend(tokenizer.encode(ex['text']))
print(f"Val tokens: {len(all_ids):,}")

n_eval = 100
seq_len = 256

losses_full = []
losses_bypass = []
all_imp_means = []
all_imp_stds = []

with torch.no_grad():
    for i in range(0, min(len(all_ids) - seq_len - 1, n_eval * seq_len), seq_len):
        inp = torch.tensor(all_ids[i:i+seq_len], dtype=torch.long).unsqueeze(0).to(device)
        lab = torch.tensor(all_ids[i+1:i+seq_len+1], dtype=torch.long).unsqueeze(0).to(device)

        # Full model
        model.ribosome_alpha = 1.0
        loss_f, _, importance = model(inp, lab)
        losses_full.append(loss_f.item())
        all_imp_means.append(importance.mean().item())
        all_imp_stds.append(importance.std().item())

        # Bypass only (lower transformer, no ribosome)
        model.ribosome_alpha = 0.0
        loss_b, _, _ = model(inp, lab)
        losses_bypass.append(loss_b.item())

print(f"\n=== RESULTS ({n_eval} samples, len={seq_len}) ===")
print(f"Full model  (alpha=1.0): CE = {np.mean(losses_full):.4f}")
print(f"Bypass only (alpha=0.0): CE = {np.mean(losses_bypass):.4f}")
print(f"Delta (full - bypass):        {np.mean(losses_full) - np.mean(losses_bypass):.4f}")
print(f"\nImportance scores:")
print(f"  mean: {np.mean(all_imp_means):.4f} +/- {np.std(all_imp_means):.4f}")
print(f"  std:  {np.mean(all_imp_stds):.4f}")

# Check if model generates coherent text
print(f"\n=== GENERATION TEST ===")
prompt = "The history of artificial intelligence"
input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
model.ribosome_alpha = 1.0

with torch.no_grad():
    for _ in range(50):
        outputs = model.base_model if hasattr(model, 'base_model') else None
        loss, logits, _ = model(input_ids)
        next_token = logits[0, -1].argmax()
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)

print(tokenizer.decode(input_ids[0]))
