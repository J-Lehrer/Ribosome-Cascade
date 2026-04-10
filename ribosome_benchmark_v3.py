"""
Ribosome-Cascade v3: Layer Unfreezing Experiment
=================================================
Tests whether unfreezing GPT-2's upper transformer layers improves
the ribosome's ability to discriminate token importance.

Conditions:
  1. frozen    — all GPT-2 layers frozen (current approach)
  2. top2      — unfreeze layers 10-11
  3. top4      — unfreeze layers 8-11
  4. top6      — unfreeze layers 6-11
  5. uniform   — no ribosome, own lm_head (control, matched unfreezing)

For each condition, we train ribosome + lm_head + unfrozen layers,
then evaluate on the same metrics as v2.
"""

import argparse
import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig
from datasets import load_dataset


class RibosomeScorer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, hidden_states):
        return self.scorer(hidden_states).squeeze(-1)


class RibosomeCascadeModel(nn.Module):
    def __init__(self, base_model, ribosome):
        super().__init__()
        self.base_model = base_model
        self.ribosome = ribosome
        self.lm_head = nn.Linear(
            base_model.config.n_embd,
            base_model.config.vocab_size,
            bias=False
        )

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        importance = self.ribosome(hidden_states)
        weighted_states = hidden_states * importance.unsqueeze(-1)
        logits = self.lm_head(weighted_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        return loss, logits, importance


class UniformBaselineModel(nn.Module):
    def __init__(self, base_model, vocab_size, hidden_size):
        super().__init__()
        self.base_model = base_model
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        logits = self.lm_head(outputs.last_hidden_state)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        return loss, logits


# ============================================================
# FREEZING UTILS
# ============================================================

def freeze_all(model):
    for p in model.base_model.parameters():
        p.requires_grad = False

def unfreeze_top_n(model, n_layers):
    """Unfreeze the top N transformer blocks of GPT-2 (12 total)."""
    freeze_all(model)
    total_layers = len(model.base_model.h)  # GPT-2 has .h for transformer blocks
    start = total_layers - n_layers
    for i in range(start, total_layers):
        for p in model.base_model.h[i].parameters():
            p.requires_grad = True
    # Also unfreeze final layer norm
    for p in model.base_model.ln_f.parameters():
        p.requires_grad = True

def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# METRICS
# ============================================================

def score_entropy(scores):
    s = scores.clamp(1e-8, 1.0)
    s = s / s.sum()
    return -(s * s.log()).sum().item()

def peak_valley_ratio(scores):
    return (scores.max() / (scores.min() + 1e-8)).item()

def sparsity_fraction(scores, threshold=0.3):
    return (scores < threshold).float().mean().item()


# ============================================================
# DATA
# ============================================================

_cached_dataset = {}

def get_dataloader(tokenizer, split="train", max_length=64, batch_size=16):
    cache_key = f"{split}_{max_length}"
    if cache_key not in _cached_dataset:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        ds = ds.filter(lambda x: len(x["text"].strip()) > 0)

        def tok_fn(examples):
            t = tokenizer(examples["text"], padding="max_length",
                          truncation=True, max_length=max_length)
            t["labels"] = t["input_ids"].copy()
            return t

        ds = ds.map(tok_fn, batched=True, remove_columns=["text"])
        ds.set_format("torch")
        _cached_dataset[cache_key] = ds

    return DataLoader(_cached_dataset[cache_key], batch_size=batch_size, shuffle=(split == "train"))


# ============================================================
# TRAINING
# ============================================================

def train_model(model, tokenizer, device, epochs, max_length, batch_size,
                lr, sparsity_coeff, label, is_ribosome=True):
    print(f"\n{'='*60}")
    print(f"TRAINING: {label}")
    print(f"Trainable params: {count_trainable(model):,}")
    print(f"{'='*60}")

    loader = get_dataloader(tokenizer, "train", max_length, batch_size)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    model.train()
    history = []
    for epoch in range(epochs):
        losses = []
        t0 = time.time()
        for step, batch in enumerate(loader):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if is_ribosome:
                recon_loss, _, importance = model(ids, mask, labels)
                sparsity_loss = importance.mean()
                loss = recon_loss + sparsity_coeff * sparsity_loss
            else:
                recon_loss, _ = model(ids, mask, labels)
                loss = recon_loss
                sparsity_loss = torch.tensor(0.0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(recon_loss.item())
            if step % 100 == 0:
                sp_str = f"  sparsity={sparsity_loss.item():.4f}" if is_ribosome else ""
                print(f"  epoch {epoch+1}/{epochs}  step {step:4d}  "
                      f"CE={recon_loss.item():.4f}{sp_str}")

        mean_loss = np.mean(losses)
        history.append(mean_loss)
        print(f"  epoch {epoch+1} done  mean_CE={mean_loss:.4f}  time={time.time()-t0:.1f}s")

    return history


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model, tokenizer, device, seq_len, n_samples=50,
                   is_ribosome=True):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ds = ds.filter(lambda x: len(x["text"].strip()) > 20)
    texts = [x["text"] for x in ds]

    inputs_list = []
    for t in texts:
        enc = tokenizer(t, truncation=True, max_length=seq_len,
                        padding="max_length", return_tensors="pt")
        real_tokens = (enc["input_ids"] != tokenizer.pad_token_id).sum().item()
        if real_tokens >= seq_len * 0.7:
            inputs_list.append(enc)
        if len(inputs_list) >= n_samples:
            break

    if not inputs_list:
        return None

    model.eval()
    losses, entropies, pvrs, sparsities = [], [], [], []

    with torch.no_grad():
        for enc in inputs_list:
            ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            labels = ids.clone()

            if is_ribosome:
                loss, _, importance = model(ids, mask, labels)
                real_mask = mask[0].bool()
                real_scores = importance[0][real_mask]
                entropies.append(score_entropy(real_scores))
                pvrs.append(peak_valley_ratio(real_scores))
                sparsities.append(sparsity_fraction(real_scores))
            else:
                loss, _ = model(ids, mask, labels)

            losses.append(loss.item())

    result = {"loss": float(np.mean(losses)), "n": len(inputs_list)}
    if is_ribosome:
        result["entropy"] = float(np.mean(entropies))
        result["pvr"] = float(np.mean(pvrs))
        result["sparsity"] = float(np.mean(sparsities))
    return result


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_train_len", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--sparsity_coeff", type=float, default=0.1)
    parser.add_argument("--eval_lengths", nargs="+", type=int,
                        default=[32, 64, 128, 256, 512])
    parser.add_argument("--eval_samples", type=int, default=50)
    parser.add_argument("--weights_dir", default="E:/Ribosome-Cascade/weights_v3")
    parser.add_argument("--output", default="E:/Ribosome-Cascade/benchmark_results_v3.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    os.makedirs(args.weights_dir, exist_ok=True)

    # -------------------------------------------------------
    # Define conditions: (label, n_unfrozen_layers)
    # -------------------------------------------------------
    conditions = [
        ("frozen",  0),
        ("top2",    2),
        ("top4",    4),
        ("top6",    6),
    ]

    all_results = {}

    for label, n_unfreeze in conditions:
        weight_path = os.path.join(args.weights_dir, f"{label}.pt")
        uniform_path = os.path.join(args.weights_dir, f"{label}_uniform.pt")

        # --- Build fresh models (each condition gets its own GPT-2 copy) ---
        print(f"\n{'#'*60}")
        print(f"CONDITION: {label} (unfreezing top {n_unfreeze} layers)")
        print(f"{'#'*60}")

        base_model = AutoModel.from_pretrained("gpt2").to(device)
        hidden_size = base_model.config.n_embd
        vocab_size = base_model.config.vocab_size

        ribosome = RibosomeScorer(hidden_size).to(device)
        ribo_model = RibosomeCascadeModel(base_model, ribosome).to(device)

        if n_unfreeze == 0:
            freeze_all(ribo_model)
        else:
            unfreeze_top_n(ribo_model, n_unfreeze)

        # Build uniform baseline with same unfreezing
        base_model_u = AutoModel.from_pretrained("gpt2").to(device)
        uni_model = UniformBaselineModel(base_model_u, vocab_size, hidden_size).to(device)
        if n_unfreeze == 0:
            for p in uni_model.base_model.parameters():
                p.requires_grad = False
        else:
            for p in uni_model.base_model.parameters():
                p.requires_grad = False
            total = len(uni_model.base_model.h)
            for i in range(total - n_unfreeze, total):
                for p in uni_model.base_model.h[i].parameters():
                    p.requires_grad = True
            for p in uni_model.base_model.ln_f.parameters():
                p.requires_grad = True

        # --- Train or load ---
        if os.path.exists(weight_path) and os.path.exists(uniform_path):
            print(f"Loading saved weights for {label}...")
            ribo_model.load_state_dict(
                torch.load(weight_path, map_location=device))
            uni_model.load_state_dict(
                torch.load(uniform_path, map_location=device))
        else:
            # Train ribosome variant
            train_model(ribo_model, tokenizer, device,
                        epochs=args.epochs, max_length=args.max_train_len,
                        batch_size=args.batch_size, lr=args.lr,
                        sparsity_coeff=args.sparsity_coeff,
                        label=f"{label}_ribosome", is_ribosome=True)

            # Train matched uniform baseline
            train_model(uni_model, tokenizer, device,
                        epochs=args.epochs, max_length=args.max_train_len,
                        batch_size=args.batch_size, lr=args.lr,
                        sparsity_coeff=0, label=f"{label}_uniform",
                        is_ribosome=False)

            torch.save(ribo_model.state_dict(), weight_path)
            torch.save(uni_model.state_dict(), uniform_path)
            print(f"Saved weights for {label}")

        # --- Evaluate ---
        print(f"\nEvaluating {label}...")
        condition_results = {}
        for seq_len in args.eval_lengths:
            ribo_res = evaluate_model(ribo_model, tokenizer, device, seq_len,
                                      n_samples=args.eval_samples, is_ribosome=True)
            uni_res = evaluate_model(uni_model, tokenizer, device, seq_len,
                                     n_samples=args.eval_samples, is_ribosome=False)
            if ribo_res and uni_res:
                condition_results[str(seq_len)] = {
                    "ribosome": ribo_res,
                    "uniform": uni_res,
                    "delta": float(ribo_res["loss"] - uni_res["loss"])
                }

        all_results[label] = condition_results

        # Free GPU memory before next condition
        del ribo_model, uni_model, base_model, base_model_u, ribosome
        torch.cuda.empty_cache()

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "="*70)
    print("RESULTS: Ribosome CE loss by condition and sequence length")
    print("="*70)

    # Header
    lens = args.eval_lengths
    header = f"  {'condition':>10s}"
    for l in lens:
        header += f"  {'len='+str(l):>10s}"
    print(header)
    print("  " + "-" * (12 + 12 * len(lens)))

    for label, cond_res in all_results.items():
        row = f"  {label:>10s}"
        for l in lens:
            sl = str(l)
            if sl in cond_res:
                row += f"  {cond_res[sl]['ribosome']['loss']:10.4f}"
            else:
                row += f"  {'n/a':>10s}"
        print(row)

    print("\n" + "="*70)
    print("RESULTS: Ribosome vs Uniform delta (negative = ribosome wins)")
    print("="*70)
    print(header)
    print("  " + "-" * (12 + 12 * len(lens)))

    for label, cond_res in all_results.items():
        row = f"  {label:>10s}"
        for l in lens:
            sl = str(l)
            if sl in cond_res:
                row += f"  {cond_res[sl]['delta']:+10.4f}"
            else:
                row += f"  {'n/a':>10s}"
        print(row)

    print("\n" + "="*70)
    print("SCORE DISCRIMINATION: Entropy ratio (lower = more peaked)")
    print("="*70)
    print(header)
    print("  " + "-" * (12 + 12 * len(lens)))

    for label, cond_res in all_results.items():
        row = f"  {label:>10s}"
        for l in lens:
            sl = str(l)
            if sl in cond_res:
                H = cond_res[sl]["ribosome"].get("entropy", float("nan"))
                H_max = np.log(l)
                ratio = H / H_max if H_max > 0 else float("nan")
                row += f"  {ratio:10.3f}"
            else:
                row += f"  {'n/a':>10s}"
        print(row)

    print("\n" + "="*70)
    print("PEAK-TO-VALLEY RATIO (higher = more discriminative)")
    print("="*70)
    print(header)
    print("  " + "-" * (12 + 12 * len(lens)))

    for label, cond_res in all_results.items():
        row = f"  {label:>10s}"
        for l in lens:
            sl = str(l)
            if sl in cond_res:
                pvr = cond_res[sl]["ribosome"].get("pvr", float("nan"))
                row += f"  {pvr:10.2f}"
            else:
                row += f"  {'n/a':>10s}"
        print(row)

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved to {args.output}")


if __name__ == "__main__":
    main()
