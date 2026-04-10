"""
Ribosome-Cascade Architecture Benchmark v2
===========================================
Fixes from v1:
  - Attention baseline: manually extract attentions from GPT-2
  - Added "retrained uniform" baseline (lm_head trained with scores=1.0)
    to control for co-adaptation confound

Baselines:
  1. Ribosome     — the learned scorer + its co-trained lm_head
  2. Uniform-co   — scores=1.0 through the ribosome's lm_head (unfair baseline)
  3. Uniform-own  — scores=1.0 with its OWN separately trained lm_head
  4. Random       — scores ~ U(0,1) through ribosome's lm_head
  5. Attention    — GPT-2 attention-derived importance + ribosome's lm_head

Run:
  python ribosome_benchmark_v2.py --device cuda --epochs 3
"""

import argparse
import os
import json
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

# ============================================================
# 1. ARCHITECTURE
# ============================================================

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


class RibosomeCascadeTrainerModel(nn.Module):
    def __init__(self, base_model, ribosome):
        super().__init__()
        self.base_model = base_model
        self.ribosome = ribosome
        self.lm_head = nn.Linear(
            base_model.config.n_embd,
            base_model.config.vocab_size,
            bias=False
        )

    def forward(self, input_ids, attention_mask=None, labels=None,
                score_override=None):
        # Attentions enabled via config (transformers 5.5+ ignores forward kwarg)
        outputs = self.base_model(
            input_ids, attention_mask=attention_mask
        )
        hidden_states = outputs.last_hidden_state

        if score_override is None:
            importance = self.ribosome(hidden_states)
        elif score_override == "uniform":
            importance = torch.ones(
                hidden_states.shape[:2], device=hidden_states.device
            )
        elif score_override == "random":
            importance = torch.rand(
                hidden_states.shape[:2], device=hidden_states.device
            )
        elif score_override == "attention":
            # outputs.attentions is tuple of (B, H, S, S) per layer
            attn_tensors = outputs.attentions
            # Stack: (L, B, H, S, S), mean over layers+heads -> (B, S, S)
            attn_stack = torch.stack(list(attn_tensors))
            attn_mean = attn_stack.mean(dim=(0, 2))  # (B, S, S)
            # Column sum = how much attention each token receives
            importance = attn_mean.sum(dim=1)  # (B, S)
            # Normalize per-sequence to [0,1]
            imp_min = importance.min(dim=-1, keepdim=True)[0]
            imp_max = importance.max(dim=-1, keepdim=True)[0]
            importance = (importance - imp_min) / (imp_max - imp_min + 1e-8)
        else:
            raise ValueError(f"Unknown score_override: {score_override}")

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
    """Separate model with its own lm_head, trained with scores=1.0 (no ribosome)."""
    def __init__(self, base_model, vocab_size, hidden_size):
        super().__init__()
        self.base_model = base_model  # shared, frozen
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        # No weighting — raw hidden states
        logits = self.lm_head(hidden_states)

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
# 2. METRICS
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
# 3. DATA
# ============================================================

def get_dataloader(tokenizer, split="train", max_length=64, batch_size=16):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    ds = ds.filter(lambda x: len(x["text"].strip()) > 0)

    def tok_fn(examples):
        t = tokenizer(examples["text"], padding="max_length",
                      truncation=True, max_length=max_length)
        t["labels"] = t["input_ids"].copy()
        return t

    ds = ds.map(tok_fn, batched=True, remove_columns=["text"])
    ds.set_format("torch")
    return DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"))


# ============================================================
# 4. TRAINING
# ============================================================

def train_ribosome(model, tokenizer, device, epochs=3, max_length=64,
                   batch_size=16, lr=3e-5, sparsity_coeff=0.1):
    print("--- Training Ribosome Model ---")
    loader = get_dataloader(tokenizer, "train", max_length, batch_size)

    for p in model.base_model.parameters():
        p.requires_grad = False
    optimizer = torch.optim.AdamW(
        list(model.ribosome.parameters()) + list(model.lm_head.parameters()),
        lr=lr
    )

    model.train()
    history = []
    for epoch in range(epochs):
        losses = []
        t0 = time.time()
        for step, batch in enumerate(loader):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            recon_loss, _, importance = model(ids, mask, labels)
            sparsity_loss = importance.mean()
            loss = recon_loss + sparsity_coeff * sparsity_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(recon_loss.item())
            if step % 100 == 0:
                print(f"  epoch {epoch+1}/{epochs}  step {step:4d}  "
                      f"CE={recon_loss.item():.4f}  sparsity={sparsity_loss.item():.4f}")

        mean_loss = np.mean(losses)
        history.append(mean_loss)
        print(f"  epoch {epoch+1} done  mean_CE={mean_loss:.4f}  time={time.time()-t0:.1f}s")
    return history


def train_uniform_baseline(uniform_model, tokenizer, device, epochs=3,
                           max_length=64, batch_size=16, lr=3e-5):
    print("\n--- Training Uniform Baseline (own lm_head, no ribosome) ---")
    loader = get_dataloader(tokenizer, "train", max_length, batch_size)

    for p in uniform_model.base_model.parameters():
        p.requires_grad = False
    optimizer = torch.optim.AdamW(uniform_model.lm_head.parameters(), lr=lr)

    uniform_model.train()
    history = []
    for epoch in range(epochs):
        losses = []
        t0 = time.time()
        for step, batch in enumerate(loader):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            loss, _ = uniform_model(ids, mask, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            if step % 100 == 0:
                print(f"  epoch {epoch+1}/{epochs}  step {step:4d}  CE={loss.item():.4f}")

        mean_loss = np.mean(losses)
        history.append(mean_loss)
        print(f"  epoch {epoch+1} done  mean_CE={mean_loss:.4f}  time={time.time()-t0:.1f}s")
    return history


# ============================================================
# 5. EVALUATION
# ============================================================

def evaluate_at_length(ribosome_model, uniform_model, tokenizer, device,
                       seq_len, n_samples=50):
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

    results = {}

    # --- Ribosome methods (ribosome, uniform-co, random, attention) ---
    ribosome_methods = [None, "uniform", "random", "attention"]
    ribosome_names = ["ribosome", "uniform_co", "random", "attention"]

    ribosome_model.eval()
    with torch.no_grad():
        for method, name in zip(ribosome_methods, ribosome_names):
            losses, entropies, pvrs, sparsities = [], [], [], []
            for enc in inputs_list:
                ids = enc["input_ids"].to(device)
                mask = enc["attention_mask"].to(device)
                labels = ids.clone()

                loss, _, importance = ribosome_model(
                    ids, mask, labels, score_override=method
                )

                real_mask = mask[0].bool()
                real_scores = importance[0][real_mask]

                losses.append(loss.item())
                entropies.append(score_entropy(real_scores))
                pvrs.append(peak_valley_ratio(real_scores))
                sparsities.append(sparsity_fraction(real_scores))

            results[name] = {
                "loss": float(np.mean(losses)),
                "entropy": float(np.mean(entropies)),
                "pvr": float(np.mean(pvrs)),
                "sparsity": float(np.mean(sparsities)),
                "n": len(inputs_list)
            }

    # --- Uniform-own (separately trained lm_head) ---
    uniform_model.eval()
    with torch.no_grad():
        losses = []
        for enc in inputs_list:
            ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            labels = ids.clone()
            loss, _ = uniform_model(ids, mask, labels)
            losses.append(loss.item())

        results["uniform_own"] = {
            "loss": float(np.mean(losses)),
            "entropy": float("nan"),
            "pvr": float("nan"),
            "sparsity": float("nan"),
            "n": len(inputs_list)
        }

    return results


# ============================================================
# 6. MAIN
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
    parser.add_argument("--weights_dir", default="E:/Ribosome-Cascade/weights_v2")
    parser.add_argument("--output", default="E:/Ribosome-Cascade/benchmark_results_v2.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    # --- Build models ---
    print("Loading GPT-2...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained("gpt2")
    config.output_attentions = True  # Must be config-level for transformers 5.5+
    base_model = AutoModel.from_pretrained("gpt2", config=config).to(device)

    hidden_size = base_model.config.n_embd
    vocab_size = base_model.config.vocab_size

    # Ribosome model
    ribosome = RibosomeScorer(hidden_size).to(device)
    ribosome_model = RibosomeCascadeTrainerModel(base_model, ribosome).to(device)

    # Uniform baseline (shares base_model, has its own lm_head)
    uniform_model = UniformBaselineModel(base_model, vocab_size, hidden_size).to(device)

    os.makedirs(args.weights_dir, exist_ok=True)
    ribo_path = os.path.join(args.weights_dir, "ribosome_model.pt")
    uniform_path = os.path.join(args.weights_dir, "uniform_model.pt")

    # --- Train or load ---
    if os.path.exists(ribo_path) and os.path.exists(uniform_path):
        print(f"Loading ribosome weights from {ribo_path}")
        ribosome_model.load_state_dict(
            torch.load(ribo_path, map_location=device), strict=False
        )
        print(f"Loading uniform weights from {uniform_path}")
        uniform_model.load_state_dict(
            torch.load(uniform_path, map_location=device), strict=False
        )
    else:
        # Train ribosome
        ribo_history = train_ribosome(
            ribosome_model, tokenizer, device,
            epochs=args.epochs, max_length=args.max_train_len,
            batch_size=args.batch_size, lr=args.lr,
            sparsity_coeff=args.sparsity_coeff
        )
        print(f"Ribosome training losses: {ribo_history}")

        # Train uniform baseline with same hyperparams
        uniform_history = train_uniform_baseline(
            uniform_model, tokenizer, device,
            epochs=args.epochs, max_length=args.max_train_len,
            batch_size=args.batch_size, lr=args.lr
        )
        print(f"Uniform training losses: {uniform_history}")

        # Save both
        torch.save(ribosome_model.state_dict(), ribo_path)
        torch.save(uniform_model.state_dict(), uniform_path)
        print(f"Weights saved to {args.weights_dir}")

    # --- Verify attention baseline works ---
    print("\nVerifying attention extraction...")
    test_ids = tokenizer("Hello world test", return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = base_model(test_ids)
        attn = out.attentions
        print(f"  Attention layers returned: {len(attn)}")
        print(f"  Shape per layer: {attn[0].shape}")
        attn_stack = torch.stack(list(attn))
        attn_mean = attn_stack.mean(dim=(0, 2))
        importance = attn_mean.sum(dim=1)
        print(f"  Importance vector: {importance[0].cpu().numpy().round(3)}")
        print(f"  Range: [{importance.min().item():.3f}, {importance.max().item():.3f}]")
        if importance.max() > importance.min() + 0.01:
            print("  OK: Attention baseline is discriminating")
        else:
            print("  WARNING: Attention scores near-uniform")

    # --- Evaluate ---
    print("\n" + "="*70)
    print("BENCHMARK v2: 5 methods x sequence lengths")
    print("="*70)

    all_results = {}
    for seq_len in args.eval_lengths:
        print(f"\n--- seq_len = {seq_len} ---")
        res = evaluate_at_length(
            ribosome_model, uniform_model, tokenizer, device,
            seq_len, n_samples=args.eval_samples
        )
        if res is None:
            print(f"  Skipped (not enough samples)")
            continue

        all_results[str(seq_len)] = res
        for name, m in res.items():
            ent_str = f"H={m['entropy']:.3f}" if not np.isnan(m['entropy']) else "H=n/a    "
            pvr_str = f"PVR={m['pvr']:.2f}" if not np.isnan(m['pvr']) else "PVR=n/a  "
            print(f"  {name:14s}  CE={m['loss']:.4f}  {ent_str}  {pvr_str}")

    # --- Summary ---
    print("\n" + "="*70)
    print("KEY COMPARISON: Ribosome vs Uniform-Own (fair, no co-adaptation)")
    print("="*70)
    print(f"  {'len':>4s}  {'ribosome':>9s}  {'unif_own':>9s}  {'attention':>9s}  "
          f"{'d(r-u)':>8s}  {'d(r-a)':>8s}  {'winner':>10s}")
    print("  " + "-"*65)
    for seq_len_str, res in all_results.items():
        r = res["ribosome"]["loss"]
        u = res["uniform_own"]["loss"]
        a = res["attention"]["loss"]
        d_ru = r - u
        d_ra = r - a
        best = min(r, u, a)
        if best == r:
            winner = "RIBOSOME"
        elif best == u:
            winner = "UNIFORM"
        else:
            winner = "ATTENTION"
        print(f"  {seq_len_str:>4s}  {r:9.4f}  {u:9.4f}  {a:9.4f}  "
              f"{d_ru:+8.4f}  {d_ra:+8.4f}  [{winner}]")

    print("\nCO-ADAPTATION CHECK (uniform_co should be worse than uniform_own):")
    for seq_len_str, res in all_results.items():
        uco = res["uniform_co"]["loss"]
        uown = res["uniform_own"]["loss"]
        gap = uco - uown
        verdict = "co-adapted (expected)" if gap > 0 else "NOT co-adapted (?)"
        print(f"  len={seq_len_str:>4s}  uniform_co={uco:.4f}  "
              f"uniform_own={uown:.4f}  gap={gap:+.4f}  [{verdict}]")

    print("\nSCORE DISCRIMINATION:")
    for seq_len_str, res in all_results.items():
        seq_len = int(seq_len_str)
        r_H = res["ribosome"]["entropy"]
        a_H = res["attention"]["entropy"]
        H_max = np.log(seq_len)
        r_ratio = r_H / H_max
        a_ratio = a_H / H_max
        print(f"  len={seq_len_str:>4s}  H_ribo={r_H:.3f} ({r_ratio:.3f})  "
              f"H_attn={a_H:.3f} ({a_ratio:.3f})  "
              f"H_max={H_max:.3f}")

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
