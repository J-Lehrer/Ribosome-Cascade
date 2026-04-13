"""
Ribosome-Cascade Architecture Benchmark
========================================
Evaluates whether the Ribosome scorer provides meaningful signal beyond
baseline importance methods, and whether it degrades gracefully with
sequence length.

Baselines:
  1. Uniform  — all scores = 1.0 (no ribosome, raw hidden states)
  2. Random   — scores ~ U(0,1) at same sparsity as ribosome
  3. Attention — use GPT-2's own attention weights as importance (free)
  4. Ribosome — the learned scorer

Metrics:
  A. Reconstruction loss (CE) vs sequence length
  B. Score entropy (is the ribosome actually discriminating?)
  C. Peak-to-valley ratio
  D. Sparsity (fraction of tokens below threshold)

Run:
  python ribosome_benchmark.py [--device cuda] [--epochs 3] [--max_len 512]
"""

import argparse
import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

# ============================================================
# 1. ARCHITECTURE (verbatim from notebook, minor cleanup)
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
        """
        score_override: if provided, use these scores instead of the ribosome.
                        This lets us swap in uniform / random / attention baselines.
        """
        need_attn = (score_override == "attention")
        outputs = self.base_model(
            input_ids, attention_mask=attention_mask,
            output_attentions=need_attn
        )
        hidden_states = outputs.last_hidden_state

        # --- Score selection ---
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
            # Average attention across heads and layers -> per-token importance
            attn = outputs.attentions
            if attn is None or len(attn) == 0:
                # Fallback: use uniform if attentions unavailable
                importance = torch.ones(
                    hidden_states.shape[:2], device=hidden_states.device
                )
            else:
                attn_stack = torch.stack(list(attn))  # (L, B, H, S, S)
                attn_mean = attn_stack.mean(dim=(0, 2)) # (B, S, S)
                importance = attn_mean.sum(dim=1)        # (B, S) -- column sums
                # Normalize to [0,1]
                importance = importance / (importance.max(dim=-1, keepdim=True)[0] + 1e-8)
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


# ============================================================
# 2. METRICS
# ============================================================

def score_entropy(scores):
    """H(s) for a 1-D score vector. Lower = more peaked."""
    s = scores.clamp(1e-8, 1.0)
    s = s / s.sum()
    return -(s * s.log()).sum().item()

def peak_valley_ratio(scores):
    return (scores.max() / (scores.min() + 1e-8)).item()

def sparsity_fraction(scores, threshold=0.3):
    return (scores < threshold).float().mean().item()


# ============================================================
# 3. TRAINING
# ============================================================

def train_ribosome(model, tokenizer, device, epochs=3, max_length=64,
                   batch_size=16, lr=3e-5, sparsity_coeff=0.1):
    """Train ribosome on wikitext-2. Returns training loss history."""
    print("Loading wikitext-2...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    ds = ds.filter(lambda x: len(x["text"].strip()) > 0)

    def tok_fn(examples):
        t = tokenizer(examples["text"], padding="max_length",
                      truncation=True, max_length=max_length)
        t["labels"] = t["input_ids"].copy()
        return t

    ds = ds.map(tok_fn, batched=True, remove_columns=["text"])
    ds.set_format("torch")

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    # Freeze base model, train ribosome + lm_head
    for p in model.base_model.parameters():
        p.requires_grad = False
    optimizer = torch.optim.AdamW(
        list(model.ribosome.parameters()) + list(model.lm_head.parameters()),
        lr=lr
    )

    model.train()
    loss_history = []
    for epoch in range(epochs):
        epoch_losses = []
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

            epoch_losses.append(recon_loss.item())
            if step % 100 == 0:
                print(f"  epoch {epoch+1}/{epochs}  step {step:4d}  "
                      f"CE={recon_loss.item():.4f}  sparsity={sparsity_loss.item():.4f}")

        mean_loss = np.mean(epoch_losses)
        loss_history.append(mean_loss)
        print(f"  epoch {epoch+1} done  mean_CE={mean_loss:.4f}  "
              f"time={time.time()-t0:.1f}s")

    return loss_history


# ============================================================
# 4. EVALUATION
# ============================================================

def evaluate_at_length(model, tokenizer, device, seq_len, n_samples=50):
    """
    Evaluate all 4 methods at a given sequence length.
    Returns dict of {method: {loss, entropy, pvr, sparsity}}.
    """
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ds = ds.filter(lambda x: len(x["text"].strip()) > 20)
    texts = [x["text"] for x in ds]

    # Build inputs at target length
    inputs_list = []
    for t in texts:
        enc = tokenizer(t, truncation=True, max_length=seq_len,
                        padding="max_length", return_tensors="pt")
        # Only keep sequences that are mostly real tokens (not padding)
        real_tokens = (enc["input_ids"] != tokenizer.pad_token_id).sum().item()
        if real_tokens >= seq_len * 0.7:
            inputs_list.append(enc)
        if len(inputs_list) >= n_samples:
            break

    if not inputs_list:
        return None

    methods = [None, "uniform", "random", "attention"]
    method_names = ["ribosome", "uniform", "random", "attention"]
    results = {}

    model.eval()
    with torch.no_grad():
        for method, name in zip(methods, method_names):
            losses, entropies, pvrs, sparsities = [], [], [], []
            for enc in inputs_list:
                ids = enc["input_ids"].to(device)
                mask = enc["attention_mask"].to(device)
                labels = ids.clone()

                loss, _, importance = model(ids, mask, labels,
                                            score_override=method)

                # Mask out padding for metrics
                real_mask = mask[0].bool()
                real_scores = importance[0][real_mask]

                losses.append(loss.item())
                entropies.append(score_entropy(real_scores))
                pvrs.append(peak_valley_ratio(real_scores))
                sparsities.append(sparsity_fraction(real_scores))

            results[name] = {
                "loss": np.mean(losses),
                "entropy": np.mean(entropies),
                "pvr": np.mean(pvrs),
                "sparsity": np.mean(sparsities),
                "n": len(inputs_list)
            }

    return results


# ============================================================
# 5. MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_train_len", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--sparsity_coeff", type=float, default=0.1)
    parser.add_argument("--eval_lengths", nargs="+", type=int,
                        default=[32, 64, 128, 256, 512])
    parser.add_argument("--eval_samples", type=int, default=50)
    parser.add_argument("--weights_path", default=None,
                        help="Load pretrained ribosome weights (state_dict)")
    parser.add_argument("--save_weights", default="E:/Ribosome-Cascade/ribosome_weights.pt")
    parser.add_argument("--output", default="E:/Ribosome-Cascade/benchmark_results.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    # --- Build model ---
    print("Loading GPT-2...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModel.from_pretrained("gpt2").to(device)

    ribosome = RibosomeScorer(base_model.config.n_embd).to(device)
    model = RibosomeCascadeTrainerModel(base_model, ribosome).to(device)

    # --- Train or load ---
    if args.weights_path and os.path.exists(args.weights_path):
        print(f"Loading weights from {args.weights_path}")
        model.load_state_dict(torch.load(args.weights_path, map_location=device))
    else:
        print("\n" + "="*60)
        print("TRAINING RIBOSOME")
        print("="*60)
        train_history = train_ribosome(
            model, tokenizer, device,
            epochs=args.epochs,
            max_length=args.max_train_len,
            batch_size=args.batch_size,
            lr=args.lr,
            sparsity_coeff=args.sparsity_coeff
        )
        print(f"\nTraining losses by epoch: {train_history}")
        if args.save_weights:
            os.makedirs(os.path.dirname(args.save_weights) or ".", exist_ok=True)
            torch.save(model.state_dict(), args.save_weights)
            print(f"Weights saved to {args.save_weights}")

    # --- Evaluate ---
    print("\n" + "="*60)
    print("BENCHMARK: 4 methods x sequence lengths")
    print("="*60)

    all_results = {}
    for seq_len in args.eval_lengths:
        print(f"\n--- seq_len = {seq_len} ---")
        res = evaluate_at_length(
            model, tokenizer, device, seq_len,
            n_samples=args.eval_samples
        )
        if res is None:
            print(f"  Skipped (not enough samples at length {seq_len})")
            continue

        all_results[seq_len] = res
        for name, metrics in res.items():
            print(f"  {name:12s}  CE={metrics['loss']:.4f}  "
                  f"H={metrics['entropy']:.3f}  "
                  f"PVR={metrics['pvr']:.2f}  "
                  f"sparsity={metrics['sparsity']:.3f}")

    # --- Summary table ---
    print("\n" + "="*60)
    print("SUMMARY: Does Ribosome beat attention-based importance?")
    print("="*60)
    for seq_len, res in all_results.items():
        r_loss = res["ribosome"]["loss"]
        a_loss = res["attention"]["loss"]
        u_loss = res["uniform"]["loss"]
        delta_vs_attn = r_loss - a_loss
        delta_vs_uniform = r_loss - u_loss
        winner = "RIBOSOME" if r_loss < a_loss else "ATTENTION"
        print(f"  len={seq_len:4d}  ribosome={r_loss:.4f}  attention={a_loss:.4f}  "
              f"uniform={u_loss:.4f}  delta(r-a)={delta_vs_attn:+.4f}  [{winner}]")

    # Entropy check
    print("\nSCORE DISCRIMINATION (entropy, lower = more peaked):")
    uniform_H_theoretical = {s: np.log(s) for s in args.eval_lengths}
    for seq_len, res in all_results.items():
        r_H = res["ribosome"]["entropy"]
        H_max = uniform_H_theoretical[seq_len]
        ratio = r_H / H_max
        verdict = "GOOD (<0.8)" if ratio < 0.8 else "WEAK (>0.8, near-uniform)"
        print(f"  len={seq_len:4d}  H_ribosome={r_H:.3f}  "
              f"H_uniform={H_max:.3f}  ratio={ratio:.3f}  [{verdict}]")

    # Save
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved to {args.output}")


if __name__ == "__main__":
    main()
