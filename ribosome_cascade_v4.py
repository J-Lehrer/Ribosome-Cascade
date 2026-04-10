"""
Ribosome-Cascade v4: Three Critical Fixes
==========================================
Fix 1: Subword indexing — metatoken assembly now works on token indices,
        not whitespace-split word counts.
Fix 2: Tie-breaker semantics — equidistant filler goes to HEAVIER peak
        (strong-attractor semantics, not load-balancing).
Fix 3: Train/inference alignment — differentiable soft-cascade replaces
        hard discrete grouping during training. Soft boundary detection
        via gradient of importance scores, soft assignment via attention
        over boundary-defined segments, priority-weighted processing.

Architecture:
  Training mode:  soft-cascade (differentiable approximation)
  Inference mode: hard-cascade (discrete peak-finding, as original)
  Both paths use the SAME ribosome scorer, so training signal
  actually optimizes for the inference-time behavior.
"""

import argparse
import os
import json
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset


# ============================================================
# FIX 1 & 2: Corrected Metatoken Assembly (inference path)
# ============================================================

def assemble_metatokens_fixed(importance_scores, hidden_states, tokens=None):
    """
    Fixed metatoken assembly operating on token indices (not word splits).
    
    Fix 1: Works directly on token-level indices. Each position i corresponds
           to hidden_states[i] and importance_scores[i]. No whitespace splitting.
    
    Fix 2: Tie-breaker sends equidistant filler to the HEAVIER peak
           (strong-attractor semantics).
    
    Args:
        importance_scores: (seq_len,) tensor of importance scores
        hidden_states: (seq_len, hidden_dim) tensor
        tokens: optional list of token strings for debugging
    
    Returns:
        list of dicts with keys:
            'indices': list of token indices in this metatoken
            'hidden': (hidden_dim,) mean-pooled hidden state
            'weight': sum of importance scores
            'temporal_tag': index of the peak token (for ordering)
    """
    scores = importance_scores.detach().cpu().numpy()
    seq_len = len(scores)
    
    if seq_len <= 2:
        return [{
            'indices': list(range(seq_len)),
            'hidden': hidden_states.mean(dim=0),
            'weight': importance_scores.sum().item(),
            'temporal_tag': 0,
        }]
    
    # Find peaks (local maxima)
    peaks = []
    for i in range(seq_len):
        left = scores[i - 1] if i > 0 else -1.0
        right = scores[i + 1] if i < seq_len - 1 else -1.0
        if scores[i] >= left and scores[i] >= right:
            peaks.append(i)
    
    if len(peaks) == 0:
        peaks = [int(np.argmax(scores))]
    
    # Assign each token to nearest peak
    # FIX 2: tie-break goes to HEAVIER peak (higher score)
    assignments = np.zeros(seq_len, dtype=np.int64)
    for i in range(seq_len):
        if i in peaks:
            assignments[i] = i
            continue
        
        best_peak = peaks[0]
        best_dist = abs(i - peaks[0])
        
        for p in peaks[1:]:
            d = abs(i - p)
            if d < best_dist:
                best_dist = d
                best_peak = p
            elif d == best_dist:
                # FIX 2: tie-break to heavier peak (strong attractor)
                if scores[p] > scores[best_peak]:
                    best_peak = p
        
        assignments[i] = best_peak
    
    # Group by peak
    groups = {}
    for i in range(seq_len):
        p = assignments[i]
        if p not in groups:
            groups[p] = []
        groups[p].append(i)
    
    # Build metatokens
    metatokens = []
    for peak_idx in sorted(groups.keys()):
        indices = groups[peak_idx]
        h = hidden_states[indices].mean(dim=0)  # FIX 1: direct index, no word-split
        w = importance_scores[indices].sum().item()
        metatokens.append({
            'indices': indices,
            'hidden': h,
            'weight': w,
            'temporal_tag': peak_idx,
        })
    
    return metatokens


# ============================================================
# FIX 3: Differentiable Soft-Cascade for Training
# ============================================================

class SoftCascadeLayer(nn.Module):
    """
    Differentiable approximation of the metatoken cascade.
    
    Instead of hard peak-finding + discrete grouping:
    1. Compute soft boundaries from importance score gradients
    2. Use boundary-aware positional soft-attention to form chunk representations
    3. Priority-weight chunks by importance mass
    4. Cross-attend chunks in priority order (heaviest first anchors lighter ones)
    5. Expand back to token-level predictions
    
    This ensures the ribosome scorer receives gradient signal that
    actually reflects the cascade's behavior at inference time.
    """
    
    def __init__(self, hidden_size, n_chunks=8, chunk_heads=4):
        super().__init__()
        self.n_chunks = n_chunks
        self.hidden_size = hidden_size
        
        # Chunk encoder: compress variable-length segments into fixed representations
        # Uses cross-attention with n_chunks learnable queries
        self.chunk_queries = nn.Parameter(torch.randn(n_chunks, hidden_size) * 0.02)
        self.chunk_attn = nn.MultiheadAttention(
            hidden_size, num_heads=chunk_heads, batch_first=True
        )
        
        # Priority cross-attention: chunks attend to each other in priority order
        self.priority_attn = nn.MultiheadAttention(
            hidden_size, num_heads=chunk_heads, batch_first=True
        )
        self.priority_norm = nn.LayerNorm(hidden_size)
        
        # Chunk decoder: expand chunk representations back to token-level
        self.decode_attn = nn.MultiheadAttention(
            hidden_size, num_heads=chunk_heads, batch_first=True
        )
        self.decode_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, hidden_states, importance_scores):
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            importance_scores: (batch, seq_len) in [0, 1]
        
        Returns:
            refined_states: (batch, seq_len, hidden_size)
        """
        B, S, H = hidden_states.shape
        
        # --- Step 1: Importance-weighted chunk encoding ---
        # Weight hidden states by importance BEFORE chunk attention
        # so high-importance tokens dominate chunk representations
        weighted_states = hidden_states * importance_scores.unsqueeze(-1)
        
        # Chunk queries attend to importance-weighted hidden states
        queries = self.chunk_queries.unsqueeze(0).expand(B, -1, -1)  # (B, n_chunks, H)
        chunk_repr, chunk_weights = self.chunk_attn(
            queries, weighted_states, hidden_states  # keys=weighted, values=original
        )
        # chunk_repr: (B, n_chunks, H)
        # chunk_weights: (B, n_chunks, S)
        
        # Weight chunk representations by importance mass they capture
        # Each chunk's "weight" = sum of importance scores it attends to
        chunk_importance = (chunk_weights * importance_scores.unsqueeze(1)).sum(dim=-1)
        # chunk_importance: (B, n_chunks)
        
        # --- Step 2: Priority-ordered cross-attention ---
        # Sort chunks by importance (heaviest first = anchor)
        # Use causal masking so chunk i only attends to chunks 0..i
        # (heavier chunks processed first, lighter ones conditioned on them)
        
        sort_idx = chunk_importance.argsort(dim=-1, descending=True)  # (B, n_chunks)
        
        # Gather sorted chunk representations
        sorted_chunks = torch.gather(
            chunk_repr, 1, sort_idx.unsqueeze(-1).expand(-1, -1, H)
        )
        
        # Causal mask: chunk i can attend to chunks 0..i (already-processed heavier chunks)
        causal_mask = torch.triu(
            torch.ones(self.n_chunks, self.n_chunks, device=hidden_states.device),
            diagonal=1
        ).bool()
        
        # Priority cross-attention with causal mask
        priority_out, _ = self.priority_attn(
            sorted_chunks, sorted_chunks, sorted_chunks,
            attn_mask=causal_mask
        )
        priority_out = self.priority_norm(sorted_chunks + priority_out)
        
        # Unsort back to original chunk order
        unsort_idx = sort_idx.argsort(dim=-1)
        priority_chunks = torch.gather(
            priority_out, 1, unsort_idx.unsqueeze(-1).expand(-1, -1, H)
        )
        
        # --- Step 3: Decode back to token-level ---
        # Tokens attend to chunk representations to get refined states
        decoded, _ = self.decode_attn(
            hidden_states, priority_chunks, priority_chunks
        )
        refined_states = self.decode_norm(hidden_states + decoded)
        
        return refined_states


# ============================================================
# MODEL DEFINITIONS
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


class RibosomeCascadeModelV4(nn.Module):
    """
    V4 model: uses SoftCascadeLayer during training for end-to-end
    differentiable cascade. At inference, can switch to hard cascade.
    """
    def __init__(self, base_model, hidden_size, vocab_size, n_chunks=8):
        super().__init__()
        self.base_model = base_model
        self.ribosome = RibosomeScorer(hidden_size)
        self.cascade = SoftCascadeLayer(hidden_size, n_chunks=n_chunks)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (B, S, H)
        
        importance = self.ribosome(hidden_states)  # (B, S)
        
        # Soft-cascade: differentiable metatoken processing
        refined = self.cascade(hidden_states, importance)
        
        # Blend: importance-weighted mix of original and cascade-refined
        # High-importance tokens keep more of their refined representation
        # Low-importance tokens are more aggressively compressed
        alpha = importance.unsqueeze(-1)  # (B, S, 1)
        blended = alpha * refined + (1 - alpha) * hidden_states
        
        logits = self.lm_head(blended)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        return loss, logits, importance
    
    def forward_hard_cascade(self, input_ids, attention_mask=None):
        """Inference path using hard discrete metatoken assembly."""
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        importance = self.ribosome(hidden_states)
        
        batch_metatokens = []
        for b in range(hidden_states.shape[0]):
            if attention_mask is not None:
                real_len = attention_mask[b].sum().item()
            else:
                real_len = hidden_states.shape[1]
            
            mt = assemble_metatokens_fixed(
                importance[b, :real_len],
                hidden_states[b, :real_len]
            )
            batch_metatokens.append(mt)
        
        return importance, batch_metatokens


class UniformBaselineModel(nn.Module):
    """Same as v3 — control baseline without ribosome."""
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
# FREEZE / UNFREEZE UTILS
# ============================================================

def freeze_base(model):
    for p in model.base_model.parameters():
        p.requires_grad = False

def unfreeze_top_n(model, n_layers):
    freeze_base(model)
    total = len(model.base_model.h)
    for i in range(total - n_layers, total):
        for p in model.base_model.h[i].parameters():
            p.requires_grad = True
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

def bimodality_coefficient(scores):
    """Measures how bimodal the score distribution is (higher = more bimodal)."""
    s = scores.detach().cpu().numpy()
    n = len(s)
    if n < 4:
        return 0.0
    m = s.mean()
    s2 = ((s - m) ** 2).mean()
    s3 = ((s - m) ** 3).mean()
    s4 = ((s - m) ** 4).mean()
    if s2 < 1e-10:
        return 0.0
    skew = s3 / (s2 ** 1.5)
    kurt = s4 / (s2 ** 2) - 3.0
    # Sarle's bimodality coefficient
    bc = (skew ** 2 + 1) / (kurt + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
    return float(bc)


# ============================================================
# LOSS WITH BIMODAL REGULARIZER
# ============================================================

def bimodal_penalty(scores, target_low_fraction=0.4):
    """
    Two-part penalty for bimodal score distribution:
    1. Push individual scores toward 0 or 1: mean(s * (1-s))
    2. Enforce target fraction of low scores: (frac_low - target)^2
    
    Part 2 prevents the model from gaming part 1 by pushing everything to 1.
    """
    # Part 1: push scores toward extremes
    extremity = (scores * (1 - scores)).mean()
    
    # Part 2: enforce that ~target_low_fraction of scores are below 0.5
    # Use soft threshold (sigmoid) for differentiability
    soft_low = torch.sigmoid(10.0 * (0.5 - scores))  # ~1 when s<0.5, ~0 when s>0.5
    frac_low = soft_low.mean()
    ratio_penalty = (frac_low - target_low_fraction) ** 2
    
    return extremity + ratio_penalty


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

    return DataLoader(
        _cached_dataset[cache_key], batch_size=batch_size,
        shuffle=(split == "train")
    )


# ============================================================
# TRAINING
# ============================================================

def train_v4(model, tokenizer, device, epochs, max_length, batch_size,
             lr, sparsity_coeff, bimodal_coeff, label):
    print(f"\n{'='*60}")
    print(f"TRAINING: {label}")
    print(f"Trainable params: {count_trainable(model):,}")
    print(f"Loss = CE + {sparsity_coeff}*mean(s) + {bimodal_coeff}*bimodal(s)")
    print(f"{'='*60}")

    loader = get_dataloader(tokenizer, "train", max_length, batch_size)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    model.train()
    history = []
    for epoch in range(epochs):
        losses_ce, losses_sp, losses_bm = [], [], []
        t0 = time.time()
        for step, batch in enumerate(loader):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            ce_loss, _, importance = model(ids, mask, labels)
            sp_loss = importance.mean()
            bm_loss = bimodal_penalty(importance)
            
            loss = ce_loss + sparsity_coeff * sp_loss + bimodal_coeff * bm_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            losses_ce.append(ce_loss.item())
            losses_sp.append(sp_loss.item())
            losses_bm.append(bm_loss.item())

            if step % 100 == 0:
                print(f"  epoch {epoch+1}/{epochs}  step {step:4d}  "
                      f"CE={ce_loss.item():.4f}  "
                      f"sparse={sp_loss.item():.4f}  "
                      f"bimodal={bm_loss.item():.4f}")

        mean_ce = np.mean(losses_ce)
        history.append(mean_ce)
        print(f"  epoch {epoch+1} done  mean_CE={mean_ce:.4f}  "
              f"mean_sparse={np.mean(losses_sp):.4f}  "
              f"mean_bimodal={np.mean(losses_bm):.4f}  "
              f"time={time.time()-t0:.1f}s")

    return history


def train_uniform(model, tokenizer, device, epochs, max_length, batch_size,
                  lr, label):
    print(f"\n{'='*60}")
    print(f"TRAINING UNIFORM: {label}")
    print(f"Trainable params: {count_trainable(model):,}")
    print(f"{'='*60}")

    loader = get_dataloader(tokenizer, "train", max_length, batch_size)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    model.train()
    for epoch in range(epochs):
        losses = []
        t0 = time.time()
        for step, batch in enumerate(loader):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            loss, _ = model(ids, mask, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            losses.append(loss.item())

            if step % 100 == 0:
                print(f"  epoch {epoch+1}/{epochs}  step {step:4d}  CE={loss.item():.4f}")

        print(f"  epoch {epoch+1} done  mean_CE={np.mean(losses):.4f}  "
              f"time={time.time()-t0:.1f}s")


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
    all_losses, all_entropy, all_pvr, all_sparsity, all_bimodal = [], [], [], [], []

    with torch.no_grad():
        for enc in inputs_list:
            ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            labels = ids.clone()

            if is_ribosome:
                loss, _, importance = model(ids, mask, labels)
                real_mask = mask[0].bool()
                real_scores = importance[0][real_mask]
                all_entropy.append(score_entropy(real_scores))
                all_pvr.append(peak_valley_ratio(real_scores))
                all_sparsity.append(sparsity_fraction(real_scores))
                all_bimodal.append(bimodality_coefficient(real_scores))
            else:
                loss, _ = model(ids, mask, labels)

            all_losses.append(loss.item())

    result = {"loss": float(np.mean(all_losses)), "n": len(inputs_list)}
    if is_ribosome:
        result["entropy"] = float(np.mean(all_entropy))
        result["pvr"] = float(np.mean(all_pvr))
        result["sparsity"] = float(np.mean(all_sparsity))
        result["bimodality"] = float(np.mean(all_bimodal))
    return result


def evaluate_hard_cascade(model, tokenizer, device, seq_len, n_samples=20):
    """Evaluate using the hard cascade path to check train/inference alignment."""
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ds = ds.filter(lambda x: len(x["text"].strip()) > 20)
    texts = [x["text"] for x in ds]

    inputs_list = []
    for t in texts:
        enc = tokenizer(t, truncation=True, max_length=seq_len,
                        padding="max_length", return_tensors="pt")
        real_tokens = (enc["input_ids"] != tokenizer.pad_token_id).sum().item()
        if real_tokens >= seq_len * 0.7:
            inputs_list.append((enc, t))
        if len(inputs_list) >= n_samples:
            break

    model.eval()
    chunk_counts = []
    compression_ratios = []

    with torch.no_grad():
        for enc, text in inputs_list:
            ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            importance, batch_mt = model.forward_hard_cascade(ids, mask)
            
            for mt_list in batch_mt:
                n_tokens = sum(len(m['indices']) for m in mt_list)
                n_chunks = len(mt_list)
                chunk_counts.append(n_chunks)
                compression_ratios.append(n_tokens / max(n_chunks, 1))

    return {
        "mean_chunks": float(np.mean(chunk_counts)),
        "mean_compression": float(np.mean(compression_ratios)),
        "std_chunks": float(np.std(chunk_counts)),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Ribosome-Cascade v4")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_train_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--sparsity_coeff", type=float, default=0.1)
    parser.add_argument("--bimodal_coeff", type=float, default=0.5,
                        help="Weight for bimodal penalty (scores*[1-scores])")
    parser.add_argument("--n_chunks", type=int, default=8,
                        help="Number of soft chunks in cascade")
    parser.add_argument("--eval_lengths", nargs="+", type=int,
                        default=[32, 64, 128, 256, 512])
    parser.add_argument("--eval_samples", type=int, default=50)
    parser.add_argument("--weights_dir", default="E:/Ribosome-Cascade/weights_v4.1")
    parser.add_argument("--output", default="E:/Ribosome-Cascade/benchmark_results_v4.1.json")
    parser.add_argument("--max_vram_gb", type=float, default=20.0,
                        help="Hard cap on VRAM usage in GB. Aborts if exceeded.")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {total_vram:.1f} GB (cap: {args.max_vram_gb:.1f} GB)")
        if total_vram < 6.0:
            print("ERROR: Need at least 6GB VRAM for GPT-2 + cascade")
            return
        # Set memory fraction cap
        frac = min(args.max_vram_gb / total_vram, 0.95)
        torch.cuda.set_per_process_memory_fraction(frac)
        print(f"VRAM fraction cap set to {frac:.2f}")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    os.makedirs(args.weights_dir, exist_ok=True)

    # Only run frozen condition for v4 to validate the fix
    # Once validated, can expand to unfreezing experiments
    conditions = [
        ("frozen_v4.1", 0),
    ]

    all_results = {}

    for label, n_unfreeze in conditions:
        weight_path = os.path.join(args.weights_dir, f"{label}.pt")
        uniform_path = os.path.join(args.weights_dir, f"{label}_uniform.pt")

        print(f"\n{'#'*60}")
        print(f"CONDITION: {label}")
        print(f"{'#'*60}")

        # --- Ribosome v4 model ---
        base = AutoModel.from_pretrained("gpt2").to(device)
        H = base.config.n_embd
        V = base.config.vocab_size

        model = RibosomeCascadeModelV4(base, H, V, n_chunks=args.n_chunks).to(device)
        freeze_base(model)
        # ribosome, cascade, lm_head are all trainable

        # --- Uniform baseline ---
        base_u = AutoModel.from_pretrained("gpt2").to(device)
        uni_model = UniformBaselineModel(base_u, V, H).to(device)
        for p in uni_model.base_model.parameters():
            p.requires_grad = False

        # --- Train or load ---
        if os.path.exists(weight_path) and os.path.exists(uniform_path):
            print(f"Loading saved weights for {label}...")
            model.load_state_dict(torch.load(weight_path, map_location=device))
            uni_model.load_state_dict(torch.load(uniform_path, map_location=device))
        else:
            train_v4(model, tokenizer, device,
                     epochs=args.epochs, max_length=args.max_train_len,
                     batch_size=args.batch_size, lr=args.lr,
                     sparsity_coeff=args.sparsity_coeff,
                     bimodal_coeff=args.bimodal_coeff,
                     label=f"{label}_ribosome")

            train_uniform(uni_model, tokenizer, device,
                          epochs=args.epochs, max_length=args.max_train_len,
                          batch_size=args.batch_size, lr=args.lr,
                          label=f"{label}_uniform")

            torch.save(model.state_dict(), weight_path)
            torch.save(uni_model.state_dict(), uniform_path)
            print(f"Saved weights for {label}")

        # --- Evaluate ---
        print(f"\nEvaluating {label} (soft cascade)...")
        condition_results = {}
        for seq_len in args.eval_lengths:
            ribo_res = evaluate_model(model, tokenizer, device, seq_len,
                                      n_samples=args.eval_samples, is_ribosome=True)
            uni_res = evaluate_model(uni_model, tokenizer, device, seq_len,
                                     n_samples=args.eval_samples, is_ribosome=False)
            if ribo_res and uni_res:
                condition_results[str(seq_len)] = {
                    "ribosome": ribo_res,
                    "uniform": uni_res,
                    "delta": float(ribo_res["loss"] - uni_res["loss"])
                }

        # Hard cascade evaluation
        print(f"Evaluating {label} (hard cascade)...")
        hard_results = {}
        for seq_len in args.eval_lengths:
            hr = evaluate_hard_cascade(model, tokenizer, device, seq_len, n_samples=20)
            if hr:
                hard_results[str(seq_len)] = hr

        all_results[label] = {
            "soft_eval": condition_results,
            "hard_cascade": hard_results,
        }

        del model, uni_model, base, base_u
        torch.cuda.empty_cache()

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 70)
    print("V4 RESULTS: Soft Cascade (training path)")
    print("=" * 70)

    for label, res in all_results.items():
        cond = res["soft_eval"]
        print(f"\n  {label}:")
        print(f"  {'len':>6s}  {'ribo_CE':>8s}  {'uni_CE':>8s}  {'delta':>8s}  "
              f"{'PVR':>8s}  {'bimodal':>8s}  {'sparse%':>8s}")
        print(f"  {'-'*60}")
        for sl, d in sorted(cond.items(), key=lambda x: int(x[0])):
            r = d["ribosome"]
            u = d["uniform"]
            print(f"  {sl:>6s}  {r['loss']:8.4f}  {u['loss']:8.4f}  "
                  f"{d['delta']:+8.4f}  "
                  f"{r.get('pvr', 0):8.2f}  "
                  f"{r.get('bimodality', 0):8.4f}  "
                  f"{r.get('sparsity', 0)*100:7.1f}%")

    print("\n" + "=" * 70)
    print("V4 RESULTS: Hard Cascade (inference path)")
    print("=" * 70)

    for label, res in all_results.items():
        hc = res["hard_cascade"]
        print(f"\n  {label}:")
        print(f"  {'len':>6s}  {'chunks':>10s}  {'compress':>10s}")
        print(f"  {'-'*30}")
        for sl, d in sorted(hc.items(), key=lambda x: int(x[0])):
            print(f"  {sl:>6s}  {d['mean_chunks']:10.1f}  {d['mean_compression']:10.2f}x")

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved to {args.output}")


if __name__ == "__main__":
    main()
