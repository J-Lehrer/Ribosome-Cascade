"""
Ribosome-Cascade v5: Importance-Modulated Sparse Attention
==========================================================
No bottleneck. All tokens preserved. Importance scores shape HOW
tokens attend to each other, not WHETHER they exist.

Core mechanism:
  - Each token's attention span is proportional to its importance score
  - High-importance tokens (anchors) attend broadly across the sequence
  - Low-importance tokens (filler) attend only to nearby high-importance tokens
  - This is the differentiable analog of the hard cascade's gravity behavior

The attention mask is continuous and derived from importance scores,
so the ribosome scorer gets direct gradient signal from the attention
pattern it creates.

Fixes carried forward from v4:
  - Fix 1: Subword indexing (token-level, no word splits)
  - Fix 2: Tie-breaker to heavier peak (strong-attractor semantics)
  - Bimodal penalty with target sparsity ratio
  - VRAM safety cap
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
# IMPORTANCE-MODULATED SPARSE ATTENTION (v5 core)
# ============================================================

class ImportanceAttentionLayer(nn.Module):
    """
    Self-attention where the attention mask is shaped by importance scores.
    
    For each token i with importance s_i:
      - Attention span ~ s_i (high importance = broad attention)
      - Attraction strength ~ s_j for each key j (high importance keys
        attract attention from everyone)
    
    The soft attention mask M[i,j] combines:
      1. Distance penalty: tokens far apart attend less (locality)
      2. Query reach: high-importance queries can reach further
      3. Key attraction: high-importance keys are visible from further away
    
    M[i,j] = sigma(reach_i + attract_j - |i-j| / temperature)
    
    Where reach_i = f(s_i) and attract_j = g(s_j) are learned functions
    of the importance scores.
    """
    
    def __init__(self, hidden_size, n_heads=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads
        
        # Standard QKV projections
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        
        # Importance -> attention reach (how far this token can see)
        # Maps importance score [0,1] to a reach value
        self.reach_proj = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, n_heads),  # per-head reach
        )
        
        # Importance -> attraction strength (how visible this token is)
        self.attract_proj = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, n_heads),  # per-head attraction
        )
        
        # Temperature for distance scaling (learned)
        self.log_temperature = nn.Parameter(torch.zeros(n_heads))
        
        self.norm = nn.LayerNorm(hidden_size)
    
    def forward(self, hidden_states, importance_scores):
        """
        Args:
            hidden_states: (B, S, H)
            importance_scores: (B, S) in [0, 1]
        Returns:
            refined_states: (B, S, H)
        """
        B, S, H = hidden_states.shape
        
        # QKV projections
        Q = self.q_proj(hidden_states).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(hidden_states).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(hidden_states).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        # Q, K, V: (B, n_heads, S, head_dim)
        
        # Standard attention scores
        scale = math.sqrt(self.head_dim)
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / scale
        # attn_logits: (B, n_heads, S, S)
        
        # --- Importance-modulated mask ---
        
        # Compute reach per token (how far query i can see)
        imp = importance_scores.unsqueeze(-1)  # (B, S, 1)
        reach = self.reach_proj(imp)  # (B, S, n_heads)
        reach = reach.permute(0, 2, 1).unsqueeze(-1)  # (B, n_heads, S, 1)
        
        # Compute attraction per token (how visible key j is)
        attract = self.attract_proj(imp)  # (B, S, n_heads)
        attract = attract.permute(0, 2, 1).unsqueeze(-2)  # (B, n_heads, 1, S)
        
        # Distance matrix |i - j|
        positions = torch.arange(S, device=hidden_states.device, dtype=torch.float32)
        dist = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()  # (S, S)
        dist = dist.unsqueeze(0).unsqueeze(0)  # (1, 1, S, S)
        
        # Temperature per head
        temperature = self.log_temperature.exp().view(1, self.n_heads, 1, 1) + 1.0
        
        # Soft mask: high reach + high attraction - normalized_distance/temp
        # Normalize distance by sequence length so learned reach/attract
        # values transfer across sequence lengths (trained at 128, eval at 512)
        normalized_dist = dist / S
        mask_logits = reach + attract - normalized_dist / temperature
        soft_mask = torch.sigmoid(mask_logits)  # (B, n_heads, S, S)
        
        # Apply mask to attention (additive in log-space)
        # Where soft_mask ≈ 0, attention should be suppressed
        # Where soft_mask ≈ 1, attention proceeds normally
        mask_bias = torch.log(soft_mask + 1e-8)
        attn_logits = attn_logits + mask_bias
        
        # Softmax and attend
        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_output = torch.matmul(attn_weights, V)  # (B, n_heads, S, head_dim)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, H)
        output = self.o_proj(attn_output)
        
        # Residual + norm
        refined = self.norm(hidden_states + output)
        
        return refined


# ============================================================
# MODEL
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


class RibosomeCascadeModelV5(nn.Module):
    def __init__(self, base_model, hidden_size, vocab_size, n_heads=4):
        super().__init__()
        self.base_model = base_model
        self.ribosome = RibosomeScorer(hidden_size)
        self.importance_attn = ImportanceAttentionLayer(hidden_size, n_heads=n_heads)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (B, S, H)
        
        importance = self.ribosome(hidden_states)  # (B, S)
        
        # Importance-modulated attention: no bottleneck, all tokens kept
        refined = self.importance_attn(hidden_states, importance)
        
        logits = self.lm_head(refined)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        return loss, logits, importance
    
    def get_attention_pattern(self, input_ids, attention_mask=None):
        """Debug: return the importance-modulated attention weights."""
        self.eval()
        with torch.no_grad():
            outputs = self.base_model(input_ids, attention_mask=attention_mask)
            h = outputs.last_hidden_state
            imp = self.ribosome(h)
            
            # Re-run attention to get weights
            B, S, H = h.shape
            layer = self.importance_attn
            Q = layer.q_proj(h).view(B, S, layer.n_heads, layer.head_dim).transpose(1, 2)
            K = layer.k_proj(h).view(B, S, layer.n_heads, layer.head_dim).transpose(1, 2)
            
            scale = math.sqrt(layer.head_dim)
            attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / scale
            
            imp_in = imp.unsqueeze(-1)
            reach = layer.reach_proj(imp_in).permute(0, 2, 1).unsqueeze(-1)
            attract = layer.attract_proj(imp_in).permute(0, 2, 1).unsqueeze(-2)
            
            positions = torch.arange(S, device=h.device, dtype=torch.float32)
            dist = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs().unsqueeze(0).unsqueeze(0)
            temp = layer.log_temperature.exp().view(1, layer.n_heads, 1, 1) + 1.0
            
            mask_logits = reach + attract - (dist / S) / temp
            soft_mask = torch.sigmoid(mask_logits)
            
            mask_bias = torch.log(soft_mask + 1e-8)
            attn_logits = attn_logits + mask_bias
            attn_weights = F.softmax(attn_logits, dim=-1)
            
        return imp, attn_weights, soft_mask


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
# UTILS
# ============================================================

def freeze_base(model):
    for p in model.base_model.parameters():
        p.requires_grad = False

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
    bc = (skew ** 2 + 1) / (kurt + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
    return float(bc)


def bimodal_penalty(scores, target_low_fraction=0.4):
    extremity = (scores * (1 - scores)).mean()
    soft_low = torch.sigmoid(10.0 * (0.5 - scores))
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
    return DataLoader(_cached_dataset[cache_key], batch_size=batch_size,
                      shuffle=(split == "train"))


# ============================================================
# TRAINING
# ============================================================

def train_v5(model, tokenizer, device, epochs, max_length, batch_size,
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


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Ribosome-Cascade v5")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_train_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--sparsity_coeff", type=float, default=0.1)
    parser.add_argument("--bimodal_coeff", type=float, default=0.5)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--eval_lengths", nargs="+", type=int,
                        default=[32, 64, 128, 256, 512])
    parser.add_argument("--eval_samples", type=int, default=50)
    parser.add_argument("--weights_dir", default="./weights_v5.1")
    parser.add_argument("--output", default="./benchmark_results_v5.1.json")
    parser.add_argument("--max_vram_gb", type=float, default=20.0)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {total_vram:.1f} GB (cap: {args.max_vram_gb:.1f} GB)")
        frac = min(args.max_vram_gb / total_vram, 0.95)
        torch.cuda.set_per_process_memory_fraction(frac)
        print(f"VRAM fraction cap: {frac:.2f}")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    os.makedirs(args.weights_dir, exist_ok=True)

    weight_path = os.path.join(args.weights_dir, "frozen_v5.1.pt")
    uniform_path = os.path.join(args.weights_dir, "frozen_v5.1_uniform.pt")

    print(f"\n{'#'*60}")
    print(f"RIBOSOME-CASCADE v5: Importance-Modulated Sparse Attention")
    print(f"{'#'*60}")

    # --- Ribosome v5 ---
    base = AutoModel.from_pretrained("gpt2").to(device)
    H = base.config.n_embd
    V = base.config.vocab_size
    model = RibosomeCascadeModelV5(base, H, V, n_heads=args.n_heads).to(device)
    freeze_base(model)

    # --- Uniform baseline ---
    base_u = AutoModel.from_pretrained("gpt2").to(device)
    uni_model = UniformBaselineModel(base_u, V, H).to(device)
    for p in uni_model.base_model.parameters():
        p.requires_grad = False

    if os.path.exists(weight_path) and os.path.exists(uniform_path):
        print("Loading saved weights...")
        model.load_state_dict(torch.load(weight_path, map_location=device))
        uni_model.load_state_dict(torch.load(uniform_path, map_location=device))
    else:
        train_v5(model, tokenizer, device,
                 epochs=args.epochs, max_length=args.max_train_len,
                 batch_size=args.batch_size, lr=args.lr,
                 sparsity_coeff=args.sparsity_coeff,
                 bimodal_coeff=args.bimodal_coeff,
                 label="v5_ribosome")

        train_uniform(uni_model, tokenizer, device,
                      epochs=args.epochs, max_length=args.max_train_len,
                      batch_size=args.batch_size, lr=args.lr,
                      label="v5_uniform")

        torch.save(model.state_dict(), weight_path)
        torch.save(uni_model.state_dict(), uniform_path)
        print("Saved weights")

    # --- Evaluate ---
    print(f"\nEvaluating v5...")
    results = {}
    for seq_len in args.eval_lengths:
        ribo_res = evaluate_model(model, tokenizer, device, seq_len,
                                  n_samples=args.eval_samples, is_ribosome=True)
        uni_res = evaluate_model(uni_model, tokenizer, device, seq_len,
                                 n_samples=args.eval_samples, is_ribosome=False)
        if ribo_res and uni_res:
            results[str(seq_len)] = {
                "ribosome": ribo_res,
                "uniform": uni_res,
                "delta": float(ribo_res["loss"] - uni_res["loss"])
            }

    # --- Summary ---
    print("\n" + "=" * 70)
    print("V5 RESULTS: Importance-Modulated Sparse Attention")
    print("=" * 70)
    print(f"  {'len':>6s}  {'ribo_CE':>8s}  {'uni_CE':>8s}  {'delta':>8s}  "
          f"{'PVR':>8s}  {'bimodal':>8s}  {'sparse%':>8s}")
    print(f"  {'-'*60}")
    for sl, d in sorted(results.items(), key=lambda x: int(x[0])):
        r = d["ribosome"]
        u = d["uniform"]
        print(f"  {sl:>6s}  {r['loss']:8.4f}  {u['loss']:8.4f}  "
              f"{d['delta']:+8.4f}  "
              f"{r.get('pvr', 0):8.2f}  "
              f"{r.get('bimodality', 0):8.4f}  "
              f"{r.get('sparsity', 0)*100:7.1f}%")

    # --- Comparison vs v2/v3 baseline ---
    print("\n" + "=" * 70)
    print("COMPARISON: v5 delta vs v2 frozen delta (-0.25)")
    print("=" * 70)
    for sl, d in sorted(results.items(), key=lambda x: int(x[0])):
        v5_delta = d["delta"]
        improvement = -0.25 - v5_delta  # positive = v5 is better than v2
        status = "BETTER" if v5_delta < -0.25 else ("MATCH" if v5_delta < 0 else "WORSE")
        print(f"  len={sl:>3s}  v5_delta={v5_delta:+.4f}  "
              f"vs_v2={improvement:+.4f}  [{status}]")

    all_results = {"frozen_v5": results}
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved to {args.output}")


if __name__ == "__main__":
    main()
