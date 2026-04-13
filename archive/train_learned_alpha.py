"""
Experiment A: Learned per-token alpha blend
============================================
Instead of a fixed alpha=1.0 that routes everything through the bottleneck,
the model learns PER TOKEN whether to use the bottleneck path or the direct
path. High-importance tokens → direct path (preserve precision).
Low-importance tokens → bottleneck path (compress away).

This is architecturally equivalent to a learned skip connection where
importance scores control the routing. The model can preserve token-level
detail for generation while still compressing filler.

alpha_i = sigma(f(importance_i))  -- learned function of importance
output_i = alpha_i * decoded_i + (1 - alpha_i) * token_states_i
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse, os, json, time, math
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from native_arch_v1 import (
    RMSNorm, RotaryEmbedding, TransformerBlock, FFN,
    RibosomeLayer, CascadeProcessor, ChunkDecoder
)
from train_native import get_wikitext_loader, get_lr, StreamingTextDataset
from torch.utils.data import DataLoader


class RibosomeLearnedAlpha(nn.Module):
    """
    Same as RibosomeCascadeNative but with learned per-token alpha.
    
    Key change: importance scores control routing.
    High importance → keep direct token representation (precise)
    Low importance → route through bottleneck (compress)
    """
    def __init__(self, vocab_size=50257, hidden_size=768, n_heads=12,
                 lower_layers=4, upper_layers=4, cascade_layers=2,
                 max_seq_len=1024, max_chunks=8):
        super().__init__()
        self.hidden_size = hidden_size

        self.tok_emb = nn.Embedding(vocab_size, hidden_size)
        self.rope = RotaryEmbedding(hidden_size // n_heads, max_seq_len)

        self.lower = nn.ModuleList([
            TransformerBlock(hidden_size, n_heads, self.rope)
            for _ in range(lower_layers)
        ])
        self.lower_norm = RMSNorm(hidden_size)

        self.ribosome = RibosomeLayer(hidden_size, max_chunks, n_heads)

        self.cascade = CascadeProcessor(hidden_size, n_heads, cascade_layers)

        self.upper = nn.ModuleList([
            TransformerBlock(hidden_size, n_heads)
            for _ in range(upper_layers)
        ])
        self.upper_norm = RMSNorm(hidden_size)

        self.decoder = ChunkDecoder(hidden_size, n_heads)

        # Learned routing: importance → alpha per token
        # High importance = high alpha = MORE direct path (preserve detail)
        # Low importance = low alpha = MORE bottleneck path (compress)
        # Note: this is INVERTED from before — important tokens BYPASS compression
        self.alpha_proj = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        # Global alpha ramp for training warmup (0→1 activates bottleneck)
        self.global_alpha = 1.0

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                torch.nn.init.normal_(m.weight, std=0.02)

    def forward(self, input_ids, labels=None):
        B, S = input_ids.shape
        x = self.tok_emb(input_ids)
        causal = torch.triu(
            torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1)

        for layer in self.lower:
            x = layer(x, causal_mask=causal)
        token_states = self.lower_norm(x)

        chunk_repr, chunk_weights, assign, importance = self.ribosome(token_states)
        chunk_repr = self.cascade(chunk_repr, chunk_weights)

        for layer in self.upper:
            chunk_repr = layer(chunk_repr)
        chunk_repr = self.upper_norm(chunk_repr)

        decoded = self.decoder(token_states, chunk_repr, assign)

        # Per-token routing based on importance
        # Important tokens keep direct path, filler goes through bottleneck
        per_token_alpha = self.alpha_proj(importance.unsqueeze(-1))  # (B, S, 1)

        # During warmup, scale by global alpha
        effective_alpha = per_token_alpha * self.global_alpha

        # INVERTED: high importance = high alpha = MORE direct path
        output = effective_alpha * token_states + (1 - effective_alpha) * decoded

        logits = self.lm_head(output)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1))

        return loss, logits, importance

    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


def train(args):
    device = torch.device(args.device)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {total_vram:.1f} GB (cap: {args.max_vram_gb:.1f} GB)")
        frac = min(args.max_vram_gb / total_vram, 0.95)
        torch.cuda.set_per_process_memory_fraction(frac)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = RibosomeLearnedAlpha(
        vocab_size=len(tokenizer), hidden_size=args.hidden_size,
        n_heads=args.n_heads, lower_layers=args.lower_layers,
        upper_layers=args.upper_layers, cascade_layers=args.cascade_layers,
        max_seq_len=args.max_length, max_chunks=args.n_chunks,
    ).to(device)

    total_p, train_p = model.count_params()
    print(f"LEARNED ALPHA params: {total_p:,}")

    if args.dataset == "openwebtext":
        train_ds = StreamingTextDataset(tokenizer, args.max_length, "openwebtext")
        train_loader = DataLoader(train_ds, batch_size=args.batch_size)
        variant = "wikitext-103-raw-v1"
        val_loader = get_wikitext_loader(
            tokenizer, args.max_length, args.batch_size, "validation", variant)
        steps_per_epoch = args.steps_per_epoch
    else:
        variant = "wikitext-2-raw-v1" if args.dataset == "wikitext2" else "wikitext-103-raw-v1"
        train_loader = get_wikitext_loader(
            tokenizer, args.max_length, args.batch_size, "train", variant)
        val_loader = get_wikitext_loader(
            tokenizer, args.max_length, args.batch_size, "validation", variant)
        steps_per_epoch = len(train_loader) // args.grad_accum

    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * 0.05)
    alpha_ramp_steps = int(total_steps * 0.10)
    print(f"Total steps: {total_steps}, warmup: {warmup_steps}, alpha ramp: {alpha_ramp_steps}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.max_lr, betas=(0.9, 0.95), weight_decay=0.1)

    os.makedirs(args.output_dir, exist_ok=True)
    model.train()
    global_step = 0
    best_val_loss = float("inf")
    log_history = []

    for epoch in range(args.epochs):
        epoch_losses = []
        t0 = time.time()
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            if args.dataset == "openwebtext" and batch_idx >= steps_per_epoch * args.grad_accum:
                break
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            lr = get_lr(global_step, total_steps, args.max_lr, args.min_lr, warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            if global_step < alpha_ramp_steps:
                model.global_alpha = global_step / alpha_ramp_steps
            else:
                model.global_alpha = 1.0

            model.ribosome.gumbel_temperature = 1.0 - 0.9 * min(global_step / max(total_steps, 1), 1.0)

            loss, logits, importance = model(input_ids, labels)
            loss = loss / args.grad_accum
            loss.backward()
            epoch_losses.append(loss.item() * args.grad_accum)

            if (batch_idx + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.log_every == 0:
                    mean_loss = np.mean(epoch_losses[-args.log_every * args.grad_accum:])
                    with torch.no_grad():
                        alpha_vals = model.alpha_proj(importance.unsqueeze(-1)).squeeze(-1)
                    entry = {
                        "step": global_step, "epoch": epoch + 1,
                        "loss": float(mean_loss), "lr": lr,
                        "imp_mean": importance.mean().item(),
                        "imp_std": importance.std().item(),
                        "alpha_mean": alpha_vals.mean().item(),
                        "alpha_std": alpha_vals.std().item(),
                        "sparsity": (importance < 0.3).float().mean().item(),
                    }
                    log_history.append(entry)
                    print(f"  step {global_step:5d}  CE={mean_loss:.4f}  "
                          f"imp={importance.mean().item():.3f}  "
                          f"alpha={alpha_vals.mean().item():.3f}+/-{alpha_vals.std().item():.3f}")

                if global_step % args.eval_every == 0:
                    model.eval()
                    val_losses = []
                    with torch.no_grad():
                        for vb in val_loader:
                            vl, _, _ = model(vb["input_ids"].to(device), vb["labels"].to(device))
                            val_losses.append(vl.item())
                    val_loss = float(np.mean(val_losses))
                    print(f"  >>> VAL loss={val_loss:.4f} (best={best_val_loss:.4f})")
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save({"step": global_step, "model": model.state_dict(),
                                    "val_loss": val_loss, "args": vars(args)},
                                   os.path.join(args.output_dir, "best.pt"))
                        print(f"  >>> Saved best (step {global_step})")
                    model.train()

        print(f"Epoch {epoch+1} done  mean_CE={np.mean(epoch_losses):.4f}  time={time.time()-t0:.1f}s")

    # Generation test at end
    model.eval()
    model.global_alpha = 1.0
    prompts = [
        "The history of artificial intelligence",
        "In a recent study, researchers found that",
        "Once upon a time, in a small village",
    ]
    print("\n=== GENERATION TEST ===")
    for p in prompts:
        ids = tokenizer.encode(p, return_tensors='pt').to(device)
        with torch.no_grad():
            for _ in range(60):
                if ids.shape[1] >= args.max_length:
                    break
                loss, logits, _ = model(ids)
                next_token = logits[0, -1].argmax()
                ids = torch.cat([ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
        print(f"\n  {p}")
        print(f"  → {tokenizer.decode(ids[0])}")

    torch.save({"step": global_step, "model": model.state_dict(),
                "val_loss": best_val_loss, "args": vars(args)},
               os.path.join(args.output_dir, "final.pt"))
    with open(os.path.join(args.output_dir, "training_log.json"), "w") as f:
        json.dump(log_history, f, indent=2)
    print(f"\nComplete. Best val loss: {best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_vram_gb", type=float, default=20.0)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--n_heads", type=int, default=12)
    parser.add_argument("--lower_layers", type=int, default=4)
    parser.add_argument("--upper_layers", type=int, default=4)
    parser.add_argument("--cascade_layers", type=int, default=2)
    parser.add_argument("--n_chunks", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps_per_epoch", type=int, default=50000)
    parser.add_argument("--max_lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=2500)
    parser.add_argument("--dataset", default="openwebtext")
    parser.add_argument("--output_dir", default="./learned_alpha_ckpt")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
