"""
Perceiver-only ablation: same bottleneck architecture as ribosome
but with FIXED uniform chunking (no importance scoring).

If the ribosome beats this at high compression, importance scoring matters.
If they're equal, the bottleneck alone explains everything.
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
from transformers import AutoTokenizer
from datasets import load_dataset

from native_arch_v1 import (
    RMSNorm, RotaryEmbedding, TransformerBlock, FFN
)
from train_native import get_wikitext_loader, get_lr, StreamingTextDataset


class FixedPerceiverModel(nn.Module):
    """
    Same architecture as RibosomeCascadeNative but:
    - No importance scorer
    - No boundary predictor
    - Fixed uniform chunking via learnable queries (standard Perceiver)
    - Same cascade processor with uniform weights
    - Same chunk decoder
    - Same alpha-ramp bypass
    """
    def __init__(self, vocab_size=50257, hidden_size=768, n_heads=12,
                 lower_layers=4, upper_layers=4, cascade_layers=2,
                 max_seq_len=1024, n_chunks=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_chunks = n_chunks

        self.tok_emb = nn.Embedding(vocab_size, hidden_size)
        self.rope = RotaryEmbedding(hidden_size // n_heads, max_seq_len)

        self.lower = nn.ModuleList([
            TransformerBlock(hidden_size, n_heads, self.rope)
            for _ in range(lower_layers)
        ])
        self.lower_norm = RMSNorm(hidden_size)

        # Fixed Perceiver: learnable queries, no importance
        self.chunk_queries = nn.Parameter(torch.randn(1, n_chunks, hidden_size) * 0.02)
        self.chunk_cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads=n_heads, batch_first=True)
        self.chunk_norm = RMSNorm(hidden_size)

        # Cascade (no priority sorting — uniform weights)
        self.cascade = nn.ModuleList([
            TransformerBlock(hidden_size, n_heads)
            for _ in range(cascade_layers)
        ])

        self.upper = nn.ModuleList([
            TransformerBlock(hidden_size, n_heads)
            for _ in range(upper_layers)
        ])
        self.upper_norm = RMSNorm(hidden_size)

        # Decoder: expand chunks back to tokens
        self.decode_attn = nn.MultiheadAttention(
            hidden_size, num_heads=n_heads, batch_first=True)
        self.decode_norm = RMSNorm(hidden_size)
        self.decode_ffn = FFN(hidden_size)
        self.decode_ffn_norm = RMSNorm(hidden_size)

        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        self.ribosome_alpha = 1.0
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

        # Fixed Perceiver compression
        queries = self.chunk_queries.expand(B, -1, -1)
        chunk_repr, _ = self.chunk_cross_attn(queries, token_states, token_states)
        chunk_repr = self.chunk_norm(chunk_repr)

        # Cascade (standard self-attention, no priority sort)
        for layer in self.cascade:
            chunk_repr = layer(chunk_repr)

        # Upper transformer on chunks
        for layer in self.upper:
            chunk_repr = layer(chunk_repr)
        chunk_repr = self.upper_norm(chunk_repr)

        # Decode back to tokens
        decoded, _ = self.decode_attn(token_states, chunk_repr, chunk_repr)
        decoded = self.decode_norm(token_states + decoded)
        decoded = decoded + self.decode_ffn(self.decode_ffn_norm(decoded))

        output = self.ribosome_alpha * decoded + (1 - self.ribosome_alpha) * token_states
        logits = self.lm_head(output)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1))
        return loss, logits

    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


def train_perceiver(args):
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

    model = FixedPerceiverModel(
        vocab_size=len(tokenizer), hidden_size=args.hidden_size,
        n_heads=args.n_heads, lower_layers=args.lower_layers,
        upper_layers=args.upper_layers, cascade_layers=args.cascade_layers,
        max_seq_len=args.max_length, n_chunks=args.n_chunks,
    ).to(device)

    total_p, train_p = model.count_params()
    print(f"PERCEIVER ABLATION params: {total_p:,} (n_chunks={args.n_chunks})")

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
    print(f"Total steps: {total_steps}, warmup: {warmup_steps}")

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
                model.ribosome_alpha = global_step / alpha_ramp_steps
            else:
                model.ribosome_alpha = 1.0

            loss, _ = model(input_ids, labels)
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
                    entry = {"step": global_step, "epoch": epoch + 1,
                             "loss": float(mean_loss), "lr": lr,
                             "alpha": model.ribosome_alpha}
                    log_history.append(entry)
                    print(f"  step {global_step:5d}  CE={mean_loss:.4f}  "
                          f"lr={lr:.2e}  alpha={model.ribosome_alpha:.3f}")

                if global_step % args.eval_every == 0:
                    model.eval()
                    val_losses = []
                    with torch.no_grad():
                        for vb in val_loader:
                            vl, _ = model(vb["input_ids"].to(device),
                                          vb["labels"].to(device))
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

        print(f"Epoch {epoch+1}/{args.epochs} done  "
              f"mean_CE={np.mean(epoch_losses):.4f}  time={time.time()-t0:.1f}s")

    torch.save({"step": global_step, "model": model.state_dict(),
                "val_loss": best_val_loss, "args": vars(args)},
               os.path.join(args.output_dir, "final.pt"))
    with open(os.path.join(args.output_dir, "training_log.json"), "w") as f:
        json.dump(log_history, f, indent=2)
    print(f"\nPERCEIVER ABLATION complete. Best val loss: {best_val_loss:.4f}")


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
    parser.add_argument("--steps_per_epoch", type=int, default=5000)
    parser.add_argument("--max_lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--dataset", default="openwebtext")
    parser.add_argument("--output_dir", default="./perceiver_ckpt")
    args = parser.parse_args()
    train_perceiver(args)


if __name__ == "__main__":
    main()
