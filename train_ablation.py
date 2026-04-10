"""
Ribosome-Cascade Track 2: Ablation Baseline
=============================================
Same model architecture but ribosome layer is permanently bypassed.
Lower transformer → directly to upper transformer → LM head.
No compression, no importance scoring, no cascade.

This gives us the true comparison: does the ribosome help or
is the CE improvement just from having more parameters?
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
    RMSNorm, RotaryEmbedding, TransformerBlock, RibosomeCascadeNative
)
from train_native import get_wikitext_loader, get_lr


class AblationModel(nn.Module):
    """
    Same param budget as RibosomeCascadeNative but no ribosome.
    Lower transformer (4L) → Upper transformer (4L) → LM head.
    Total ~10 layers of standard transformer = fair comparison.
    """
    def __init__(self, vocab_size=50257, hidden_size=768, n_heads=12,
                 n_layers=10, max_seq_len=1024):
        super().__init__()
        self.hidden_size = hidden_size
        self.tok_emb = nn.Embedding(vocab_size, hidden_size)
        self.rope = RotaryEmbedding(hidden_size // n_heads, max_seq_len)
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, n_heads, self.rope)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
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
        for layer in self.layers:
            x = layer(x, causal_mask=causal)
        x = self.norm(x)
        logits = self.lm_head(x)
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


def train_ablation(args):
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

    model = AblationModel(
        vocab_size=len(tokenizer),
        hidden_size=args.hidden_size,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_seq_len=args.max_length,
    ).to(device)

    total_p, train_p = model.count_params()
    print(f"ABLATION model params: {total_p:,} total, {train_p:,} trainable")

    train_loader = get_wikitext_loader(
        tokenizer, args.max_length, args.batch_size, "train")
    val_loader = get_wikitext_loader(
        tokenizer, args.max_length, args.batch_size, "validation")

    steps_per_epoch = len(train_loader) // args.grad_accum
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * 0.05)

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
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            lr = get_lr(global_step, total_steps, args.max_lr, args.min_lr, warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            loss, logits = model(input_ids, labels)
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
                             "loss": float(mean_loss), "lr": lr}
                    log_history.append(entry)
                    print(f"  step {global_step:5d}  CE={mean_loss:.4f}  lr={lr:.2e}")

                if global_step % args.eval_every == 0:
                    model.eval()
                    val_losses = []
                    with torch.no_grad():
                        for vb in val_loader:
                            vl, _ = model(vb["input_ids"].to(device), vb["labels"].to(device))
                            val_losses.append(vl.item())
                    val_loss = float(np.mean(val_losses))
                    print(f"  >>> VAL loss={val_loss:.4f} (best={best_val_loss:.4f})")

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save({
                            "step": global_step, "model": model.state_dict(),
                            "val_loss": val_loss, "args": vars(args),
                        }, os.path.join(args.output_dir, "best.pt"))
                        print(f"  >>> Saved best (step {global_step})")
                    model.train()

        print(f"Epoch {epoch+1}/{args.epochs} done  "
              f"mean_CE={np.mean(epoch_losses):.4f}  time={time.time()-t0:.1f}s")

    torch.save({
        "step": global_step, "model": model.state_dict(),
        "val_loss": best_val_loss, "args": vars(args),
    }, os.path.join(args.output_dir, "final.pt"))

    with open(os.path.join(args.output_dir, "training_log.json"), "w") as f:
        json.dump(log_history, f, indent=2)

    print(f"\nABLATION complete. Best val loss: {best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Ablation baseline")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_vram_gb", type=float, default=7.0)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--n_heads", type=int, default=12)
    parser.add_argument("--n_layers", type=int, default=10)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--output_dir", default="./ablation_ckpt")
    args = parser.parse_args()
    train_ablation(args)


if __name__ == "__main__":
    main()
