"""
Ribosome-Cascade Track 2: Training Script
==========================================
Trains the native architecture from scratch on OpenWebText.

Features:
  - Streaming OpenWebText via HuggingFace datasets
  - Cosine LR schedule with warmup
  - Ribosome alpha ramp (0→1 over first 10% of training)
  - Gumbel temperature annealing (1.0→0.1)
  - Gradient accumulation for effective larger batches
  - VRAM safety cap
  - Periodic eval + checkpoint saving
  - Colab-compatible (no Windows paths when run remotely)
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
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer
from datasets import load_dataset

from native_arch_v1 import RibosomeCascadeNative


# ============================================================
# STREAMING DATASET
# ============================================================

class StreamingTextDataset(IterableDataset):
    """
    Streams from OpenWebText (or any HF text dataset), packs
    tokenized text into fixed-length chunks with no padding waste.
    """
    def __init__(self, tokenizer, max_length=512, dataset_name="openwebtext",
                 split="train", buffer_size=1000):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset_name = dataset_name
        self.split = split
        self.buffer_size = buffer_size

    def __iter__(self):
        ds = load_dataset(self.dataset_name, split=self.split, streaming=True,
                          trust_remote_code=True)
        ds = ds.shuffle(buffer_size=self.buffer_size)

        token_buffer = []
        for example in ds:
            text = example.get("text", "")
            if not text.strip():
                continue
            ids = self.tokenizer.encode(text)
            token_buffer.extend(ids)

            while len(token_buffer) >= self.max_length + 1:
                chunk = token_buffer[:self.max_length + 1]
                token_buffer = token_buffer[self.max_length:]
                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)
                yield {"input_ids": input_ids, "labels": labels}


# ============================================================
# WIKITEXT FALLBACK (for quick validation)
# ============================================================

def get_wikitext_loader(tokenizer, max_length=512, batch_size=8, split="train",
                        variant="wikitext-103-raw-v1"):
    ds = load_dataset("wikitext", variant, split=split)
    ds = ds.filter(lambda x: len(x["text"].strip()) > 20)

    all_ids = []
    for ex in ds:
        all_ids.extend(tokenizer.encode(ex["text"]))

    # Pack into chunks
    chunks = []
    for i in range(0, len(all_ids) - max_length, max_length):
        input_ids = torch.tensor(all_ids[i:i + max_length], dtype=torch.long)
        labels = torch.tensor(all_ids[i + 1:i + max_length + 1], dtype=torch.long)
        chunks.append({"input_ids": input_ids, "labels": labels})

    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return self.data[idx]

    return DataLoader(SimpleDataset(chunks), batch_size=batch_size, shuffle=(split == "train"))


# ============================================================
# LR SCHEDULE
# ============================================================

def get_lr(step, total_steps, max_lr, min_lr, warmup_steps):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ============================================================
# TRAINING
# ============================================================

def train(args):
    device = torch.device(args.device)
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {total_vram:.1f} GB (cap: {args.max_vram_gb:.1f} GB)")
        frac = min(args.max_vram_gb / total_vram, 0.95)
        torch.cuda.set_per_process_memory_fraction(frac)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Model
    model = RibosomeCascadeNative(
        vocab_size=len(tokenizer),
        hidden_size=args.hidden_size,
        n_heads=args.n_heads,
        lower_layers=args.lower_layers,
        upper_layers=args.upper_layers,
        cascade_layers=args.cascade_layers,
        max_seq_len=args.max_length,
        max_chunks=args.max_chunks,
    ).to(device)

    total_p, train_p = model.count_params()
    print(f"Model params: {total_p:,} total, {train_p:,} trainable")

    # Data
    if args.dataset in ("wikitext", "wikitext2"):
        variant = "wikitext-2-raw-v1" if args.dataset == "wikitext2" else "wikitext-103-raw-v1"
        train_loader = get_wikitext_loader(
            tokenizer, args.max_length, args.batch_size, "train", variant)
        val_loader = get_wikitext_loader(
            tokenizer, args.max_length, args.batch_size, "validation", variant)
        steps_per_epoch = len(train_loader) // args.grad_accum
    else:
        train_ds = StreamingTextDataset(
            tokenizer, args.max_length, args.dataset)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size)
        val_loader = get_wikitext_loader(
            tokenizer, args.max_length, args.batch_size, "validation")
        steps_per_epoch = args.steps_per_epoch

    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * 0.05)
    alpha_ramp_steps = int(total_steps * 0.10)

    print(f"Total steps: {total_steps}, warmup: {warmup_steps}, "
          f"alpha ramp: {alpha_ramp_steps}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.max_lr,
        betas=(0.9, 0.95), weight_decay=0.1
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # Training loop
    model.train()
    global_step = 0
    best_val_loss = float("inf")
    log_history = []

    for epoch in range(args.epochs):
        epoch_losses = []
        epoch_start = time.time()
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            if args.dataset != "wikitext" and batch_idx >= steps_per_epoch * args.grad_accum:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Update schedules
            lr = get_lr(global_step, total_steps, args.max_lr, args.min_lr, warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # Alpha ramp: 0→1 over first 10%
            if global_step < alpha_ramp_steps:
                model.ribosome_alpha = global_step / alpha_ramp_steps
            else:
                model.ribosome_alpha = 1.0

            # Gumbel temperature annealing: 1.0→0.1
            progress = min(global_step / max(total_steps, 1), 1.0)
            model.ribosome.gumbel_temperature = 1.0 - 0.9 * progress

            # Forward
            loss, logits, importance = model(input_ids, labels)
            
            # Only apply importance regularization after alpha ramp
            # (don't penalize scores before the ribosome is even active)
            if model.ribosome_alpha > 0.5:
                imp_reg = 0.001 * importance.mean()
                loss = loss + imp_reg
            
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
                    imp_mean = importance.mean().item()
                    imp_std = importance.std().item()
                    sparsity = (importance < 0.3).float().mean().item()
                    gumbel_t = model.ribosome.gumbel_temperature
                    alpha = model.ribosome_alpha

                    entry = {
                        "step": global_step, "epoch": epoch + 1,
                        "loss": float(mean_loss), "lr": lr,
                        "alpha": alpha, "gumbel_t": gumbel_t,
                        "imp_mean": imp_mean, "imp_std": imp_std,
                        "sparsity": sparsity,
                    }
                    log_history.append(entry)

                    print(f"  step {global_step:5d}  "
                          f"CE={mean_loss:.4f}  lr={lr:.2e}  "
                          f"alpha={alpha:.3f}  gumbel={gumbel_t:.3f}  "
                          f"imp={imp_mean:.3f}±{imp_std:.3f}  "
                          f"sparse={sparsity*100:.1f}%")

                # Eval + checkpoint
                if global_step % args.eval_every == 0:
                    val_loss = evaluate(model, val_loader, device)
                    print(f"  >>> VAL loss={val_loss:.4f} "
                          f"(best={best_val_loss:.4f})")

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        ckpt_path = os.path.join(args.output_dir, "best.pt")
                        torch.save({
                            "step": global_step,
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "val_loss": val_loss,
                            "args": vars(args),
                        }, ckpt_path)
                        print(f"  >>> Saved best checkpoint (step {global_step})")

                    model.train()

        epoch_time = time.time() - epoch_start
        epoch_mean = np.mean(epoch_losses)
        print(f"Epoch {epoch+1}/{args.epochs} done  "
              f"mean_CE={epoch_mean:.4f}  time={epoch_time:.1f}s")

    # Final save
    final_path = os.path.join(args.output_dir, "final.pt")
    torch.save({
        "step": global_step,
        "model": model.state_dict(),
        "val_loss": best_val_loss,
        "args": vars(args),
    }, final_path)

    log_path = os.path.join(args.output_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(log_history, f, indent=2)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {final_path}")
    print(f"Log: {log_path}")


def evaluate(model, val_loader, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            loss, _, _ = model(input_ids, labels)
            losses.append(loss.item())
    return float(np.mean(losses))


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train Ribosome-Cascade Native")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_vram_gb", type=float, default=20.0)

    # Model
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--n_heads", type=int, default=12)
    parser.add_argument("--lower_layers", type=int, default=4)
    parser.add_argument("--upper_layers", type=int, default=4)
    parser.add_argument("--cascade_layers", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_chunks", type=int, default=64)

    # Data
    parser.add_argument("--dataset", default="wikitext",
                        help="'wikitext' for wikitext-103, 'wikitext2' for small, 'openwebtext' for real training")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--steps_per_epoch", type=int, default=1000,
                        help="Steps per epoch for streaming datasets")

    # Training
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)

    # Logging
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--output_dir", default="./native_v1_checkpoints")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
