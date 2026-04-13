"""
Experiment: Extended BigBaseline to 500K steps on OWT
Resume from frank's 100K checkpoint, continue on main (5090).
VRAM cap: 30GB
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, sys, json, time, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp2_lighter import BigBaseline
from train_native import get_wikitext_loader, get_lr, StreamingTextDataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

def main():
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name()}")
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    cap = 30.0
    frac = min(cap / total_vram, 0.95)
    torch.cuda.set_per_process_memory_fraction(frac)
    print(f"VRAM cap: {cap}GB ({frac:.0%} of {total_vram:.0f}GB)")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    V = len(tokenizer)

    model = BigBaseline(vocab_size=V, hidden_size=512, n_heads=8,
                        n_layers=12, max_seq_len=256).to(device)
    print(f"BigBaseline: {model.count_params():,} params")

    # Resume from checkpoint if available
    out_dir = "./exp_extended_baseline"
    os.makedirs(out_dir, exist_ok=True)
    resume_path = "./weights_frank/best.pt"
    start_step = 0

    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        start_step = ckpt["step"]
        print(f"Resumed from {resume_path}: step={start_step}, val_loss={ckpt['val_loss']:.4f}")
    else:
        print("No checkpoint found, training from scratch")

    # Training config
    total_steps = 500000
    batch_size = 8
    grad_accum = 4  # effective batch = 32
    max_lr = 3e-4
    min_lr = 3e-5
    warmup_steps = int(total_steps * 0.05)
    log_every = 500
    eval_every = 10000
    max_length = 256

    train_ds = StreamingTextDataset(tokenizer, max_length, "openwebtext")
    train_loader = DataLoader(train_ds, batch_size=batch_size)
    val_loader = get_wikitext_loader(tokenizer, max_length, batch_size, "validation", "wikitext-103-raw-v1")

    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, betas=(0.9, 0.95), weight_decay=0.1)
    model.train()
    global_step = start_step
    best_val_loss = float("inf")
    log_history = []

    print(f"\nTraining from step {start_step} to {total_steps}")
    print(f"Batch: {batch_size} x {grad_accum} = {batch_size * grad_accum}")
    print(f"{'='*60}")

    epoch_losses = []
    optimizer.zero_grad()
    t0 = time.time()

    for batch_idx, batch in enumerate(train_loader):
        if global_step >= total_steps:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        lr = get_lr(global_step, total_steps, max_lr, min_lr, warmup_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        loss, logits = model(input_ids, labels)
        loss = loss / grad_accum
        loss.backward()
        epoch_losses.append(loss.item() * grad_accum)

        if (batch_idx + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % log_every == 0:
                mean_loss = np.mean(epoch_losses[-log_every * grad_accum:])
                elapsed = time.time() - t0
                steps_per_sec = (global_step - start_step) / elapsed if elapsed > 0 else 0
                eta_hrs = (total_steps - global_step) / steps_per_sec / 3600 if steps_per_sec > 0 else 0
                print(f"  step={global_step:>7d}  CE={mean_loss:.4f}  lr={lr:.2e}  "
                      f"spd={steps_per_sec:.1f}s/s  ETA={eta_hrs:.1f}h")
                log_history.append({"step": global_step, "train_loss": float(mean_loss), "lr": lr})

            if global_step % eval_every == 0:
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for vb in val_loader:
                        vl, _ = model(vb["input_ids"].to(device), vb["labels"].to(device))
                        val_losses.append(vl.item())
                val_loss = np.mean(val_losses)
                print(f"  >>> VAL={val_loss:.4f} (best={best_val_loss:.4f}) PPL={np.exp(val_loss):.1f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({"step": global_step, "model": model.state_dict(),
                                "val_loss": val_loss, "params": model.count_params()},
                               os.path.join(out_dir, "best.pt"))
                model.train()

    # Save final
    torch.save({"step": global_step, "model": model.state_dict(),
                "val_loss": best_val_loss, "params": model.count_params()},
               os.path.join(out_dir, "final.pt"))
    with open(os.path.join(out_dir, "training_log.json"), "w") as f:
        json.dump(log_history, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"DONE: BigBaseline extended to {global_step} steps")
    print(f"Best val CE: {best_val_loss:.4f}  PPL: {np.exp(best_val_loss):.1f}")
    print(f"Time: {elapsed/3600:.1f}h")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
