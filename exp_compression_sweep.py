"""
Compression Ratio Sweep: RibosomeTiny with n_chunks ∈ {4, 8, 32}
We already have n_chunks=16 from olares. This maps the Pareto frontier.
Runs sequentially on side (3060 Ti, 8GB).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, sys, json, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp2_lighter import RibosomeTiny
from train_native import get_wikitext_loader, get_lr, StreamingTextDataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader


def train_one(n_chunks, device, tokenizer, V):
    print(f"\n{'#'*60}")
    print(f"# COMPRESSION SWEEP: n_chunks={n_chunks} (256 -> {n_chunks})")
    print(f"{'#'*60}")

    model = RibosomeTiny(vocab_size=V, hidden_size=512, n_heads=8,
                         embed_layers=2, upper_layers=4,
                         max_seq_len=256, n_chunks=n_chunks).to(device)
    print(f"Params: {model.count_params():,}")

    total_steps = 100000
    batch_size = 4
    grad_accum = 4
    max_lr = 3e-4
    min_lr = 3e-5
    warmup_steps = int(total_steps * 0.05)
    alpha_ramp_steps = int(total_steps * 0.10)
    log_every = 500
    eval_every = 5000
    max_length = 256
    out_dir = f"./exp_compression_sweep/chunks_{n_chunks}"
    os.makedirs(out_dir, exist_ok=True)

    train_ds = StreamingTextDataset(tokenizer, max_length, "openwebtext")
    train_loader = DataLoader(train_ds, batch_size=batch_size)
    val_loader = get_wikitext_loader(tokenizer, max_length, batch_size, "validation", "wikitext-103-raw-v1")

    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, betas=(0.9, 0.95), weight_decay=0.1)
    model.train()
    global_step = 0
    best_val_loss = float("inf")
    log_history = []
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

        if global_step < alpha_ramp_steps:
            model.ribosome_alpha = global_step / alpha_ramp_steps
        else:
            model.ribosome_alpha = 1.0
        model.ribosome.gumbel_temperature = 1.0 - 0.9 * min(global_step / max(total_steps, 1), 1.0)

        loss, logits, importance = model(input_ids, labels)
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
                sps = global_step / elapsed if elapsed > 0 else 0
                eta = (total_steps - global_step) / sps / 3600 if sps > 0 else 0
                print(f"  step={global_step:>7d}  CE={mean_loss:.4f}  lr={lr:.2e}  ETA={eta:.1f}h")
                log_history.append({"step": global_step, "train_loss": float(mean_loss)})

            if global_step % eval_every == 0:
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for vb in val_loader:
                        vl, _, _ = model(vb["input_ids"].to(device), vb["labels"].to(device))
                        val_losses.append(vl.item())
                val_loss = np.mean(val_losses)
                print(f"  >>> VAL={val_loss:.4f} PPL={np.exp(val_loss):.1f} (best={best_val_loss:.4f})")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({"step": global_step, "model": model.state_dict(),
                                "val_loss": val_loss, "params": model.count_params()},
                               os.path.join(out_dir, "best.pt"))
                model.train()

    torch.save({"step": global_step, "model": model.state_dict(),
                "val_loss": best_val_loss, "params": model.count_params()},
               os.path.join(out_dir, "final.pt"))
    with open(os.path.join(out_dir, "training_log.json"), "w") as f:
        json.dump(log_history, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n  chunks={n_chunks}: best_val={best_val_loss:.4f} PPL={np.exp(best_val_loss):.1f} time={elapsed/3600:.1f}h")
    del model, optimizer
    torch.cuda.empty_cache()
    return {"n_chunks": n_chunks, "val_ce": float(best_val_loss),
            "val_ppl": float(np.exp(best_val_loss)), "time_h": round(elapsed/3600, 2)}


def main():
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name()}")
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    cap = min(total_vram * 0.85, 7.0)
    torch.cuda.set_per_process_memory_fraction(cap / total_vram)
    print(f"VRAM cap: {cap:.0f}GB")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    V = len(tokenizer)

    chunks_to_test = [4, 8, 32]
    results = []
    for nc in chunks_to_test:
        r = train_one(nc, device, tokenizer, V)
        results.append(r)
        print(f"\n{'='*60}")
        print(f"PROGRESS: {len(results)}/{len(chunks_to_test)} done")
        for rr in results:
            print(f"  chunks={rr['n_chunks']:>3d}: CE={rr['val_ce']:.4f} PPL={rr['val_ppl']:.1f}")
        print(f"{'='*60}")

    with open("./exp_compression_sweep/summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nAll done. Saved to exp_compression_sweep/summary.json")

if __name__ == "__main__":
    main()
