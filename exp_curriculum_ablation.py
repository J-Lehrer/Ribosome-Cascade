"""
Curriculum Ablation: Does staged layer introduction explain the ribosome's advantage?

12-layer transformer with alpha-ramp bypass:
- During first 10% of training, output comes from layer 4 (bypass upper 8)
- Alpha ramps from 0→1, blending in upper layer contribution
- Same schedule as RibosomeTiny's ribosome_alpha ramp
- If this matches RibosomeTiny performance, the curriculum is the hero, not compression
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, sys, json, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from native_arch_v1 import RMSNorm, RotaryEmbedding, TransformerBlock
from train_native import get_wikitext_loader, get_lr, StreamingTextDataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader


class CurriculumBaseline(nn.Module):
    """12-layer transformer with alpha-ramp layer bypass.
    First `split` layers always active. Remaining layers blended in via alpha."""
    def __init__(self, vocab_size=50257, hidden_size=512, n_heads=8,
                 n_layers=12, max_seq_len=256, split=4):
        super().__init__()
        self.split = split
        self.tok_emb = nn.Embedding(vocab_size, hidden_size)
        self.rope = RotaryEmbedding(hidden_size // n_heads, max_seq_len)
        self.lower = nn.ModuleList([
            TransformerBlock(hidden_size, n_heads, self.rope) for _ in range(split)
        ])
        self.upper = nn.ModuleList([
            TransformerBlock(hidden_size, n_heads, self.rope) for _ in range(n_layers - split)
        ])
        self.norm = RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self.alpha = 0.0  # ramps 0→1 during training
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

        # Lower layers (always active)
        for layer in self.lower:
            x = layer(x, causal_mask=causal)
        lower_out = x

        # Upper layers (blended in via alpha)
        for layer in self.upper:
            x = layer(x, causal_mask=causal)

        # Alpha blend: output = alpha * upper_out + (1-alpha) * lower_out
        x = self.alpha * x + (1.0 - self.alpha) * lower_out

        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            # No shift: loader already provides aligned input/label pairs
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1))
        return loss, logits

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

def main():
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name()}")
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    cap = min(total_vram * 0.9, 22.0)
    frac = cap / total_vram
    torch.cuda.set_per_process_memory_fraction(frac)
    print(f"VRAM cap: {cap:.0f}GB ({frac:.0%} of {total_vram:.0f}GB)")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    V = len(tokenizer)

    model = CurriculumBaseline(vocab_size=V, hidden_size=512, n_heads=8,
                               n_layers=12, max_seq_len=256, split=4).to(device)
    print(f"CurriculumBaseline: {model.count_params():,} params (split=4, alpha-ramp)")

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
    out_dir = "./exp_curriculum_ablation"
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

    print(f"\nTraining {total_steps} steps, alpha ramp over first {alpha_ramp_steps}")
    print(f"{'='*60}")

    for batch_idx, batch in enumerate(train_loader):
        if global_step >= total_steps:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        lr = get_lr(global_step, total_steps, max_lr, min_lr, warmup_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Alpha ramp — matches RibosomeTiny schedule exactly
        if global_step < alpha_ramp_steps:
            model.alpha = global_step / alpha_ramp_steps
        else:
            model.alpha = 1.0

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
                sps = global_step / elapsed if elapsed > 0 else 0
                eta = (total_steps - global_step) / sps / 3600 if sps > 0 else 0
                print(f"  step={global_step:>7d}  CE={mean_loss:.4f}  alpha={model.alpha:.3f}  "
                      f"lr={lr:.2e}  spd={sps:.1f}s/s  ETA={eta:.1f}h")
                log_history.append({"step": global_step, "train_loss": float(mean_loss),
                                    "lr": lr, "alpha": model.alpha})

            if global_step % eval_every == 0:
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for vb in val_loader:
                        vl, _ = model(vb["input_ids"].to(device), vb["labels"].to(device))
                        val_losses.append(vl.item())
                val_loss = np.mean(val_losses)
                ppl = np.exp(val_loss)
                print(f"  >>> VAL={val_loss:.4f} PPL={ppl:.1f} (best={best_val_loss:.4f})")
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
    print(f"\n{'='*60}")
    print(f"DONE: CurriculumBaseline {global_step} steps")
    print(f"Best val CE: {best_val_loss:.4f}  PPL: {np.exp(best_val_loss):.1f}")
    print(f"Time: {elapsed/3600:.1f}h")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
