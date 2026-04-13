"""
Experiment 2: Lighter — tiny model on metatokens vs big model on raw tokens
=============================================================================
Can a 4-layer transformer processing 16 metatokens match a 12-layer
transformer processing 256 raw tokens?

Setup:
  A) Big model: 12-layer transformer, 256 raw tokens (GPT-2-like baseline)
  B) Ribosome + Tiny: ribosome compresses 256→16, then 4-layer transformer
     processes metatokens. Much fewer params, much less compute.

Both trained from scratch on OpenWebText, evaluated on wikitext-103 val.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse, os, json, time, math
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from native_arch_v1 import (
    RMSNorm, RotaryEmbedding, TransformerBlock, RibosomeLayer, CascadeProcessor, ChunkDecoder
)
from train_native import get_wikitext_loader, get_lr, StreamingTextDataset, PreloadedTextDataset


class BigBaseline(nn.Module):
    """Standard 12-layer transformer on raw tokens."""
    def __init__(self, vocab_size=50257, hidden_size=512, n_heads=8,
                 n_layers=12, max_seq_len=256):
        super().__init__()
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
        return sum(p.numel() for p in self.parameters())


class RibosomeTiny(nn.Module):
    """
    Ribosome compresses 256→16 tokens, then a tiny 4-layer transformer
    processes metatokens. Much smaller total compute than the big baseline.
    """
    def __init__(self, vocab_size=50257, hidden_size=512, n_heads=8,
                 embed_layers=2, upper_layers=4, max_seq_len=256, n_chunks=16):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, hidden_size)
        self.rope = RotaryEmbedding(hidden_size // n_heads, max_seq_len)

        # Light embedding layers (build token representations)
        self.embed = nn.ModuleList([
            TransformerBlock(hidden_size, n_heads, self.rope)
            for _ in range(embed_layers)
        ])
        self.embed_norm = RMSNorm(hidden_size)

        # Ribosome: compress
        self.ribosome = RibosomeLayer(hidden_size, n_chunks, n_heads)

        # Tiny upper transformer on metatokens
        self.upper = nn.ModuleList([
            TransformerBlock(hidden_size, n_heads)
            for _ in range(upper_layers)
        ])
        self.upper_norm = RMSNorm(hidden_size)

        # Decode back to tokens
        self.decoder = ChunkDecoder(hidden_size, n_heads)

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

        for layer in self.embed:
            x = layer(x, causal_mask=causal)
        token_states = self.embed_norm(x)

        chunk_repr, chunk_weights, assign, importance = self.ribosome(token_states)

        for layer in self.upper:
            chunk_repr = layer(chunk_repr)
        chunk_repr = self.upper_norm(chunk_repr)

        decoded = self.decoder(token_states, chunk_repr, assign)
        output = self.ribosome_alpha * decoded + (1 - self.ribosome_alpha) * token_states

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
        return sum(p.numel() for p in self.parameters())


def train_model(model, name, tokenizer, device, args, is_ribosome=False):
    use_streaming = getattr(args, "streaming", True)
    if args.dataset == "openwebtext":
        if use_streaming:
            train_ds = StreamingTextDataset(tokenizer, args.max_length, "openwebtext")
        else:
            max_tok = getattr(args, "max_tokens", None)
            train_ds = PreloadedTextDataset(tokenizer, args.max_length, "openwebtext",
                                            max_tokens=max_tok)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size)
        val_loader = get_wikitext_loader(
            tokenizer, args.max_length, args.batch_size, "validation", "wikitext-103-raw-v1")
        steps_per_epoch = args.steps_per_epoch
    elif args.dataset == "local_wikitext103":
        # Load pre-tokenized tokens from disk (Colab-friendly, no HF downloads at train time)
        token_path = getattr(args, "token_cache", "/content/wikitext103_tokens.pt")
        import torch as _torch
        print(f"  Loading pre-tokenized data from {token_path}...")
        all_ids = _torch.load(token_path, weights_only=True)
        if not isinstance(all_ids, list):
            all_ids = all_ids.tolist() if hasattr(all_ids, 'tolist') else list(all_ids)
        chunks = []
        for i in range(0, len(all_ids) - args.max_length, args.max_length):
            inp = torch.tensor(all_ids[i:i + args.max_length], dtype=torch.long)
            lab = torch.tensor(all_ids[i + 1:i + args.max_length + 1], dtype=torch.long)
            chunks.append({"input_ids": inp, "labels": lab})
        print(f"  {len(all_ids):,} tokens -> {len(chunks):,} chunks")

        class _DS(torch.utils.data.Dataset):
            def __init__(self, d): self.d = d
            def __len__(self): return len(self.d)
            def __getitem__(self, i): return self.d[i]

        train_loader = DataLoader(_DS(chunks), batch_size=args.batch_size, shuffle=True)
        val_loader = get_wikitext_loader(
            tokenizer, args.max_length, args.batch_size, "validation", "wikitext-103-raw-v1")
        steps_per_epoch = len(train_loader) // args.grad_accum
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

    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print(f"Params: {model.count_params():,}")
    print(f"Steps: {total_steps}, warmup: {warmup_steps}")
    print(f"{'='*60}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.max_lr, betas=(0.9, 0.95), weight_decay=0.1)

    out_dir = os.path.join(args.output_dir, name)
    os.makedirs(out_dir, exist_ok=True)
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

            if is_ribosome:
                if global_step < alpha_ramp_steps:
                    model.ribosome_alpha = global_step / alpha_ramp_steps
                else:
                    model.ribosome_alpha = 1.0
                model.ribosome.gumbel_temperature = 1.0 - 0.9 * min(
                    global_step / max(total_steps, 1), 1.0)
                loss, logits, importance = model(input_ids, labels)
            else:
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
                    entry = {"step": global_step, "loss": float(mean_loss), "lr": lr}
                    if is_ribosome:
                        entry["imp_mean"] = importance.mean().item()
                        entry["imp_std"] = importance.std().item()
                    log_history.append(entry)
                    extra = f"  imp={importance.mean().item():.3f}" if is_ribosome else ""
                    print(f"  [{name}] step {global_step:5d}  CE={mean_loss:.4f}{extra}")

                if global_step % args.eval_every == 0:
                    model.eval()
                    val_losses = []
                    with torch.no_grad():
                        for vb in val_loader:
                            ids = vb["input_ids"].to(device)
                            lab = vb["labels"].to(device)
                            if is_ribosome:
                                vl, _, _ = model(ids, lab)
                            else:
                                vl, _ = model(ids, lab)
                            val_losses.append(vl.item())
                    val_loss = float(np.mean(val_losses))
                    print(f"  [{name}] >>> VAL={val_loss:.4f} (best={best_val_loss:.4f})")
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save({"step": global_step, "model": model.state_dict(),
                                    "val_loss": val_loss, "params": model.count_params()},
                                   os.path.join(out_dir, "best.pt"))
                    model.train()

        print(f"  [{name}] Epoch {epoch+1} done  CE={np.mean(epoch_losses):.4f}  "
              f"time={time.time()-t0:.1f}s")

    torch.save({"step": global_step, "model": model.state_dict(),
                "val_loss": best_val_loss, "params": model.count_params()},
               os.path.join(out_dir, "final.pt"))
    with open(os.path.join(out_dir, "training_log.json"), "w") as f:
        json.dump(log_history, f, indent=2)

    return best_val_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_vram_gb", type=float, default=7.0)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps_per_epoch", type=int, default=20000)
    parser.add_argument("--max_lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=2000)
    parser.add_argument("--dataset", default="openwebtext")
    parser.add_argument("--output_dir", default="./lighter_experiment")
    parser.add_argument("--model", default="both", choices=["big", "tiny", "both"])
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {total_vram:.1f} GB (cap: {args.max_vram_gb:.1f} GB)")
        frac = min(args.max_vram_gb / total_vram, 0.95)
        torch.cuda.set_per_process_memory_fraction(frac)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    V = len(tokenizer)

    results = {}

    if args.model in ("big", "both"):
        big = BigBaseline(vocab_size=V, hidden_size=512, n_heads=8,
                          n_layers=12, max_seq_len=args.max_length).to(device)
        print(f"\nBig baseline: {big.count_params():,} params, 12 layers, 256 tokens")
        results["big"] = train_model(big, "big_12L", tokenizer, device, args)
        del big
        torch.cuda.empty_cache()

    if args.model in ("tiny", "both"):
        tiny = RibosomeTiny(vocab_size=V, hidden_size=512, n_heads=8,
                            embed_layers=2, upper_layers=4,
                            max_seq_len=args.max_length, n_chunks=16).to(device)
        print(f"\nRibosome tiny: {tiny.count_params():,} params, "
              f"2 embed + 4 upper layers, 16 metatokens")
        results["tiny"] = train_model(tiny, "ribosome_tiny", tokenizer, device, args,
                                       is_ribosome=True)
        del tiny
        torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for name, val in results.items():
        print(f"  {name}: val CE = {val:.4f}")


if __name__ == "__main__":
    main()
