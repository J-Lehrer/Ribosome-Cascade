"""
Experiment 1: Preprocessor — compression fidelity
===================================================
Fixed version: ribosome compresses 256→16 tokens, then a small
learned decoder head predicts from the compressed representation.

The loss is KL divergence between:
  - Teacher: frozen GPT-2 on full 256 tokens
  - Student: ribosome compression → small transformer → predict

Key fix from previous version: don't feed compressed tokens through
GPT-2's transformer blocks (they expect position IDs, causal masks,
specific seq lengths). Instead, use a small dedicated decoder on top
of the compressed representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse, os, json, time, math
import numpy as np
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from native_arch_v1 import RMSNorm, TransformerBlock, RotaryEmbedding
from train_native import get_wikitext_loader, get_lr, StreamingTextDataset


class RibosomeCompressor(nn.Module):
    """Compresses token embeddings into metatokens via importance-weighted cross-attention."""
    def __init__(self, hidden_size, n_chunks=16, n_heads=4):
        super().__init__()
        self.n_chunks = n_chunks
        self.scorer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        self.chunk_queries = nn.Parameter(torch.randn(1, n_chunks, hidden_size) * 0.02)
        self.compress_attn = nn.MultiheadAttention(
            hidden_size, num_heads=n_heads, batch_first=True)
        self.compress_norm = RMSNorm(hidden_size)

    def forward(self, token_embeds):
        B, S, H = token_embeds.shape
        importance = self.scorer(token_embeds).squeeze(-1)
        weighted = token_embeds * importance.unsqueeze(-1)
        queries = self.chunk_queries.expand(B, -1, -1)
        compressed, attn_weights = self.compress_attn(queries, weighted, token_embeds)
        compressed = self.compress_norm(compressed)
        return compressed, importance, attn_weights

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


class CompressedDecoder(nn.Module):
    """Small transformer that processes metatokens and predicts vocab distributions."""
    def __init__(self, hidden_size, vocab_size, n_heads=4, n_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, n_heads)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, compressed):
        x = compressed
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.head(x)


class PreprocessorPipelineV2(nn.Module):
    def __init__(self, base_model_name="gpt2", n_chunks=16, n_heads=4, decoder_layers=2):
        super().__init__()
        self.base = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.base.config.n_embd
        vocab_size = self.base.config.vocab_size

        for p in self.base.parameters():
            p.requires_grad = False

        self.ribosome = RibosomeCompressor(hidden_size, n_chunks, n_heads)
        self.decoder = CompressedDecoder(hidden_size, vocab_size, n_heads, decoder_layers)

        # Teacher LM head (frozen, weight-tied)
        self.teacher_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.teacher_head.weight = self.base.wte.weight
        self.teacher_head.weight.requires_grad = False

    def forward_teacher(self, input_ids):
        with torch.no_grad():
            outputs = self.base(input_ids)
            logits = self.teacher_head(outputs.last_hidden_state)
        return logits

    def forward_student(self, input_ids):
        with torch.no_grad():
            # Use full frozen GPT-2 to get contextual embeddings
            outputs = self.base(input_ids)
            hidden = outputs.last_hidden_state

        compressed, importance, attn_weights = self.ribosome(hidden)
        student_logits = self.decoder(compressed)
        return student_logits, importance, attn_weights

    def forward(self, input_ids, labels=None):
        teacher_logits = self.forward_teacher(input_ids)  # (B, S, V)
        student_logits, importance, attn_weights = self.forward_student(input_ids)  # (B, K, V)

        B, S, V = teacher_logits.shape
        K = student_logits.shape[1]
        chunk_size = S // K
        trim_S = chunk_size * K

        # Pool teacher logits into K chunks
        teacher_chunked = teacher_logits[:, :trim_S, :].view(B, K, chunk_size, V).mean(dim=2)

        # KL divergence (student should match teacher)
        T = 2.0
        teacher_probs = F.softmax(teacher_chunked / T, dim=-1)
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
        kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (T ** 2)

        # CE on chunk-level next-token prediction
        if labels is not None:
            labels_chunked = labels[:, :trim_S].view(B, K, chunk_size)
            chunk_labels = labels_chunked[:, :, -1]
            ce_loss = F.cross_entropy(student_logits.view(-1, V), chunk_labels.view(-1))
        else:
            ce_loss = torch.tensor(0.0, device=input_ids.device)

        loss = kl_loss + 0.5 * ce_loss

        # Compute fidelity metric: top-1 agreement between teacher and student
        with torch.no_grad():
            teacher_top1 = teacher_chunked.argmax(dim=-1)
            student_top1 = student_logits.argmax(dim=-1)
            agreement = (teacher_top1 == student_top1).float().mean().item()

        return loss, student_logits, importance, kl_loss.item(), ce_loss.item(), agreement


def train_preprocessor(args):
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

    model = PreprocessorPipelineV2(
        n_chunks=args.n_chunks, decoder_layers=args.decoder_layers
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Trainable: {trainable:,} | Frozen (GPT-2): {frozen:,}")
    print(f"Compression: {args.max_length}→{args.n_chunks} ({args.max_length//args.n_chunks}:1)")

    if args.dataset == "openwebtext":
        train_ds = StreamingTextDataset(tokenizer, args.max_length, "openwebtext")
        train_loader = DataLoader(train_ds, batch_size=args.batch_size)
        val_loader = get_wikitext_loader(
            tokenizer, args.max_length, args.batch_size, "validation", "wikitext-103-raw-v1")
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
    print(f"Total steps: {total_steps}, warmup: {warmup_steps}")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.max_lr,
                                  betas=(0.9, 0.95), weight_decay=0.01)

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

            loss, logits, importance, kl, ce, agreement = model(input_ids, labels)
            loss = loss / args.grad_accum
            loss.backward()
            epoch_losses.append(loss.item() * args.grad_accum)

            if (batch_idx + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.log_every == 0:
                    mean_loss = np.mean(epoch_losses[-args.log_every * args.grad_accum:])
                    entry = {
                        "step": global_step, "loss": float(mean_loss),
                        "kl": kl, "ce": ce, "agreement": agreement,
                        "imp_mean": importance.mean().item(),
                        "imp_std": importance.std().item(),
                    }
                    log_history.append(entry)
                    print(f"  step {global_step:5d}  loss={mean_loss:.4f}  "
                          f"KL={kl:.4f}  CE={ce:.4f}  "
                          f"agree={agreement*100:.1f}%  "
                          f"imp={importance.mean().item():.3f}+/-{importance.std().item():.3f}")

                if global_step % args.eval_every == 0:
                    model.eval()
                    val_losses, val_agrees = [], []
                    with torch.no_grad():
                        for vb in val_loader:
                            vl, _, _, _, _, va = model(
                                vb["input_ids"].to(device), vb["labels"].to(device))
                            val_losses.append(vl.item())
                            val_agrees.append(va)
                    val_loss = float(np.mean(val_losses))
                    val_agree = float(np.mean(val_agrees))
                    print(f"  >>> VAL loss={val_loss:.4f} agree={val_agree*100:.1f}% "
                          f"(best={best_val_loss:.4f})")
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save({
                            "step": global_step,
                            "ribosome": model.ribosome.state_dict(),
                            "decoder": model.decoder.state_dict(),
                            "val_loss": val_loss, "val_agreement": val_agree,
                            "args": vars(args),
                        }, os.path.join(args.output_dir, "best.pt"))
                        print(f"  >>> Saved best (step {global_step})")
                    model.train()

        print(f"Epoch {epoch+1} done  mean_loss={np.mean(epoch_losses):.4f}  "
              f"time={time.time()-t0:.1f}s")

    torch.save({
        "step": global_step,
        "ribosome": model.ribosome.state_dict(),
        "decoder": model.decoder.state_dict(),
        "val_loss": best_val_loss, "args": vars(args),
    }, os.path.join(args.output_dir, "final.pt"))
    with open(os.path.join(args.output_dir, "training_log.json"), "w") as f:
        json.dump(log_history, f, indent=2)
    print(f"\nDone. Best val loss: {best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_vram_gb", type=float, default=20.0)
    parser.add_argument("--n_chunks", type=int, default=16)
    parser.add_argument("--decoder_layers", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps_per_epoch", type=int, default=20000)
    parser.add_argument("--max_lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-4)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--dataset", default="openwebtext")
    parser.add_argument("--output_dir", default="./preprocessor_v2_ckpt")
    args = parser.parse_args()
    train_preprocessor(args)


if __name__ == "__main__":
    main()
