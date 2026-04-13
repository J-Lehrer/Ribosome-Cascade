"""
Experiment B: Lightweight front-pass preprocessor
===================================================
The ribosome as a standalone compression module that sits in front
of a frozen pretrained LLM. NOT trying to beat the LLM on reconstruction —
trying to COMPRESS the input while preserving downstream quality.

Architecture:
  Input tokens (seq_len)
      │
  [Ribosome: score + group + compress]
      │
  Metatokens (n_chunks << seq_len)
      │
  [Frozen GPT-2: processes compressed input]
      │
  Output logits

Loss = KL divergence between:
  - Teacher: frozen GPT-2 on FULL uncompressed input
  - Student: frozen GPT-2 on compressed metatoken input

The ribosome learns to compress such that GPT-2's output is preserved.
This is distillation-based compression — the right objective for Track 1.
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
from native_arch_v1 import RMSNorm
from train_native import get_wikitext_loader, get_lr, StreamingTextDataset


class RibosomePreprocessor(nn.Module):
    """
    Lightweight ribosome that compresses token sequences for a frozen LLM.
    
    Input: token embeddings from frozen LLM (B, S, H)
    Output: compressed metatoken embeddings (B, K, H)
    
    The compressed output is fed back into the frozen LLM's transformer
    layers in place of the original token sequence.
    """
    def __init__(self, hidden_size, n_chunks=8, n_heads=4):
        super().__init__()
        self.n_chunks = n_chunks

        # Importance scorer
        self.scorer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )

        # Chunk encoder: cross-attention compression
        self.chunk_queries = nn.Parameter(torch.randn(1, n_chunks, hidden_size) * 0.02)
        self.compress_attn = nn.MultiheadAttention(
            hidden_size, num_heads=n_heads, batch_first=True)
        self.compress_norm = RMSNorm(hidden_size)

        # Position encoding for compressed sequence
        self.chunk_pos = nn.Parameter(torch.randn(1, n_chunks, hidden_size) * 0.02)

    def forward(self, token_embeds):
        """
        Args:
            token_embeds: (B, S, H) from frozen LLM's embedding layer
        Returns:
            compressed: (B, K, H) metatoken embeddings
            importance: (B, S) importance scores
        """
        B, S, H = token_embeds.shape
        K = self.n_chunks

        importance = self.scorer(token_embeds).squeeze(-1)  # (B, S)

        # Weight embeddings by importance before compression
        weighted = token_embeds * importance.unsqueeze(-1)

        # Cross-attention: K queries compress S tokens
        queries = self.chunk_queries.expand(B, -1, -1)
        compressed, _ = self.compress_attn(queries, weighted, token_embeds)
        compressed = self.compress_norm(compressed)

        # Add positional info for the compressed sequence
        compressed = compressed + self.chunk_pos

        return compressed, importance

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


class PreprocessorPipeline(nn.Module):
    """
    Full pipeline: ribosome compresses → frozen GPT-2 processes compressed.
    Compared against: frozen GPT-2 on full uncompressed input (teacher).
    """
    def __init__(self, base_model_name="gpt2", n_chunks=8, n_heads=4):
        super().__init__()
        self.base = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.base.config.n_embd

        # Freeze base model entirely
        for p in self.base.parameters():
            p.requires_grad = False

        self.ribosome = RibosomePreprocessor(hidden_size, n_chunks, n_heads)

        # LM head (frozen, from pretrained)
        self.lm_head = nn.Linear(hidden_size, self.base.config.vocab_size, bias=False)
        # Initialize from pretrained weights
        self.lm_head.weight = self.base.wte.weight
        self.lm_head.weight.requires_grad = False

    def forward_teacher(self, input_ids):
        """Full uncompressed forward (teacher signal)."""
        with torch.no_grad():
            outputs = self.base(input_ids)
            logits = self.lm_head(outputs.last_hidden_state)
        return logits

    def forward_student(self, input_ids):
        """Compressed forward through ribosome."""
        with torch.no_grad():
            embeds = self.base.wte(input_ids)  # (B, S, H)

        compressed, importance = self.ribosome(embeds)  # (B, K, H)

        # Feed compressed sequence through frozen transformer layers
        # We bypass the embedding layer and feed directly into the blocks
        hidden = compressed
        for block in self.base.h:
            with torch.no_grad():
                # Frozen transformer processes compressed input
                outputs = block(hidden)
                hidden = outputs[0]
        with torch.no_grad():
            hidden = self.base.ln_f(hidden)

        logits = self.lm_head(hidden)  # (B, K, vocab)
        return logits, importance, compressed

    def forward(self, input_ids, labels=None):
        teacher_logits = self.forward_teacher(input_ids)  # (B, S, V)
        student_logits, importance, compressed = self.forward_student(input_ids)  # (B, K, V)

        # Teacher provides soft targets from full sequence
        # Student predicts from compressed sequence
        # Since K != S, we compare at the sequence level:
        # Average teacher logits over chunks of S/K tokens each
        B, S, V = teacher_logits.shape
        K = student_logits.shape[1]
        chunk_size = S // K

        # Reshape teacher into K chunks and mean-pool
        # Trim to exact multiple of K
        trim_S = chunk_size * K
        teacher_chunked = teacher_logits[:, :trim_S, :].view(B, K, chunk_size, V).mean(dim=2)

        # KL divergence: student should match teacher's output distribution
        teacher_probs = F.softmax(teacher_chunked / 2.0, dim=-1)  # temperature=2
        student_log_probs = F.log_softmax(student_logits / 2.0, dim=-1)
        kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (2.0 ** 2)

        # Also compute CE on the student's own predictions
        # Map each chunk back to its original tokens for CE
        if labels is not None:
            labels_chunked = labels[:, :trim_S].view(B, K, chunk_size)
            # Use the LAST token in each chunk as the target (next-token prediction)
            chunk_labels = labels_chunked[:, :, -1]  # (B, K)
            ce_loss = F.cross_entropy(
                student_logits.view(-1, V), chunk_labels.view(-1))
        else:
            ce_loss = torch.tensor(0.0, device=input_ids.device)

        loss = kl_loss + 0.5 * ce_loss

        return loss, student_logits, importance, kl_loss.item(), ce_loss.item()


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

    model = PreprocessorPipeline(
        base_model_name="gpt2", n_chunks=args.n_chunks, n_heads=4
    ).to(device)

    ribo_params = model.ribosome.count_params()
    print(f"Ribosome params: {ribo_params:,} (only these are trained)")
    print(f"Compression: {args.max_length} tokens → {args.n_chunks} chunks "
          f"({args.max_length // args.n_chunks}:1)")

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
    print(f"Total steps: {total_steps}, warmup: {warmup_steps}")

    # Only optimize ribosome parameters
    optimizer = torch.optim.AdamW(
        model.ribosome.parameters(), lr=args.max_lr,
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

            loss, logits, importance, kl, ce = model(input_ids, labels)
            loss = loss / args.grad_accum
            loss.backward()
            epoch_losses.append(loss.item() * args.grad_accum)

            if (batch_idx + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.ribosome.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.log_every == 0:
                    mean_loss = np.mean(epoch_losses[-args.log_every * args.grad_accum:])
                    entry = {
                        "step": global_step, "epoch": epoch + 1,
                        "loss": float(mean_loss), "kl": kl, "ce": ce,
                        "lr": lr,
                        "imp_mean": importance.mean().item(),
                        "imp_std": importance.std().item(),
                        "sparsity": (importance < 0.3).float().mean().item(),
                    }
                    log_history.append(entry)
                    print(f"  step {global_step:5d}  loss={mean_loss:.4f}  "
                          f"KL={kl:.4f}  CE={ce:.4f}  "
                          f"imp={importance.mean().item():.3f}+/-{importance.std().item():.3f}")

                if global_step % args.eval_every == 0:
                    model.eval()
                    val_losses = []
                    with torch.no_grad():
                        for vb in val_loader:
                            vl, _, _, _, _ = model(
                                vb["input_ids"].to(device), vb["labels"].to(device))
                            val_losses.append(vl.item())
                    val_loss = float(np.mean(val_losses))
                    print(f"  >>> VAL loss={val_loss:.4f} (best={best_val_loss:.4f})")
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save({
                            "step": global_step,
                            "ribosome": model.ribosome.state_dict(),
                            "val_loss": val_loss, "args": vars(args),
                        }, os.path.join(args.output_dir, "best.pt"))
                        print(f"  >>> Saved best (step {global_step})")
                    model.train()

        print(f"Epoch {epoch+1} done  mean_loss={np.mean(epoch_losses):.4f}  "
              f"time={time.time()-t0:.1f}s")

    # Save
    torch.save({
        "step": global_step,
        "ribosome": model.ribosome.state_dict(),
        "val_loss": best_val_loss, "args": vars(args),
    }, os.path.join(args.output_dir, "final.pt"))
    with open(os.path.join(args.output_dir, "training_log.json"), "w") as f:
        json.dump(log_history, f, indent=2)
    print(f"\nPREPROCESSOR complete. Best val loss: {best_val_loss:.4f}")
    print(f"Ribosome params: {ribo_params:,}")
    print(f"Compression: {args.max_length}→{args.n_chunks} ({args.max_length//args.n_chunks}:1)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_vram_gb", type=float, default=20.0)
    parser.add_argument("--n_chunks", type=int, default=16)
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
    parser.add_argument("--output_dir", default="./preprocessor_ckpt")
    args = parser.parse_args()
    train_preprocessor(args)


if __name__ == "__main__":
    main()
