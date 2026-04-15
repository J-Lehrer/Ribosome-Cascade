"""exp_causal_v1.py — test whether fixing the future-info leak rescues LAMBADA.

Hypothesis
----------
Prior runs all achieved suspiciously low wt103 PPL (1.14 at chunks=64) with
LAMBADA stuck at chance. Architectural audit located a future-info leak:
  1. Upper transformer over chunks runs bidirectional self-attention.
  2. ReverseRibosome.cross_attn lets token t attend to ALL K chunks.
  3. ChunkDecoder.cross_attn same issue.
Combined: token t's prediction can access info from tokens > t via the chunks
pathway, bypassing the causal mask in the embed layers.

Fix (in native_arch_v1.py, backward-compat via causal_chunks=True flag):
  - Token-to-chunk cross-attn is masked so token t attends only to chunks
    c <= floor(t*K/S) + slack.
  - Upper transformer over chunks is given a causal mask on chunk index.

This script:
  - Trains RibosomeTinyCausal at chunks=64, reverse=2 (matches olares rev2_64c_25k config).
  - Evaluates wt103 PPL + LAMBADA acc.
  - If PPL lands in the 20-60 range (honest) AND LAMBADA > 5%, the architecture
    is vindicated and we escalate scale. If PPL is still <2, there's a second
    leak. If PPL is honest but LAMBADA is still zero, clean negative result.
"""
import argparse, json, math, os, time, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the (patched) native arch + existing RibosomeTiny
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from native_arch_v1 import (
    RMSNorm, RotaryEmbedding, TransformerBlock,
    RibosomeLayer, ReverseRibosome, ChunkDecoder,
)
from exp2_lighter import RibosomeTiny


class RibosomeTinyCausal(RibosomeTiny):
    """RibosomeTiny with all three causal patches enabled.

    Patches:
      1. ReverseRibosome is rebuilt with causal_chunks=True.
      2. ChunkDecoder (if used) is rebuilt with causal_chunks=True.
      3. forward() passes a causal mask to the upper transformer over chunks.
    """
    def __init__(self, vocab_size=50257, hidden_size=512, n_heads=8,
                 embed_layers=2, upper_layers=4, max_seq_len=256, n_chunks=16,
                 reverse_layers=0, chunk_mask_slack=1):
        super().__init__(vocab_size=vocab_size, hidden_size=hidden_size,
                         n_heads=n_heads, embed_layers=embed_layers,
                         upper_layers=upper_layers, max_seq_len=max_seq_len,
                         n_chunks=n_chunks, reverse_layers=reverse_layers)
        self.chunk_mask_slack = chunk_mask_slack
        # Swap decoder for causal-aware version. We re-init a fresh module; the
        # lm_head/tok_emb tie is preserved because lm_head is a separate attr.
        if reverse_layers > 0:
            self.decoder = ReverseRibosome(
                hidden_size, n_heads, n_layers=reverse_layers, rope=self.rope,
                causal_chunks=True, chunk_mask_slack=chunk_mask_slack)
        else:
            self.decoder = ChunkDecoder(
                hidden_size, n_heads,
                causal_chunks=True, chunk_mask_slack=chunk_mask_slack)
        # Re-apply the standard init so the new decoder has the same init scheme
        self._init_weights()

    def forward(self, input_ids, labels=None, padding_mask=None):
        B, S = input_ids.shape
        if padding_mask is None and labels is not None:
            if (labels == -100).any():
                padding_mask = (labels != -100)

        x = self.tok_emb(input_ids)
        causal = torch.triu(
            torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1)

        for layer in self.embed:
            x = layer(x, causal_mask=causal)
        token_states = self.embed_norm(x)

        chunk_repr, chunk_weights, assign, importance, chunk_positions = self.ribosome(
            token_states, padding_mask=padding_mask)

        # FIX #3: causal mask over chunks in the upper transformer.
        K = chunk_repr.size(1)
        chunk_causal = torch.triu(
            torch.ones(K, K, device=chunk_repr.device, dtype=torch.bool),
            diagonal=1)
        for layer in self.upper:
            chunk_repr = layer(chunk_repr, causal_mask=chunk_causal)
        chunk_repr = self.upper_norm(chunk_repr)

        # Decoder (already causal_chunks=True after __init__ swap).
        decoded = self.decoder(token_states, chunk_repr, assign)
        output = self.ribosome_alpha * decoded + (1 - self.ribosome_alpha) * token_states

        logits = self.lm_head(output)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1))
        return loss, logits, importance


# ─── eval + runner ──────────────────────────────────────────────────────

import numpy as np
from exp2_lighter import train_model
from eval_cross_dataset import get_lambada_loader
from train_native import get_wikitext_loader
from transformers import AutoTokenizer


@torch.no_grad()
def eval_lambada(model, loader, device):
    model.eval()
    correct = total = 0
    ces = []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        lab = batch["labels"].to(device)
        _, logits, _ = model(ids, lab)
        flat = logits.view(-1, logits.size(-1))
        flat_lab = lab.view(-1)
        mask = flat_lab != -100
        if mask.sum() > 0:
            ces.append(F.cross_entropy(flat[mask], flat_lab[mask]).item())
        for i in range(ids.size(0)):
            valid = (lab[i] != -100).nonzero(as_tuple=True)[0]
            if len(valid) == 0:
                continue
            last = valid[-1].item()
            if logits[i, last].argmax().item() == lab[i, last].item():
                correct += 1
            total += 1
    ce = float(np.mean(ces)) if ces else 99.0
    return ce, math.exp(ce), correct / max(total, 1), correct, total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True, help="run label → subdir")
    p.add_argument("--chunks", type=int, default=64)
    p.add_argument("--reverse", type=int, default=2)
    p.add_argument("--embed", type=int, default=3)
    p.add_argument("--upper", type=int, default=3)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--seq", type=int, default=256)
    p.add_argument("--steps", type=int, default=25000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--max_lr", type=float, default=3e-4)
    p.add_argument("--chunk_mask_slack", type=int, default=1,
                   help="extra chunks of right-context allowed (>=1 required; "
                        "0 would block token 0 from seeing chunk 0)")
    p.add_argument("--max_vram_frac", type=float, default=0.85)
    p.add_argument("--output_dir", default="./exp_causal_v1")
    args = p.parse_args()

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name()}")
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM: {total_vram:.1f} GB  cap frac={args.max_vram_frac}")
    torch.cuda.set_per_process_memory_fraction(args.max_vram_frac)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    V = len(tokenizer)

    print("Loading eval datasets...")
    lambada_loader = get_lambada_loader(tokenizer, args.seq, args.batch)
    wt_loader = get_wikitext_loader(tokenizer, args.seq, args.batch,
                                     "validation", "wikitext-103-raw-v1")

    tag_desc = (f"chunks={args.chunks} rev={args.reverse} slack={args.chunk_mask_slack} "
                f"hidden={args.hidden} heads={args.heads} seq={args.seq}")
    print(f"\n{'='*72}\nCAUSAL Run: {args.tag}  ({tag_desc})\n{'='*72}")

    model = RibosomeTinyCausal(
        vocab_size=V, hidden_size=args.hidden, n_heads=args.heads,
        embed_layers=args.embed, upper_layers=args.upper,
        max_seq_len=args.seq, n_chunks=args.chunks,
        reverse_layers=args.reverse,
        chunk_mask_slack=args.chunk_mask_slack,
    ).to(device)
    total_p = model.count_params()
    print(f"Params: {total_p:,}")

    steps_per_epoch = args.steps // args.grad_accum
    run_args = argparse.Namespace(
        device="cuda", max_length=args.seq, batch_size=args.batch,
        grad_accum=args.grad_accum,
        epochs=1, steps_per_epoch=steps_per_epoch,
        max_lr=args.max_lr, min_lr=args.max_lr / 10,
        log_every=500, eval_every=2500,
        dataset="openwebtext", streaming=True,
        output_dir=args.output_dir,
    )
    val_ce = train_model(model, args.tag, tokenizer, device, run_args,
                         is_ribosome=True)

    # Final eval
    lam_ce, lam_ppl, lam_acc, lam_c, lam_t = eval_lambada(
        model, lambada_loader, device)
    model.eval()
    wt_losses = []
    with torch.no_grad():
        for vb in wt_loader:
            loss, _, _ = model(vb["input_ids"].to(device),
                               vb["labels"].to(device))
            wt_losses.append(loss.item())
    wt_ce = float(np.mean(wt_losses))

    print(f"\n{'='*72}\nRESULTS: {args.tag}  ({tag_desc})\n{'='*72}")
    print(f"  Params:       {total_p:,}")
    print(f"  wikitext-103: CE={wt_ce:.4f}  PPL={math.exp(wt_ce):.2f}")
    print(f"  LAMBADA:      CE={lam_ce:.4f}  PPL={lam_ppl:.2f}  "
          f"Acc={lam_acc*100:.2f}% ({lam_c}/{lam_t})")
    print(f"\n  Reference — non-causal rev2_64c_25k (pre-fix, leaked):")
    print(f"    wt103 PPL 1.14,  LAMBADA 0.08% (4/5153)")
    print(f"  Reference — GPT-2 Small 124M:")
    print(f"    wt103 PPL 44.4,  LAMBADA 58.1%")
    print(f"\n  Verdict heuristic:")
    print(f"    PPL in [20,80] → leak is plugged (honest LM signal).")
    print(f"    PPL < 5        → second leak likely (investigate further).")
    print(f"    LAMBADA > 5%   → architecture has real gradient to learn from.")

    results = {
        "tag": args.tag,
        "config": dict(
            chunks=args.chunks, reverse=args.reverse,
            embed=args.embed, upper=args.upper,
            hidden=args.hidden, heads=args.heads, seq=args.seq,
            steps=args.steps, chunk_mask_slack=args.chunk_mask_slack,
            causal_chunks=True,
        ),
        "params": total_p,
        "wt103_ce": round(wt_ce, 4),
        "wt103_ppl": round(math.exp(wt_ce), 2),
        "lambada_ce": round(lam_ce, 4),
        "lambada_ppl": round(lam_ppl, 2),
        "lambada_acc": round(lam_acc, 4),
        "lambada_correct": lam_c,
        "lambada_total": lam_t,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"results_{args.tag}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
