"""
ReverseRibosome v2 — parameterized ablation.

Tests three theories for why reverse_layers=2, n_chunks=16 still gave LAMBADA~0%:

  (A) "bypass": explicit learned per-token output gate blending
      reverse_output with token_states directly, giving the model a
      full-bandwidth path back to embedding-layer identity.

  (B) "chunks": less compression. Test n_chunks=32 with reverse=2L.
      If chunks carry more info, reverse has more to work with.

  (C) "deeper": more reconstruction capacity. reverse_layers=4 at
      n_chunks=16. If 2L was too shallow, 4L should help.

Usage:
  python exp_reverse_v2.py --tag bypass --chunks 16 --reverse 2 --bypass
  python exp_reverse_v2.py --tag chunks32 --chunks 32 --reverse 2
  python exp_reverse_v2.py --tag deep4 --chunks 16 --reverse 4
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math, sys, os, argparse, json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from native_arch_v1 import (
    RMSNorm, RotaryEmbedding, TransformerBlock, RibosomeLayer,
    ReverseRibosome,
)
from exp2_lighter import RibosomeTiny, train_model
from eval_cross_dataset import get_lambada_loader
from train_native import get_wikitext_loader
from transformers import AutoTokenizer


class ReverseRibosomeGated(ReverseRibosome):
    """
    ReverseRibosome + learned per-token output gate.

    output = gate(token_states) * reverse_path + (1 - gate(token_states)) * token_states

    The gate sees only token_states (NOT reverse_path), so it decides
    how much to trust the chunk-routed reconstruction vs. the raw
    embedding identity on a per-token basis. This gives fine-grained
    tokens (e.g. LAMBADA name completions) a clean escape hatch.
    """
    def __init__(self, hidden_size, n_heads, n_layers=2, rope=None):
        super().__init__(hidden_size, n_heads, n_layers, rope)
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, token_states, chunk_repr, token_to_chunk):
        reverse_out = super().forward(token_states, chunk_repr, token_to_chunk)
        g = self.gate(token_states)  # (B, S, 1) in [0,1]
        return g * reverse_out + (1 - g) * token_states


class RibosomeTinyBypass(RibosomeTiny):
    """RibosomeTiny that uses the gated reverse decoder."""
    def __init__(self, *, vocab_size, hidden_size, n_heads,
                 embed_layers, upper_layers, max_seq_len, n_chunks,
                 reverse_layers):
        # Build parent with reverse_layers=reverse_layers, then swap decoder
        super().__init__(
            vocab_size=vocab_size, hidden_size=hidden_size, n_heads=n_heads,
            embed_layers=embed_layers, upper_layers=upper_layers,
            max_seq_len=max_seq_len, n_chunks=n_chunks,
            reverse_layers=reverse_layers,
        )
        # Replace the ReverseRibosome decoder with the gated variant
        self.decoder = ReverseRibosomeGated(
            hidden_size, n_heads, n_layers=reverse_layers, rope=self.rope,
        )
        # Re-init the new gate layers
        for m in self.decoder.gate.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)


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
    ce = np.mean(ces) if ces else 99.0
    return ce, math.exp(ce), correct / max(total, 1), correct, total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True, help="run label, becomes output subdir")
    p.add_argument("--chunks", type=int, default=16)
    p.add_argument("--reverse", type=int, default=2, help="reverse layers")
    p.add_argument("--bypass", action="store_true", help="use gated bypass decoder")
    p.add_argument("--embed", type=int, default=3)
    p.add_argument("--upper", type=int, default=3)
    p.add_argument("--steps", type=int, default=25000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--max_lr", type=float, default=3e-4)
    p.add_argument("--max_vram_frac", type=float, default=0.85,
                   help="fraction of total VRAM to use (main=32G, laptop=24G, 3060=8G)")
    p.add_argument("--output_dir", default="./exp_reverse_v2")
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
    lambada_loader = get_lambada_loader(tokenizer, 256, args.batch)
    wt_loader = get_wikitext_loader(tokenizer, 256, args.batch, "validation",
                                     "wikitext-103-raw-v1")

    cls = RibosomeTinyBypass if args.bypass else RibosomeTiny
    tag_desc = f"chunks={args.chunks} rev={args.reverse} bypass={args.bypass}"
    print(f"\n{'='*66}\nRun: {args.tag}  ({tag_desc})\n{'='*66}")

    model = cls(
        vocab_size=V, hidden_size=512, n_heads=8,
        embed_layers=args.embed, upper_layers=args.upper,
        max_seq_len=256, n_chunks=args.chunks,
        reverse_layers=args.reverse,
    ).to(device)
    total_p = model.count_params()
    print(f"Params: {total_p:,}")

    # Effective steps = steps_per_epoch * grad_accum (train_model increments
    # global_step every grad_accum batches, so steps_per_epoch is the
    # optimizer-step count).
    steps_per_epoch = args.steps // args.grad_accum
    run_args = argparse.Namespace(
        device="cuda", max_length=256, batch_size=args.batch,
        grad_accum=args.grad_accum,
        epochs=1, steps_per_epoch=steps_per_epoch,
        max_lr=args.max_lr, min_lr=args.max_lr / 10,
        log_every=500, eval_every=2500,
        dataset="openwebtext", streaming=True,
        output_dir=args.output_dir,
    )
    val_ce = train_model(model, args.tag, tokenizer, device, run_args,
                         is_ribosome=True)

    # Eval
    lam_ce, lam_ppl, lam_acc, lam_c, lam_t = eval_lambada(model, lambada_loader, device)
    model.eval()
    wt_losses = []
    with torch.no_grad():
        for vb in wt_loader:
            loss, _, _ = model(vb["input_ids"].to(device), vb["labels"].to(device))
            wt_losses.append(loss.item())
    wt_ce = float(np.mean(wt_losses))

    print(f"\n{'='*66}\nRESULTS: {args.tag}  ({tag_desc})\n{'='*66}")
    print(f"  Params:       {total_p:,}")
    print(f"  wikitext-103: CE={wt_ce:.4f}  PPL={math.exp(wt_ce):.1f}")
    print(f"  LAMBADA:      CE={lam_ce:.4f}  PPL={lam_ppl:.1f}  "
          f"Acc={lam_acc*100:.2f}% ({lam_c}/{lam_t})")
    print(f"\n  Reference — ReverseRibosome 2L chunks=16 (olares):")
    print(f"    wt103 PPL 8.2,  LAMBADA 0.2% (11/5153)")
    print(f"  Reference — GPT-2 Small 124M:")
    print(f"    wt103 PPL 44.4, LAMBADA 58.1%")

    results = {
        "tag": args.tag,
        "config": dict(chunks=args.chunks, reverse=args.reverse, bypass=args.bypass,
                       embed=args.embed, upper=args.upper, steps=args.steps),
        "params": total_p,
        "wt103_ce": round(wt_ce, 4), "wt103_ppl": round(math.exp(wt_ce), 2),
        "lambada_ce": round(lam_ce, 4), "lambada_ppl": round(lam_ppl, 2),
        "lambada_acc": round(lam_acc, 4), "lambada_correct": lam_c,
        "lambada_total": lam_t,
    }
    out_path = os.path.join(args.output_dir, f"results_{args.tag}.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
