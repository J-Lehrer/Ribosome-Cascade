"""
Test: RibosomeTiny 3+3 WITH chunk positional encoding.
Train 25K steps, then test LAMBADA.
Does sequence ordering metadata fix the long-range context problem?
"""
import torch
import torch.nn.functional as F
import numpy as np
import math
import sys, os, argparse, json, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp2_lighter import RibosomeTiny, train_model
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
        # Pass labels so padding_mask auto-derives
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
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name()}")
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    cap = min(total_vram * 0.85, 22.0)
    torch.cuda.set_per_process_memory_fraction(cap / total_vram)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    V = len(tokenizer)

    print("Loading LAMBADA for post-training eval...")
    lambada_loader = get_lambada_loader(tokenizer, 256, 8)
    wt_loader = get_wikitext_loader(tokenizer, 256, 8, "validation", "wikitext-103-raw-v1")

    # Train with positional encoding
    print("\n" + "=" * 60)
    print("RibosomeTiny 3+3 WITH chunk positional encoding")
    print("=" * 60)

    model = RibosomeTiny(
        vocab_size=V, hidden_size=512, n_heads=8,
        embed_layers=3, upper_layers=3,
        max_seq_len=256, n_chunks=16
    ).to(device)
    print(f"Params: {model.count_params():,}")

    args = argparse.Namespace(
        device="cuda", max_length=256, batch_size=8, grad_accum=4,
        epochs=1, steps_per_epoch=6250,  # 25K effective steps
        max_lr=3e-4, min_lr=3e-5,
        log_every=500, eval_every=2500,
        dataset="openwebtext", streaming=True,
        output_dir="./exp_pos_encoding/ribosome_3p3_pos",
    )
    val_ce = train_model(model, "ribo_3p3_pos", tokenizer, device, args,
                         is_ribosome=True)

    # LAMBADA eval
    lam_ce, lam_ppl, lam_acc, lam_c, lam_t = eval_lambada(
        model, lambada_loader, device)

    # wikitext-103 eval
    model.eval()
    wt_losses = []
    with torch.no_grad():
        for vb in wt_loader:
            loss, _, _ = model(vb["input_ids"].to(device), vb["labels"].to(device))
            wt_losses.append(loss.item())
    wt_ce = np.mean(wt_losses)

    print(f"\n{'='*60}")
    print(f"RESULTS: RibosomeTiny 3+3 + chunk positional encoding")
    print(f"{'='*60}")
    print(f"  wikitext-103: CE={wt_ce:.4f}  PPL={math.exp(wt_ce):.1f}")
    print(f"  LAMBADA:      CE={lam_ce:.4f}  PPL={lam_ppl:.1f}  Acc={lam_acc*100:.1f}% ({lam_c}/{lam_t})")
    print(f"\n  Compare to PRIOR (no pos encoding):")
    print(f"  wikitext-103: CE=0.8533  PPL=2.3")
    print(f"  LAMBADA:      CE=10.0    PPL=22175  Acc=0.0%")

    results = {
        "wt103_ce": round(wt_ce, 4), "wt103_ppl": round(math.exp(wt_ce), 2),
        "lambada_ce": round(lam_ce, 4), "lambada_ppl": round(lam_ppl, 2),
        "lambada_acc": round(lam_acc, 4), "lambada_correct": lam_c,
    }
    os.makedirs("exp_pos_encoding", exist_ok=True)
    with open("exp_pos_encoding/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to exp_pos_encoding/results.json")


if __name__ == "__main__":
    main()
