"""
Test LAMBADA with padding mask fix.
Loads the existing corrected 3+3 checkpoint (trained without mask)
and tests if just masking at eval time helps.
Then tests if it also helps the 64-chunk model.
"""
import torch
import torch.nn.functional as F
import math
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp2_lighter import RibosomeTiny
from eval_cross_dataset import get_lambada_loader
from transformers import AutoTokenizer, AutoModelForCausalLM


@torch.no_grad()
def eval_lambada(model, loader, device, use_mask=True):
    model.eval()
    correct = total = 0
    ces = []

    for batch in loader:
        ids = batch["input_ids"].to(device)
        lab = batch["labels"].to(device)

        if use_mask:
            # Pass labels so padding_mask auto-derives from -100 positions
            _, logits, _ = model(ids, lab)
        else:
            # No mask (original behavior)
            _, logits, _ = model(ids)

        # CE on valid positions
        flat_logits = logits.view(-1, logits.size(-1))
        flat_labels = lab.view(-1)
        mask = flat_labels != -100
        if mask.sum() > 0:
            ces.append(F.cross_entropy(flat_logits[mask], flat_labels[mask]).item())

        # Accuracy on last valid token
        for i in range(ids.size(0)):
            valid = (lab[i] != -100).nonzero(as_tuple=True)[0]
            if len(valid) == 0:
                continue
            last = valid[-1].item()
            if logits[i, last].argmax().item() == lab[i, last].item():
                correct += 1
            total += 1

    ce = np.mean(ces) if ces else 99.0
    acc = correct / max(total, 1)
    return ce, math.exp(ce), acc, correct, total


def main():
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name()}")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    V = len(tokenizer)

    print("Loading LAMBADA...")
    loader = get_lambada_loader(tokenizer, 256, 8)

    # Also check that wikitext-103 isn't affected by the mask
    from train_native import get_wikitext_loader
    wt_loader = get_wikitext_loader(tokenizer, 256, 8, "validation", "wikitext-103-raw-v1")

    configs = [
        ("RibosomeTiny 3+3 (16 chunks)", 3, 3, 16,
         "corrected_v1/ribosome_3p3/best.pt"),
        ("RibosomeTiny 3+3 (64 chunks)", 3, 3, 64,
         "overnight_chunks/chunks_64/ribo_3p3_c64/best.pt"),
    ]

    print(f"\n{'='*70}")
    print(f"LAMBADA WITH vs WITHOUT PADDING MASK")
    print(f"{'='*70}")

    for name, el, ul, nc, ckpt_path in configs:
        if not os.path.exists(ckpt_path):
            print(f"\n  {name}: checkpoint not found ({ckpt_path})")
            continue

        m = RibosomeTiny(V, 512, 8, embed_layers=el, upper_layers=ul,
                         max_seq_len=256, n_chunks=nc).to(device)
        m.load_state_dict(torch.load(ckpt_path, map_location=device,
                                      weights_only=False)["model"])

        print(f"\n  {name}")
        print(f"  {'':>4} {'LAMBADA CE':>10} {'PPL':>10} {'Acc':>8}")
        print(f"  {'-'*40}")

        # Without mask (original)
        ce, ppl, acc, c, t = eval_lambada(m, loader, device, use_mask=False)
        print(f"  {'no mask':>4} {ce:>10.4f} {ppl:>10.1f} {acc*100:>7.1f}%  ({c}/{t})")

        # With mask
        ce, ppl, acc, c, t = eval_lambada(m, loader, device, use_mask=True)
        print(f"  {'MASK':>4} {ce:>10.4f} {ppl:>10.1f} {acc*100:>7.1f}%  ({c}/{t})")

        # Sanity: wikitext-103 shouldn't change (no padding)
        m.eval()
        wt_losses = []
        with torch.no_grad():
            for vb in wt_loader:
                vi = vb["input_ids"].to(device)
                vl = vb["labels"].to(device)
                loss, _, _ = m(vi, vl)
                wt_losses.append(loss.item())
        wt_ce = np.mean(wt_losses)
        print(f"  wt103 CE={wt_ce:.4f} PPL={math.exp(wt_ce):.1f} (should be unchanged)")

        del m; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
