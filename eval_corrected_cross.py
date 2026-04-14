"""
Cross-dataset eval on corrected checkpoints.
Runs on olares. Tests generalization on C4 and LAMBADA.
Also includes GPT-2 Small as calibration reference.
"""
import torch
import torch.nn.functional as F
import numpy as np
import math
import json
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp2_lighter import RibosomeTiny, BigBaseline
from train_native import get_wikitext_loader
from eval_cross_dataset import get_c4_loader, get_lambada_loader
from transformers import AutoTokenizer, AutoModelForCausalLM


@torch.no_grad()
def eval_ce(model, loader, device, is_hf=False):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        if is_hf:
            out = model(input_ids)
            logits = out.logits
        else:
            result = model(input_ids)
            logits = result[1]  # (loss, logits, ...) or (loss, logits)

        # No shift: loader pre-aligns input/label pairs
        flat_logits = logits.view(-1, logits.size(-1))
        flat_labels = labels.view(-1)
        mask = flat_labels != -100
        if mask.sum() == 0:
            continue
        if (flat_labels == -100).any():
            loss = F.cross_entropy(flat_logits[mask], flat_labels[mask])
            n = mask.sum().item()
        else:
            loss = F.cross_entropy(flat_logits, flat_labels)
            n = flat_labels.numel()
        total_loss += loss.item() * n
        total_tokens += n

    ce = total_loss / max(total_tokens, 1)
    return ce, math.exp(ce), total_tokens


@torch.no_grad()
def eval_lambada_acc(model, loader, device, is_hf=False):
    model.eval()
    correct = total = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        if is_hf:
            logits = model(input_ids).logits
        else:
            logits = model(input_ids)[1]
        for i in range(input_ids.size(0)):
            valid_pos = (labels[i] != -100).nonzero(as_tuple=True)[0]
            if len(valid_pos) == 0:
                continue
            last = valid_pos[-1].item()
            if logits[i, last].argmax().item() == labels[i, last].item():
                correct += 1
            total += 1
    return correct / max(total, 1), correct, total


def main():
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name()}")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    V = len(tokenizer)
    ml = 256
    bs = 8

    # Load datasets
    print("\n=== Loading Datasets ===")
    wt103 = get_wikitext_loader(tokenizer, ml, bs, "validation", "wikitext-103-raw-v1")
    print(f"  wikitext-103: {len(wt103)} batches")
    c4 = get_c4_loader(tokenizer, ml, bs, max_examples=2000)
    lambada = get_lambada_loader(tokenizer, ml, bs)

    # Models
    models = []

    # GPT-2 Small reference
    print("\nLoading GPT-2 Small...")
    gpt2 = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    models.append(("GPT2-Small (124M)", gpt2, True))

    # Corrected RibosomeTiny 3+3
    p = "corrected_v1/ribosome_3p3/best.pt"
    if os.path.exists(p):
        m = RibosomeTiny(V, 512, 8, embed_layers=3, upper_layers=3,
                         max_seq_len=256, n_chunks=16).to(device)
        m.load_state_dict(torch.load(p, map_location=device, weights_only=False)["model"])
        models.append(("RibosomeTiny 3+3 (49M)", m, False))
        print(f"  Loaded {p}")

    # Corrected BigBaseline 12L
    p = "corrected_v1/baseline_12L/best.pt"
    if os.path.exists(p):
        m = BigBaseline(V, 512, 8, n_layers=12, max_seq_len=256).to(device)
        m.load_state_dict(torch.load(p, map_location=device, weights_only=False)["model"])
        models.append(("BigBaseline 12L (63M)", m, False))
        print(f"  Loaded {p}")

    # Run evals
    print(f"\n{'='*80}")
    print(f"CROSS-DATASET EVALUATION (corrected models)")
    print(f"{'='*80}")

    results = {}
    for name, model, is_hf in models:
        model.eval()
        r = {}

        t0 = time.time()
        ce, ppl, n = eval_ce(model, wt103, device, is_hf)
        r["wt103"] = {"ce": round(ce, 4), "ppl": round(ppl, 2)}
        print(f"\n  {name}")
        print(f"    wikitext-103: CE={ce:.4f}  PPL={ppl:.1f}  ({n:,} tok, {time.time()-t0:.1f}s)")

        t0 = time.time()
        ce, ppl, n = eval_ce(model, c4, device, is_hf)
        r["c4"] = {"ce": round(ce, 4), "ppl": round(ppl, 2)}
        print(f"    C4:           CE={ce:.4f}  PPL={ppl:.1f}  ({n:,} tok, {time.time()-t0:.1f}s)")

        t0 = time.time()
        ce, ppl, n = eval_ce(model, lambada, device, is_hf)
        acc, correct, total = eval_lambada_acc(model, lambada, device, is_hf)
        r["lambada"] = {"ce": round(ce, 4), "ppl": round(ppl, 2),
                        "acc": round(acc, 4), "correct": correct, "total": total}
        print(f"    LAMBADA:      CE={ce:.4f}  PPL={ppl:.1f}  Acc={acc*100:.1f}% ({correct}/{total}, {time.time()-t0:.1f}s)")

        results[name] = r
        if not is_hf:
            del model
            torch.cuda.empty_cache()

    # Summary table
    print(f"\n{'='*80}")
    print(f"{'Model':<25} {'wt103':>8} {'C4':>8} {'LAMB CE':>8} {'LAMB Acc':>8}")
    print(f"{'':<25} {'PPL':>8} {'PPL':>8} {'PPL':>8} {'%':>8}")
    print("-" * 80)
    for name, r in results.items():
        print(f"{name:<25} {r['wt103']['ppl']:>8.1f} {r['c4']['ppl']:>8.1f} "
              f"{r['lambada']['ppl']:>8.1f} {r['lambada']['acc']*100:>7.1f}%")
    print("=" * 80)

    with open("corrected_v1/cross_dataset_eval.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to corrected_v1/cross_dataset_eval.json")


if __name__ == "__main__":
    main()
