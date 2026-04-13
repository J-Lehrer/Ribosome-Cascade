"""
Calibration v2: Fix the double-shift bug and re-evaluate everything.

Bug: data loader pre-shifts (input=tokens[:-1], labels=tokens[1:]),
then model.forward() shifts AGAIN. Net effect: predict-2-ahead.

Fix: evaluate with corrected loss (no extra shift).
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
def eval_ce_fixed(model, loader, device, is_ribosome=False):
    """
    Corrected CE eval. Since data loader already shifts
    (input=tokens[:-1], labels=tokens[1:]), we compare
    logits directly against labels with NO extra shift.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # Get logits WITHOUT using model's internal loss (which double-shifts)
        if is_ribosome:
            _, logits, _ = model(input_ids)  # no labels -> no loss computation
        elif isinstance(model, BigBaseline):
            _, logits = model(input_ids)
        else:
            # HuggingFace model
            out = model(input_ids)
            logits = out.logits

        # Correct loss: logits[i] predicts next token, labels[i] IS the next token
        # No shift needed - loader already aligned them
        mask = labels != -100
        if mask.sum() == 0:
            continue
        flat_logits = logits.view(-1, logits.size(-1))
        flat_labels = labels.view(-1)
        if (flat_labels == -100).any():
            valid = flat_labels != -100
            loss = F.cross_entropy(flat_logits[valid], flat_labels[valid])
            n = valid.sum().item()
        else:
            loss = F.cross_entropy(flat_logits, flat_labels)
            n = flat_labels.numel()
        total_loss += loss.item() * n
        total_tokens += n

    ce = total_loss / max(total_tokens, 1)
    return ce, math.exp(ce), total_tokens


def main():
    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name()}")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    V = len(tokenizer)
    max_length = 256
    batch_size = 16

    # Load datasets once
    print("\n=== Loading Datasets ===")
    wt103_loader = get_wikitext_loader(tokenizer, max_length, batch_size,
                                        "validation", "wikitext-103-raw-v1")
    c4_loader = get_c4_loader(tokenizer, max_length, batch_size, max_examples=2000)
    # Skip LAMBADA for now - focus on CE/PPL calibration

    # Models to test
    models_to_eval = []

    # GPT-2 Small (the calibration reference)
    print("\nLoading GPT-2 Small (124M, pretrained)...")
    gpt2 = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    models_to_eval.append(("GPT2-Small (124M)", gpt2, False))

    # Our models
    if os.path.exists("checkpoints/e3_u3_best.pt"):
        m = RibosomeTiny(vocab_size=V, hidden_size=512, n_heads=8,
                         embed_layers=3, upper_layers=3,
                         max_seq_len=256, n_chunks=16).to(device)
        ckpt = torch.load("checkpoints/e3_u3_best.pt", map_location=device,
                           weights_only=False)
        m.load_state_dict(ckpt["model"])
        models_to_eval.append(("RibosomeTiny 3+3 (49M)", m, True))

    if os.path.exists("weights_olares/best.pt"):
        m = RibosomeTiny(vocab_size=V, hidden_size=512, n_heads=8,
                         embed_layers=2, upper_layers=4,
                         max_seq_len=256, n_chunks=16).to(device)
        ckpt = torch.load("weights_olares/best.pt", map_location=device,
                           weights_only=False)
        m.load_state_dict(ckpt["model"])
        models_to_eval.append(("RibosomeTiny 2+4 (49M)", m, True))

    if os.path.exists("exp_extended_baseline/best.pt"):
        m = BigBaseline(vocab_size=V, hidden_size=512, n_heads=8,
                        n_layers=12, max_seq_len=256).to(device)
        ckpt = torch.load("exp_extended_baseline/best.pt", map_location=device,
                           weights_only=False)
        m.load_state_dict(ckpt["model"])
        models_to_eval.append(("BigBaseline 12L (63M)", m, False))

    # Run evals
    print(f"\n{'='*70}")
    print(f"{'CORRECTED EVAL (no double-shift)':^70}")
    print(f"{'='*70}")
    print(f"{'Model':<30} {'wt103 CE':>10} {'PPL':>8} {'C4 CE':>10} {'PPL':>8}")
    print("-" * 70)

    results = {}
    for name, model, is_ribo in models_to_eval:
        model.eval()
        wt_ce, wt_ppl, wt_n = eval_ce_fixed(model, wt103_loader, device, is_ribo)
        c4_ce, c4_ppl, c4_n = eval_ce_fixed(model, c4_loader, device, is_ribo)
        print(f"{name:<30} {wt_ce:>10.4f} {wt_ppl:>8.1f} {c4_ce:>10.4f} {c4_ppl:>8.1f}")
        results[name] = {
            "wikitext103": {"ce": round(wt_ce, 4), "ppl": round(wt_ppl, 2)},
            "c4": {"ce": round(c4_ce, 4), "ppl": round(c4_ppl, 2)},
        }
        # Free memory
        del model
        torch.cuda.empty_cache()

    print(f"{'='*70}")
    print(f"\nReference: GPT-2 Small published wikitext-103 PPL ≈ 29.4")
    print(f"(If GPT-2 reads ~29-35, our eval pipeline is correct.)")

    with open("results/calibrated_eval.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to results/calibrated_eval.json")


if __name__ == "__main__":
    main()
