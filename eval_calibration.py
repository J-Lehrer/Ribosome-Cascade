"""
Calibration: Run pretrained GPT-2 Small (124M) through the exact same
eval pipeline to sanity-check our numbers.

Expected PPL on wikitext-103: ~29-30 (published)
If we get that, our eval is clean. If not, we have a bug.
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
from train_native import get_wikitext_loader
from eval_cross_dataset import get_c4_loader, get_lambada_loader
from transformers import AutoTokenizer, AutoModelForCausalLM


@torch.no_grad()
def eval_ce_hf(model, loader, device, max_length=256):
    """Eval CE for a HuggingFace causal LM."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        # HF models want labels in the input
        out = model(input_ids, labels=input_ids)
        # Manually compute CE on shifted targets to match our pipeline
        logits = out.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        mask = shift_labels != -100
        if mask.sum() == 0:
            continue
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        # For standard eval (no -100 masking)
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
    max_length = 256
    batch_size = 16

    print("\nLoading GPT-2 Small (124M, pretrained)...")
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}")

    # Also test with max_length=1024 (GPT-2's native context)
    for ml in [256, 1024]:
        print(f"\n{'='*60}")
        print(f"  GPT-2 Small @ max_length={ml}")
        print(f"{'='*60}")

        print("  wikitext-103 val...")
        loader = get_wikitext_loader(tokenizer, ml, batch_size,
                                      "validation", "wikitext-103-raw-v1")
        ce, ppl, n = eval_ce_hf(model, loader, device, ml)
        print(f"    CE={ce:.4f}  PPL={ppl:.1f}  ({n:,} tokens)")

        print("  C4 val...")
        loader = get_c4_loader(tokenizer, ml, batch_size, max_examples=2000)
        ce, ppl, n = eval_ce_hf(model, loader, device, ml)
        print(f"    CE={ce:.4f}  PPL={ppl:.1f}  ({n:,} tokens)")

        print("  LAMBADA...")
        loader = get_lambada_loader(tokenizer, ml, batch_size)
        ce, ppl, n = eval_ce_hf(model, loader, device, ml)
        print(f"    CE={ce:.4f}  PPL={ppl:.1f}  ({n:,} tokens)")

    print("\nDone. Compare these against published numbers:")
    print("  GPT-2 Small wikitext-103 PPL (published): ~29.4")
    print("  GPT-2 Small LAMBADA acc (published): ~52%")


if __name__ == "__main__":
    main()
