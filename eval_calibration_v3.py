"""
Calibration: Run GPT-2 Small through the EXACT same eval path
that train_model uses. Same loader, same loss computation.

If GPT-2 Small gives PPL ~44 here, our numbers are real.
If it gives PPL ~2, we still have a bug.
"""
import torch
import torch.nn.functional as F
import numpy as np
import math
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_native import get_wikitext_loader
from transformers import AutoTokenizer, AutoModelForCausalLM


class GPT2Wrapper(torch.nn.Module):
    """Wrap HF GPT-2 to match our eval interface."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, labels=None):
        out = self.model(input_ids)
        logits = out.logits
        loss = None
        if labels is not None:
            # Same loss as our corrected models: no shift
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1))
        return loss, logits


def evaluate_like_train_model(model, val_loader, device):
    """Exact same eval loop as train_model in exp2_lighter.py."""
    model.eval()
    val_losses = []
    with torch.no_grad():
        for vb in val_loader:
            vi = vb["input_ids"].to(device)
            vl = vb["labels"].to(device)
            vv, _ = model(vi, vl)
            val_losses.append(vv.item())
    val_loss = np.mean(val_losses)
    return val_loss


def main():
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name()}")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    max_length = 256
    batch_size = 8  # same as corrected runs

    # Same val loader as corrected training runs
    print("Loading wikitext-103 val (same loader as train_model)...")
    val_loader = get_wikitext_loader(
        tokenizer, max_length, batch_size, "validation", "wikitext-103-raw-v1")
    print(f"  {len(val_loader)} batches")

    # GPT-2 Small
    print("\nLoading GPT-2 Small (124M)...")
    gpt2 = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    wrapped = GPT2Wrapper(gpt2).to(device)

    val_ce = evaluate_like_train_model(wrapped, val_loader, device)
    val_ppl = math.exp(val_ce)

    print(f"\n{'='*60}")
    print(f"GPT-2 Small through train_model eval path:")
    print(f"  Val CE  = {val_ce:.4f}")
    print(f"  Val PPL = {val_ppl:.1f}")
    print(f"{'='*60}")
    print(f"\nExpected: PPL ~29-44")
    print(f"If PPL >> 44: bug in eval loader or loss")
    print(f"If PPL << 10: bug in eval loader or loss")
    print(f"If PPL ~29-44: our corrected results are real")


if __name__ == "__main__":
    main()
