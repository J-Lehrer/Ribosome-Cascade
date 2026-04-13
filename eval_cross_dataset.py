"""
Cross-Dataset Evaluation for Ribosome-Cascade
==============================================
Evaluates trained checkpoints on held-out datasets to test generalization:
  - wikitext-103 val (already known, included for completeness)
  - C4 validation (different web text distribution)
  - LAMBADA (long-range word prediction)

Runs on main (5090) for speed.
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
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader


# ============================================================
# DATASET LOADERS
# ============================================================

def get_c4_loader(tokenizer, max_length, batch_size, max_examples=2000):
    """Load C4 validation split, tokenize and pack into chunks."""
    print("  Loading C4 validation...")
    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True,
                      trust_remote_code=True)
    chunks = []
    token_buffer = []
    count = 0
    for example in ds:
        if count >= max_examples:
            break
        text = example.get("text", "")
        if not text.strip():
            continue
        ids = tokenizer.encode(text)
        token_buffer.extend(ids)
        while len(token_buffer) >= max_length + 1:
            chunk = token_buffer[:max_length + 1]
            token_buffer = token_buffer[max_length:]
            chunks.append({
                "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
                "labels": torch.tensor(chunk[1:], dtype=torch.long),
            })
        count += 1

    class ListDS(torch.utils.data.Dataset):
        def __init__(self, data): self.data = data
        def __len__(self): return len(self.data)
        def __getitem__(self, i): return self.data[i]

    print(f"  C4: {len(chunks)} chunks from {count} examples")
    return DataLoader(ListDS(chunks), batch_size=batch_size, shuffle=False)


def get_lambada_loader(tokenizer, max_length, batch_size):
    """Load LAMBADA test set for last-word prediction eval."""
    print("  Loading LAMBADA...")
    ds = load_dataset("cimec/lambada", split="test", trust_remote_code=True)
    chunks = []
    for example in ds:
        text = example.get("text", "")
        if not text.strip():
            continue
        ids = tokenizer.encode(text)
        if len(ids) > max_length:
            ids = ids[-max_length:]  # take last max_length tokens
        if len(ids) < 4:
            continue
        # Pad to max_length for batching
        pad_len = max_length - len(ids)
        input_ids = [tokenizer.eos_token_id] * pad_len + ids[:-1]
        labels = [-100] * pad_len + ids[1:]  # only predict actual tokens
        chunks.append({
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "last_word_ids": tokenizer.encode(" " + example["text"].split()[-1]),
            "seq_len": len(ids),
        })

    class ListDS(torch.utils.data.Dataset):
        def __init__(self, data): self.data = data
        def __len__(self): return len(self.data)
        def __getitem__(self, i): return self.data[i]

    print(f"  LAMBADA: {len(chunks)} examples")
    return DataLoader(ListDS(chunks), batch_size=batch_size, shuffle=False,
                      collate_fn=lambda batch: {
                          "input_ids": torch.stack([b["input_ids"] for b in batch]),
                          "labels": torch.stack([b["labels"] for b in batch]),
                          "seq_len": [b["seq_len"] for b in batch],
                      })


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def eval_ce(model, loader, device, is_ribosome=False):
    """Standard cross-entropy evaluation."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        if is_ribosome:
            loss, logits, _ = model(input_ids, labels)
        else:
            loss, logits = model(input_ids, labels)
        # Count non-masked tokens
        mask = labels != -100
        n_tokens = mask.sum().item()
        if n_tokens == 0:
            continue
        # Recompute loss only on valid tokens for LAMBADA
        if (labels == -100).any():
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            valid = shift_labels != -100
            if valid.sum() == 0:
                continue
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))[valid.view(-1)]
            flat_labels = shift_labels.view(-1)[valid.view(-1)]
            loss = F.cross_entropy(flat_logits, flat_labels)
            n_tokens = valid.sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens
    ce = total_loss / max(total_tokens, 1)
    return ce, math.exp(ce), total_tokens

@torch.no_grad()
def eval_lambada_accuracy(model, loader, device, is_ribosome=False):
    """LAMBADA last-word accuracy: can the model predict the final word?"""
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        if is_ribosome:
            _, logits, _ = model(input_ids, labels)
        else:
            _, logits = model(input_ids, labels)
        # Check last valid position per sequence
        for i in range(input_ids.size(0)):
            seq_labels = labels[i]
            valid_pos = (seq_labels != -100).nonzero(as_tuple=True)[0]
            if len(valid_pos) == 0:
                continue
            last_pos = valid_pos[-1].item()
            # The prediction for position last_pos comes from logits at last_pos-1
            pred_pos = last_pos - 1 if last_pos > 0 else 0
            pred = logits[i, pred_pos].argmax().item()
            target = seq_labels[last_pos].item()
            if pred == target:
                correct += 1
            total += 1
    return correct / max(total, 1), correct, total


# ============================================================
# MAIN
# ============================================================

def load_ribosome(path, embed_layers, upper_layers, device, vocab_size):
    model = RibosomeTiny(
        vocab_size=vocab_size, hidden_size=512, n_heads=8,
        embed_layers=embed_layers, upper_layers=upper_layers,
        max_seq_len=256, n_chunks=16
    ).to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    step = ckpt.get("step", "?")
    print(f"  Loaded {path} (step {step}, {model.count_params():,} params)")
    return model


def load_baseline(path, device, vocab_size):
    model = BigBaseline(
        vocab_size=vocab_size, hidden_size=512, n_heads=8,
        n_layers=12, max_seq_len=256
    ).to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    step = ckpt.get("step", "?")
    print(f"  Loaded {path} (step {step}, {model.count_params():,} params)")
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name() if device.type == 'cuda' else 'CPU'})")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    V = len(tokenizer)
    max_length = 256
    batch_size = 16  # 5090 can handle more

    # --- Load models ---
    print("\n=== Loading Models ===")
    models = {}

    if os.path.exists("checkpoints/e3_u3_best.pt"):
        models["RibosomeTiny_3+3"] = {
            "model": load_ribosome("checkpoints/e3_u3_best.pt", 3, 3, device, V),
            "is_ribosome": True,
        }
    if os.path.exists("weights_olares/best.pt"):
        models["RibosomeTiny_2+4"] = {
            "model": load_ribosome("weights_olares/best.pt", 2, 4, device, V),
            "is_ribosome": True,
        }
    if os.path.exists("exp_extended_baseline/best.pt"):
        models["BigBaseline_12L"] = {
            "model": load_baseline("exp_extended_baseline/best.pt", device, V),
            "is_ribosome": False,
        }

    if not models:
        print("No checkpoints found!")
        return

    # --- Load datasets ---
    print("\n=== Loading Datasets ===")
    datasets_eval = {}

    print("  [wikitext-103 val]")
    datasets_eval["wikitext103"] = {
        "loader": get_wikitext_loader(tokenizer, max_length, batch_size, "validation",
                                       "wikitext-103-raw-v1"),
        "type": "ce",
    }

    print("  [C4 val]")
    datasets_eval["c4"] = {
        "loader": get_c4_loader(tokenizer, max_length, batch_size, max_examples=2000),
        "type": "ce",
    }

    print("  [LAMBADA]")
    datasets_eval["lambada"] = {
        "loader": get_lambada_loader(tokenizer, max_length, batch_size),
        "type": "lambada",
    }

    # --- Run evals ---
    print("\n=== Running Evaluations ===")
    results = {}
    for model_name, minfo in models.items():
        model = minfo["model"]
        is_ribo = minfo["is_ribosome"]
        results[model_name] = {}

        for ds_name, dinfo in datasets_eval.items():
            t0 = time.time()
            print(f"\n  {model_name} on {ds_name}...", end=" ", flush=True)

            if dinfo["type"] == "ce":
                ce, ppl, n_tok = eval_ce(model, dinfo["loader"], device, is_ribo)
                elapsed = time.time() - t0
                results[model_name][ds_name] = {
                    "ce": round(ce, 4), "ppl": round(ppl, 2),
                    "tokens": n_tok, "time_s": round(elapsed, 1),
                }
                print(f"CE={ce:.4f}  PPL={ppl:.1f}  ({n_tok:,} tokens, {elapsed:.1f}s)")

            elif dinfo["type"] == "lambada":
                # CE + accuracy
                ce, ppl, n_tok = eval_ce(model, dinfo["loader"], device, is_ribo)
                acc, correct, total = eval_lambada_accuracy(
                    model, dinfo["loader"], device, is_ribo)
                elapsed = time.time() - t0
                results[model_name][ds_name] = {
                    "ce": round(ce, 4), "ppl": round(ppl, 2),
                    "accuracy": round(acc, 4),
                    "correct": correct, "total": total,
                    "time_s": round(elapsed, 1),
                }
                print(f"CE={ce:.4f}  PPL={ppl:.1f}  Acc={acc*100:.1f}%  "
                      f"({correct}/{total}, {elapsed:.1f}s)")

        # Free GPU memory between models
        del model
        minfo["model"] = None
        torch.cuda.empty_cache()

    # --- Summary ---
    print("\n" + "=" * 80)
    print("CROSS-DATASET EVALUATION RESULTS")
    print("=" * 80)

    # Table header
    ds_names = list(datasets_eval.keys())
    header = f"{'Model':<25}"
    for ds in ds_names:
        if ds == "lambada":
            header += f"  {'LAMBADA CE':>10} {'PPL':>8} {'Acc%':>6}"
        else:
            header += f"  {ds+' CE':>15} {'PPL':>8}"
    print(header)
    print("-" * len(header))

    for model_name in results:
        row = f"{model_name:<25}"
        for ds in ds_names:
            r = results[model_name].get(ds, {})
            if ds == "lambada":
                row += f"  {r.get('ce','?'):>10}  {r.get('ppl','?'):>7}  {r.get('accuracy',0)*100:>5.1f}"
            else:
                row += f"  {r.get('ce','?'):>15}  {r.get('ppl','?'):>7}"
        print(row)

    print("=" * 80)

    # Save
    out_path = "results/cross_dataset_eval.json"
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
