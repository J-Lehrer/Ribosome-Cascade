"""
Overnight Experiment A: Chunk sweep on corrected loss.
Maps PPL vs LAMBADA Pareto frontier at n_chunks = {4, 8, 32, 64}.
Sequential runs on olares (5090 Laptop).
"""
import torch
import torch.nn.functional as F
import numpy as np
import math
import sys, os, json, time, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp2_lighter import RibosomeTiny, train_model
from train_native import get_wikitext_loader
from eval_cross_dataset import get_c4_loader, get_lambada_loader
from transformers import AutoTokenizer


@torch.no_grad()
def quick_lambada(model, loader, device):
    model.eval()
    correct = total = 0
    ces = []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        lab = batch["labels"].to(device)
        _, logits, _ = model(ids)
        flat_logits = logits.view(-1, logits.size(-1))
        flat_labels = lab.view(-1)
        mask = flat_labels != -100
        if mask.sum() > 0:
            ces.append(F.cross_entropy(flat_logits[mask], flat_labels[mask]).item())
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

    # Load LAMBADA once
    print("Loading LAMBADA for post-training eval...")
    lambada_loader = get_lambada_loader(tokenizer, 256, 8)

    chunk_counts = [4, 8, 32, 64]
    results = []

    for nc in chunk_counts:
        print(f"\n{'#'*60}")
        print(f"# CHUNK SWEEP: n_chunks={nc}")
        print(f"{'#'*60}")

        model = RibosomeTiny(
            vocab_size=V, hidden_size=512, n_heads=8,
            embed_layers=3, upper_layers=3,
            max_seq_len=256, n_chunks=nc
        ).to(device)
        print(f"Params: {model.count_params():,}")

        args = argparse.Namespace(
            device="cuda", max_length=256, batch_size=8, grad_accum=4,
            epochs=1, steps_per_epoch=6250,  # 25K effective steps
            max_lr=3e-4, min_lr=3e-5,
            log_every=500, eval_every=2500,
            dataset="openwebtext", streaming=True,
            output_dir=f"./overnight_chunks/chunks_{nc}",
        )
        val_ce = train_model(model, f"ribo_3p3_c{nc}", tokenizer, device, args,
                             is_ribosome=True)

        # LAMBADA eval
        lam_ce, lam_ppl, lam_acc, lam_c, lam_t = quick_lambada(
            model, lambada_loader, device)

        r = {
            "n_chunks": nc, "params": model.count_params(),
            "val_ce": round(float(val_ce), 4), "val_ppl": round(math.exp(float(val_ce)), 2),
            "lambada_ce": round(lam_ce, 4), "lambada_ppl": round(lam_ppl, 2),
            "lambada_acc": round(lam_acc, 4), "lambada_correct": lam_c,
        }
        results.append(r)
        print(f"\n  chunks={nc}: val_PPL={r['val_ppl']}  LAMBADA_PPL={r['lambada_ppl']}  "
              f"LAMBADA_acc={lam_acc*100:.1f}%")

        del model
        torch.cuda.empty_cache()

        # Save progress
        with open("overnight_chunks/sweep_results.json", "w") as f:
            json.dump(results, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print("CHUNK SWEEP RESULTS")
    print(f"{'='*60}")
    print(f"{'Chunks':>6} {'Val PPL':>8} {'LAM PPL':>10} {'LAM Acc':>8}")
    print("-" * 40)
    for r in results:
        print(f"{r['n_chunks']:>6} {r['val_ppl']:>8.1f} {r['lambada_ppl']:>10.1f} "
              f"{r['lambada_acc']*100:>7.1f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
