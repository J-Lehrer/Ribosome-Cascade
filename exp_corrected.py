"""
Corrected training run: RibosomeTiny 3+3 with fixed loss (no double-shift).
Then BigBaseline 12L for comparison. Sequential on olares.
"""
import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp2_lighter import RibosomeTiny, BigBaseline, train_model
from transformers import AutoTokenizer
import argparse

def main():
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name()}")
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    cap = min(total_vram * 0.85, 22.0)
    torch.cuda.set_per_process_memory_fraction(cap / total_vram)
    print(f"VRAM cap: {cap:.0f}GB")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    V = len(tokenizer)

    args = argparse.Namespace(
        device="cuda", max_length=256, batch_size=8, grad_accum=4,
        epochs=1, steps_per_epoch=25000,  # 100K effective steps
        max_lr=3e-4, min_lr=3e-5,
        log_every=500, eval_every=2500,
        dataset="openwebtext", streaming=True,
        output_dir="./corrected_v1",
    )

    # --- RibosomeTiny 3+3 ---
    print("\n" + "="*60)
    print("CORRECTED RibosomeTiny 3+3 (fixed next-token loss)")
    print("="*60)
    model = RibosomeTiny(
        vocab_size=V, hidden_size=512, n_heads=8,
        embed_layers=3, upper_layers=3,
        max_seq_len=256, n_chunks=16
    ).to(device)
    train_model(model, "ribosome_3p3", tokenizer, device, args, is_ribosome=True)
    del model
    torch.cuda.empty_cache()

    # --- BigBaseline 12L ---
    print("\n" + "="*60)
    print("CORRECTED BigBaseline 12L (fixed next-token loss)")
    print("="*60)
    model = BigBaseline(
        vocab_size=V, hidden_size=512, n_heads=8,
        n_layers=12, max_seq_len=256
    ).to(device)
    train_model(model, "baseline_12L", tokenizer, device, args, is_ribosome=False)
    del model
    torch.cuda.empty_cache()

    print("\nBoth done. Compare results in ./corrected_v1/")

if __name__ == "__main__":
    main()
