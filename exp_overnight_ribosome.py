"""
Overnight Experiment C: Extended RibosomeTiny 3+3, 100K steps.
Runs on side (3060 Ti, 8GB). Tests if ribosome keeps improving.
"""
import torch, sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp2_lighter import RibosomeTiny, train_model
from transformers import AutoTokenizer

def main():
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name()}")
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    cap = min(total_vram * 0.85, 7.0)
    torch.cuda.set_per_process_memory_fraction(cap / total_vram)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    V = len(tokenizer)
    args = argparse.Namespace(
        device="cuda", max_length=256, batch_size=4, grad_accum=4,
        epochs=1, steps_per_epoch=100000,
        max_lr=3e-4, min_lr=3e-5,
        log_every=500, eval_every=5000,
        dataset="openwebtext", streaming=True,
        output_dir=r"C:\ribosome-cascade\overnight_extended",
    )
    model = RibosomeTiny(vocab_size=V, hidden_size=512, n_heads=8,
                         embed_layers=3, upper_layers=3,
                         max_seq_len=256, n_chunks=16).to(device)
    print(f"RibosomeTiny 3+3: {model.count_params():,} params, 100K steps")
    train_model(model, "ribosome_3p3_100K", tokenizer, device, args,
                is_ribosome=True)
    print("Done.")

if __name__ == "__main__":
    main()
