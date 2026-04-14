"""
LAMBADA diagnostic: trace through individual examples to see if
the eval is broken or the model genuinely can't predict last words.

Tests:
1. Is the label alignment correct?
2. What is the model actually predicting?
3. Does GPT-2 work correctly as a control?
"""
import torch
import torch.nn.functional as F
import math
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp2_lighter import RibosomeTiny
from eval_cross_dataset import get_lambada_loader
from transformers import AutoTokenizer, AutoModelForCausalLM


def diagnose(model_name, model, loader, tokenizer, device, is_hf=False, n=10):
    model.eval()
    count = 0
    correct = 0
    print(f"\n{'='*70}")
    print(f"LAMBADA DIAGNOSIS: {model_name}")
    print(f"{'='*70}")

    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            lab = batch["labels"].to(device)

            if is_hf:
                logits = model(ids).logits
            else:
                _, logits, _ = model(ids)

            for i in range(ids.size(0)):
                if count >= n:
                    break
                valid = (lab[i] != -100).nonzero(as_tuple=True)[0]
                if len(valid) == 0:
                    continue

                # Context: last few input tokens before the target
                first_valid = valid[0].item()
                last_valid = valid[-1].item()
                n_valid = len(valid)

                # What does the model predict at the last valid position?
                pred_token = logits[i, last_valid].argmax().item()
                target_token = lab[i, last_valid].item()

                # Also check: what does it predict at last_valid-1?
                # (in case there's an off-by-one)
                if last_valid > 0:
                    pred_prev = logits[i, last_valid - 1].argmax().item()
                else:
                    pred_prev = -1

                # Decode
                pred_word = tokenizer.decode([pred_token])
                target_word = tokenizer.decode([target_token])
                prev_pred_word = tokenizer.decode([pred_prev]) if pred_prev >= 0 else "N/A"

                # Context snippet (last 8 input tokens before prediction)
                ctx_start = max(first_valid, last_valid - 8)
                ctx_tokens = ids[i, ctx_start:last_valid + 1].tolist()
                ctx_text = tokenizer.decode(ctx_tokens)

                # Top 5 predictions
                top5 = logits[i, last_valid].topk(5)
                top5_words = [tokenizer.decode([t.item()]) for t in top5.indices]
                top5_probs = F.softmax(logits[i, last_valid], dim=-1)
                top5_p = [top5_probs[t.item()].item() for t in top5.indices]

                is_correct = pred_token == target_token
                if is_correct:
                    correct += 1
                count += 1

                mark = "OK" if is_correct else "MISS"
                print(f"\n  [{count}] {mark}")
                print(f"    Context: ...{ctx_text}")
                print(f"    Target:  '{target_word}' (id={target_token})")
                print(f"    Pred:    '{pred_word}' (id={pred_token})")
                print(f"    Pred@-1: '{prev_pred_word}' (id={pred_prev})")
                print(f"    Top5:    {list(zip(top5_words, [f'{p:.3f}' for p in top5_p]))}")
                print(f"    Valid tokens: {n_valid}/{ids.size(1)}, "
                      f"padding: {first_valid} positions")

            if count >= n:
                break

    print(f"\n  Summary: {correct}/{count} correct ({correct/max(count,1)*100:.1f}%)")
    return correct, count


def main():
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name()}")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    V = len(tokenizer)

    print("Loading LAMBADA...")
    loader = get_lambada_loader(tokenizer, 256, 1)  # batch=1 for clarity

    # GPT-2 Small (control)
    print("Loading GPT-2 Small...")
    gpt2 = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    diagnose("GPT-2 Small (124M)", gpt2, loader, tokenizer, device, is_hf=True, n=10)
    del gpt2; torch.cuda.empty_cache()

    # RibosomeTiny 3+3
    p = "corrected_v1/ribosome_3p3/best.pt"
    if os.path.exists(p):
        m = RibosomeTiny(V, 512, 8, embed_layers=3, upper_layers=3,
                         max_seq_len=256, n_chunks=16).to(device)
        m.load_state_dict(torch.load(p, map_location=device, weights_only=False)["model"])
        diagnose("RibosomeTiny 3+3 (49M, 16 chunks)", m, loader, tokenizer, device, n=10)
        del m; torch.cuda.empty_cache()

    # Also test the 64-chunk model if available
    p64 = "overnight_chunks/chunks_64/ribo_3p3_c64/best.pt"
    if os.path.exists(p64):
        m = RibosomeTiny(V, 512, 8, embed_layers=3, upper_layers=3,
                         max_seq_len=256, n_chunks=64).to(device)
        m.load_state_dict(torch.load(p64, map_location=device, weights_only=False)["model"])
        diagnose("RibosomeTiny 3+3 (49M, 64 chunks)", m, loader, tokenizer, device, n=10)
        del m; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
