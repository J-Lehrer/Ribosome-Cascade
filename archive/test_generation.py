"""Generation quality test for the 50K OWT ribosome model."""
import torch
import sys, os
os.environ['HF_HOME'] = '/var/hf_cache'
sys.path.insert(0, '/var/ribosome-cascade')

from native_arch_v1 import RibosomeCascadeNative
from transformers import AutoTokenizer

device = torch.device('cuda')
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

model = RibosomeCascadeNative(
    vocab_size=len(tokenizer), hidden_size=768, n_heads=12,
    lower_layers=4, upper_layers=4, cascade_layers=2,
    max_seq_len=256, max_chunks=8
).to(device)

ckpt = torch.load('/var/ribosome-cascade/ribosome_owt_50k/best.pt',
                   map_location=device, weights_only=False)
model.load_state_dict(ckpt['model'])
model.eval()
model.ribosome_alpha = 1.0
print(f"Loaded step={ckpt['step']}, val={ckpt['val_loss']:.4f}")

prompts = [
    "The history of artificial intelligence",
    "In a recent study, researchers found that",
    "The president announced today that",
    "Python is a programming language that",
    "The capital of France is",
    "Once upon a time, in a small village",
]

def generate(prompt, max_new=80, temperature=0.8, top_k=50):
    ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        for _ in range(max_new):
            if ids.shape[1] >= 256:
                break
            loss, logits, _ = model(ids)
            next_logits = logits[0, -1] / temperature
            # top-k filtering
            topk_vals, topk_idx = torch.topk(next_logits, top_k)
            probs = torch.zeros_like(next_logits).fill_(float('-inf'))
            probs.scatter_(0, topk_idx, topk_vals)
            probs = torch.softmax(probs, dim=0)
            next_token = torch.multinomial(probs, 1)
            ids = torch.cat([ids, next_token.unsqueeze(0)], dim=1)
    return tokenizer.decode(ids[0])

def generate_greedy(prompt, max_new=80):
    ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        for _ in range(max_new):
            if ids.shape[1] >= 256:
                break
            loss, logits, _ = model(ids)
            next_token = logits[0, -1].argmax()
            ids = torch.cat([ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
    return tokenizer.decode(ids[0])

# Check importance scores on a real sentence
print("\n=== IMPORTANCE SCORES ON SAMPLE ===")
sample = "The quick brown fox jumps over the lazy dog"
sample_ids = tokenizer.encode(sample, return_tensors='pt').to(device)
with torch.no_grad():
    _, _, importance = model(sample_ids)
tokens = tokenizer.convert_ids_to_tokens(sample_ids[0])
scores = importance[0].cpu().numpy()
for tok, sc in zip(tokens, scores):
    bar = '#' * int(sc * 40)
    print(f"  {tok:>12s}  {sc:.4f}  {bar}")

print("\n=== GREEDY GENERATION ===")
for p in prompts:
    print(f"\nPrompt: {p}")
    print(f"Output: {generate_greedy(p)}")

print("\n=== SAMPLED GENERATION (temp=0.8, top_k=50) ===")
for p in prompts[:3]:
    print(f"\nPrompt: {p}")
    print(f"Output: {generate(p)}")
