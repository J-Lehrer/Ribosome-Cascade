"""
FLOP-matched comparison: compute FLOPs per forward pass for each model.

The key question: is the ribosome winning because of a better architecture,
or just because it has more effective compute per token?

FLOPs for a transformer layer (self-attention + FFN):
  Self-attention: 2 * seq_len * hidden^2 * 4 (QKV + output projections)
                + 2 * seq_len^2 * hidden     (attention scores + weighted sum)
  FFN (SwiGLU):   2 * hidden * ff_dim * 3    (w1, w2, w3 projections)
                  * seq_len

For RibosomeTiny:
  - embed layers process seq_len=256 tokens
  - ribosome layer: scorer + boundary + cross-attention
  - upper layers process n_chunks=16 metatokens (16x shorter sequences!)
  - decoder: cross-attention back to 256 tokens

For BigBaseline:
  - 12 layers all process seq_len=256 tokens
"""
import json, math


def transformer_layer_flops(seq_len, hidden, n_heads, ff_mult=4):
    """FLOPs for one transformer block (attn + SwiGLU FFN)."""
    ff_dim = int(hidden * ff_mult * 2 / 3)  # SwiGLU correction

    # Self-attention
    qkv_proj = 2 * seq_len * hidden * 3 * hidden  # QKV
    attn_scores = 2 * seq_len * seq_len * hidden    # Q @ K^T
    attn_values = 2 * seq_len * seq_len * hidden    # scores @ V
    out_proj = 2 * seq_len * hidden * hidden         # output projection
    attn_total = qkv_proj + attn_scores + attn_values + out_proj

    # SwiGLU FFN: w1 (gate), w3 (up), w2 (down)
    ffn_total = 2 * seq_len * hidden * ff_dim * 3  # 3 projections

    return attn_total + ffn_total


def cross_attention_flops(q_len, kv_len, hidden, n_heads):
    """FLOPs for cross-attention (queries attend to keys/values)."""
    q_proj = 2 * q_len * hidden * hidden
    kv_proj = 2 * kv_len * hidden * 2 * hidden  # K and V
    scores = 2 * q_len * kv_len * hidden
    values = 2 * q_len * kv_len * hidden
    out_proj = 2 * q_len * hidden * hidden
    return q_proj + kv_proj + scores + values + out_proj


def ribosome_tiny_flops(seq_len=256, hidden=512, n_heads=8,
                         embed_layers=3, upper_layers=3, n_chunks=16):
    """Total FLOPs for one forward pass of RibosomeTiny."""
    flops = {}

    # Embedding: just lookup, negligible FLOPs
    flops["embedding"] = 0

    # Embed layers: process full seq_len
    flops["embed_layers"] = embed_layers * transformer_layer_flops(
        seq_len, hidden, n_heads)

    # Ribosome layer
    scorer = 2 * seq_len * hidden * (hidden // 2) * 2  # two linear layers
    boundary = 2 * seq_len * hidden * (hidden // 2) * 2
    chunk_cross = cross_attention_flops(n_chunks, seq_len, hidden, n_heads)
    flops["ribosome"] = scorer + boundary + chunk_cross

    # Upper layers: process n_chunks (much shorter!)
    flops["upper_layers"] = upper_layers * transformer_layer_flops(
        n_chunks, hidden, n_heads)

    # Decoder: cross-attention from tokens to chunks
    flops["decoder"] = cross_attention_flops(seq_len, n_chunks, hidden, n_heads)
    # + FFN in decoder
    ff_dim = int(hidden * 4 * 2 / 3)
    flops["decoder"] += 2 * seq_len * hidden * ff_dim * 3

    # LM head (tied weights)
    vocab = 50257
    flops["lm_head"] = 2 * seq_len * hidden * vocab

    total = sum(flops.values())
    return flops, total


def big_baseline_flops(seq_len=256, hidden=512, n_heads=8, n_layers=12):
    """Total FLOPs for one forward pass of BigBaseline."""
    flops = {}
    flops["embedding"] = 0
    flops["transformer"] = n_layers * transformer_layer_flops(
        seq_len, hidden, n_heads)
    vocab = 50257
    flops["lm_head"] = 2 * seq_len * hidden * vocab
    total = sum(flops.values())
    return flops, total


def format_flops(f):
    if f >= 1e12: return f"{f/1e12:.2f}T"
    if f >= 1e9: return f"{f/1e9:.2f}G"
    if f >= 1e6: return f"{f/1e6:.2f}M"
    return f"{f:.0f}"


def main():
    print("=" * 70)
    print("FLOP COMPARISON: RibosomeTiny vs BigBaseline")
    print("=" * 70)

    configs = [
        ("RibosomeTiny 3+3", lambda: ribosome_tiny_flops(
            embed_layers=3, upper_layers=3)),
        ("RibosomeTiny 2+4", lambda: ribosome_tiny_flops(
            embed_layers=2, upper_layers=4)),
        ("BigBaseline 12L", lambda: big_baseline_flops()),
    ]

    results = {}
    for name, fn in configs:
        breakdown, total = fn()
        results[name] = {"breakdown": breakdown, "total": total}
        print(f"\n{name}: {format_flops(total)} total")
        for component, f in breakdown.items():
            pct = f / total * 100
            print(f"  {component:20s}: {format_flops(f):>10s}  ({pct:5.1f}%)")

    # Comparison
    r33 = results["RibosomeTiny 3+3"]["total"]
    r24 = results["RibosomeTiny 2+4"]["total"]
    big = results["BigBaseline 12L"]["total"]

    print(f"\n{'=' * 70}")
    print("COMPARISON")
    print(f"{'=' * 70}")
    print(f"  BigBaseline 12L:    {format_flops(big):>10s}  (1.00x)")
    print(f"  RibosomeTiny 3+3:   {format_flops(r33):>10s}  ({r33/big:.2f}x)")
    print(f"  RibosomeTiny 2+4:   {format_flops(r24):>10s}  ({r24/big:.2f}x)")

    print(f"\n  Upper layer savings (processing {16} chunks vs {256} tokens):")
    full_layer = transformer_layer_flops(256, 512, 8)
    chunk_layer = transformer_layer_flops(16, 512, 8)
    print(f"    Full-seq layer:  {format_flops(full_layer)}")
    print(f"    Chunk layer:     {format_flops(chunk_layer)}")
    print(f"    Ratio:           {chunk_layer/full_layer:.4f}x ({(1-chunk_layer/full_layer)*100:.1f}% savings)")

    print(f"\n  Key insight: upper layers are {full_layer/chunk_layer:.0f}x cheaper per layer")
    print(f"  But ribosome + decoder add overhead. Net effect shown above.")

    # Save
    out = {name: {"total": t["total"], "breakdown": {k: int(v) for k, v in t["breakdown"].items()}}
           for name, t in results.items()}
    with open("results/flop_comparison.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to results/flop_comparison.json")


if __name__ == "__main__":
    main()
