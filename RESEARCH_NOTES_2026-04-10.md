# Ribosome-Cascade: Research Session Notes — April 10, 2026
**Date:** April 10, 2026
**Session Focus:** v4/v5 differentiable cascade experiments, Track 1 conclusion

---

## 1. Problem Statement

The v2/v3 results showed ~0.25 CE improvement with soft-gating (`h × σ(s)`), but:
- Scores were near-uniform (0.94+), meaning the ribosome wasn't really scoring importance
- The hard cascade (metatoken assembly, priority sort) was never trained — only used at inference
- Three bugs identified: subword indexing, tie-breaker semantics, train/inference mismatch

Today's goal: fix all three bugs and make the cascade differentiable.

---

## 2. Experiments Run

### v4: Perceiver-Style Bottleneck (run on main RTX 5090)
**Architecture:** Compress seq_len tokens → 8 learnable chunk queries via cross-attention → priority-ordered causal cross-attention → expand back to seq_len via decode attention.
**Bug found:** `importance_bias` computed but never used — dead code. Chunk queries attended freely regardless of importance.
**Result:** Delta = +0.37 (WORSE than uniform). The 8-chunk bottleneck destroyed positional/contextual information. 

### v4.1: Fixed Importance Injection (run on olares RTX 5090 Laptop)
**Fix:** Keys = `hidden_states × importance_scores`, values = original hidden states. Added target sparsity ratio to bimodal penalty.
**Result:** Delta = +0.37 still. The bottleneck itself is the problem, not the importance injection.

### v5: Importance-Modulated Sparse Attention (run on olares)
**Architecture:** No bottleneck. All tokens preserved. Added a self-attention layer where:
- Each token's attention span ∝ its importance score (high = broad reach)
- Each token's visibility ∝ its importance score (high = attracts attention)
- Soft mask: `M[i,j] = σ(reach_i + attract_j - |i-j|/temperature)`
- Reach and attract are learned functions of importance scores
- Temperature is learned per-head

**Result:** Mixed. Delta = −0.21 at len=32 (close to v2's −0.25), but positive at len=128+. The distance scaling didn't generalize across sequence lengths.

**Bimodal penalty working:** Sparsity went from 0.2% (v4) to 97-99%. Real peaks and valleys for the first time.

### v5.1: Normalized Distance (run on olares)
**Fix:** `dist / S` instead of raw `dist` so reach/attract values transfer across lengths.
**Result:** Delta = +0.58 (worst yet). The normalization made the model too aggressive — nearly all tokens scored near zero, information-starved.

---

## 3. Key Finding: The Frozen-Base Limitation

Pattern across all experiments:

| Version | Mechanism | Delta (512) |
|---------|-----------|-------------|
| v2/v3 | h × σ(s), scores ~0.94 | **−0.25** |
| v4 | Perceiver bottleneck | +0.37 |
| v4.1 | Fixed bottleneck | +0.37 |
| v5 | Importance sparse attn | +0.04 |
| v5.1 | + normalized distance | +0.58 |

**Every time we push the ribosome to actually do importance scoring, performance degrades.** The only winning version had near-uniform scores (essentially identity).

**Interpretation:** On frozen GPT-2, the hidden states already encode everything needed. Any non-trivial manipulation hurts reconstruction. The v2/v3 improvement was from extra learnable parameters, not from importance scoring.

---

## 4. Strategic Decision

**Track 1 pivot:** Stop optimizing for LM reconstruction loss. The preprocessor's value is compression-at-acceptable-quality, not beating uniform on perplexity. New objective: distillation loss + downstream task accuracy at compression ratios 2x, 3x, 4x.

**Track 2 greenlit:** Native architecture where the ribosome is built into the model from scratch. The v3 unfreezing experiment already showed that when the base model can adapt, it absorbs importance scoring into its own representations. A native design avoids the frozen-base problem entirely.

---

## 5. Infrastructure Notes

- **VRAM safety cap added:** `torch.cuda.set_per_process_memory_fraction()` — prevents crashing host machines
- **Multi-machine workflow established:** main (dev, no training), olares (primary GPU), side (RTX 3060 Ti), frank (GTX 1070)
- **SSH pipeline working:** SCP scripts → nohup run → poll for results

---

## 6. Files from This Session

| File | Location | Description |
|------|----------|-------------|
| v4 script | E:\Ribosome-Cascade\ribosome_cascade_v4.py | Perceiver bottleneck + importance attn |
| v5 script | E:\Ribosome-Cascade\ribosome_cascade_v5.py | Importance-modulated sparse attention |
| v4 results | benchmark_results_v4.json, v4.1.json | Bottleneck experiment data |
| v5 results | benchmark_results_v5.json, v5.1.json | Sparse attention experiment data |
