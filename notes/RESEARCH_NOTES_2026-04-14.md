# Research Notes — April 14, 2026 (Session 4: Morning)

## Overnight Results

### Chunk Sweep (olares, completed)
Mapped PPL vs LAMBADA as f(n_chunks) at 25K steps, 3+3 split:

| Chunks | Compression | Val PPL | LAMBADA PPL | LAMBADA Acc |
|--------|------------|---------|-------------|-------------|
| 4 | 64:1 | 322.1 | 2,344 | 0.3% |
| 8 | 32:1 | 95.3 | 9,915 | 0.1% |
| 16 | 16:1 | 2.3 | 21,912 | 0.0% |
| 32 | 8:1 | 2.3 | 6.2e13 | 0.1% |
| 64 | 4:1 | 1.4 | 2.9 | 0.2% |

Key finding: LAMBADA accuracy is ~0% at ALL chunk counts. This is not a
compression ratio problem — it's architectural.

### Extended Baseline (frank, still running)
BigBaseline 12L at step 45K: val CE 5.82, PPL 336. Still training to 100K.

## LAMBADA Diagnostic

Ran token-level trace on 10 LAMBADA examples comparing GPT-2 vs RibosomeTiny.

GPT-2 Small: 8/10 correct. High confidence on correct predictions (>99% on
name completions like "hel" -> "en" for Helen).

RibosomeTiny (16 chunks): 0/10 correct. Probability distributions are
essentially uniform noise. The model has no idea what token to predict.

RibosomeTiny (64 chunks): 0/10 correct. Better calibrated probabilities
but still wrong predictions.

### Root Cause Analysis

Tested two hypotheses:

**Hypothesis 1: Padding contamination**
LAMBADA examples are padded with EOS tokens (150-190 of 255 positions).
The ribosome treats padding as real tokens, wasting chunks on nothing.
- Fix attempted: padding_mask that zeros importance/boundaries for -100 labels
- Result: No improvement. 16-chunk went from 0/5153 to 0/5153.
- Conclusion: Padding is not the primary issue.

**Hypothesis 2: Missing positional metadata**
The upper layers have no positional encoding — chunks are processed as an
unordered set. Chunks don't know WHERE in the sequence they came from.
- Fix attempted: sinusoidal position encoding from weighted mean position
- Result: wikitext PPL degraded 2.3 -> 12.4, LAMBADA 0/5153 -> 7/5153
- Conclusion: Additive encoding interfered with learned representations.
  Not the right approach.

### True Root Cause: Reconstruction Bottleneck

The problem is the ChunkDecoder. It's a single cross-attention + FFN that
tries to reconstruct 256 token-level predictions from 16 chunk vectors.
This is simply not enough capacity. The chunks capture semantic gist but
lose fine-grained token identity (subword completions, exact word choices).

Evidence: The model produces near-uniform distributions over the vocabulary
for LAMBADA targets — it genuinely has no signal about which token to predict,
not even wrong-but-confident predictions.

## ReverseRibosome: Hourglass Architecture (running)

Jeff's insight: "what if we had something on the way out? a reverse ribosome?"

Replace the thin ChunkDecoder with a ReverseRibosome that has real capacity:
1. Cross-attention: each token attends to processed chunks (injects context)
2. Residual add with original token_states (preserves token identity)
3. 2 causal self-attention layers with RoPE at full 256-token resolution
   (recovers sequential dependencies with proper positional encoding)

Architecture becomes hourglass:
```
Embed (3L, RoPE, 256 tokens)        <- full resolution, token identity
    | compress
Upper (3L, 16 chunks)               <- cheap reasoning on concepts
    | expand + refine
ReverseRibosome (2L, RoPE, 256 tok) <- reconstruct fine-grained predictions
    |
LM head
```

Param count: ~55M (vs 49M without reverse layers, vs 63M baseline)
FLOP budget: still favorable vs baseline — reverse layers are output-only

Currently training on olares. ETA ~1-2 hours.

## Currently Running

| Machine | GPU | Experiment | Status |
|---------|-----|-----------|--------|
| olares | 5090 Laptop | ReverseRibosome test (25K steps) | Training |
| frank | GTX 1070 | Extended BigBaseline 100K steps | ~step 50K |
| side | RTX 3060 Ti | Idle (needs manual launch) | - |
| Colab | A100 | ScaleUp v3 corrected notebook | Running |

## Next Steps

1. Collect ReverseRibosome results — does LAMBADA accuracy improve?
2. Collect extended baseline 100K results from frank
3. If ReverseRibosome helps: run extended 100K version for fair comparison
4. If ReverseRibosome doesn't help: characterize as architectural tradeoff
5. Collect Colab A100 scale-up results
6. Cross-dataset eval on all corrected models
7. Update RESEARCH_SUMMARY.md with corrected results + LAMBADA analysis
8. Re-draft LinkedIn post with honest numbers
