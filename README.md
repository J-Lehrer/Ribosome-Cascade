# Ribosome-Cascade

A novel architecture for priority-driven token processing in large language models. Instead of treating all tokens equally, the Ribosome-Cascade scores tokens by semantic importance, groups them into Metatokens around importance peaks, and processes the heaviest concepts first to establish semantic anchors.

## Architecture

The pipeline has three stages:

### Stage I: The Ribosome (Scorer)
A lightweight MLP (`Linear → GELU → Linear → Sigmoid`) attached to a base LLM's hidden states. Outputs an importance score ∈ [0, 1] per token.

### Stage II: Metatoken Assembly (Gravity Engine)
A deterministic algorithm that treats importance scores as topography:
- **Peaks** = local maxima in the score landscape
- **Valleys** slide downhill to the nearest peak (gravity)
- **Tie-breaker**: equidistant tokens go to the *heavier* peak (strong-attractor semantics)
- Each group is stamped with a `temporal_tag` to preserve original order

### Stage III: The Cascade (Priority Decoder)
- Mean-pools hidden states per metatoken into a single concept vector
- Sorts by weight — heaviest concept processed first (the anchor)
- Re-sorts by temporal tag for final articulation

## Research Status

This project is active research exploring two tracks:

### Track 1: Preprocessor (Current Priority)
Build the ribosome as a **standalone compression layer** that sits in front of any existing LLM. Reduces token count while preserving semantic content. No base model retraining required.

**Target**: 3–5x compression on typical text, measured against downstream task accuracy at various compression ratios.

### Track 2: Native Architecture (Follow-up)
Integrate the ribosome directly into the transformer stack as a first-class architectural component. Train from scratch so the model's attention mechanism *is* the ribosome.

## Experimental Results

All experiments use frozen GPT-2 (124M params) on wikitext-2, measuring cross-entropy (CE) loss vs. a matched uniform baseline (same architecture, no importance scoring).

### v2: Fair Baseline (frozen GPT-2)
Soft-gating: `h_weighted = h × σ(scores)`, separate lm_heads for ribosome and uniform.

| Seq Length | Ribosome CE | Uniform CE | Delta |
|------------|-------------|------------|-------|
| 32 | 6.297 | 6.601 | **−0.304** |
| 64 | 6.210 | 6.457 | **−0.247** |
| 128 | 6.063 | 6.286 | **−0.224** |
| 256 | 5.579 | 5.805 | **−0.226** |
| 512 | 5.126 | 5.382 | **−0.255** |

**Finding**: Consistent ~0.25 CE improvement. But score entropy ratio is 0.95–0.99 (near-uniform). The ribosome is doing mild contrast enhancement, not producing real peaks and valleys.

### v3: Layer Unfreezing
Unfreezing GPT-2's top layers dramatically improves absolute CE (~1 point) but the ribosome's *marginal* advantage collapses from −0.33 to −0.04. The base model absorbs the ribosome's function into its own representations.

| Condition | Delta at len=512 |
|-----------|-----------------|
| frozen | **−0.326** |
| top-2 unfrozen | −0.049 |
| top-4 unfrozen | −0.040 |
| top-6 unfrozen | −0.034 |

### v4–v5: Differentiable Cascade Attempts
Multiple approaches to make the cascade end-to-end differentiable:

| Version | Approach | Delta (512) | Issue |
|---------|----------|-------------|-------|
| v4 | Perceiver bottleneck (8 chunks) | +0.37 | Info destroyed in compression |
| v4.1 | + importance-weighted keys | +0.37 | Same bottleneck problem |
| v5 | Importance-modulated sparse attention | +0.04 | Worked at len=32 (−0.21), failed at longer |
| v5.1 | + normalized distance scaling | +0.58 | Over-aggressive sparsity |

**Key insight**: On frozen GPT-2, the hidden states already encode everything needed. Any manipulation that deviates from near-identity hurts reconstruction. The v2/v3 "win" came from additional learnable parameters, not from importance scoring *per se*.

**Implication for Track 1**: Stop optimizing for LM reconstruction loss. Instead, target compression-at-acceptable-quality — a fundamentally different objective (distillation loss, downstream task accuracy at compression ratios).

## Key Differentiators vs Existing Work

| Concept | This Work | Prior Art |
|---------|-----------|-----------|
| Token grouping | Gravity-based (respects semantic boundaries) | Token Merging (ToMe): greedy bipartite by similarity |
| Compression | Importance-aware grouping | LLMLingua: simple token dropping |
| Architecture | Can be applied to any model without retraining | Funnel Transformer: requires architectural change |
| Priority | Heaviest concepts processed first (anchor semantics) | Standard: all tokens equal |

## Files

| File | Description |
|------|-------------|
| `Project_Ribosome.ipynb` | Original PoC notebook (Colab) |
| `Project_Ribosome_large.ipynb` | Extended notebook |
| `ribosome_benchmark.py` | v1 benchmark (initial comparison) |
| `ribosome_benchmark_v2.py` | v2 benchmark (fair baseline) |
| `ribosome_benchmark_v3.py` | v3 benchmark (layer unfreezing) |
| `ribosome_cascade_v4.py` | v4/4.1 (differentiable cascade) |
| `ribosome_cascade_v5.py` | v5/5.1 (importance-modulated attention) |
| `RESEARCH_NOTES_2026-04-09.md` | Detailed session notes |
| `benchmark_results_*.json` | Raw experimental data |

## Hardware

- Development: NVIDIA RTX 5090 (32GB), RTX 5090 Laptop (25GB)
- Also tested on: RTX 3060 Ti, GTX 1070
- Python 3.12/3.13, PyTorch 2.x, transformers 5.5.0

## Literature

| Concept | Reference |
|---------|-----------|
| Progressive token downsampling | Funnel Transformer (Dai et al. 2020) |
| Soft token merging | Charformer / GBST (Tay et al. 2022) |
| Non-autoregressive decoding | Mask-Predict (Ghazvininejad et al. 2019) |
| Sparse routing | Switch Transformer / MoE |
| Token merging for inference | ToMe (Bolya et al. 2023) |
| Prompt compression | LLMLingua (Jiang et al. 2023) |
| Perceiver bottleneck | Perceiver (Jaegle et al. 2021) |

## License

MIT
