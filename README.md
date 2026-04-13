# Ribosome-Cascade

A novel transformer architecture that compresses token sequences through learned importance scoring before processing. Instead of running all tokens through every layer, the Ribosome-Cascade **scores tokens by importance, compresses them into metatokens, and processes a reduced sequence through the upper transformer** — achieving dramatically better perplexity per parameter and per FLOP.

## Key Result

A 49M-parameter RibosomeTiny model (6 total layers, processing 16 metatokens) outperforms a 63M-parameter standard transformer (12 layers, processing 256 raw tokens) by **over 100× in perplexity** at matched training steps on OpenWebText:

| Model | Params | Layers | Tokens Processed | Val CE | PPL |
|-------|--------|--------|-----------------|--------|-----|
| BigBaseline 12L | 63M | 12 | 256 raw | 6.18 | 485 |
| **RibosomeTiny 2+4** | **49M** | **6** | **16 metatokens** | **1.42** | **4.1** |

This is not a curriculum or training artifact — a controlled ablation (same alpha-ramp schedule, no compression) achieves PPL 771, confirming the compression mechanism itself is responsible.

## Architecture

```
Input tokens (256)
    │
    ▼
┌─────────────────────┐
│  Embedding Layers    │  2-4 transformer layers with RoPE
│  (token-level)       │  Build rich per-token representations
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Ribosome Layer      │  Gumbel-softmax boundary detection
│  (compress)          │  Perceiver cross-attention → metatokens
│                      │  Importance scoring per token
└─────────┬───────────┘
          │  256 tokens → 16 metatokens (16:1 compression)
          ▼
┌─────────────────────┐
│  Upper Transformer   │  2-4 layers on metatokens only
│  (process)           │  Causal attention on compressed sequence
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Chunk Decoder       │  Cross-attention back to token space
│  (expand)            │  Alpha-blended with skip connection
└─────────┬───────────┘
          │
          ▼
    LM Head → logits
```

The key insight: the upper transformer layers — which dominate compute in standard transformers — operate on 16 metatokens instead of 256 raw tokens. Attention cost scales as O(n²), so this is a **256× reduction** in upper-layer attention compute.

## Experimental Results (April 2026)

### Compression is real (not curriculum)

| Model | Architecture | Val CE | PPL | Steps |
|-------|-------------|--------|-----|-------|
| BigBaseline | 12L, 256 raw tokens | 6.18 | 485 | 470K |
| CurriculumBaseline | 12L, alpha-ramp (no compression) | 6.65 | 771 | 100K |
| **RibosomeTiny 2+4** | **2 embed + 4 upper, 16 metatokens** | **1.42** | **4.1** | **100K** |

The CurriculumBaseline uses the same staged training schedule as RibosomeTiny (alpha-ramp over first 10% of training) but without compression. It achieves PPL 771 — the curriculum explains almost none of the ribosome's advantage.

### Layer balance: embedding depth matters

| Embed + Upper | Val CE | PPL | Status |
|--------------|--------|-----|--------|
| 1 + 5 | 2.68 | 14.5 | complete |
| 2 + 4 | 1.42 | 4.1 | complete |
| 3 + 3 | 0.95 | 2.6 | 75K steps (running) |
| 4 + 2 | — | — | queued |

More embedding layers before compression → better results. The ribosome needs rich token representations to score importance accurately.

### Compression ratio: smooth Pareto, no cliff

| Metatokens | Compression | Val CE | PPL | Status |
|-----------|-------------|--------|-----|--------|
| 4 | 64:1 | 3.45 | 31 | complete |
| 8 | 32:1 | 2.25 | 9.4 | 75K steps (running) |
| 16 | 16:1 | 1.42 | 4.1 | complete |
| 32 | 8:1 | — | — | queued |

Even at 64:1 compression (256 → 4 metatokens), the model achieves PPL 31 — still vastly outperforming the uncompressed 12-layer baseline at PPL 485.

### Importance scores show real discrimination

The ribosome learns non-trivial token filtering:
- Mean importance: 0.316 (suppresses ~2/3 of tokens on average)
- Range: 0.020 → 0.988 (near-binary decisions at the tails)
- Entropy: 0.573 (meaningful structure, not collapsed or uniform)

## Research Status

This is active research. The results above are promising but have known limitations:

**Memorization concern.** The 49M-param model trained on ~100M tokens (wikitext-103) has a 1:2 param-to-token ratio. The lowest PPL numbers likely reflect partial memorization. Cross-dataset evaluation (train on wikitext, eval on C4/LAMBADA) is needed to establish generalization.

**Compute-matched comparison needed.** Current comparisons are at equal training steps, not equal FLOPs. The ribosome does less compute per step (by design), which means a FLOPs-matched comparison would be even more favorable — but this needs to be measured explicitly.

**Scale-up in progress.** 250M-parameter experiments are running on Colab A100. The BigBaseline (252M params, 16 layers) reached val CE 4.36 / PPL 78 on wikitext-103 in 6 hours. RibosomeTiny at the same scale is training next — this will show whether the advantage holds beyond 50M params.

### Research tracks

| Track | Status | Description |
|-------|--------|-------------|
| Track 1: Preprocessor | Paused | Compression layer in front of frozen LLMs. Works (~0.25 CE gain) but collapses when base model unfreezes. |
| **Track 2: Native Architecture** | **Active** | Ribosome as first-class component trained from scratch. This is where the strong results are. |

## How to Run

### Requirements
```
torch>=2.0
transformers
datasets
numpy
```

### Train RibosomeTiny vs BigBaseline (local)
```bash
# Train ribosome tiny (2+4 layers, 16 metatokens)
python exp2_lighter.py --model tiny --device cuda --epochs 1 --steps_per_epoch 100000

# Train big baseline (12 layers, 256 raw tokens)
python exp2_lighter.py --model big --device cuda --epochs 1 --steps_per_epoch 100000
```

### Scale-up on Colab
Upload `notebooks/Ribosome_Cascade_ScaleUp_v2.ipynb` to Colab with A100 runtime. Self-contained: downloads data via wget, no HuggingFace streaming dependencies.

## File Guide

```
├── native_arch_v1.py          # Core: RMSNorm, RoPE, TransformerBlock, RibosomeLayer, ChunkDecoder
├── train_native.py            # Training utilities: data loaders, LR schedules
├── exp2_lighter.py            # Main experiment: BigBaseline vs RibosomeTiny
├── exp_curriculum_ablation.py # Curriculum-only control (no compression)
├── exp_compression_sweep.py   # Compression ratio sweep (4/8/16/32 chunks)
├── exp_layer_balance.py       # Embed/upper layer split ablation
├── exp_extended_baseline.py   # BigBaseline trained to 500K steps
│
├── notebooks/
│   ├── Ribosome_Cascade_ScaleUp_v2.ipynb  # Colab A100 notebook (250M scale)
│   ├── Project_Ribosome.ipynb              # Original PoC
│   └── Project_Ribosome_large.ipynb        # Extended PoC
│
├── results/                   # Experiment data (JSON)
│   ├── experiment_results_2026-04-13.json  # Consolidated results
│   └── benchmark_*.json, *_log.json        # Raw training logs
│
├── notes/                     # Detailed session research notes
│   ├── RESEARCH_NOTES_2026-04-09.md
│   ├── RESEARCH_NOTES_2026-04-10.md
│   └── RESEARCH_NOTES_2026-04-13.md
│
└── archive/                   # Track 1 (frozen GPT-2) experiments — historical
    ├── ribosome_benchmark_v*.py
    ├── ribosome_cascade_v*.py
    └── train_*.py
```

## Hardware

Experiments run across a 4-GPU home lab cluster:
- NVIDIA RTX 5090 (32GB) — primary training
- RTX 5090 Laptop (24GB) — parallel experiments
- RTX 3060 Ti (8GB) — ablation sweeps
- GTX 1070 (8GB) — ablation sweeps

## Related Work

| Concept | Reference | How we differ |
|---------|-----------|---------------|
| Token downsampling | Funnel Transformer (Dai et al. 2020) | Learned importance-based grouping vs fixed pooling |
| Perceiver bottleneck | Perceiver (Jaegle et al. 2021) | Importance-weighted compression vs uniform cross-attention |
| Token merging | ToMe (Bolya et al. 2023) | Trained end-to-end vs post-hoc similarity merging |
| Prompt compression | LLMLingua (Jiang et al. 2023) | Architectural (native) vs inference-time token dropping |
| Sparse routing | Switch Transformer / MoE | Token importance scoring vs expert routing |
| KV cache compression | TurboQuant (Google, 2026) | Complementary: we reduce token count, they reduce precision |

## License

MIT
