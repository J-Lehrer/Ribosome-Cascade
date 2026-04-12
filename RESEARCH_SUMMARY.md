# Ribosome-Cascade: Research Summary
**Date:** April 10, 2026  
**Author:** Jeff Lehrer  
**Status:** Active research — Track 2 experiments running

---

## 1. The Idea

Standard transformers process all tokens equally. The Ribosome-Cascade architecture introduces a learned importance hierarchy: tokens are scored, grouped into semantic chunks ("metatokens"), and processed in priority order — heaviest concepts first. The heaviest chunk becomes the semantic anchor; lighter chunks are interpreted through the anchor's lens.

The biological metaphor is deliberate: like a ribosome reading mRNA and assembling proteins by codon priority, the architecture reads token sequences and assembles meaning by semantic weight.

## 2. Architecture Evolution

### 2.1 Original PoC (Colab, GPT-2)
Three-stage pipeline bolted onto frozen GPT-2 (124M params):

**Stage I — Ribosome Scorer:** MLP (Linear→GELU→Linear→Sigmoid) on hidden states. Outputs importance ∈ [0,1] per token.

**Stage II — Metatoken Assembly:** Deterministic gravity algorithm. Importance scores treated as topography — peaks are semantic anchors, valleys slide to nearest peak. Produces variable-length token groups stamped with temporal tags.

**Stage III — Cascade Decoder:** Mean-pools hidden states per group into concept vectors. Sorts by weight (heaviest first). Cosine similarity against vocab embeddings for articulation.

**Training:** Soft-gating approximation `h_weighted = h × σ(scores)` with cross-entropy + sparsity penalty. The actual cascade was never trained — only used at inference.

### 2.2 Track 1: Preprocessor Experiments (v2–v5)

Goal: make the ribosome work as a bolt-on compression layer for existing LLMs.

| Version | Mechanism | Result (Δ CE vs uniform, len=512) |
|---------|-----------|-----------------------------------|
| v2 | Soft gating, fair baseline | **−0.25** (ribosome wins) |
| v3 | + layer unfreezing | −0.04 (advantage collapses) |
| v4 | Perceiver bottleneck | +0.37 (ribosome loses) |
| v4.1 | + importance-weighted keys | +0.37 (same) |
| v5 | Importance-modulated sparse attention | +0.04 (mixed) |
| v5.1 | + normalized distance | +0.58 (worse) |

**Key finding:** On frozen GPT-2, every attempt to make the ribosome actively score importance degraded performance. The only winning version (v2) had near-uniform scores (~0.94) — essentially an identity function with extra parameters. The improvement came from added capacity, not importance scoring.

**Conclusion:** The preprocessor approach on frozen models is a dead end for reconstruction loss. Pivoted to compression-at-acceptable-quality as the Track 1 objective (different loss function, future work).

### 2.3 Track 2: Native Architecture (Current)

Goal: build a transformer from scratch where the ribosome is a first-class component.

**Architecture (119M params):**
- Token embedding (weight-tied with LM head)
- Lower transformer (4 layers, RoPE, SwiGLU, RMSNorm)
- **Ribosome layer:** Gumbel-softmax boundary prediction, Perceiver-style chunk encoding, importance-weighted assignment
- **Cascade processor** (2 layers): causal attention sorted by chunk weight — heaviest chunk processed first
- Upper transformer (4 layers on metatokens)
- Chunk decoder: cross-attention expanding chunks back to tokens
- Alpha-ramp bypass: model trains as 4-layer LM during warmup, gradually activates full cascade

**Training innovations:**
- Gumbel-softmax for differentiable discrete boundary placement (no two-phase training needed)
- Alpha ramp 0→1 over first 10% of steps (progressive complexity curriculum)
- Gumbel temperature annealing 1.0→0.1 (soft boundaries → hard boundaries)
- VRAM safety cap via `torch.cuda.set_per_process_memory_fraction()`

## 3. Results

### 3.1 Wikitext-2 (memorizable — proof of concept only)

| Model | Epochs | Val CE | Perplexity | Hardware |
|-------|--------|--------|------------|----------|
| Ribosome native | 3 | **3.76** | 43 | GPU-C (3060 Ti) |
| Ablation (10L standard) | 3 | **6.43** | 620 | GPU-C (3060 Ti) |
| Ribosome native | 10 | **0.69** | 2.0 | GPU-B (5090 Laptop) |

The ribosome architecture outperforms a matched 10-layer standard transformer by **14× in perplexity** at 3 epochs on identical hardware with comparable parameter counts (119M vs 109M).

### 3.2 Open Question: Curriculum vs Compression

The ribosome's alpha-ramp bypass creates a built-in curriculum: the model pre-trains as a shallow 4-layer LM, then gradually adds the compression pipeline. The ablation trains all 10 layers from scratch — a harder optimization problem.

**Three-way experiment currently running on wikitext-103 (103M tokens, non-memorizable):**

| Machine | GPU | Experiment | Controls for |
|---------|-----|-----------|-------------|
| GPU-B | RTX 5090 Laptop | Ribosome native | — |
| GPU-C | RTX 3060 Ti | Standard ablation (10L, no bypass) | Everything |
| GPU-D | GTX 1070 | Curriculum ablation (10L + bypass) | Curriculum effect |

If ribosome > curriculum ablation > standard ablation → both compression and curriculum matter.
If ribosome ≈ curriculum ablation > standard ablation → curriculum alone explains the gap.
If ribosome > curriculum ablation ≈ standard ablation → compression matters, curriculum doesn't.

## 4. Theoretical Framework

### 4.1 Why Priority Processing Might Work

Standard attention is O(n²) and treats all tokens equally. The ribosome introduces an information-theoretic prior: not all tokens carry equal semantic load. Function words ("the", "of", "is") are low-entropy given context; content words ("quantum", "collapse", "boundary") are high-entropy and carry more meaning.

By scoring importance and processing heavy concepts first, the cascade establishes a semantic frame before filling in details. This mirrors how humans process language — we grasp the main concepts first, then integrate supporting details.

The mathematical analog: the ribosome performs a learned, content-aware basis transformation. Standard attention operates in the token basis. The cascade operates in a compressed semantic basis where the dominant eigenvectors (heavy metatokens) are processed first.

### 4.2 The Curriculum Effect

The alpha-ramp bypass is not just a training trick — it's architecturally meaningful. A 4-layer LM learns local patterns (n-grams, syntax) quickly. The cascade layer then learns to compress and reorganize these local representations into global semantic structure. This is analogous to how CNNs learn: early layers capture edges, later layers capture objects.

The progressive activation schedule ensures the model doesn't try to learn compression before it has representations worth compressing. This is a form of the information bottleneck principle applied to training dynamics.

### 4.3 Compression Bounds

English text has approximately 1.0–1.5 bits/character of entropy. Standard BPE tokenization uses ~4–5 bits/token. The theoretical maximum compression ratio before information loss is ~5–8×.

The hard cascade at inference achieves ~3.1× compression (observed in v4/v5 experiments). This is well within theoretical bounds, suggesting room for higher compression with better boundary learning.

For hierarchical compression (the long-term vision):
- Level 0: 1,000,000 tokens
- Level 1: ~200,000 metatokens (word→phrase, ~5×)
- Level 2: ~30,000 metatokens (phrase→clause, ~7×)
- Level 3: ~5,000 metatokens (clause→topic, ~6×)
- Total: ~200× compression

## 5. Related Work

| Concept | Prior Art | How Ribosome Differs |
|---------|-----------|---------------------|
| Token compression | Token Merging (ToMe, Bolya 2023) | Gravity-based grouping respects semantic boundaries vs greedy bipartite matching |
| Prompt compression | LLMLingua (Jiang 2023) | Importance-aware grouping vs simple token dropping |
| Progressive downsampling | Funnel Transformer (Dai 2020) | Can be applied without retraining base model (Track 1) |
| Non-autoregressive | Mask-Predict (Ghazvininejad 2019) | Priority ordering (heaviest first) vs parallel with iterative refinement |
| Sparse attention | Switch Transformer / MoE | Importance-modulated span vs routing to expert subnetworks |
| Perceiver bottleneck | Perceiver (Jaegle 2021) | Chunk encoding preserves variable-length semantics vs fixed cross-attention |
| Soft token merging | Charformer / GBST (Tay 2022) | Learned boundaries via Gumbel-softmax vs fixed downsample factors |

## 6. Infrastructure

### Compute Fleet
| Machine | Role | GPU | VRAM |
|---------|------|-----|------|
| GPU-A | Development only (no training) | RTX 5090 | 32 GB |
| GPU-B | Primary training | RTX 5090 Laptop | 25 GB |
| GPU-C | Secondary training | RTX 3060 Ti | 8 GB |
| GPU-D | Tertiary training | GTX 1070 | 8 GB |

### Safety Measures
- VRAM cap: `torch.cuda.set_per_process_memory_fraction()` on all machines
- Dev machine excluded from training runs (development/analysis only)
- HuggingFace cache redirected to NVMe on primary training machine
- All training via SSH with nohup/background processes
- Checkpoints saved periodically for crash recovery

### Software
- Python 3.12/3.13, PyTorch 2.x, transformers 5.5.0
- GPT-2 tokenizer (50,257 vocab)
- HuggingFace datasets for data loading

## 7. Next Steps

### Immediate (pending wikitext-103 results)
1. Analyze 3-way comparison (ribosome vs standard ablation vs curriculum ablation)
2. Determine whether compression or curriculum drives the improvement
3. Push results to GitHub

### Short-term
4. Scale to OpenWebText or larger corpus if wikitext-103 results are promising
5. Inspect learned boundaries — are metatokens linguistically meaningful?
6. Compression ratio analysis at different Gumbel temperatures

### Medium-term
7. If compression proves valuable: scale model to 250M+ params, longer context
8. Benchmark on downstream tasks (QA, summarization, retrieval)
9. Google Colab A100 runs for larger-scale validation
10. Track 1 revival: compression-at-acceptable-quality with distillation loss

### Long-term
11. Hierarchical compression (multi-level ribosome)
12. Cross-model transfer (train ribosome on one model, apply to another)
13. Paper submission

## 8. Repository

**GitHub:** `J-Lehrer/Ribosome-Cascade`  
**License:** MIT

All code, benchmark scripts, training logs, and experimental data are version-controlled and publicly available.
