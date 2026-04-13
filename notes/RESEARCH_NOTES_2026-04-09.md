# Ribosome-Cascade: Research Session Notes
**Date:** April 9, 2026
**Session Focus:** Architecture evaluation, benchmarking, and strategic direction

---

## 1. What We Built and Tested

### Original Architecture (from notebook)
Three-stage pipeline on frozen GPT-2 (124M params):
- **Ribosome Scorer:** Linear(768->384) -> GELU -> Linear(384->1) -> Sigmoid. Outputs importance score in [0,1] per token.
- **Metatoken Assembly:** Deterministic peak-finding on score topography. Valley tokens slide to nearest peak via gravity. Tie-breaker assigns equidistant tokens to the lighter peak (semantic load balancing). Temporal tags preserve original order.
- **Cascade Decoder:** Mean-pool hidden states per metatoken -> priority-sort by weight (heaviest first) -> cosine similarity against vocab embeddings for articulation.

Training: Soft-gating approximation (h_weighted = h * sigma(s)), lm_head projection to vocab, loss = CE + 0.1 * mean(scores).

### Critical Issue Identified
**Train/inference mismatch.** Training uses smooth multiplicative gating. Inference uses hard peak-finding, discrete grouping, and mean-pooling. The ribosome has no gradient signal to produce peaks -- it's optimized for soft reweighting but evaluated on topographic structure. This explains the near-uniform score distributions observed in the notebook (all scores >0.94 at inference).

---

## 2. Benchmark Results

### v2: Fair Baseline Comparison (frozen GPT-2)
Fixed the co-adaptation confound by training a separate uniform baseline with its own lm_head.

| Length | Ribosome CE | Uniform (own lm_head) CE | Delta |
|--------|-------------|--------------------------|-------|
| 32     | 6.297       | 6.601                    | -0.304 |
| 64     | 6.210       | 6.457                    | -0.247 |
| 128    | 6.063       | 6.286                    | -0.224 |
| 256    | 5.579       | 5.805                    | -0.226 |
| 512    | 5.126       | 5.382                    | -0.255 |

**Finding:** Ribosome consistently beats uniform by ~0.25 CE. Modest but real. Gap does NOT widen with sequence length (roughly flat).

Co-adaptation confirmed: uniform scores through the ribosome's lm_head (uniform_co) were 1.0-1.4 CE worse than uniform with its own decoder.

Score entropy ratios: 0.95-0.99 (near-uniform). The ribosome is doing a mild contrast enhancement, not producing real peaks and valleys.

### v3: Layer Unfreezing Experiment
Tested unfreezing top 0/2/4/6 GPT-2 layers. Each condition trained both ribosome and matched uniform baseline.

**Absolute CE (ribosome, at len=512):**
| Condition | CE   |
|-----------|------|
| frozen    | 5.01 |
| top2      | 4.17 |
| top4      | 4.18 |
| top6      | 4.23 |

**Delta (ribosome - uniform, at len=512):**
| Condition | Delta  |
|-----------|--------|
| frozen    | -0.326 |
| top2      | -0.049 |
| top4      | -0.040 |
| top6      | -0.034 |

**Key finding:** Unfreezing layers dramatically improves absolute model quality (~1 CE point), but the ribosome's marginal advantage collapses from -0.33 to -0.04. When the base model can adapt, it absorbs the ribosome's function into its own representations. The importance information migrates into the hidden states, making the external scorer redundant.

PVR drops from ~164 (frozen) to ~1.2-1.4 (unfrozen) -- the unfrozen ribosome produces even more uniform scores.

---

## 3. Strategic Direction

### Two development tracks identified:

---

### TRACK 1: Preprocessor Integration (PRIORITY -- IMMEDIATE)
**Goal:** Build the ribosome as a standalone preprocessing module that sits in front of any existing LLM to reduce token count while preserving semantic content.

**Rationale:**
- Largest immediate real-world impact
- Model-agnostic -- works with any existing LLM
- No retraining of base models required
- Addresses a real deployed cost problem (KV cache, attention cost, context window limits)

**Two operational modes:**

**Mode A -- Compression:** Ribosome groups and compresses tokens into metatokens. LLM sees fewer tokens. Trades small accuracy hit for large speed gain. Target: 3-5x compression on typical text.

**Mode B -- Enrichment (annotation):** Ribosome scores and tags tokens with importance metadata but does NOT compress. LLM sees same tokens plus structural/importance information. Effectively a learned, content-aware importance encoding.

**Theoretical compression limits:**
- English text entropy: ~1.0-1.5 bits/character
- Standard tokenization: ~4-5 bits/token
- Theoretical max compression: ~5-8x without information loss
- For 1M token context: could represent in 125K-200K metatokens

**Hierarchical compression for very long contexts:**
```
Level 0:  1,000,000 tokens
Level 1:  ~200,000 metatokens  (word -> phrase, ~5x)
Level 2:  ~30,000 metatokens   (phrase -> clause/sentence, ~7x)
Level 3:  ~5,000 metatokens    (sentence -> paragraph/topic, ~6x)
Total: ~200x compression
```

**Key differentiator vs existing work:**
- vs Token Merging (ToMe): gravity-based grouping respects semantic boundaries rather than greedy bipartite matching by similarity
- vs LLMLingua: importance-aware grouping rather than simple token dropping
- vs Funnel Transformer: can be applied to any model without retraining

**Proposed validation experiment:**
1. Take pretrained model (Llama 3, Phi, or similar that fits on RTX 5090 32GB)
2. Freeze base model entirely
3. Train only the ribosome as front-end compressor
4. Measure downstream task accuracy at compression ratios 2x, 3x, 4x
5. Compare against: random token dropping, every-Nth pruning, attention-based pruning
6. Target benchmarks: needle-in-haystack retrieval, long-document QA, summarization

**Timeline estimate:** 2-3 weeks to working prototype

---

### TRACK 2: Native Ribosome Architecture (FOLLOW-UP -- RESEARCH)
**Goal:** Build a transformer architecture from scratch where the ribosome mechanism is a native component, not an add-on.

**Rationale:**
- v3 results show that when the base model can adapt, it absorbs importance scoring into its own representations
- A native architecture would be designed so the model's attention mechanism IS the ribosome
- Could achieve fundamentally better long-context performance

**Proposed architecture concept:**

```
Layers 1-4:   Standard token-level transformer processing
Layer 5:      RIBOSOME LAYER -- scores, groups, compresses tokens into metatokens
Layers 6-12:  Chunk-level transformer processing on metatokens
```

**Key components that need design:**

1. **Chunk Encoder** -- Replaces mean-pooling. Small attention layer (Perceiver-style) that takes variable-length token subsequence and produces a single learned chunk representation. Not a geometric centroid but a learned compression.

2. **Cascade Processor** -- Makes priority ordering computational, not just organizational. Heaviest chunk processed first through cross-attention. Each subsequent chunk attends to already-processed chunks. The anchor genuinely shapes interpretation of lighter concepts.

3. **Chunk Decoder** -- Inverse of chunk encoder. Expands processed chunk representations back to variable-length token predictions. Could be autoregressive per chunk or non-autoregressive.

**Ribosome layer placement:**
- Too early (layer 1): insufficient contextual information for good grouping
- Too late (layer 10): already paid full O(n^2) cost, no savings
- Sweet spot: layers 3-4 (enough context, early enough to save compute)

**Training challenges:**
- Grouping step is discrete and non-differentiable
- Options: Gumbel-softmax boundaries, straight-through estimator, or two-phase training
- Two-phase probably most stable: Phase 1 trains ribosome boundaries, Phase 2 trains cascade

**Open questions:**
- Whether the preprocessor (Track 1) should work IN CONJUNCTION with the native architecture
- Whether we want the ribosome layer at a fixed position or learnable/adaptive
- How to handle the variable-length metatoken problem in batched training
- Whether the gravity-based assembly algorithm itself should be differentiable

**Feasibility:** RTX 5090 32GB can handle ~200-300M param training from scratch. 4-layer base encoder + 1-layer chunk encoder/decoder + 1-layer cascade + 6-layer upper transformer is ~250M params.

**Timeline estimate:** 2-3 months research effort

---

## 4. Existing Literature to Review

| Concept | Reference |
|---------|-----------|
| Progressive token downsampling | Funnel Transformer (Dai et al. 2020) |
| Soft token merging | Charformer / GBST (Tay et al. 2022) |
| Non-autoregressive decoding with priority | Mask-Predict (Ghazvininejad et al. 2019) |
| Sparse routing | Switch Transformer / MoE |
| Token merging for inference | ToMe (Bolya et al. 2023) |
| Prompt compression | LLMLingua (Jiang et al. 2023) |
| Perceiver bottleneck | Perceiver (Jaegle et al. 2021) |

---

## 5. Files and Artifacts from This Session

| File | Location | Description |
|------|----------|-------------|
| Original notebook | E:\Ribosome-Cascade\Project_Ribosome_large.ipynb | PoC with GPT-2 |
| Benchmark v1 | E:\Ribosome-Cascade\ribosome_benchmark.py | Initial 4-method comparison |
| Benchmark v2 | E:\Ribosome-Cascade\ribosome_benchmark_v2.py | Fixed attention baseline + uniform control |
| Benchmark v3 | E:\Ribosome-Cascade\ribosome_benchmark_v3.py | Layer unfreezing experiment |
| v2 results | E:\Ribosome-Cascade\benchmark_results_v2.json | Fair baseline comparison data |
| v3 results | E:\Ribosome-Cascade\benchmark_results_v3.json | Unfreezing experiment data |
| v2 weights | E:\Ribosome-Cascade\weights_v2\ | Trained model weights |
| v3 weights | E:\Ribosome-Cascade\weights_v3\ | Per-condition weights |

---

## 6. Hardware

- **Local machine:** NVIDIA RTX 5090, 32GB VRAM
- **Python:** 3.13, PyTorch with CUDA, transformers 5.5.0
- **Note:** transformers 5.5.0 requires config-level `output_attentions=True` (forward kwarg is silently ignored for GPT-2)
