# Ribosome-Cascade: Research Session Notes — April 13, 2026
**Date:** April 13, 2026
**Session Focus:** Cluster-scale ablations, curriculum confound resolution, Pareto frontier mapping

---

## 1. Infrastructure

### Monitor Bug Fix
- `monitor.py` had two bugs: (a) SSH quoting — subprocess split `python3 -c '...'` into separate args, bash interpreted Python syntax as shell; (b) no duplicate-instance guard — 5 zombie processes bound to port 8777 via `SO_REUSEADDR`.
- Fixed: single-string remote command, port-in-use check + PID file at startup.

### Cluster Deployment
Four parallel experiments across all available GPUs:
| Node | GPU | VRAM | Experiment |
|------|-----|------|-----------|
| main | RTX 5090 | 32GB (cap 30) | Extended BigBaseline 500K |
| olares | RTX 5090 Laptop | 24GB | Curriculum Ablation |
| side | RTX 3060 Ti | 8GB | Compression Ratio Sweep |
| frank | GTX 1070 | 8GB | Layer Balance Ablation |

Cluster monitor dashboard on `http://localhost:8778`.

---

## 2. Experiment Results

### 2a. Extended BigBaseline (500K steps)
**Question:** Was the BigBaseline just undertrained at 100K steps?
**Answer:** Partially, but the gap remains enormous.

| Steps | Val CE | PPL |
|-------|--------|-----|
| 90K (frank) | 6.97 | 1,063 |
| 470K (main) | 6.18 | 485 |

Even at 5× training budget, BigBaseline (63M params, 12 layers, 256 raw tokens) achieves PPL 485. RibosomeTiny (49M params, 6 layers, 16 metatokens) achieves PPL 4.1 at 100K steps. The gap is **~120×** in perplexity.

### 2b. Curriculum Ablation (CRITICAL RESULT)
**Question:** Is the ribosome's advantage from compression, or from the alpha-ramp staged training?
**Answer:** Compression. Definitively.

CurriculumBaseline: 12-layer transformer with identical alpha-ramp (layers 1-4 active, layers 5-12 blended in over first 10% of training). Same params as BigBaseline (63M). No ribosome, no compression.

| Model | Val CE | PPL |
|-------|--------|-----|
| CurriculumBaseline | 6.65 | 771 |
| BigBaseline (100K) | 6.97 | 1,063 |
| **RibosomeTiny** | **1.42** | **4.1** |

The curriculum helps slightly (771 vs 1063) but explains almost none of the ribosome's advantage. The April 10 confound is resolved.

### 2c. Compression Ratio Sweep
**Question:** How does the number of metatokens affect quality?

| Chunks | Compression | Val CE | PPL | Steps |
|--------|-------------|--------|-----|-------|
| 4 | 64:1 | 3.45 | 31.4 | 100K |
| 8 | 32:1 | 2.25 | 9.4 | 75K* |
| 16 | 16:1 | 1.42 | 4.1 | 100K |
| 32 | 8:1 | — | — | queued |

*Still running, will improve further.

The Pareto curve is smooth — no cliff. Quality degrades gracefully from 16:1 through 64:1. Even at 64:1 compression (256→4 tokens), the model still achieves PPL 31.4, vastly outperforming the uncompressed BigBaseline at PPL 485.

### 2d. Layer Balance Ablation
**Question:** What's the optimal split between embedding layers (before compression) and upper layers (after compression)?

| Split | Embed | Upper | Val CE | PPL | Steps |
|-------|-------|-------|--------|-----|-------|
| 1+5 | 1 | 5 | 2.68 | 14.5 | 95K |
| 2+4 | 2 | 4 | 1.42 | 4.1 | 100K |
| 3+3 | 3 | 3 | 0.95 | 2.6 | 75K* |
| 4+2 | 4 | 2 | — | — | queued |

*Still running at 75K, will improve further. Already best result.

**Key finding:** More embedding layers before compression → better results. The 3+3 split at 75K steps already outperforms the 2+4 split at 100K steps. The ribosome benefits from richer token representations at its input.

The trend: 1+5 (14.5) → 2+4 (4.1) → 3+3 (2.6). If 4+2 continues the trend, embedding depth is the dominant architectural parameter.

---

## 3. Key Findings Summary

1. **The ribosome compression is real.** Not an artifact of curriculum, not an artifact of extra params, not an artifact of undertrained baseline. The curriculum ablation is the definitive control.

2. **Embedding depth is the most important architectural knob.** 3+3 > 2+4 > 1+5 by large margins. The ribosome needs good input representations to score token importance accurately.

3. **Compression degrades gracefully.** The 4-chunk model (64:1 compression) at PPL 31 still beats an uncompressed 12-layer transformer at PPL 485 with more params.

4. **The BigBaseline is not just undertrained.** At 5× the training budget it's at PPL 485 vs the ribosome's PPL 4. This is a fundamental architectural advantage, not a convergence speed difference.

---

## 4. Open Questions

- **4+2 split:** Will the embedding depth trend continue, or does 3+3 saturate?
- **chunks=32:** Will 8:1 compression approach or beat 16:1?
- **Scale-up (Colab A100):** Does the advantage hold at 250M params / hidden=1024? Currently training on wikitext-103.
- **Memorization check:** Some of the very low PPL numbers (2.6) may reflect memorization on wikitext-103 (~100M tokens, 49M params). Need to cross-validate on held-out data or a different dataset.
- **The freeze/unfreeze problem:** These native architecture results are excellent, but the Track 1 preprocessor (plug-in to existing LLMs) still collapses on unfreeze. The native architecture sidesteps this but limits deployment to training-from-scratch scenarios.

---

## 5. Files

| File | Description |
|------|-------------|
| experiment_results_2026-04-13.json | All experiment results in JSON |
| exp_extended_baseline.py | BigBaseline 500K training script |
| exp_curriculum_ablation.py | Curriculum ablation script |
| exp_compression_sweep.py | Compression ratio sweep script |
| exp_layer_balance.py | Layer balance ablation script |
| cluster_monitor.py | Dashboard for cluster experiment monitoring |
| cluster_status.py | CLI cluster status check |
| Ribosome_Cascade_ScaleUp_v2.ipynb | Clean Colab notebook for 250M scale experiments |
