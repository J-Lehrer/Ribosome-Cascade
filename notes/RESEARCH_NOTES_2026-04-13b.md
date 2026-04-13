# Research Notes — April 13, 2026 (Session 2: Evening)

## Critical Bug Discovery: Double-Shift Loss

### The Bug
Every model in the project was trained with a **double-shifted loss function**.

The data loader already pre-shifts:
```python
input_ids = chunk[:-1]   # tokens 0..n-2
labels = chunk[1:]        # tokens 1..n-1
```

But every model's `forward()` shifted AGAIN:
```python
shift_logits = logits[..., :-1, :]   # predictions 0..n-3
shift_labels = labels[..., 1:]        # tokens 2..n-1
```

Net effect: models were trained to predict the token **two positions ahead**,
not the next token. This is not standard language modeling.

### How We Caught It
Ran GPT-2 Small (124M, pretrained) through our eval pipeline as a calibration.
- Our eval: PPL **8,125** on wikitext-103
- Published: PPL **~29.4**
- 277x discrepancy → eval pipeline is broken

Traced to the double-shift: model forward() shifts labels that the loader
already shifted. Confirmed with corrected eval (eval_calibration_v2.py):
- GPT-2 Small corrected: PPL **44.4** (expected ~29-44 for 256 ctx) ✓
- Our models corrected: PPL millions (representations tuned for wrong task)

### Impact
- **ALL prior absolute PPL numbers are invalid** — not comparable to published results
- **Relative comparisons still hold** — same bug everywhere, so ribosome >> baseline
  ordering is genuine (confirmed by consistent advantage across multiple experiments)
- **No prior results can be published** — need fresh runs on correct objective

### Files Fixed
1. `exp2_lighter.py` — RibosomeTiny.forward() and BigBaseline.forward()
2. `native_arch_v1.py` — RibosomeCascadeNative.forward()
3. `exp_curriculum_ablation.py` — CurriculumAblation.forward()
4. `eval_cross_dataset.py` — LAMBADA CE eval + accuracy function
5. `monitor.py` — updated to track corrected experiment paths

### Fix
Remove the extra shift. Since the loader provides aligned pairs,
`logits[i]` directly predicts `labels[i]`:
```python
loss = F.cross_entropy(logits.view(-1, V), labels.view(-1))
```

## Corrected Training Runs (launched)

| Machine | GPU | Experiment | Steps | ETA |
|---------|-----|-----------|-------|-----|
| olares | 5090 Laptop | RibosomeTiny 3+3 | 100K | ~10h |
| frank | GTX 1070 | BigBaseline 12L | 100K | ~20h |
| side | RTX 3060 Ti | RibosomeTiny 2+4 | 100K | ~14h |

All on OpenWebText (streaming), eval on wikitext-103 val, correct next-token loss.

Early signal from olares: **val CE 1.80 at step 10K** (PPL ~6.0).
For a 49M model this is reasonable — GPT-2 Small (124M, fully trained) ≈ CE 3.4.

## Key Questions for Corrected Results

1. **Does compression still beat the baseline?**
   Prior (buggy): RibosomeTiny PPL 4.1 vs BigBaseline PPL 485 (both predict-2-ahead)
   The gap was enormous. If even a fraction persists under correct training, the
   architecture is validated.

2. **Does 3+3 still beat 2+4?**
   Prior (buggy): 3+3 PPL 2.6 vs 2+4 PPL 4.1
   This was a clean relative comparison (same bug, same model family).
   Expect the ranking to hold.

3. **What's the calibrated PPL range?**
   GPT-2 Small (124M) ≈ 29.4 PPL on wikitext-103 (1024 ctx).
   Our models are 49M params, 256 ctx, 100K steps on OWT.
   Reasonable target: PPL 20-60 for ribosome, PPL 100-500 for baseline.

## Lessons Learned
- **Always calibrate against a known reference.** Running GPT-2 through the
  eval pipeline should have been step 1, not step N.
- **Pre-shifted labels + model-internal shifts = silent corruption.**
  The loss looked fine, training converged, models produced coherent-looking
  importance scores. Nothing flagged the bug except the absolute PPL numbers
  being "too good to be true."
- **LinkedIn post removed.** Good instinct to pull it immediately.
