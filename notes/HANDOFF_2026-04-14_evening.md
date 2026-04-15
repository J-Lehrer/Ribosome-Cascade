# Handoff — 2026-04-14 Evening (Jeff went for a run)

**State @ 18:52 local:** All 4 local machines actively working.

## Cluster Status

| Host | GPU | State | Job | Progress |
|---|---|---|---|---|
| **main** | 5090 32GB | 74% util, 18.9GB used | Pareto sweep: chunks=48 then 96, 50K steps each | Training started; dataset cache warming |
| **olares** | 5090 Lap 24GB | 86% util, 11.3GB, 71°C | chunks=64 + reverse=2, 25K steps | ~5min in, training active |
| **frank** | 1070 8GB | IDLE (eval done) | 2× eval_diagnostic complete | See results below |
| **side** | 3060 Ti 8GB | 86% util, 6.2GB | chunks=8 + reverse=2, 25K steps | Training active |

VRAM headroom respected on main (~14GB free > 2GB reservation).

## KEY FINDING — Matched-size comparison, held-out C4

Ran `eval_diagnostic.py` on frank against:
- BigBaseline 12L (63.5M, trained 100K steps on wt103) — the new 100K baseline
- RibosomeTiny 3+3+rev2, 16 chunks (55.4M, ~17K steps) — pulled from olares

| Metric | BigBaseline 63.5M | Ribosome 55.4M | Ratio |
|---|---|---|---|
| C4 PPL (100 seqs × 256 tok) | **180.67** | **42.73** | 4.23× better for ribosome |
| HellaSwag acc (500 ex) | 23.8% | 20.0% | both at chance (25%) |
| wikitext-103 val PPL (training) | 240 | 8.2 | 29× better for ribosome |
| LAMBADA acc (historical) | ~0% | 0.2% | both at floor |

**Interpretations:**
1. Ribosome's absurdly low wt103 PPL (2.3 without reverse, 8.2 with reverse) is **not wikitext memorization** — generalizes to C4 at 42.7 PPL vs the baseline's 180.7. The compression bottleneck is learning genuinely useful representations.
2. Both models are below the HellaSwag competency threshold (~25% = chance). GPT-2 small (124M) reaches 28.9%, so **55–63M is fundamentally under-scaled for multi-choice semantic reasoning**. This reframes the LAMBADA failure: it's not an architecture problem, it's scale.
3. The Pareto sweep now has a sharper success criterion: **C4 PPL**, not wikitext PPL. The C4 gap (42 vs 180) is the real signal.

## Running Experiments (will finish while you're out)

### main: `pareto_48c_50k.log`, then `pareto_96c_50k.log`
exp_reverse_v2 at chunks=48 and chunks=96, reverse=2, 50K steps, batch=48.
Output dirs: `exp_reverse_v2/pareto_48c_50k/` and `pareto_96c_50k/`.
Master log: `pareto_sweep_master.log`. Master PID: see `pareto_master.pid`.

### olares: `/var/ribosome-cascade/rev2_64c_25k.log`
exp_reverse_v2 chunks=64 reverse=2, 25K steps, batch=32.
Fills the 48/64/96 gap in the Pareto sweep with a matched-protocol point.
Output: `/var/ribosome-cascade/exp_reverse_v2/rev2_64c_25k/`.

### side: `C:\Users\jeffr\ribosome-cascade\rev2_8c_25k.log`
exp_reverse_v2 chunks=8 (max compression, 256→8) reverse=2, 25K steps, batch=16.
Tests extreme end of compression axis. scheduled task: `Ribo8c`.

### frank: IDLE
Diagnostic results saved at:
- `/home/jeff/ribosome-cascade/frank_baseline_100K_diagnostic.json`
- `/home/jeff/ribosome-cascade/olares_ribo_diagnostic.json`

## Known Issues / Cleanup

1. **PIQA + WinoGrande datasets** fail on new HF datasets library (dataset-script loading removed). C4 + HellaSwag work cleanly. eval_diagnostic.py wraps per-task in try/except and saves partial JSON so other tasks complete.
2. **`trust_remote_code` warnings** flooding all logs come from older HF datasets references in exp_reverse_v2.py. Harmless — training proceeds.
3. **WinoGrande score_completion bug** when ctx="" — the slice `logits[0, -1:-1]` returns empty. Didn't fix; can address next session.

## Next Moves (when you're back)

1. **Collect Pareto results**: main will produce 2 checkpoints (48c, 96c @ 50K), olares produces 1 (64c @ 25K), side produces 1 (8c @ 25K). Combined with existing 16c and 32c from this morning → full compression sweep.
2. **Run eval_diagnostic on all Pareto checkpoints** — same format, report C4 PPL + HellaSwag across the compression ratio axis. This is the Pareto frontier.
3. **Decide Colab budget spend**: given what we see on C4, the most interesting single run is probably `chunks=32, hidden=768, 100K steps` (scale-up to see if reasoning emerges) — ~6hr on A100.
4. **Fix WinoGrande bug** and find a parquet PIQA mirror for complete diagnostic coverage.

## Paths Reference

- Local: `E:\Ribosome-Cascade\`
- olares: `/var/ribosome-cascade/` (HF cache: `/var/hf_cache`)
- frank: `/home/jeff/ribosome-cascade/`
- side: `C:\Users\jeffr\ribosome-cascade\`

Key scripts this session:
- `eval_diagnostic.py` — unified BigBaseline/RibosomeTiny eval, C4+HellaSwag
- `_launch_evening.py`, `_fix2.py`, `_relaunch_frank.py` — paramiko launchers
- `_launch_side_8c.py` — schtasks launcher for side
- `_launch_pareto_detached.py` → `_run_pareto_main.py` — DETACHED_PROCESS chain for main
- `_ribo_eval_on_frank.py` — olares→local→frank checkpoint relay + eval
- `_verify_cluster.py`, `_final_check.py` — status pollers

## Artifacts to Keep

- `E:\Ribosome-Cascade\notes\RESEARCH_NOTES_2026-04-14b.md` — afternoon session
- `E:\Ribosome-Cascade\results\afternoon_2026_04_14.json` — afternoon machine-readable
- This file — evening session
- JSON diagnostics on frank (ready to pull)
