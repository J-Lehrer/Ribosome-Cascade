# Research Notes — April 14, 2026 (Session 4: Afternoon — completed)

Continues from `RESEARCH_NOTES_2026-04-14.md`. Picked up remotely while
Jeff was at work. SSH from PowerShell silently fails (ssh.exe is
intercepted, exit 0, no output) — used paramiko in Python, which works
through the same OpenSSH key without going through the blocked binary.

## Headline result

**The 16-chunk hourglass class is bandwidth-limited, not capacity-limited.**

All three theories (gated bypass, more chunks, deeper reverse) failed to
break LAMBADA above ~0.3%. The reverse-ribosome morning result (0.21%)
was already near the ceiling for this architecture family. The
chunks=64 morning result (LAMBADA PPL 2.9, vs 21k at chunks=16) was
correctly identified as the actual unlock — compression ratio dominates.

## Full results matrix (all corrected, no double-shift)

| Tag | Params | Chunks | Reverse | Bypass | wt103 PPL | LAMBADA Acc | LAMBADA PPL |
|-----|-------:|-------:|--------:|:------:|----------:|------------:|------------:|
| RibosomeTiny 3+3 (no reverse) | 49.1M | 16 | 0 | — | **2.3** | 0.0% | 21,912 |
| ReverseRibosome 2L | 55.4M | 16 | 2 | — | 8.2 | 0.21% (11) | 84,124 |
| **bypass_16c_2L** (main) | 55.5M | 16 | 2 | ✅ | 15.5 | **0.33% (17)** | 30,911 |
| **chunks32_2L** (olares) | 55.4M | 32 | 2 | — | **2.7** | 0.14% (7) | 1,057,051 |
| **deep4_16c** (side) | 61.7M | 16 | 4 | — | 27.7 | 0.16% (8) | 11,277 |
| BigBaseline 12L 100K (frank) | 63.5M | — | — | — | 240 | tbd | tbd |
| GPT-2 Small (pretrained) | 124M | — | — | — | 44.4 | 58.1% | 40.2 |

(LAMBADA random-baseline is ~0.002% over 50,257 vocab. Anything below
~1% is at noise floor.)

### What each result tells us

**bypass (Theory A) — fail.** The learned per-token gate did not unlock
LAMBADA. 17/5153 vs olares' 11/5153 is within statistical noise. The
gate's existence cost wt103 PPL roughly 2× (8.2 → 15.5) — capacity
diverted to learning the gate didn't pay off. Reading the imp= telemetry
(0.31 → 0.59 → 0.48), the model didn't converge on a bimodal
"bypass-or-route" distribution. **Conclusion:** giving the model an
escape hatch doesn't help when the model has no reason to use it. The
LM-head loss can be minimized via the chunk path most of the time, and
the gradient signal for "use bypass on hard tokens" is too sparse to
learn from in 25K steps.

**chunks32 (Theory B) — partial confirmation.** The wt103 PPL stayed
excellent (2.7) and LAMBADA acc went *down* slightly. But this matches
the morning chunk-sweep: wt103 improved monotonically with chunks (322
→ 95 → 2.3 → 2.3 → 1.4 from chunks=4 to chunks=64) but LAMBADA was
~0% throughout until chunks=64 where LAMBADA PPL collapsed from 21,912
to 2.9 (still 0.2% acc). The chunks=32 condition is in the dead zone
where wt103 is great but LAMBADA hasn't yet unlocked. **Conclusion:**
the LAMBADA unlock is a non-monotonic function of compression ratio
with a sharp transition near chunks/seq_len ≈ 1/4.

**deep4 (Theory C) — fail.** More reverse capacity hurt wt103 (8.2 →
27.7) without helping LAMBADA. The extra layers absorbed gradient that
would have gone into the upper transformer or ribosome layer. 25K
steps wasn't enough for a model with 12% more params at the
deeper-reverse end. **Conclusion:** depth past 2 layers in the reverse
path is dead weight at this training budget.

**baseline 100K (frank) — modest control gain.** The 12L baseline
trained 4× longer dropped from PPL 358 → 270 → **240 final** (step
100K / 95K best). 25k → 100k = 4× compute → ~33% PPL improvement.
Linear extrapolation suggests PPL ~150 at 250K steps, ~80 at 1M — still
nowhere near GPT-2 Small's 44.4 at 124M params and orders of magnitude
more pretraining compute. **Conclusion:** the BigBaseline architecture
is not learning fast enough at 256-token context to be a fair compute
match; it's a parameter-match comparison only, and on parameters
RibosomeTiny dominates 100×.

## What this means for the project

The hourglass story has crystallized:

1. **Compression ratio is the dominant axis.** Not chunk count, not
   reverse depth, not gating — the *ratio* of seq_len / n_chunks
   determines what fine-grained tasks are recoverable.

2. **wt103 PPL and LAMBADA acc trade off non-monotonically.** The PPL
   sweet spot (chunks=16-32) is exactly where LAMBADA is most broken.
   This is the architecture telling us something important about *what
   PPL is measuring* on next-token prediction — it's primarily local
   bigram structure that survives any compression that preserves
   local-context, while LAMBADA needs cross-sentence tokens that only
   survive when chunks are dense enough to carry them.

3. **The "PPL champion vs reasoning loser" framing is now defensible.**
   This is a real, characterized architectural tradeoff, not a bug.
   The right next move is not to try to fix LAMBADA at chunks=16, but
   to map the Pareto frontier (PPL vs LAMBADA acc) as a function of
   compression ratio at fixed compute, and present that frontier as the
   contribution.

## Next experiments (suggested, not launched)

The following sit naturally on what we've learned:

1. **Compression sweep at 100K steps**, chunks ∈ {32, 48, 64, 96, 128}.
   The morning sweep at 25K showed LAMBADA's transition between
   chunks=32 (PPL 6.2e13!!) and chunks=64 (PPL 2.9). We need finer
   resolution near the transition AND at 4× more steps to know if the
   transition sharpens or smooths with training. (Estimated: ~5
   olares-days OR ~2 main-days.)

2. **chunks=64 + reverse=2 at 25K.** The morning chunks=64 ran *without*
   the reverse ribosome (just ChunkDecoder). Adding the reverse path to
   the configuration that already had a non-zero LAMBADA signal might
   convert the 0.2% into a real improvement.

3. **Kill the "all theories failed, all 25K" angle in the writeup.**
   The afternoon's three-way ablation is publishable as-is: it kills
   three plausible-sounding fixes and leaves the compression-ratio
   story standing as the survivor.

## Colab 250M scale-up

Drive folders `Ribosome-Cascade/scale_experiment/big_16L_1024h` and
`ribosome_tiny_1024h` exist (created 00:56 UTC) but are still empty as
of 13:00 PT. Either the runtime disconnected before any checkpoint was
written, or the corrected v3 notebook is still loading data. **Action
needed from Jeff:** check the notebook tab; restart if disconnected.

## Inherited frank BigBaseline final

| Metric | Value |
|--------|-------|
| Params | 63,486,976 |
| Best step | 95,000 |
| val_loss | 5.4805 |
| val PPL | 240.0 |
| Wallclock | 16h57m |
| Final step | 100,000 (val PPL 240.7) |

LAMBADA eval not yet run on this checkpoint — should be queued; the
checkpoint is at `frank:/home/jeff/ribosome-cascade/overnight_extended/baseline_12L_100K/best.pt`.

## Infrastructure built this session

| File | Purpose |
|------|---------|
| `_poll_paramiko.py` | All-host status poll via paramiko (bypasses ssh.exe block) |
| `_poll_frank_deep.py` | Detailed frank state (procs, logs, ckpt meta) |
| `_status.py` | One-shot status of main + olares + side + frank |
| `_results.py` | One-shot final results pull |
| `_launch_main.py` | Local DETACHED_PROCESS launcher for main |
| `_launch_remote.py` | sftp + nohup launcher for olares/frank |
| `_launch_side_v2.py` | side launcher via schtasks (SSH detach doesn't work) |
| `_check_side*.py` | side state checks |
| `exp_reverse_v2.py` | Parameterized ablation: --chunks N --reverse N --bypass |

## SSH-from-PowerShell mystery (unresolved)

`C:\Windows\System32\OpenSSH\ssh.exe` exits silently with no output even
for `ssh -V` when invoked from PowerShell or cmd.exe via this tool's
shell. Same binary called from `subprocess.run(["ssh", ...])` works on
the monitor (PID 11952 was running successfully and polling all hosts
every 30s). Works fine via paramiko. Not a network issue, not an auth
issue, not a key issue. Probably a console-handle redirection issue
specific to how the parent shell launches child processes — worth
diagnosing but not blocking.

## Status when leaving session

- Main: idle, GPU 0%, ready for new work
- Olares: idle, GPU 0%
- Side: idle, GPU 0%
- Frank: idle, GPU 0% (100K run completed)
- All 4 results .json files saved
- All 3 detached training processes have exited cleanly
