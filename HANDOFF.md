# Ribosome-Cascade: Session Handoff — April 14, 2026

## What You Are
You're picking up an active ML research project. The repo is at `E:\Ribosome-Cascade` (GitHub: `J-Lehrer/Ribosome-Cascade`). You have access to a 4-machine compute cluster via SSH (main, olares, frank, side) and Desktop Commander for local file operations.

## The Architecture
A "hourglass" transformer that compresses tokens into metatokens for efficient processing:

```
Embed layers (3L, RoPE, causal, 256 tokens)   <- full resolution
    | Ribosome: compress 256 -> 16 metatokens
Upper layers (3L, 16 chunks)                   <- 17x cheaper per layer
    | ReverseRibosome: cross-attn + 2L causal self-attn with RoPE
LM head                                        <- full resolution output
```

Key files: `native_arch_v1.py` (architecture), `exp2_lighter.py` (RibosomeTiny + BigBaseline models), `train_native.py` (training infrastructure).

## Critical Bug (Fixed Apr 13)
A double-shift in the loss function meant all models before Apr 13 were trained on predict-2-ahead, not next-token prediction. Fixed in all .py files and the Colab notebook (v3_corrected). GPT-2 Small calibration confirms the corrected eval pipeline is accurate (PPL 44.4 vs published ~29).

## Corrected Results (Apr 13-14)

| Model | Params | FLOPs | wt103 PPL | C4 PPL | LAMBADA Acc |
|-------|--------|-------|-----------|--------|-------------|
| GPT-2 Small (pretrained) | 124M | -- | 44.4 | 40.2 | 58.1% |
| RibosomeTiny 3+3 | 49M | 20.6G (0.6x) | **2.3** | **2.2** | **0.0%** |
| BigBaseline 12L | 63M | 34.1G (1.0x) | 242.8 | 145.4 | 24.2% |

The ribosome is phenomenal at local next-token prediction but completely fails long-range tasks. This is an architectural limitation: the compress-then-decompress pipeline destroys fine-grained sequential detail. The ReverseRibosome (2L causal self-attention with RoPE after decompression) was added to fix this.

## What's Currently Running

| Machine | SSH | GPU | Experiment | Check Command |
|---------|-----|-----|-----------|---------------|
| olares | ssh olares | 5090 Laptop | ReverseRibosome 3+3+2L, 25K steps | `ssh olares "tail -20 /var/ribosome-cascade/reverse_ribosome.log"` |
| frank | ssh frank | GTX 1070 | BigBaseline 12L, 100K steps | `ssh frank "python3 /home/jeff/ribosome-cascade/_qcheck.py"` |
| Colab | browser | A100 | ScaleUp v3 corrected 250M params | Check Google Drive |
| side | ssh side | 3060 Ti | IDLE - needs manual launch | `ssh side "nvidia-smi"` |

Main (RTX 5090 32GB) is free for eval/analysis.

## Immediate TODO When Results Land

1. Check ReverseRibosome results on olares - does LAMBADA accuracy improve from 0%?
2. Check frank BigBaseline 100K - does PPL improve from 243 with 4x more training?
3. Launch side manually for extended RibosomeTiny 100K
4. Collect Colab results from Drive
5. If ReverseRibosome works: run extended 100K, cross-dataset eval, update summary
6. If not: document as characterized tradeoff, explore bypass alpha tuning

## Key Research Notes
- notes/RESEARCH_NOTES_2026-04-14.md - today: chunk sweep, LAMBADA diagnostic, reverse ribosome
- notes/RESEARCH_NOTES_2026-04-13b.md - double-shift bug discovery
- notes/RESEARCH_NOTES_2026-04-13c.md - corrected cross-dataset results

## Cluster Gotchas
- olares root disk full: always set HF_HOME=/var/hf_cache HF_DATASETS_CACHE=/var/hf_cache/datasets
- side runs Windows: use python not python3, VBS detach unreliable
- Monitor at localhost:8777 may have stale paths
