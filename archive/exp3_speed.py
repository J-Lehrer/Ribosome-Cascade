"""
Experiment 3: Speed benchmark — concrete speedup numbers
==========================================================
Measures actual wall-clock inference time and memory for processing
different numbers of tokens through GPT-2.

Compares:
  - Raw token counts: 64, 128, 256, 512, 1024
  - Metatoken counts: 8, 16, 32, 64 (simulating ribosome compression)

Reports: tokens/sec, memory usage, attention FLOPs, latency per sample.
"""

import torch
import time
import json
import argparse
import numpy as np
from transformers import AutoModel, AutoTokenizer


def benchmark_inference(model, input_ids, n_runs=50, warmup=10, device='cuda'):
    """Time inference for a given input."""
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_ids)

    if device == 'cuda':
        torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(n_runs):
        if device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(input_ids)
        if device == 'cuda':
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    return {
        "mean_ms": float(np.mean(times) * 1000),
        "std_ms": float(np.std(times) * 1000),
        "median_ms": float(np.median(times) * 1000),
        "p99_ms": float(np.percentile(times, 99) * 1000),
    }


def measure_memory(model, input_ids, device='cuda'):
    """Measure peak GPU memory for a forward pass."""
    if device != 'cuda':
        return {"peak_mb": 0}

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    with torch.no_grad():
        _ = model(input_ids)

    peak = torch.cuda.max_memory_allocated() / 1e6
    return {"peak_mb": float(peak)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_runs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--output", default="./speed_benchmark.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    model = AutoModel.from_pretrained("gpt2").to(device)
    model.eval()
    print(f"GPT-2: {sum(p.numel() for p in model.parameters()):,} params")

    # Sequence lengths to benchmark
    raw_lengths = [64, 128, 256, 512, 1024]
    meta_lengths = [8, 16, 32, 64]

    results = {"raw_tokens": {}, "metatokens": {}, "compression_ratios": {}}

    print(f"\n{'='*60}")
    print(f"RAW TOKEN INFERENCE (standard GPT-2)")
    print(f"{'='*60}")

    for seq_len in raw_lengths:
        input_ids = torch.randint(0, 50257, (1, seq_len)).to(device)
        timing = benchmark_inference(model, input_ids, args.n_runs, args.warmup, str(device))
        memory = measure_memory(model, input_ids, str(device))
        flops = 12 * model.config.n_layer * model.config.n_embd ** 2 * seq_len + \
                2 * model.config.n_layer * model.config.n_embd * seq_len ** 2

        results["raw_tokens"][seq_len] = {
            **timing, **memory,
            "seq_len": seq_len,
            "attn_flops": flops,
            "tokens_per_sec": float(seq_len / (timing["mean_ms"] / 1000)),
        }
        print(f"  len={seq_len:5d}  {timing['mean_ms']:7.2f}ms +/- {timing['std_ms']:.2f}ms  "
              f"mem={memory['peak_mb']:.0f}MB  "
              f"tok/s={seq_len/(timing['mean_ms']/1000):.0f}")

    print(f"\n{'='*60}")
    print(f"METATOKEN INFERENCE (simulating ribosome compression)")
    print(f"{'='*60}")

    for n_chunks in meta_lengths:
        input_ids = torch.randint(0, 50257, (1, n_chunks)).to(device)
        timing = benchmark_inference(model, input_ids, args.n_runs, args.warmup, str(device))
        memory = measure_memory(model, input_ids, str(device))
        flops = 12 * model.config.n_layer * model.config.n_embd ** 2 * n_chunks + \
                2 * model.config.n_layer * model.config.n_embd * n_chunks ** 2

        results["metatokens"][n_chunks] = {
            **timing, **memory,
            "n_chunks": n_chunks,
            "attn_flops": flops,
            "tokens_per_sec": float(n_chunks / (timing["mean_ms"] / 1000)),
        }
        print(f"  chunks={n_chunks:4d}  {timing['mean_ms']:7.2f}ms +/- {timing['std_ms']:.2f}ms  "
              f"mem={memory['peak_mb']:.0f}MB  "
              f"tok/s={n_chunks/(timing['mean_ms']/1000):.0f}")

    # Compute compression ratios
    print(f"\n{'='*60}")
    print(f"COMPRESSION SPEEDUP (raw → metatoken)")
    print(f"{'='*60}")
    print(f"  {'raw':>6s} → {'meta':>6s}  {'ratio':>6s}  {'time_speedup':>12s}  "
          f"{'mem_savings':>12s}  {'attn_speedup':>12s}")
    print(f"  {'-'*65}")

    for raw_len in [256, 512, 1024]:
        for n_chunks in meta_lengths:
            if n_chunks >= raw_len:
                continue
            raw = results["raw_tokens"].get(raw_len)
            meta = results["metatokens"].get(n_chunks)
            if raw and meta:
                time_speedup = raw["mean_ms"] / meta["mean_ms"]
                mem_ratio = raw["peak_mb"] / max(meta["peak_mb"], 1)
                attn_speedup = raw["attn_flops"] / meta["attn_flops"]
                compression = raw_len / n_chunks

                key = f"{raw_len}_to_{n_chunks}"
                results["compression_ratios"][key] = {
                    "raw_len": raw_len, "n_chunks": n_chunks,
                    "compression": compression,
                    "time_speedup": float(time_speedup),
                    "mem_ratio": float(mem_ratio),
                    "attn_speedup": float(attn_speedup),
                }
                print(f"  {raw_len:>6d} → {n_chunks:>6d}  {compression:5.0f}:1  "
                      f"{time_speedup:11.1f}x  {mem_ratio:11.1f}x  {attn_speedup:11.1f}x")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
