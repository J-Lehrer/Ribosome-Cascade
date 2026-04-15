"""Diagnostic evaluation: HellaSwag, WinoGrande, PIQA, C4 PPL.

Distinguishes 'model can't do long-range binding' (all tasks fail) vs
'model can't produce exact tokens' (multiple-choice above chance but LAMBADA fails).

Loads a BigBaseline(12L) checkpoint trained via exp2_lighter.py.
Budget: ~1000 examples per MC task, 100 C4 sequences.

Usage:
    python eval_diagnostic.py --ckpt path/to/best.pt --tag NAME --out results.json
"""
import argparse, json, os, math, time, random
import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast
from datasets import load_dataset

# Local
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp2_lighter import BigBaseline, RibosomeTiny

def score_completion(model, tok, ctx, completion, device, max_len=256):
    """Sum of -log p(completion | ctx). Lower = better."""
    ctx_ids = tok.encode(ctx)
    comp_ids = tok.encode(completion)
    full = ctx_ids + comp_ids
    if len(full) > max_len:
        full = full[-max_len:]
        ctx_len = len(full) - len(comp_ids)
    else:
        ctx_len = len(ctx_ids)
    if len(comp_ids) == 0:
        return 0.0, 0
    x = torch.tensor([full], device=device)
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.float16):
            out = model(x)
    logits = out[1] if isinstance(out, tuple) else out
    log_probs = F.log_softmax(logits[0, ctx_len - 1:-1].float(), dim=-1)
    target = torch.tensor(comp_ids, device=device)
    nll = -log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1).sum().item()
    return nll, len(comp_ids)

def eval_hellaswag(model, tok, device, n=1000, seed=0):
    ds = load_dataset("Rowan/hellaswag", split="validation")
    random.seed(seed)
    idx = random.sample(range(len(ds)), min(n, len(ds)))
    correct = 0
    total = 0
    for i, j in enumerate(idx):
        ex = ds[j]
        ctx = ex["ctx"]
        label = int(ex["label"])
        scores = []
        for ending in ex["endings"]:
            nll, ntok = score_completion(model, tok, ctx + " ", ending, device)
            scores.append(nll / max(ntok, 1))  # length-normalized
        pred = min(range(4), key=lambda k: scores[k])
        correct += (pred == label)
        total += 1
        if (i + 1) % 100 == 0:
            print(f"  hellaswag {i+1}/{len(idx)}: acc={correct/total:.4f}", flush=True)
    return correct, total

def eval_winogrande(model, tok, device, n=1000, seed=0):
    # Parquet-hosted mirror (HF dropped script-based loading)
    try:
        ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")
    except Exception:
        ds = load_dataset("coref-data/winogrande_raw", "winogrande_xl", split="validation")
    random.seed(seed)
    idx = random.sample(range(len(ds)), min(n, len(ds)))
    correct = 0
    total = 0
    for i, j in enumerate(idx):
        ex = ds[j]
        sent = ex["sentence"]
        opt1, opt2 = ex["option1"], ex["option2"]
        ans = int(ex["answer"])  # "1" or "2"
        # Replace _ with each option, score full sentence (context-free length-matched)
        scores = []
        for opt in [opt1, opt2]:
            filled = sent.replace("_", opt)
            # score the whole sentence; compare
            nll, ntok = score_completion(model, tok, "", filled, device)
            scores.append(nll / max(ntok, 1))
        pred = 1 + min(range(2), key=lambda k: scores[k])
        correct += (pred == ans)
        total += 1
        if (i + 1) % 100 == 0:
            print(f"  winogrande {i+1}/{len(idx)}: acc={correct/total:.4f}", flush=True)
    return correct, total

def eval_piqa(model, tok, device, n=1000, seed=0):
    # Parquet mirror
    try:
        ds = load_dataset("ybisk/piqa", split="validation")
    except Exception:
        ds = load_dataset("cnut1648/piqa", split="validation")
    random.seed(seed)
    idx = random.sample(range(len(ds)), min(n, len(ds)))
    correct = 0
    total = 0
    for i, j in enumerate(idx):
        ex = ds[j]
        goal = ex["goal"]
        sol1, sol2 = ex["sol1"], ex["sol2"]
        label = int(ex["label"])
        scores = []
        for sol in [sol1, sol2]:
            nll, ntok = score_completion(model, tok, goal + " ", sol, device)
            scores.append(nll / max(ntok, 1))
        pred = min(range(2), key=lambda k: scores[k])
        correct += (pred == label)
        total += 1
        if (i + 1) % 100 == 0:
            print(f"  piqa {i+1}/{len(idx)}: acc={correct/total:.4f}", flush=True)
    return correct, total

def eval_c4_ppl(model, tok, device, n=100, seq_len=256, seed=0):
    # Use allenai/c4 'en' validation stream
    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    it = iter(ds)
    total_nll = 0.0
    total_tokens = 0
    random.seed(seed)
    count = 0
    for ex in it:
        if count >= n:
            break
        text = ex["text"]
        if len(text) < 100:
            continue
        ids = tok.encode(text)[:seq_len]
        if len(ids) < 32:
            continue
        x = torch.tensor([ids], device=device)
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.float16):
                out = model(x)
        logits = out[1] if isinstance(out, tuple) else out
        log_probs = F.log_softmax(logits[0, :-1].float(), dim=-1)
        target = torch.tensor(ids[1:], device=device)
        nll = -log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1).sum().item()
        total_nll += nll
        total_tokens += len(ids) - 1
        count += 1
        if count % 20 == 0:
            avg = total_nll / total_tokens
            print(f"  c4 {count}/{n}: avg_ce={avg:.4f} ppl={math.exp(avg):.2f}", flush=True)
    avg_ce = total_nll / max(total_tokens, 1)
    return avg_ce, math.exp(avg_ce), total_tokens

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", default="eval")
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", default="baseline", choices=["baseline", "ribosome"])
    ap.add_argument("--vocab_size", type=int, default=50257)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--embed_layers", type=int, default=3)
    ap.add_argument("--upper_layers", type=int, default=3)
    ap.add_argument("--reverse_layers", type=int, default=2)
    ap.add_argument("--n_chunks", type=int, default=16)
    ap.add_argument("--hellaswag_n", type=int, default=1000)
    ap.add_argument("--winogrande_n", type=int, default=1000)
    ap.add_argument("--piqa_n", type=int, default=1000)
    ap.add_argument("--c4_n", type=int, default=100)
    ap.add_argument("--skip", default="", help="comma-separated tasks to skip")
    args = ap.parse_args()

    skip = set(args.skip.split(",")) if args.skip else set()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{args.tag}] device={device}, loading ckpt {args.ckpt}", flush=True)
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    if args.model == "baseline":
        model = BigBaseline(
            vocab_size=args.vocab_size,
            hidden_size=args.hidden,
            n_layers=args.layers,
            n_heads=args.heads,
        ).to(device)
    else:  # ribosome
        model = RibosomeTiny(
            vocab_size=args.vocab_size,
            hidden_size=args.hidden,
            n_heads=args.heads,
            embed_layers=args.embed_layers,
            upper_layers=args.upper_layers,
            reverse_layers=args.reverse_layers,
            n_chunks=args.n_chunks,
        ).to(device)
    sd = torch.load(args.ckpt, map_location=device)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    model.load_state_dict(sd, strict=False)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{args.tag}] params={n_params:,}", flush=True)

    results = {"tag": args.tag, "ckpt": args.ckpt, "params": n_params}
    t0 = time.time()

    def try_task(name, fn):
        try:
            return fn()
        except Exception as e:
            print(f"[{args.tag}] {name} FAILED: {type(e).__name__}: {e}", flush=True)
            results[f"{name}_error"] = f"{type(e).__name__}: {e}"
            # Save partial results after each task
            with open(args.out or f"{args.tag}_diagnostic.json", "w") as f:
                json.dump(results, f, indent=2)
            return None

    def save_partial():
        with open(args.out or f"{args.tag}_diagnostic.json", "w") as f:
            json.dump(results, f, indent=2)

    if "c4" not in skip:
        print(f"[{args.tag}] C4 PPL (n={args.c4_n})", flush=True)
        def run_c4():
            ce, ppl, toks = eval_c4_ppl(model, tok, device, n=args.c4_n)
            results["c4_ce"] = ce; results["c4_ppl"] = ppl; results["c4_tokens"] = toks
            print(f"[{args.tag}] C4: ce={ce:.4f} ppl={ppl:.2f} tokens={toks}", flush=True)
        try_task("c4", run_c4); save_partial()

    if "piqa" not in skip:
        print(f"[{args.tag}] PIQA (n={args.piqa_n})", flush=True)
        def run_piqa():
            c, t = eval_piqa(model, tok, device, n=args.piqa_n)
            results["piqa_correct"] = c; results["piqa_total"] = t
            results["piqa_acc"] = c / t if t else 0
            print(f"[{args.tag}] PIQA: {c}/{t} = {c/t:.4f}", flush=True)
        try_task("piqa", run_piqa); save_partial()

    if "winogrande" not in skip:
        print(f"[{args.tag}] WinoGrande (n={args.winogrande_n})", flush=True)
        def run_wg():
            c, t = eval_winogrande(model, tok, device, n=args.winogrande_n)
            results["winogrande_correct"] = c; results["winogrande_total"] = t
            results["winogrande_acc"] = c / t if t else 0
            print(f"[{args.tag}] WinoGrande: {c}/{t} = {c/t:.4f}", flush=True)
        try_task("winogrande", run_wg); save_partial()

    if "hellaswag" not in skip:
        print(f"[{args.tag}] HellaSwag (n={args.hellaswag_n})", flush=True)
        def run_hs():
            c, t = eval_hellaswag(model, tok, device, n=args.hellaswag_n)
            results["hellaswag_correct"] = c; results["hellaswag_total"] = t
            results["hellaswag_acc"] = c / t if t else 0
            print(f"[{args.tag}] HellaSwag: {c}/{t} = {c/t:.4f}", flush=True)
        try_task("hellaswag", run_hs); save_partial()

    results["elapsed_s"] = time.time() - t0
    out = args.out or f"{args.tag}_diagnostic.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[{args.tag}] WROTE {out} in {results['elapsed_s']:.1f}s", flush=True)
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
