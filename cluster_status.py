"""Cluster status check for all 4 experiments."""
import subprocess, json, time, os

def ssh_cmd(host, cmd, timeout=15):
    try:
        r = subprocess.run(["ssh", "-o", "ConnectTimeout=3", host, cmd],
                          capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip()
    except:
        return "TIMEOUT"

def check_ckpt(host, path):
    py = f"python3 -c \"import torch,json; c=torch.load('{path}',map_location='cpu',weights_only=False); print(json.dumps(dict(s=c['step'],v=round(c['val_loss'],4),p=c['params'])))\""
    out = ssh_cmd(host, py, timeout=30)
    try:
        return json.loads(out.strip().split('\n')[-1])
    except:
        return None

def check_gpu(host):
    out = ssh_cmd(host, "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader")
    return out.strip().split('\n')[-1] if out else "?"

print(f"\n{'='*70}")
print(f"  RIBOSOME-CASCADE CLUSTER STATUS — {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*70}")

# 1. Main — Extended BigBaseline
print(f"\n  [main] RTX 5090 — Extended BigBaseline 500K")
log = "E:\\Ribosome-Cascade\\exp_extended_baseline.log"
if os.path.exists(log):
    with open(log) as f:
        lines = f.readlines()
    steps = [l for l in lines if l.strip().startswith("step=")]
    if steps:
        print(f"    Last: {steps[-1].strip()}")
    else:
        print(f"    Training started, no step logs yet")
else:
    print(f"    NOT STARTED")

# 2. Olares — Curriculum ablation
print(f"\n  [olares] RTX 5090 Laptop — Curriculum Ablation")
gpu = check_gpu("olares")
print(f"    GPU: {gpu}")
ckpt = check_ckpt("olares", "/var/ribosome-cascade/exp_curriculum_ablation/best.pt")
if ckpt:
    print(f"    Best: step={ckpt['s']:,}  val_loss={ckpt['v']:.4f}  PPL={2.718**ckpt['v']:.1f}")
else:
    print(f"    No checkpoint yet")

# 3. Side — Compression sweep
print(f"\n  [side] RTX 3060 Ti — Compression Sweep {{4, 8, 32}}")
gpu = check_gpu("side")
print(f"    GPU: {gpu}")
for nc in [4, 8, 32]:
    ckpt = check_ckpt("side", f"C:/Users/jeffr/ribosome-cascade/exp_compression_sweep/chunks_{nc}/best.pt")
    if ckpt:
        print(f"    chunks={nc:>2d}: step={ckpt['s']:,}  val_loss={ckpt['v']:.4f}  PPL={2.718**ckpt['v']:.1f}")

# 4. Frank — Layer balance
print(f"\n  [frank] GTX 1070 — Layer Balance {{1+5, 3+3, 4+2}}")
gpu = check_gpu("frank")
print(f"    GPU: {gpu}")
for e, u in [(1,5), (3,3), (4,2)]:
    tag = f"e{e}_u{u}"
    ckpt = check_ckpt("frank", f"/home/jeff/ribosome-cascade/exp_layer_balance/{tag}/best.pt")
    if ckpt:
        print(f"    {tag}: step={ckpt['s']:,}  val_loss={ckpt['v']:.4f}  PPL={2.718**ckpt['v']:.1f}")

# Reference
print(f"\n  {'—'*50}")
print(f"  REFERENCE (completed):")
print(f"    RibosomeTiny (olares):  val_CE=1.4151  PPL=4.12   49M params")
print(f"    BigBaseline  (frank):   val_CE=6.9694  PPL=1063   63M params")
print(f"{'='*70}")

if __name__ == "__main__":
    pass  # already runs at import
