"""Cluster experiment monitor. http://localhost:8778"""
import subprocess, json, time, http.server, threading, socket, sys, os, atexit

PORT = 8778
PIDFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cluster_monitor.pid")

def port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

# Experiment definitions
EXPERIMENTS = [
    {"machine": "main", "gpu": "RTX 5090", "exp": "BigBaseline 500K",
     "host": None,  # local
     "ckpt": r"E:\Ribosome-Cascade\exp_extended_baseline\best.pt",
     "log": r"E:\Ribosome-Cascade\exp_extended_baseline.log",
     "python": "python"},
    {"machine": "olares", "gpu": "5090 Laptop", "exp": "Curriculum Ablation",
     "host": "olares",
     "ckpt": "/var/ribosome-cascade/exp_curriculum_ablation/best.pt",
     "python": "python3"},
    {"machine": "side", "gpu": "RTX 3060 Ti", "exp": "Compression Sweep",
     "host": "side",
     "ckpts": [f"C:/Users/jeffr/ribosome-cascade/exp_compression_sweep/chunks_{n}/best.pt"
               for n in [4, 8, 32]],
     "labels": ["4 chunks", "8 chunks", "32 chunks"],
     "python": "python"},
    {"machine": "frank", "gpu": "GTX 1070", "exp": "Layer Balance",
     "host": "frank",
     "ckpts": [f"/home/jeff/ribosome-cascade/exp_layer_balance/{t}/best.pt"
               for t in ["e1_u5", "e3_u3", "e4_u2"]],
     "labels": ["1+5", "3+3", "4+2"],
     "python": "python3"},
]

PY_CHECK = 'import torch,os,json;bp="{}";c=torch.load(bp,map_location="cpu",weights_only=False) if os.path.exists(bp) else None;print(json.dumps(dict(v=round(c["val_loss"],4),s=c["step"],p=c.get("params",0))) if c else json.dumps(dict(v=None,s=0,p=0)))'

def ssh_check(host, ckpt_path, py="python3"):
    try:
        cmd = f"{py} -c '{PY_CHECK.format(ckpt_path)}'"
        r = subprocess.run(["ssh", "-o", "ConnectTimeout=3", host, cmd],
                          capture_output=True, text=True, timeout=20)
        for line in r.stdout.strip().split("\n"):
            if line.strip().startswith("{"):
                return json.loads(line.strip())
    except:
        pass
    return {"v": None, "s": 0, "p": 0}

def local_check(ckpt_path):
    try:
        import torch
        if os.path.exists(ckpt_path):
            c = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            return {"v": round(c["val_loss"], 4), "s": c["step"], "p": c.get("params", 0)}
    except:
        pass
    return {"v": None, "s": 0, "p": 0}

def parse_local_log(log_path):
    """Parse last step line from local log for live progress."""
    try:
        if not os.path.exists(log_path):
            return None
        with open(log_path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 2000))
            tail = f.read().decode("utf-8", errors="ignore")
        lines = [l.strip() for l in tail.split("\n") if "step=" in l and "CE=" in l]
        if lines:
            last = lines[-1]
            # Parse key=value pairs, handling spaces: "step=  96000  CE=5.86"
            import re
            step_m = re.search(r'step=\s*(\d+)', last)
            ce_m = re.search(r'CE=\s*([\d.]+)', last)
            eta_m = re.search(r'ETA=\s*([\d.]+h)', last)
            if step_m and ce_m:
                return {"step": int(step_m.group(1)),
                        "ce": float(ce_m.group(1)),
                        "eta": eta_m.group(1) if eta_m else "?"}
    except:
        pass
    return None

def ssh_gpu(host):
    try:
        r = subprocess.run(["ssh", "-o", "ConnectTimeout=3", host,
            "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        return r.stdout.strip().split("\n")[-1].strip()
    except:
        return "?"

cached_json = '{"cards":[],"ts":"starting..."}'

def poll():
    global cached_json
    while True:
        cards = []
        # 1. Main (local)
        exp = EXPERIMENTS[0]
        d = local_check(exp["ckpt"])
        live = parse_local_log(exp["log"])
        cards.append({"machine": exp["machine"], "gpu": exp["gpu"], "exp": exp["exp"],
                       "v": d["v"], "s": live["step"] if live else d["s"],
                       "p": d["p"], "ce_live": live["ce"] if live else None,
                       "eta": live["eta"] if live else None, "target": 500000,
                       "gpu_util": None, "sub": None})

        # 2. Olares (single experiment)
        exp = EXPERIMENTS[1]
        d = ssh_check(exp["host"], exp["ckpt"], exp["python"])
        gpu = ssh_gpu(exp["host"])
        cards.append({"machine": exp["machine"], "gpu": exp["gpu"], "exp": exp["exp"],
                       "v": d["v"], "s": d["s"], "p": d["p"],
                       "ce_live": None, "eta": None, "target": 100000,
                       "gpu_util": gpu, "sub": None})

        # 3. Side (multi-checkpoint sweep)
        exp = EXPERIMENTS[2]
        gpu = ssh_gpu(exp["host"])
        subs = []
        for i, cp in enumerate(exp["ckpts"]):
            d = ssh_check(exp["host"], cp, exp["python"])
            subs.append({"label": exp["labels"][i], "v": d["v"], "s": d["s"]})
        cards.append({"machine": exp["machine"], "gpu": exp["gpu"], "exp": exp["exp"],
                       "v": None, "s": 0, "p": 0, "ce_live": None, "eta": None,
                       "target": 100000, "gpu_util": gpu, "sub": subs})

        # 4. Frank (multi-checkpoint sweep)
        exp = EXPERIMENTS[3]
        gpu = ssh_gpu(exp["host"])
        subs = []
        for i, cp in enumerate(exp["ckpts"]):
            d = ssh_check(exp["host"], cp, exp["python"])
            subs.append({"label": exp["labels"][i], "v": d["v"], "s": d["s"]})
        cards.append({"machine": exp["machine"], "gpu": exp["gpu"], "exp": exp["exp"],
                       "v": None, "s": 0, "p": 0, "ce_live": None, "eta": None,
                       "target": 100000, "gpu_util": gpu, "sub": subs})

        cached_json = json.dumps({"cards": cards, "ts": time.strftime("%H:%M:%S")})
        print(f'[{time.strftime("%H:%M:%S")}] Polled {len(cards)} experiments')
        time.sleep(45)

PAGE = rb"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Ribosome Cluster</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,sans-serif;background:#111;color:#e0e0e0;padding:24px}
h1{font-size:22px;font-weight:500;color:#fff;margin-bottom:2px}
.sub{font-size:12px;color:#666;margin-bottom:20px}
.ref{font-size:11px;color:#555;padding:12px 16px;border:1px solid #222;border-radius:8px;margin-bottom:20px;background:#161616}
.ref b{color:#1d9e75}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}
.card{background:#1a1a1a;border-radius:12px;padding:16px 20px;border:1px solid #2a2a2a}
.card:hover{border-color:#333}
.top{display:flex;justify-content:space-between;align-items:center;margin-bottom:4px}
.machine{font-weight:600;font-size:15px;color:#fff}
.gpu-tag{font-size:10px;padding:2px 8px;border-radius:6px;background:#1a2a1a;color:#1d9e75}
.exp{font-size:13px;color:#888;margin-bottom:10px}
.val{font-size:32px;font-weight:500;font-family:'SF Mono',monospace;margin:4px 0}
.val.good{color:#1d9e75}.val.mid{color:#d4a843}.val.bad{color:#888}
.meta{font-size:11px;color:#555;margin-top:2px}
.bar-bg{height:4px;background:#222;border-radius:2px;margin-top:10px;overflow:hidden}
.bar{height:100%;border-radius:2px;transition:width 1s ease}
.sub-items{margin-top:10px;border-top:1px solid #222;padding-top:8px}
.sub-row{display:flex;justify-content:space-between;font-size:12px;padding:3px 0;color:#999}
.sub-row .sv{font-family:'SF Mono',monospace;color:#1d9e75}
.sub-row .sv.none{color:#444}
.eta-tag{font-size:10px;padding:2px 6px;border-radius:4px;background:#1a1a2a;color:#6688cc;margin-left:6px}
</style></head><body>
<h1>Ribosome-Cascade Cluster</h1>
<div class="sub" id="sub">Polling...</div>
<div class="ref"><b>Reference:</b> RibosomeTiny val_CE=1.4151 PPL=4.12 (49M) &nbsp;|&nbsp; BigBaseline val_CE=6.97 PPL=1063 (63M, 100K steps)</div>
<div class="grid" id="g"></div>
<script>
function fmt(v){return v!==null?v.toFixed(4):"\u2014";}
function cls(v){if(v===null)return"bad";if(v<3)return"good";if(v<6)return"mid";return"bad";}
function go(){
  var x=new XMLHttpRequest();
  x.open("GET","/api",true);
  x.onload=function(){
    if(x.status!==200)return;
    var d=JSON.parse(x.responseText);
    document.getElementById("sub").innerText="Updated "+d.ts+" \u2022 refreshes every 45s";
    var h="";
    for(var i=0;i<d.cards.length;i++){
      var c=d.cards[i];
      var val=c.ce_live!==null?c.ce_live:c.v;
      var step=c.s||0;
      var pct=Math.min(step/c.target*100,100);
      var done=pct>=99.9;
      h+='<div class="card">';
      h+='<div class="top"><span class="machine">'+c.machine+'</span>';
      if(c.gpu_util)h+='<span class="gpu-tag">'+c.gpu_util+'</span>';
      h+='</div>';
      h+='<div class="exp">'+c.exp+' \u2022 '+c.gpu+'</div>';
      if(c.sub){
        h+='<div class="sub-items">';
        for(var j=0;j<c.sub.length;j++){
          var s=c.sub[j];
          var sv=s.v!==null?s.v.toFixed(4):"\u2022\u2022\u2022";
          var sc=s.v!==null?"sv":"sv none";
          var ss=s.s>0?" (step "+s.s.toLocaleString()+")":"";
          h+='<div class="sub-row"><span>'+s.label+'</span><span class="'+sc+'">'+sv+ss+'</span></div>';
        }
        h+='</div>';
      } else {
        h+='<div class="val '+cls(val)+'">'+fmt(val)+'</div>';
        h+='<div class="meta">step '+step.toLocaleString()+' / '+c.target.toLocaleString();
        if(c.eta&&c.eta!=="?")h+='<span class="eta-tag">ETA '+c.eta+'</span>';
        h+='</div>';
      }
      h+='<div class="bar-bg"><div class="bar" style="width:'+pct+'%;background:'+(done?'#1d9e75':'#2a6aaa')+'"></div></div>';
      h+='</div>';
    }
    document.getElementById("g").innerHTML=h;
  };
  x.send();
}
go();setInterval(go,45000);
</script></body></html>"""

class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(cached_json.encode())
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(PAGE)
    def log_message(self, *a): pass

if __name__ == "__main__":
    if port_in_use(PORT):
        print(f"ERROR: port {PORT} already in use.")
        sys.exit(1)
    with open(PIDFILE, "w") as f:
        f.write(str(os.getpid()))
    atexit.register(lambda: os.remove(PIDFILE) if os.path.exists(PIDFILE) else None)
    threading.Thread(target=poll, daemon=True).start()
    print(f"Cluster monitor at http://localhost:{PORT}  (PID {os.getpid()})")
    http.server.HTTPServer(("0.0.0.0", PORT), H).serve_forever()
