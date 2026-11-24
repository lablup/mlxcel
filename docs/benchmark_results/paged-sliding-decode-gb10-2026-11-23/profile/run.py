import json, subprocess, sys, time, urllib.request, socket, signal, os
# usage: run.py out_prefix binary model prompt_tokens
out, binary, model, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
WORDS = ["alpha", "river", "stone", "cloud", "ember", "field", "glass", "harbor"]
s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
cmd = ["nsys", "profile", "-o", out, "--force-overwrite=true", "--trace=cuda,osrt", "--sample=cpu", "--cpuctxsw=none",
       "--cuda-memory-usage=false", "--cuda-graph-trace=node", binary, "-m", model, "--host", "127.0.0.1", "--port", str(port), "--parallel", "4", "--no-prompt-cache", "-c", "65536"]
log = open(out + ".log", "w")
proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
try:
    for _ in range(400):
        try: urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2); break
        except Exception: time.sleep(1)
    def req(prompt, mt):
        body = json.dumps({"model": "default", "prompt": prompt, "max_tokens": mt, "temperature": 0, "ignore_eos": True}).encode()
        t = time.perf_counter()
        r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=body, headers={"Content-Type": "application/json"}), timeout=900))
        return time.perf_counter() - t, r["usage"]
    req("alpha river", 16)
    p = " ".join(WORDS[(i*7+3) % 8] for i in range(n))
    a = req(p, 16)       # prefill + 16
    b = req(p, 528)      # prefill + 528
    # decode time per step ~ (b - a)/512
    print(json.dumps({"n": n, "short_s": a[0], "long_s": b[0], "per_step_ms": (b[0]-a[0])/512*1000, "usage": b[1]}))
finally:
    os.killpg(proc.pid, signal.SIGINT)
    try: proc.wait(timeout=120)
    except Exception: os.killpg(proc.pid, signal.SIGKILL)
