import json, subprocess, sys, time, urllib.request, socket, signal, os
binary, model = sys.argv[1], sys.argv[2]
WORDS = ["alpha", "river", "stone", "cloud", "ember", "field", "glass", "harbor"]
s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
proc = subprocess.Popen([binary, "-m", model, "--host", "127.0.0.1", "--port", str(port), "--parallel", "4", "--no-prompt-cache", "-c", "65536"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
try:
    for _ in range(400):
        try: urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2); break
        except Exception: time.sleep(1)
    def req(prompt, mt):
        body = json.dumps({"model": "default", "prompt": prompt, "max_tokens": mt, "temperature": 0, "ignore_eos": True}).encode()
        t = time.perf_counter(); json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=body, headers={"Content-Type": "application/json"}), timeout=900)); return time.perf_counter() - t
    req("alpha river", 16)
    out = {}
    for n in (256, 8192):
        p = " ".join(WORDS[(i*7+3) % 8] for i in range(n))
        vals = []
        for _ in range(3):
            a = req(p, 16); b = req(p, 528); vals.append((b - a) / 512 * 1000)
        out[n] = round(sorted(vals)[1], 3)
    print(json.dumps({"bin": os.path.basename(binary), "env_mb": os.environ.get("MLX_MAX_MB_PER_BUFFER"), "ms_per_step": out}))
finally:
    os.killpg(proc.pid, signal.SIGTERM)
    try: proc.wait(timeout=60)
    except Exception: os.killpg(proc.pid, signal.SIGKILL)
