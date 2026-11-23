import json, subprocess, sys, threading, time, urllib.request, socket, signal
binary, model, out = sys.argv[1], sys.argv[2], sys.argv[3]
WORDS = ["alpha", "river", "stone", "cloud", "ember", "field", "glass", "harbor"]
def prompt_of(n, salt): return " ".join(WORDS[(i*7+salt) % 8] for i in range(n))
s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
log = open(out + ".server.log", "w")
proc = subprocess.Popen([binary, "-m", model, "--host", "127.0.0.1", "--port", str(port), "--parallel", "4", "--no-prompt-cache", "-c", "65536"], stdout=log, stderr=subprocess.STDOUT)
for _ in range(300):
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2); break
    except Exception: time.sleep(1)
def run(i, res):
    body = json.dumps({"model": "default", "prompt": prompt_of(256, 124 + i), "max_tokens": 1024, "temperature": 0, "ignore_eos": True, "logprobs": 1}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=body, headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=600))
    res[i] = r
results = {}
for batch in (1, 4):
    res = [None] * batch
    ts = [threading.Thread(target=run, args=(i, res)) for i in range(batch)]
    [t.start() for t in ts]; [t.join() for t in ts]
    results[batch] = res
health = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/health"))
proc.send_signal(signal.SIGTERM); proc.wait(timeout=30)
summary = {}
for b, res in results.items():
    summary[b] = [{"finish_reason": r["choices"][0]["finish_reason"], "usage": r["usage"], "tail": r["choices"][0]["text"][-120:], "text": r["choices"][0]["text"], "head": r["choices"][0]["text"][:200]} for r in res]
json.dump({"summary": summary, "health": health}, open(out, "w"), indent=1)
print(json.dumps(summary, indent=1))
