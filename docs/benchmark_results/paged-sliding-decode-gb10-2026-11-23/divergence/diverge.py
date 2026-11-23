import json, subprocess, sys, threading, time, urllib.request, socket, signal
# usage: diverge.py out.json binary model salt[,salt...] batch max_tokens
out, binary, model, salts, maxt = sys.argv[1], sys.argv[2], sys.argv[3], [int(x) for x in sys.argv[4].split(",")], int(sys.argv[5])
WORDS = ["alpha", "river", "stone", "cloud", "ember", "field", "glass", "harbor"]
def prompt_of(n, salt): return " ".join(WORDS[(i*7+salt) % 8] for i in range(n))
s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
log = open(out + ".server.log", "w")
proc = subprocess.Popen([binary, "-m", model, "--host", "127.0.0.1", "--port", str(port), "--parallel", "4", "--no-prompt-cache", "-c", "65536"], stdout=log, stderr=subprocess.STDOUT)
for _ in range(300):
    try: urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2); break
    except Exception: time.sleep(1)
res = [None]*len(salts)
def run(i):
    body = json.dumps({"model": "default", "prompt": prompt_of(256, salts[i]), "max_tokens": maxt, "temperature": 0, "ignore_eos": True, "logprobs": 3}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=body, headers={"Content-Type": "application/json"})
    res[i] = json.load(urllib.request.urlopen(req, timeout=900))
ts = [threading.Thread(target=run, args=(i,)) for i in range(len(salts))]
[t.start() for t in ts]; [t.join() for t in ts]
proc.send_signal(signal.SIGTERM); proc.wait(timeout=30)
json.dump(res, open(out, "w"))
