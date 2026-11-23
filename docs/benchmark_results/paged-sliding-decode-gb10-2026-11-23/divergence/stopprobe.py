import json, subprocess, sys, time, urllib.request, socket, signal
binary, model, src = sys.argv[1], sys.argv[2], sys.argv[3]
WORDS = ["alpha", "river", "stone", "cloud", "ember", "field", "glass", "harbor"]
prompt = " ".join(WORDS[(i*7+125) % 8] for i in range(256))
gen = json.load(open(src))[1]["choices"][0]["text"]
s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
proc = subprocess.Popen([binary, "-m", model, "--host", "127.0.0.1", "--port", str(port), "--parallel", "4", "--no-prompt-cache", "-c", "65536"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
for _ in range(300):
    try: urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2); break
    except Exception: time.sleep(1)
for ie, bias in ((True, None), (True, {"106": -100}), (True, {"1": -100}), (True, {"106": -100, "1": -100})):
    req = {"model": "default", "prompt": prompt + gen, "max_tokens": 8, "temperature": 0, "ignore_eos": ie, "logprobs": 3}
    if bias: req["logit_bias"] = bias
    body = json.dumps(req).encode()
    r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=body, headers={"Content-Type": "application/json"}), timeout=300))
    c = r["choices"][0]
    print("ignore_eos", ie, "bias", bias, c["finish_reason"], r["usage"]["completion_tokens"], repr(c["text"]), c.get("logprobs"))
proc.send_signal(signal.SIGTERM); proc.wait(timeout=30)
