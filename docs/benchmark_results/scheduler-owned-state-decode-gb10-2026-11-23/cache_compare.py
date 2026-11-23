#!/usr/bin/env python3
"""Prompt-cache behaviour, main vs branch, real checkpoints (#3254).

Per arm and model: a server with the prompt cache on and --ctx-checkpoints 4,
greedy. Turn 1, a turn 2 that extends it, an edit of turn 1's tail, turn 2
again, then four concurrent fresh requests. Records every reply and
/v1/cache/stats after each step.
"""
import urllib.error
import json, signal, socket, subprocess, sys, threading, time, urllib.request

WORDS = ["alpha", "river", "stone", "cloud", "ember", "field", "glass", "harbor"]

def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); return s.getsockname()[1]

def text(n, salt):
    return " ".join(WORDS[(i * 7 + salt) % len(WORDS)] for i in range(n))

def post(port, path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as err:
        raise RuntimeError(f"{path} -> {err.code}: {err.read().decode()[:400]}") from err

def get(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=30) as r:
        return json.loads(r.read())

def chat(port, messages, max_tokens=48):
    out = post(port, "/v1/chat/completions", {"model": "default", "messages": messages, "max_tokens": max_tokens,
                                              "temperature": 0, "user": "cachecmp"})
    return out["choices"][0]["message"]["content"], out.get("usage", {})

def run(binary, model, log):
    port = free_port()
    proc = subprocess.Popen([binary, "-m", model, "--host", "127.0.0.1", "--port", str(port),
                             "--parallel", "4", "-c", "65536", "--ctx-checkpoints", "4"],
                            stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(300):
            if proc.poll() is not None:
                raise RuntimeError("server exited")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2); break
            except Exception:
                time.sleep(1)
        steps = []
        def record(name, reply, usage):
            steps.append({"step": name, "reply": reply, "usage": usage,
                          "stats": get(port, "/v1/cache/stats")})
        a = "Summarize this list in one sentence: " + text(1500, 1)
        t1 = [{"role": "user", "content": a}]
        r1, u = chat(port, t1); record("turn1", r1, u)
        t2 = t1 + [{"role": "assistant", "content": r1},
                   {"role": "user", "content": "Now continue with: " + text(900, 5)}]
        r2, u = chat(port, t2); record("turn2", r2, u)
        t3 = t2 + [{"role": "assistant", "content": r2},
                   {"role": "user", "content": "And once more: " + text(900, 9)}]
        r3, u = chat(port, t3); record("turn3", r3, u)
        # Edit turn 2's question (keeps turn 1 and its reply): a history boundary restore.
        e2 = t1 + [{"role": "assistant", "content": r1},
                   {"role": "user", "content": "Now instead continue with: " + text(600, 3)}]
        re, u = chat(port, e2); record("edit_turn2", re, u)
        r2b, u = chat(port, t2); record("turn2_again", r2b, u)
        outs = [None] * 4
        def one(i):
            outs[i] = chat(port, [{"role": "user", "content": f"Request {i}: " + text(300, 20 + i)}])
        th = [threading.Thread(target=one, args=(i,)) for i in range(4)]
        [t.start() for t in th]; [t.join() for t in th]
        record("four_concurrent", [o[0] for o in outs], [o[1] for o in outs])
        return steps
    finally:
        proc.send_signal(signal.SIGTERM)
        try: proc.wait(timeout=30)
        except subprocess.TimeoutExpired: proc.kill()

def main():
    main_bin, branch_bin, out = sys.argv[1], sys.argv[2], sys.argv[3]
    models = sys.argv[4:]
    result = {}
    log = open(out + ".server.log", "w")
    for model in models:
        result[model] = {"main": run(main_bin, model, log), "branch": run(branch_bin, model, log)}
    json.dump(result, open(out, "w"), indent=1)
    for model, arms in result.items():
        for m, b in zip(arms["main"], arms["branch"]):
            same_text = m["reply"] == b["reply"]
            same_keys = sorted(m["stats"]) == sorted(b["stats"])
            diff = {k: (m["stats"].get(k), b["stats"].get(k)) for k in m["stats"]
                    if m["stats"].get(k) != b["stats"].get(k)}
            print(json.dumps({"model": model.split("/")[-1], "step": m["step"], "same_text": same_text,
                              "same_stat_keys": same_keys, "stat_diffs": diff,
                              "usage_main": m["usage"], "usage_branch": b["usage"]})[:1500])

main()
