"""B=1 chat-template requests for #2255 measurement (a): realistic chat, little copying.

Usage: m_chat.py PORT MAX_TOKENS. Streams each question in m_client.QUESTIONS once, sequentially,
through /v1/chat/completions (Qwen3 reasons first), greedy. The chat response carries server
timings only under speculative dispatch, so decode speed is measured on the client for both
arms: (completion_tokens - 1) / (time of the last delta - time of the first delta).
"""
import http.client, json, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m_client import QUESTIONS

port, max_tokens = int(sys.argv[1]), int(sys.argv[2])
for i, q in enumerate(QUESTIONS):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=1200)
    body = {"model": "default", "messages": [{"role": "user", "content": q}], "max_tokens": max_tokens,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    conn.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    resp = conn.getresponse()
    first = last = None; usage = None; timings = {}
    for raw in resp:
        line = raw.decode().strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        ev = json.loads(data)
        if ev.get("usage"):
            usage = ev["usage"]
        if ev.get("timings"):
            timings = ev["timings"]
        for ch in ev.get("choices", []):
            d = ch.get("delta", {})
            if d.get("content") or d.get("reasoning_content"):
                now = time.perf_counter()
                first = first or now
                last = now
    conn.close()
    n = (usage or {}).get("completion_tokens", 0)
    rate = (n - 1) / (last - first) if first and last and last > first else 0.0
    print(json.dumps({"q": i, "completion_tokens": n, "decode_s": round((last or 0) - (first or 0), 4),
                      "decode_tok_s": round(rate, 2), "draft_n": timings.get("draft_n") or 0,
                      "draft_n_accepted": timings.get("draft_n_accepted") or 0,
                      "draft_rounds": timings.get("draft_rounds") or 0}), flush=True)
