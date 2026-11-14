#!/usr/bin/env python3
"""Send one greedy streaming chat request and print TTFT and the completion.

Usage: server_ttft.py <port> <prompt_file> [max_tokens] [model]

Prints a JSON line with `ttft_s` (request start to first streamed content or
reasoning delta), `total_s`, `prompt_tokens` when the server reports usage, and
the concatenated `text` (reasoning plus content), so two runs can be compared
byte for byte.
"""

import json
import sys
import time
import urllib.request


def main() -> None:
    port, prompt_file = sys.argv[1], sys.argv[2]
    max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 48
    model = sys.argv[4] if len(sys.argv) > 4 else "default"
    with open(prompt_file, encoding="utf-8") as fh:
        prompt = fh.read()
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.monotonic()
    ttft = None
    text = []
    usage = None
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                piece = (delta.get("reasoning_content") or "") + (delta.get("content") or "")
                if piece:
                    if ttft is None:
                        ttft = time.monotonic() - start
                    text.append(piece)
    print(
        json.dumps(
            {
                "ttft_s": ttft,
                "total_s": time.monotonic() - start,
                "prompt_tokens": (usage or {}).get("prompt_tokens"),
                "text": "".join(text),
            }
        )
    )


if __name__ == "__main__":
    main()
