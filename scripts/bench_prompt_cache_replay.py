#!/usr/bin/env python3
# Copyright 2025-2026 Lablup Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Replay an agent-shaped chat workload against mlxcel-server (#2885 Part B).

The workload is deterministic and built from the repository's docs, so two
server configurations see byte-identical requests:

* one system prompt plus a tool list, shared by every chat;
* session A: ten turns that each append a user message and a scripted
  assistant reply, where turn 8 rewrites the user message of turn 3 (an agent
  compacting its history a few thousand tokens back; --edit-turn and
  --edit-target move both);
* sessions B and C: new chats with the same system prompt and tools, three
  turns each.

Assistant replies in the history are scripted rather than taken from the
model, so the prompts do not depend on what an arm generated. Each request
streams a short greedy completion and records time to first token, prompt
tokens and the cached (reused) tokens the server reports.

With --verify, every request is sent again after `POST /v1/cache/reset` and its
completion compared with the cached run's: a cache hit that adopts a snapshot
at a split point of the request's own plan must reproduce the cold run.

Usage:
  scripts/bench_prompt_cache_replay.py --port 8080 --label root-on --out r.json --verify

Run once per server configuration (for example with and without
MLXCEL_PROMPT_CACHE_SYSTEM_ROOT=1) and compare the summaries.
"""

import argparse
import hashlib
import json
import pathlib
import sys
import time
import urllib.request

CHARS_PER_TOKEN = 3.6

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": {p: {"type": "string", "description": f"The {p} argument."} for p in params},
                "required": params[:1],
            },
        },
    }
    for name, desc, params in [
        ("read_file", "Read a file from the workspace and return its contents with line numbers.", ["path", "offset", "limit"]),
        ("grep", "Search the workspace for a regular expression and return matching lines.", ["pattern", "path", "glob"]),
        ("edit_file", "Replace an exact string in a file with a new string.", ["path", "old_string", "new_string"]),
        ("run_tests", "Run a test command and return its output and exit status.", ["command", "timeout"]),
        ("list_dir", "List the entries of a directory.", ["path"]),
        ("fetch_url", "Fetch a URL and return its text content.", ["url"]),
    ]
]


def corpus(repo):
    docs = sorted((repo / "docs").glob("*.md"))
    return "\n\n".join(p.read_text(errors="ignore") for p in docs)


def build_workload(repo, system_tokens, turn_tokens, edit_turn=8, edit_target=3):
    text = corpus(repo)
    cursor = 0

    def take(tokens):
        nonlocal cursor
        n = int(tokens * CHARS_PER_TOKEN)
        piece = text[cursor : cursor + n]
        cursor += n
        return piece

    system = (
        "You are a coding agent working in the mlxcel repository. Use the tools to read, search, "
        "edit and test code. Answer briefly.\n\nProject reference:\n\n" + take(system_tokens)
    )

    def user(i):
        return f"Step {i}. Here is the next part of the material:\n\n{take(turn_tokens)}\n\nWhat is the single most important point in it?"

    def reply(i):
        return f"Noted step {i}. The key point is recorded; continue."

    requests = []

    # Session A: ten turns, turn `edit_turn` rewrites turn `edit_target`'s user message.
    history = [{"role": "system", "content": system}]
    for turn in range(1, 11):
        if turn == edit_turn:
            # Compaction: the user message of the target turn is replaced by a summary.
            idx = 1 + 2 * (edit_target - 1)
            history[idx] = {"role": "user", "content": f"Step {edit_target} (compacted). Earlier material summarised: build, test and serve flow."}
        history.append({"role": "user", "content": user(f"A{turn}")})
        requests.append({"session": "A", "turn": turn, "edited": turn == edit_turn, "messages": [dict(m) for m in history]})
        history.append({"role": "assistant", "content": reply(f"A{turn}")})

    # Sessions B and C: new chats with the same system prompt and tools.
    for name in ["B", "C"]:
        history = [{"role": "system", "content": system}]
        for turn in range(1, 4):
            history.append({"role": "user", "content": user(f"{name}{turn}")})
            requests.append({"session": name, "turn": turn, "edited": False, "messages": [dict(m) for m in history]})
            history.append({"role": "assistant", "content": reply(f"{name}{turn}")})
    return requests


def post(url, body=None, timeout=3600):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="POST", headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def stream(base, messages, max_tokens):
    body = {
        "model": "m",
        "messages": messages,
        "tools": TOOLS,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    ttft = None
    text = []
    usage = None
    with post(base + "/v1/chat/completions", body) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            obj = json.loads(payload)
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices", []):
                d = ch.get("delta", {})
                piece = (d.get("content") or "") + (d.get("reasoning_content") or "")
                for call in d.get("tool_calls") or []:
                    piece += json.dumps(call, sort_keys=True)
                if piece:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    text.append(piece)
    out = "".join(text)
    usage = usage or {}
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    return {
        "ttft_s": ttft,
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": cached,
        "completion_tokens": usage.get("completion_tokens"),
        "sha": hashlib.sha256(out.encode()).hexdigest()[:16],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--repo", default=str(pathlib.Path(__file__).resolve().parent.parent))
    ap.add_argument("--system-tokens", type=int, default=4096)
    ap.add_argument("--turn-tokens", type=int, default=1536)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--edit-turn", type=int, default=8, help="session A turn that rewrites an earlier user message")
    ap.add_argument("--edit-target", type=int, default=3, help="session A turn whose user message is rewritten")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--verify", action="store_true", help="resend each request after a cache reset and compare")
    a = ap.parse_args()

    base = f"http://127.0.0.1:{a.port}"
    if not 1 <= a.edit_target < a.edit_turn <= 10:
        ap.error("need 1 <= --edit-target < --edit-turn <= 10")
    requests = build_workload(pathlib.Path(a.repo), a.system_tokens, a.turn_tokens, a.edit_turn, a.edit_target)
    post(base + "/v1/cache/reset", {}).read()

    rows = []
    for req in requests:
        r = stream(base, req["messages"], a.max_tokens)
        r.update(session=req["session"], turn=req["turn"], edited=req["edited"])
        rows.append(r)
        print(json.dumps(r), flush=True)
    stats = get_json(base + "/v1/cache/stats")

    mismatches = []
    if a.verify:
        for req, row in zip(requests, rows):
            post(base + "/v1/cache/reset", {}).read()
            cold = stream(base, req["messages"], a.max_tokens)
            row["cold_ttft_s"] = cold["ttft_s"]
            row["cold_sha"] = cold["sha"]
            if cold["sha"] != row["sha"]:
                mismatches.append(f"{row['session']}{row['turn']}")

    summary = {
        "label": a.label,
        "requests": len(rows),
        "prompt_tokens": sum(r["prompt_tokens"] or 0 for r in rows),
        "cached_tokens": sum(r["cached_tokens"] or 0 for r in rows),
        "ttft_total_s": round(sum(r["ttft_s"] or 0 for r in rows), 3),
        "cache_stats": stats,
    }
    if a.verify:
        summary["cold_ttft_total_s"] = round(sum(r["cold_ttft_s"] or 0 for r in rows), 3)
        summary["identical_to_cold"] = len(rows) - len(mismatches)
        summary["mismatches"] = mismatches
    json.dump({"summary": summary, "rows": rows}, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "cache_stats"}), flush=True)
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
