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

"""Scripted stdio MCP server for the server-side MCP tool loop tests (#1457).

Speaks newline-delimited JSON-RPC 2.0 on stdin/stdout. The well-behaved path
is the `add`, `lookup`, `env_names` and `pid` tools; every other tool, and
every `--startup` mode, misbehaves in one specific way so a test can assert
that the failure stays inside the one call that hit it:

    hang      never answers
    crash     exits without answering
    big       answers with far more text than any output cap
    wordy     answers with `size` bytes of text (for the truncation path)
    garbage   answers with a line that is not JSON-RPC
    fail      answers with `isError: true`

`--startup hang|crash|garbage` misbehaves during `initialize` instead.
`--stderr-flood` writes a megabyte to stderr before serving, which must not
block or grow the client. Runs on Python 3.9.
"""

import json
import os
import sys
import time

TOOLS = [
    {
        "name": "add",
        "description": "Add two integers and return the sum.",
        "inputSchema": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
    },
    {
        "name": "lookup",
        "description": "Look up the secret number stored under a key.",
        "inputSchema": {
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
        },
    },
    {"name": "env_names", "description": "List environment variable names."},
    {"name": "pid", "description": "Return this server's process id."},
    {"name": "hang", "description": "Never answers."},
    {"name": "crash", "description": "Exits without answering."},
    {"name": "big", "description": "Returns an oversized result."},
    {
        "name": "wordy",
        "description": "Returns `size` bytes of text.",
        "inputSchema": {"type": "object", "properties": {"size": {"type": "integer"}}},
    },
    {"name": "garbage", "description": "Answers with invalid JSON-RPC."},
    {"name": "fail", "description": "Reports a tool error."},
]

SECRETS = {"alpha": 17, "beta": 25}


def send(message):
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def text_result(request_id, text, is_error=False):
    send(
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"content": [{"type": "text", "text": text}], "isError": is_error},
        }
    )


def call_tool(request_id, name, arguments):
    if name == "add":
        text_result(request_id, str(int(arguments["a"]) + int(arguments["b"])))
    elif name == "lookup":
        key = arguments.get("key")
        if key in SECRETS:
            text_result(request_id, str(SECRETS[key]))
        else:
            text_result(request_id, "unknown key: {}".format(key), is_error=True)
    elif name == "env_names":
        text_result(request_id, "\n".join(sorted(os.environ)))
    elif name == "pid":
        text_result(request_id, str(os.getpid()))
    elif name == "hang":
        time.sleep(3600)
    elif name == "crash":
        sys.stderr.write("fake MCP server: crashing on request\n")
        sys.stderr.flush()
        os._exit(3)
    elif name == "big":
        text_result(request_id, "x" * (8 * 1024 * 1024))
    elif name == "wordy":
        text_result(request_id, "w" * int(arguments.get("size", 1024)))
    elif name == "garbage":
        sys.stdout.write("this is not json-rpc\n")
        sys.stdout.flush()
    elif name == "fail":
        text_result(request_id, "the tool could not do that", is_error=True)
    else:
        send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32602, "message": "unknown tool: {}".format(name)},
            }
        )


def main():
    args = sys.argv[1:]
    startup = None
    if "--startup" in args:
        startup = args[args.index("--startup") + 1]
    if "--stderr-flood" in args:
        sys.stderr.write("e" * (1024 * 1024))
        sys.stderr.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        message = json.loads(line)
        method = message.get("method")
        request_id = message.get("id")
        if method == "initialize":
            if startup == "hang":
                time.sleep(3600)
            if startup == "crash":
                os._exit(4)
            if startup == "garbage":
                sys.stdout.write("{not json\n")
                sys.stdout.flush()
                continue
            send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "protocolVersion": message["params"]["protocolVersion"],
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "fake-mcp", "version": "1"},
                    },
                }
            )
        elif method == "notifications/initialized":
            # A notification of our own, which the client must skip.
            send({"jsonrpc": "2.0", "method": "notifications/message", "params": {}})
        elif method == "tools/list":
            send({"jsonrpc": "2.0", "id": request_id, "result": {"tools": TOOLS}})
        elif method == "tools/call":
            params = message.get("params", {})
            call_tool(request_id, params.get("name"), params.get("arguments") or {})
        elif request_id is not None:
            send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32601, "message": "method not found"},
                }
            )


if __name__ == "__main__":
    main()
