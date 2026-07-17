# Python client

`mlxcel` ships a pure-Python client that drives the OpenAI-compatible server (`mlxcel serve`) from Python. It either spawns and supervises a local server process (managed mode) or connects to an already-running one (connect mode), auto-discovers the served model id, and exposes the raw `openai` client for the full API surface (tools, `response_format`, logprobs, multimodal).

This is Phase 1 of Python integration. It builds entirely on the existing HTTP server: no native extension, and no changes to the Rust inference core. The package lives in [`python/`](../python) and is published under the import name `mlxcel`.

## Install

```bash
pip install ./python          # from a repo checkout
pip install ./python[dev]     # adds pytest, ruff, mypy
```

Requires Python 3.9 or newer. Runtime dependencies are `openai>=1.40` and `httpx>=0.27`. Managed mode additionally needs the `mlxcel` binary; the client finds it via the `binary=` argument, the `MLXCEL_BIN` environment variable, or `mlxcel` on `PATH`, in that order. See [Installation](installation.md) for building the binary.

## Managed mode

Pass a model and the client spawns `mlxcel serve`, waits until the server reports ready, and stops it on exit. The model loads and warms up before the listener binds, so a first run that downloads weights can take a while; the client polls `/health` with a generous `startup_timeout` (default 600 seconds) and watches the child process for an early exit.

```python
import mlxcel

with mlxcel.LLM("mlx-community/Qwen3-4B-4bit") as llm:
    text = llm.generate("def fib(n):", max_tokens=128, temperature=0.7)
    print(text)

    reply = llm.chat([{"role": "user", "content": "Hello"}], max_tokens=64)
    print(reply)

    print(llm.model)       # resolved model id, auto-discovered from /v1/models
    print(llm.models())    # ["<id>"]
    ids = llm.tokenize("hello world")
    print(llm.detokenize(ids))
```

The context manager (`with`) is the recommended form; it guarantees shutdown. The client also registers an `atexit` cleanup and a best-effort finalizer so a leaked handle still stops the server.

Useful managed-mode arguments: `binary=`, `host=`/`port=` (forces TCP and binds there), `socket=` (an explicit Unix socket path), `api_key=`, `ctx_size=`, `n_predict=`, `alias=`, `warmup=`, `extra_args=[...]` (forwarded verbatim to `mlxcel serve`), and `startup_timeout=`.

## Connect mode

Pass `base_url=` or `socket=` (but not a model) to talk to a server you started yourself. No subprocess is launched or managed.

```python
# TCP
llm = mlxcel.LLM(base_url="http://localhost:8080/v1")

# Unix domain socket
llm = mlxcel.LLM(socket="/tmp/mlxcel.sock")

print(llm.generate("Hello"))
llm.close()
```

Start a matching server with the CLI:

```bash
# TCP
mlxcel serve -m mlx-community/Qwen3-4B-4bit --host 127.0.0.1 --port 8080

# Unix domain socket: --port 0 reinterprets --host as the socket path
mlxcel serve -m mlx-community/Qwen3-4B-4bit --host /tmp/mlxcel.sock --port 0
```

Passing both a model and a connect target raises `MlxcelError`, because the mode would be ambiguous.

## Security: multi-user hosts

On a shared machine, the default socket path under `/tmp` is world-readable: any local user can connect to the server you spawned and send requests. If that matters for your deployment, pass an explicit `socket=` path under a directory only you can read, for example one under `$XDG_RUNTIME_DIR` (mode `0700`, owned by your uid):

```python
import os, pathlib, mlxcel

runtime_dir = pathlib.Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"))
runtime_dir.mkdir(mode=0o700, parents=True, exist_ok=True)

with mlxcel.LLM("mlx-community/Qwen3-4B-4bit", socket=str(runtime_dir / "mlxcel.sock")) as llm:
    print(llm.generate("hello"))
```

The CLI equivalent is `mlxcel serve --host "$XDG_RUNTIME_DIR/mlxcel.sock" --port 0`.

On macOS, `$TMPDIR` already expands to a per-user path under `/var/folders`, so the default socket is private there. On Linux without an active login session, `$XDG_RUNTIME_DIR` may be absent; fall back to a `0700` subdirectory under your home directory if needed.

## Streaming

`stream` and `chat_stream` yield text deltas as they arrive.

```python
with mlxcel.LLM("mlx-community/Qwen3-4B-4bit") as llm:
    for delta in llm.stream("Write a haiku about autumn"):
        print(delta, end="", flush=True)

    for delta in llm.chat_stream([{"role": "user", "content": "List three uses for a Pi."}]):
        print(delta, end="", flush=True)
```

## Chat

`chat` returns the assistant message content as a string.

```python
messages = [
    {"role": "system", "content": "You are concise."},
    {"role": "user", "content": "What is MLX?"},
]
print(llm.chat(messages, max_tokens=128, temperature=0.3))
```

## Sampling parameters

Generation methods accept OpenAI sampling fields directly: `max_tokens`, `temperature`, `top_p`, `stop`, `seed`, `presence_penalty`, `frequency_penalty`, `logit_bias`, and `response_format`. Server-specific knobs that are not part of the OpenAI schema (`top_k`, `min_p`, `repetition_penalty`, DRY settings, `xtc_probability` / `xtc_threshold`, and the vLLM-compatible loop-detection fields `max_pattern_size` / `min_pattern_size` / `min_count`) are forwarded in the request body. `xtc_probability` (`0.0`-`1.0`, default `0.0`, disabled) is the per-step chance that XTC (Exclude Top Choices) removes all but the single least-probable token among those whose probability exceeds `xtc_threshold` (`0.0`-`0.5`, default `0.1`); both are validated on `/v1/chat/completions`, `/v1/completions`, and `/v1/responses`, and an out-of-range value returns a 400 before generation starts. You can also pass an explicit `extra_body={...}` for arbitrary server fields; values you set there win on conflict.

```python
llm.generate("Once upon a time", max_tokens=200, top_p=0.9, top_k=40, min_p=0.05)
llm.generate("...", extra_body={"repetition_penalty": 1.1})
```

## Structured output

The server's llguidance-backed constrained decoding honors `response_format`, so you can require schema-valid JSON.

```python
import json

schema = {
    "type": "json_schema",
    "json_schema": {
        "name": "person",
        "schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
            "additionalProperties": False,
        },
    },
}

reply = llm.chat(
    [{"role": "user", "content": "Invent a person as JSON."}],
    response_format=schema,
    max_tokens=128,
)
person = json.loads(reply)
```

## The `openai_client` escape hatch

For anything the convenience methods do not cover (tools, logprobs, vision and audio inputs, the Responses API), reach for the configured OpenAI client directly. It is wired to the same transport, so TCP and Unix-socket setups both work.

```python
oai = llm.openai_client   # openai.OpenAI (or openai.AsyncOpenAI for AsyncLLM)
oai.chat.completions.create(
    model=llm.model,
    messages=[{"role": "user", "content": "Weather in SF?"}],
    tools=[...],
)
```

## Async usage

`mlxcel.AsyncLLM` mirrors the synchronous API with `async`/`await` and async iterators, backed by `openai.AsyncOpenAI` and `httpx.AsyncClient`.

```python
import asyncio
import mlxcel

async def main():
    async with mlxcel.AsyncLLM("mlx-community/Qwen3-4B-4bit") as llm:
        print(await llm.generate("def fib(n):", max_tokens=128))
        async for delta in llm.stream("Write a haiku"):
            print(delta, end="", flush=True)

asyncio.run(main())
```

The managed-server lifecycle (spawn and readiness polling) runs synchronously inside the constructor since it is a one-time blocking setup; every generation call is async. The model id resolves lazily on the first request, so read `llm.model` only after a call has run, or pass an explicit `model=` override.

## Errors

| Exception | Raised when |
|-----------|-------------|
| `MlxcelError` | base class for all client errors (also used for ambiguous arguments) |
| `MlxcelServerError` | a managed server fails to launch, become ready, or stay alive (carries the captured stderr tail) |
| `MlxcelTimeoutError` | the managed server does not become ready within `startup_timeout` |

HTTP and API errors from the server propagate as native `openai` SDK exceptions (for example `openai.APIStatusError`), so status codes and response bodies stay visible. Only lifecycle concerns are wrapped in `Mlxcel*` types.

## Troubleshooting

- **Server never becomes ready.** The first run downloads weights before binding the listener. Increase `startup_timeout`, and enable logging to watch progress: `logging.getLogger("mlxcel.server").setLevel(logging.INFO)`. Server stderr (including download and load progress) is forwarded to that logger.
- **`MlxcelServerError` on startup with a stderr tail.** The child process exited before becoming ready. The attached stderr usually names the cause (model not found, out of memory, bad flag).
- **Binary not found.** Pass `binary="/path/to/mlxcel"`, set `MLXCEL_BIN`, or put `mlxcel` on `PATH`.
- **Unix socket path too long.** `sun_path` is about 104 bytes on macOS and 108 on Linux. On macOS, `$TMPDIR` resolves to a long `/var/folders/...` path, so the client defaults the socket to a short name under `/tmp` instead. If your explicit `socket=` path exceeds the limit, the client raises a clear error; pass a shorter path.
- **Port already in use (TCP).** In managed mode without an explicit `port=`, the client picks a free ephemeral port. Pass `port=` only when you need a fixed one.
