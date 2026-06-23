# mlxcel (Python client)

A thin, pure-Python client for the [mlxcel](https://github.com/lablup/mlxcel) OpenAI-compatible inference server. It spawns and manages a local `mlxcel serve` process (managed mode) or connects to an already-running one (connect mode), auto-discovers the served model id, and exposes the raw `openai` client for the full API surface.

This is Phase 1 of Python integration: it builds entirely on the existing HTTP server, so it needs no native extension and no changes to the Rust inference core.

## Install

```bash
pip install ./python          # from a repo checkout
pip install ./python[dev]     # with pytest, ruff, mypy for development
```

Requires Python 3.9+. The client itself is pure Python (`openai>=1.40`, `httpx>=0.27`). Managed mode additionally needs the `mlxcel` binary on `PATH`, or pass `binary=` / set `MLXCEL_BIN`.

## Usage

```python
import mlxcel

# Managed mode: spawn and supervise a local server.
with mlxcel.LLM("mlx-community/Qwen3-4B-4bit") as llm:
    print(llm.generate("def fib(n):", max_tokens=128, temperature=0.7))

    for delta in llm.stream("Write a haiku about autumn"):
        print(delta, end="", flush=True)

    print(llm.chat([{"role": "user", "content": "Hello"}], max_tokens=64))

    print(llm.model)        # resolved model id (auto-discovered)
    print(llm.models())     # ["<id>"]
    ids = llm.tokenize("hello world")
    print(llm.detokenize(ids))

    # Escape hatch: the full OpenAI surface (tools, response_format, logprobs).
    oai = llm.openai_client
    oai.chat.completions.create(model=llm.model, messages=[...], tools=[...])

# Connect mode: talk to an already-running server.
llm = mlxcel.LLM(base_url="http://localhost:8080/v1")  # TCP
llm = mlxcel.LLM(socket="/tmp/mlxcel.sock")            # Unix socket
```

Async usage mirrors the sync API via `mlxcel.AsyncLLM` (`await llm.generate(...)`, `async for delta in llm.stream(...)`).

## Modes

- **Managed mode** (default when `model=` is given): the client spawns `mlxcel serve`, waits until `/health` returns ready, forwards server logs to the `mlxcel.server` Python logger, and stops the process on exit.
- **Connect mode** (when `base_url=` or `socket=` is given): no subprocess; the client just talks to the server.
- Passing both a model and a connect target raises `MlxcelError`.

On POSIX, managed mode defaults to a Unix domain socket for low-overhead local IPC. Keep socket paths short (`sun_path` is about 104 bytes on macOS, 108 on Linux); the default lives under `/tmp`. Pass `socket=` to override. Windows uses TCP.

## Sampling parameters

`generate`, `stream`, `chat`, and `chat_stream` accept OpenAI sampling fields directly: `max_tokens`, `temperature`, `top_p`, `stop`, `seed`, `presence_penalty`, `frequency_penalty`, `logit_bias`, `response_format`. Server-specific knobs (`top_k`, `min_p`, `repetition_penalty`, DRY settings) are forwarded in the request body; you can also pass an explicit `extra_body={...}`.

## Errors

- `MlxcelError`: base class.
- `MlxcelServerError`: launch, readiness, or crash failures (carries the server stderr tail).
- `MlxcelTimeoutError`: readiness timeout.

HTTP and API errors surface as native `openai` SDK exceptions (for example `openai.APIStatusError`), not as `Mlxcel*` types.

## Tests

```bash
pip install -e ./python[dev]
ruff check python
ruff format --check python
mypy python/src
pytest python/tests -m "not e2e"      # unit + lifecycle, no binary needed
```

The end-to-end test (`-m e2e`) is skipped unless `MLXCEL_BIN` points at a built binary:

```bash
MLXCEL_BIN=/path/to/mlxcel pytest python/tests/test_e2e.py -m e2e
```
