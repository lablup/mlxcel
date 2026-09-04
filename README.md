# mlxcel

[![License: Apache 2.0](https://img.shields.io/github/license/lablup/mlxcel)](LICENSE)
[![Latest Release](https://img.shields.io/github/v/release/lablup/mlxcel)](https://github.com/lablup/mlxcel/releases/latest)
[![CI](https://github.com/lablup/mlxcel/actions/workflows/ci.yml/badge.svg)](https://github.com/lablup/mlxcel/actions/workflows/ci.yml)

High-performance LLM, VLM, embedding, reranking, and audio inference for Apple Silicon and NVIDIA CUDA systems. The CLI and server are implemented in Rust and execute MLX SafeTensors checkpoints through native MLX C++ bindings, without Python in the request path or a checkpoint-conversion step.

## Overview

`mlxcel` is both a local inference CLI and a production-oriented model server. It runs text generation, multimodal input, embeddings, reranking, speech workloads, continuous batching, prompt caching, speculative decoding, and distributed inference in one native runtime.

Apple Silicon is the primary target. Linux/CUDA is supported as a secondary target, and an opt-in OpenXLA/IREE backend is available as an alpha development path.

## Current main highlights

The current `main` branch is v0.7.0-beta.1 plus unreleased work. Install from source to use features that have not reached the latest tagged release; see the [changelog](CHANGELOG.md) for the release-by-release record.

- **Verified `llama-server` compatibility.** A frozen b10621 manifest classifies all 376 pinned options, routes, and native request fields, with no deferred entries. Native completion, embedding, tokenization, template, infill, props, slots, metrics, resumable-stream, router, and LoRA surfaces are implemented or explicitly classified instead of being silently ignored.
- **OpenAI, Anthropic, and Vertex-compatible serving.** Chat Completions, Completions, Responses, Embeddings, Reranking, Audio, and Anthropic Messages are available alongside the native `llama-server` routes. Optional Vertex AI custom-container routing is supported through the standard `AIP_*` variables.
- **Multi-model routing and live adapters.** Router mode discovers checkpoints from the model store, a model directory, or INI presets; loads them on demand; and bounds the resident set with LRU eviction. Multiple LoRA adapters can remain unfused for per-request or live scale changes, or be fused for zero decode overhead.
- **Broader multimodal and retrieval coverage.** Embedding and reranker families include BERT, ModernBERT, SigLIP, Qwen3/Qwen3-VL, Llama/Nemotron, LFM2.5, ColBERT-style models, cross-encoders, and generative rerankers. Qwen-VL video input and Responses-native image parts are supported.
- **Expanded audio serving.** The compatible transcription boundary recognizes WAV, MP3, and FLAC by content, and chat-model transcription streams one ASR delta per decoded token. Phi-4 Multimodal and Gemma 3n decode all three containers; other current audio families and dedicated Whisper remain WAV-only. Kokoro provides text-to-speech.
- **Measured speculative decoding.** Gemma 4 and Qwen MTP paths probe greedy exactness before enabling the fast path, and the server exposes the adaptive decision at `GET /v1/internal/mtp-policy`.
- **Operational controls.** Optional authenticated live settings, prompt/cache and slot observability, bounded response stores, idle model sleep/wake, GBNF grammars, expanded sampling controls, runtime reasoning placement, and API-key/CORS/TLS controls are available.
- **DeepSeek-V4 support.** The `deepseek_v4` architecture includes HyperConnections, rotating shared-KV attention, per-layer compression, HiSA sparse selection, and hash-routed early MoE layers.
- **Inkling across four modalities.** The text backbone runs hybrid sliding/global NoPE attention with per-layer short-convolution state and logsigmoid-normalized experts, and carries HMLP image tiling, adjacent-frame video, dMel audio, and a native MTP drafter.

See [Server features](docs/server-features.md) for the route and deployment map and [llama-server compatibility](docs/llama-server-compat.md) for the exact b10621 boundary.

## Why mlxcel

- **Native, compact runtime surface.** Loading, scheduling, and inference stay in one process, with no Python environment or interpreter layer to provision in production.
- **Direct checkpoint loading.** MLX SafeTensors checkpoints from HuggingFace, including `mlx-community`, load directly. Many standard SafeTensors embedding checkpoints also load without conversion.
- **Simple deployment artifacts.** `mlxcel` and `mlxcel-server` are native executables suitable for packaging and service supervision. Platform runtime libraries are still required.
- **Compatibility without overclaiming.** `mlxcel-server` accepts a broad `llama-server` flag and `LLAMA_ARG_*` environment surface, while the checked-in manifest records every supported, aliased, not-applicable, or intentional-difference case.
- **Serving features enabled for real workloads.** Continuous batching, prompt-prefix caching, and automatic prefix caching are on by default. Speculative decoding, KV-cache compression, router mode, live LoRA, and distributed modes are available where the model and backend support them.
- **Reproducible model surgery.** Default builds support YAML load-time weight edits through `--surgery` / `MLXCEL_SURGERY`, including `scale`, `add`, `prune`, `replace`, and `interpolate`.
- **Broad architecture coverage.** Dense transformers, sparse MoE, hybrid SSM, VLM/OCR, block diffusion, embeddings, rerankers, ASR, and TTS are represented. Run `mlxcel arch` for the binary's architecture catalog and consult [Supported models](docs/supported-models.md) for checkpoint-level notes.

## Quick start

### Install with Homebrew

The Homebrew formula installs the latest released `mlxcel` and `mlxcel-server` binaries on macOS and Linux:

```bash
brew tap lablup/tap
brew install mlxcel
```

### Run a model

`mlxcel run` resolves the model, downloads it on first use, reuses it afterward, and starts an interactive chat REPL. A bare model name resolves under `mlx-community` by default.

```bash
# Interactive chat.
mlxcel run Qwen3.5-0.8B-4bit

# One-shot generation, then exit.
mlxcel run Qwen3.5-0.8B-4bit -p "Hello, world!" -n 100

# With no model argument, use mlx-community/gemma-4-e2b-it-4bit.
mlxcel run
```

`generate`, `serve`, and `inspect` accept the same model forms through `-m`: an existing local path, a HuggingFace `owner/name` repository id, or a bare name. `MLXCEL_DEFAULT_ORG` changes the bare-name organization.

Thinking-capable checkpoints may generate reasoning before the final answer. `run` and `generate` hide reasoning from terminal output by default; pass `--show-reasoning` to display it without the raw marker tokens.

With no `-n/--max-tokens`, generation continues until EOS or the effective context limit. Pass `-n N` to set a cap.

```bash
mlxcel generate -m Qwen3.5-0.8B-4bit -p "Hello, world!" -n 100

# Read-only memory estimate before loading.
mlxcel inspect -m Qwen3.5-0.8B-4bit --max-tokens 32768

# Emit the same estimate as machine-readable bytes for recipe builders or schedulers.
mlxcel inspect --json -m Qwen3.5-0.8B-4bit --max-tokens 32768 | python3 -m json.tool

# Abort before generation if the estimated model and KV cache do not fit.
mlxcel generate -m Qwen3.5-0.8B-4bit -p "Hello" -n 32768 --estimate-memory
```

`mlxcel inspect --json` prints a single JSON object with byte-exact `weights_bytes`, `kv_bytes_total`, `activation_bytes`, `headroom_bytes`, `budget_bytes`, `total_bytes`, `fits`, input flags, and per-token FP16/INT8 KV rates when the model config exposes KV geometry. TurboQuant per-token sizing is reported as `null` until the estimator models those widths directly.

### Start a server

`mlxcel-server` and `mlxcel serve` start the same server. Their server flag surfaces are kept aligned; `mlxcel serve` additionally has the one-shot preflight flags `--estimate-memory` and `--force`.

```bash
mlxcel-server -m Qwen3.5-0.8B-4bit --port 8080

curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.5-0.8B-4bit","messages":[{"role":"user","content":"Hello"}]}'
```

Common deployment controls work on both server entry points:

```bash
# API keys from a list or file; both sources are combined.
mlxcel-server -m Qwen3.5-0.8B-4bit --api-key alice-key,bob-key
mlxcel-server -m Qwen3.5-0.8B-4bit --api-key-file /etc/mlxcel/api-keys

# Restrict browser CORS, or serve HTTPS.
mlxcel-server -m Qwen3.5-0.8B-4bit \
  --allowed-origins https://app.example.com,https://admin.example.com
mlxcel-server -m Qwen3.5-0.8B-4bit \
  --ssl-cert-file cert.pem --ssl-key-file key.pem

# Serve below a path prefix or on a Unix domain socket.
mlxcel-server -m Qwen3.5-0.8B-4bit --api-prefix /llama
mlxcel-server -m Qwen3.5-0.8B-4bit --host /run/mlxcel.sock
```

Downloaded models normally live at `${MLXCEL_CACHE_DIR:-$HOME/.cache/mlxcel}/models/<owner>/<name>`. On `generate`, `run`, `inspect`, `download`, `list`, and `rm`, use `--models-dir` to override that store. On the two server entry points, the store override is `--model-store-root`; `--models-dir` now selects multi-model router discovery and cannot be combined with `-m`.

```bash
# Router mode: discover direct checkpoint subdirectories under /srv/models,
# use the global store as another source, and keep at most four models loaded.
mlxcel-server --models-dir /srv/models \
  --model-store-root /var/lib/mlxcel/models --models-max 4
```

Router management, live LoRA, native routes, runtime settings, context sizing, idle sleep, and compatibility migration details are in [Server features](docs/server-features.md).

### Embed and rerank

The offline commands use the same loaders and batching paths as the HTTP endpoints.

```bash
# Embed two texts and print vectors plus cosine similarity.
mlxcel embed -m sentence-transformers/all-MiniLM-L6-v2 \
  -p "The weather is lovely" -p "It is sunny"

# Rank documents against one query.
mlxcel rerank -m BAAI/bge-reranker-v2-m3 \
  -q "what is panda?" -d "hi" -d "The giant panda is a bear species."

# Serve embeddings alone, or a reranker beside a chat model.
mlxcel-server -m sentence-transformers/all-MiniLM-L6-v2 --port 8080
mlxcel-server -m Qwen3.5-0.8B-4bit \
  --reranker-model mlx-community/Qwen3-Reranker-0.6B-4bit --port 8080
```

See [Embeddings and reranking](docs/embeddings.md) for multimodal inputs, supported families, pooling, request schemas, and side-model serving.

### Transcribe and synthesize audio

An audio-capable chat checkpoint serves transcription. Phi-4 Multimodal and Gemma 3n accept WAV, MP3, and FLAC detected from their bytes; other current audio families accept WAV. A dedicated Whisper checkpoint also provides a WAV-only transcription server.

```bash
mlxcel-server -m models/whisper-base --port 8080

curl http://localhost:8080/v1/audio/transcriptions \
  -F file=@recording.wav -F model=whisper-base
```

Kokoro-82M checkpoints serve `POST /v1/audio/speech`. See [Audio API](docs/audio-api.md) for model layout, request formats, limits, streaming, and current backend constraints.

### Manage downloaded models

```bash
mlxcel list                  # name, size, and modified time
mlxcel list --sort size      # largest first
mlxcel list --json           # stable machine-readable array
mlxcel list -q               # repository ids only
mlxcel list -v               # include absolute paths

mlxcel rm mlx-community/Qwen3.5-0.8B-4bit
mlxcel rm mlx-community/Qwen3.5-0.8B-4bit --yes
```

`mlxcel rm` deletes only from the mlxcel-managed store. A checkpoint that exists solely in the read-only HuggingFace cache is reported but not removed.

### Other CLI tools

```bash
# Object detection with an RT-DETRv2 checkpoint.
mlxcel detect -m models/rt-detr-v2 -i image.jpg --format json

# Inspect the hardware-specific kernel tuning matrix without profiling it.
mlxcel tune --dry-run
```

`mlxcel tune` can profile supported kernel tactics into the local autotune cache; set `MLXCEL_AUTOTUNE=cache` to consume recorded winners. Use `mlxcel --help` or a subcommand's `--help` for the complete flag surface.

### Build from source

Apple Silicon prerequisites are Rust, Xcode Command Line Tools, CMake, and the Metal toolchain component:

```bash
xcodebuild -downloadComponent MetalToolchain
git clone https://github.com/lablup/mlxcel.git
cd mlxcel
cargo build --release --features metal,accelerate
```

Linux/NVIDIA builds require the CUDA toolkit and MLX's CUDA system dependencies:

```bash
cargo build --release --features cuda
```

A plain Linux build has no CUDA feature and runs on the CPU, which is not a validated release target and is much slower. See [Installation](docs/installation.md) for the complete prerequisite, CUDA architecture, runtime-header, and packaging matrix.

## Performance

The last full same-host reference campaign measured faster short-prompt prefill and near-reference decode throughput. These are aggregate results, not guarantees for an individual checkpoint:

| Workload | Host | Reference | Result |
|----------|------|-----------|-------:|
| Text prefill, 67 comparable pairs | M5 Max 128 GB | `mlx-lm` median | **2.78x** |
| Text prefill, 74 comparable pairs | M1 Ultra | `mlx-lm` median | **1.79x** |
| Text decode, 67 comparable pairs | M5 Max 128 GB | `mlx-lm` | **99% average, 100% median** |
| VLM decode, 24 comparable pairs | M5 Max 128 GB | `mlx-vlm` | **98% average, 98% median** |

The multi-host v0.4.0 sweep covers Apple Silicon and GB10 CUDA systems, while focused reports cover paged attention, MoE, KV compression, embeddings, and speculative decoding. Read [Benchmark results](docs/benchmark_results/model_tests.md), the [benchmark report](docs/benchmark_results/benchmark-report.md), and the [methodology](docs/benchmarks.md) before comparing hosts or planning capacity.

MTP performance depends on the target/drafter pair, prompt, verify width, and hardware generation. With the v0.6.0 exactness policy, Gemma 4 12B plus its 4-bit assistant measured 93.2 tok/s against 43.8 tok/s classic decode on M5 Max (**2.13x**, greedy output byte-identical). The faster inexact kernel measured 120.4 tok/s but is not the default. The runtime measures exactness and profitability instead of treating either speedup as universal; see [Speculative-decoding acceptance](docs/speculative-acceptance.md), [MTP policy API](docs/mtp-policy-api.md), and the benchmark records under `docs/benchmark_results/`.

Re-run the supplied harnesses on the target checkpoint, prompt shape, context length, and hardware before treating any published number as a deployment expectation.

## Supported models

Model support is architecture- and checkpoint-dependent. Run:

```bash
mlxcel arch
mlxcel arch --json
```

for the architecture catalog compiled into the current binary. The `--json` form emits the stable recipes registry snapshot used by downstream site builds and automation, including standalone runtime families such as detector-only checkpoints. [Supported models](docs/supported-models.md) maintains the family table, quantization support, distributed coverage, checkpoint qualifications, and known caveats.

## Python

The pure-Python `mlxcel` client manages a local `mlxcel serve` process or connects to an existing server, discovers the served model id, and exposes the underlying OpenAI client for the full API surface.

```python
import mlxcel

with mlxcel.LLM("mlx-community/Qwen3-4B-4bit") as llm:
    print(llm.generate("def fib(n):", max_tokens=128))
    for delta in llm.stream("Write a haiku about autumn"):
        print(delta, end="", flush=True)
```

Install with `pip install ./python`. See [Python client](docs/python-client.md) for managed and connect modes, async use, streaming, structured output, and troubleshooting.

## Optional GUI

`mlxcel-server` can be used directly through HTTP clients. [Backend.AI Go](https://go.backend.ai) is available as a companion UI for local chat, model management, and multi-model routing.

## Documentation

- [Server features and route map](docs/server-features.md)
- [llama-server b10621 compatibility boundary](docs/llama-server-compat.md)
- [Installation and build prerequisites](docs/installation.md)
- [Environment variables](docs/environment-variables.md)
- [Supported models](docs/supported-models.md)
- [Architecture overview](docs/architecture.md)
- [Continuous batching and disaggregated serving](docs/CONTINUOUS_BATCHING.md)
- [Tensor and pipeline parallelism](docs/distributed.md)
- [OpenAI Responses API](docs/responses-api.md)
- [Embeddings and reranking](docs/embeddings.md)
- [Audio API](docs/audio-api.md)
- [TurboQuant KV cache](docs/turbo-kv-cache.md)
- [Speculative-decoding acceptance](docs/speculative-acceptance.md)
- [Adaptive MTP policy API](docs/mtp-policy-api.md)
- [Benchmarks and methodology](docs/benchmarks.md)
- [Python client](docs/python-client.md)
- [Adding a new model](docs/adding-models.md)

The [documentation index](docs/README.md) lists design notes, benchmark records, and specialized implementation guides that do not need to live in this README.

## Contributing

Issues and pull requests are welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the contributor workflow, local quality gates, and commit conventions. For larger changes, open an issue first so the scope and validation plan can be discussed.

For security vulnerabilities, see [`SECURITY.md`](SECURITY.md); do not file them as public issues.

## License

Apache License 2.0 unless otherwise noted. See [LICENSE](LICENSE). Third-party attributions carried forward under Apache-2.0 Section 4(d) are listed in [NOTICE](NOTICE).

## Acknowledgments

- [MLX](https://github.com/ml-explore/mlx), Apple's machine learning framework.
- [mlx-lm](https://github.com/ml-explore/mlx-lm), [mlx-vlm](https://github.com/Blaizzy/mlx-vlm), and [mlx-audio](https://github.com/Blaizzy/mlx-audio), whose model coverage and behavior mlxcel ports and mirrors. See [NOTICE](NOTICE).
- [MLX Community](https://huggingface.co/mlx-community), pre-converted MLX checkpoints.
- [turboquant_plus](https://github.com/TheTom/turboquant_plus), whose TurboQuant KV-cache algorithms are ported under Apache-2.0. See [NOTICE](NOTICE).
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer), whose paged-attention, split-KV, cascade state-merge, and sampling designs inform mlxcel's serving kernels.
