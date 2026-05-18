# mlxcel

[![License: Apache 2.0](https://img.shields.io/github/license/lablup/mlxcel)](LICENSE)
[![Latest Release](https://img.shields.io/github/v/release/lablup/mlxcel)](https://github.com/lablup/mlxcel/releases/latest)
[![CI](https://github.com/lablup/mlxcel/actions/workflows/ci.yml/badge.svg)](https://github.com/lablup/mlxcel/actions/workflows/ci.yml)

High-performance LLM/VLM inference runtime and server for Apple Silicon. The CLI and server are implemented in Rust and execute models through native MLX C++ bindings. Linux/CUDA builds are supported as a secondary target.

## Overview

`mlxcel` provides a Rust command-line runtime and an OpenAI-compatible model server for MLX-format checkpoints. Loading, scheduling, and inference stay in one native process while model execution goes through MLX C++ bindings. The project tracks the model coverage of [mlx-lm](https://github.com/ml-explore/mlx-lm) and [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) where practical.

The project started as work on structural model fine-tuning and has grown into a general-purpose serving runtime for local and small-cluster inference.

## Why mlxcel

- **Smaller runtime surface.** Model loading, scheduling, and inference stay in a single native server process. Deployments do not need to provision a Python environment, keep package versions in sync, or route requests through an interpreter layer.
- **Simple deployment artifact.** `mlxcel` and `mlxcel-server` build as native executables, which makes packaging, service supervision, and upgrades straightforward. Platform runtime libraries are still required: for example macOS frameworks on Apple Silicon, and CUDA/OpenBLAS/LAPACK components for Linux builds.
- **`llama-server`-style operation.** `mlxcel-server` accepts many `llama-server`-compatible flags and `LLAMA_ARG_*` environment variables, which makes migration from llama.cpp-based scripts simpler. Treat this as compatibility-oriented, not a guarantee that every llama.cpp option has identical behavior.
- **OpenAI-compatible HTTP API subset.** The server supports SSE streaming and the `/v1/chat/completions`, `/v1/completions`, and `/v1/responses` endpoints.
- **Serving features for real deployments.** Continuous batching, prompt-prefix caching, automatic prefix caching, speculative decoding, and KV-cache compression are available for supported model/runtime combinations.
- **Multi-device and distributed modes.** Tensor parallelism and pipeline parallelism are implemented for selected model families, including zero-config pipeline startup with static or mDNS-based discovery.
- **Broad model-family coverage.** The runtime includes loaders for Llama, Qwen, Gemma, Phi, Mistral/Mixtral, DeepSeek, Cohere, InternLM, GLM, ExaOne, OLMo, ERNIE, Hunyuan, Mamba/RWKV/Jamba, Nemotron, MiniMax, Step, Kimi, and multiple VLM families. See [Supported models](docs/supported-models.md) for the maintained list.

## Quick start

### Install with Homebrew (macOS/Linux)

The Homebrew formula installs both `mlxcel` and `mlxcel-server`:

```bash
brew tap lablup/tap
brew install mlxcel
```

### Run a model

```bash
# Download an MLX-format checkpoint from Hugging Face.
mlxcel download mlx-community/Qwen3.5-0.8B-OptiQ-4bit

# One-off generation.
mlxcel generate \
    -m models/Qwen3.5-0.8B-OptiQ-4bit \
    -p "Hello, world!" -n 100

# OpenAI-compatible server.
mlxcel-server \
    -m models/Qwen3.5-0.8B-OptiQ-4bit \
    --port 8080
```

If you build from source instead, use `./target/release/mlxcel` and
`./target/release/mlxcel-server` in place of the installed commands above.

### Build from source on Apple Silicon

Prerequisites:

- Rust toolchain
- Xcode Command Line Tools
- CMake-compatible build environment
- Apple Metal toolchain component

```bash
xcodebuild -downloadComponent MetalToolchain   # one-time, if not already installed
git clone https://github.com/lablup/mlxcel.git
cd mlxcel
cargo build --release --features metal,accelerate
```

Linux/CUDA builds use the `cuda` feature and require the CUDA toolkit plus the system libraries used by MLX. See [Installation](docs/installation.md) for the detailed prerequisite matrix.

## Performance

Benchmark results depend on model family, quantization, prompt length, decode
length, batch shape, and hardware. See [Benchmarks](docs/benchmarks.md) for
methodology notes and caveats.

As a reference-runtime comparison, a 2026-05-08 run on Mac Studio M1 Ultra
compared 37 4-bit text checkpoints against `mlx-lm` using mlxcel 0.0.25, MLX
0.31.2, and mlx-lm 0.31.3. Decode throughput averaged about **1.19x** of the
mlx-lm baseline; **35 / 37** text models were faster than `mlx-lm`, and all 37
stayed within 90% of parity. In the same M1 Ultra benchmark set, 5 / 6
comparable VLMs were at or above `mlx-vlm` decode parity.

Current Apple Silicon capacity snapshots are shown below as absolute decode
throughput, not as cross-runtime comparisons. Numbers are selected rows from
full-suite `mlxcel generate --profile` runs.

| Text model | M1 Ultra 128GB<br>2026-05-08 | M5 Max 128GB<br>2026-05-18 |
|------------|-----------------------------:|---------------------------:|
| SmolLM-135M 4bit | 365 tok/s | 919 tok/s |
| Llama 3.1 8B 4bit | 109 tok/s | 113 tok/s |
| Qwen2.5 7B 4bit | 113 tok/s | 126 tok/s |
| Gemma 3 4B 4bit | 104 tok/s | 145 tok/s |
| Gemma 4 E2B 4bit | 124 tok/s | 222 tok/s |
| Gemma 4 26B-A4B 4bit | 72 tok/s | 136 tok/s |
| Qwen3 MoE 30B 4bit | 72 tok/s | 152 tok/s |
| Nemotron-H 30B 4bit | 90 tok/s | 156 tok/s |
| Mixtral 8x7B 4bit | 54 tok/s | 63 tok/s |
| Llama 4 Scout 17B 4bit | 37 tok/s | 48 tok/s |
| Solar Open 100B 4bit | 14 tok/s | 16 tok/s |

| VLM model | M1 Ultra 128GB<br>2026-05-08 | M5 Max 128GB<br>2026-05-18 |
|-----------|-----------------------------:|---------------------------:|
| LLaVA Interleave Qwen 0.5B bf16 | 262 tok/s | 342 tok/s |
| Qwen3-VL 2B 4bit | 158 tok/s | 276 tok/s |
| Gemma 4 E2B 4bit | 106 tok/s | 216 tok/s |
| Gemma 4 26B-A4B 4bit | 66 tok/s | 129 tok/s |
| Phi 3.5 Vision 4bit | 88 tok/s | 121 tok/s |
| Llama 4 Scout 17B 4bit | 35 tok/s | 48 tok/s |

The 2026-05-18 M5 Max sweep tested 98 text model directories with 89 pass, 4
partial, and 5 fail results; 62 of 93 numeric text runs reached at least 100
decode tok/s. VLM runs used a 224x224 benchmark fixture image and should be
read separately because vision preprocessing, image size, and prompt
construction differ by family. Re-run the benchmark suite on your target
hardware before using these numbers for capacity planning.

## Supported models

Model support is architecture- and checkpoint-dependent. Run:

```bash
mlxcel list
```

for the CLI summary, and see [Supported models](docs/supported-models.md) for the maintained architecture table, known limitations, and VLM coverage notes.

## Optional GUI

`mlxcel-server` can be used directly through HTTP clients. For a local graphical front-end, [Backend.AI Go](https://go.backend.ai) can be used as a companion UI for chat, model management, and multi-model routing.

## Documentation

- [Installation](docs/installation.md)
- [Environment variables](docs/environment-variables.md)
- [Benchmarks](docs/benchmarks.md)
- [Supported models](docs/supported-models.md)
- [Architecture overview](docs/architecture.md)
- [Tensor and pipeline parallelism](docs/distributed.md)
- [TurboQuant KV cache](docs/turbo-kv-cache.md)
- [OpenAI Responses API](docs/responses-api.md)
- [Adding a new model](docs/adding-models.md)

## Contributing

Issues and pull requests are welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the contributor workflow, local quality gates (`cargo fmt`, `clippy`, `cargo test`, `cargo deny check`), and commit conventions. New model architectures, performance work, bug fixes, and documentation improvements are all useful. For larger changes, please open an issue first so the scope and validation plan can be discussed.

For security vulnerabilities, see [`SECURITY.md`](SECURITY.md) — do **not** file these as public issues.

## License

Apache License 2.0 unless otherwise noted — see [LICENSE](LICENSE).

## Acknowledgments

- [MLX](https://github.com/ml-explore/mlx) — Apple's machine learning framework
- [mlx-lm](https://github.com/ml-explore/mlx-lm) and [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) — Python projects that guide model-family compatibility
- [MLX Community](https://huggingface.co/mlx-community) — pre-converted MLX model checkpoints
