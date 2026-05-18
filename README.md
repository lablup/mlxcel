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

The current M5 Max reference snapshot is from the 2026-05-18 full sweep, with
Qwen3-VL and several targeted rows refreshed on 2026-05-19. It used mlxcel
0.0.28, MLX 0.31.2, mlx-lm 0.31.3 (`ed1fca4`), and mlx-vlm 0.4.4 on a 128GB
MacBook Pro M5 Max. Against `mlx-lm`, 67 text pairs were comparable: average
mlxcel/mlx-lm decode throughput was **95%** (median 97%, range 71%-129%),
**50 / 67** rows reached at least 90% parity, and **16 / 67** were at or above
`mlx-lm`. The comparable VLM set averaged **94%** of `mlx-vlm` (median 95%,
range 76%-107%), with **12 / 17** rows at or above 90% parity.

Recent hot-path work targeted rows that had fallen well behind the Python
references. These M5 Max spot checks show the post-fix decode rates for the
affected families. Values are decode tok/s; the ratio is `mlxcel / reference`.

| Refreshed row | mlxcel | mlx-lm | Ratio |
|---------------|-------:|-------:|------:|
| GPT-OSS 120B 4bit | 113 tok/s | 110 tok/s | 102% |
| GLM-4 Flash 4bit | 111 tok/s | 108 tok/s | 103% |
| Solar Open 100B 4bit | 66 tok/s | 66 tok/s | 99% |
| Gemma 2B 4bit | 214 tok/s | 223 tok/s | 96% |
| Falcon Mamba 7B 4bit | 94 tok/s | 94 tok/s | 100% |
| StarCoder2 3B 4bit | 215 tok/s | 214 tok/s | 100% |
| Qwen3.5 0.8B 4bit | 535 tok/s | 555 tok/s | 96% |
| Qwen3-VL 30B-A3B 4bit, text path | 146 tok/s | 148 tok/s | 99% |
| Qwen3-VL 32B 4bit, text path | 27 tok/s | 29 tok/s | 94% |

Current Apple Silicon capacity snapshots are selected rows from full-suite
`mlxcel generate --profile` runs, with the 2026-05-19 targeted refreshes used
where applicable. The M1 Ultra column is mlxcel-only; M5 Max columns include
the same-host Python reference when available. `FAIL` in a reference column
means the reference runtime failed to load or run that checkpoint under the
recorded configuration.

| Text model | M1 Ultra mlxcel<br>2026-05-08 | M5 Max mlxcel<br>2026-05-18/19 | M5 Max mlx-lm | mlxcel / mlx-lm |
|------------|------------------------------:|-------------------------------:|--------------:|----------------:|
| SmolLM-135M 4bit | 365 tok/s | 919 tok/s | 712 tok/s | 129% |
| Llama 3.1 8B 4bit | 109 tok/s | 113 tok/s | 117 tok/s | 96% |
| Qwen2.5 7B 4bit | 113 tok/s | 126 tok/s | 124 tok/s | 102% |
| Gemma 2B 4bit | 82 tok/s | 214 tok/s | 223 tok/s | 96% |
| Gemma 3 4B 4bit | 104 tok/s | 145 tok/s | 182 tok/s | 80% |
| Gemma 4 E2B 4bit | 124 tok/s | 222 tok/s | FAIL | - |
| Gemma 4 26B-A4B 4bit | 72 tok/s | 136 tok/s | 141 tok/s | 97% |
| Qwen3 MoE 30B 4bit | 72 tok/s | 153 tok/s | 147 tok/s | 104% |
| GLM-4 Flash 4bit | 36 tok/s | 111 tok/s | 108 tok/s | 103% |
| Nemotron-H 30B 4bit | 90 tok/s | 156 tok/s | 179 tok/s | 87% |
| Mixtral 8x7B 4bit | 54 tok/s | 63 tok/s | 66 tok/s | 96% |
| StarCoder2 3B 4bit | 105 tok/s | 215 tok/s | 214 tok/s | 100% |
| Qwen3.5 0.8B 4bit | 183 tok/s | 535 tok/s | 555 tok/s | 96% |
| Llama 4 Scout 17B 4bit | 37 tok/s | 48 tok/s | FAIL | - |
| GPT-OSS 120B 4bit | 20 tok/s | 113 tok/s | 110 tok/s | 102% |
| Solar Open 100B 4bit | 14 tok/s | 66 tok/s | 66 tok/s | 99% |

| VLM model | M1 Ultra mlxcel<br>2026-05-08 | M5 Max mlxcel<br>2026-05-18 | M5 Max mlx-vlm | mlxcel / mlx-vlm |
|-----------|------------------------------:|----------------------------:|---------------:|-----------------:|
| LLaVA Interleave Qwen 0.5B bf16 | 262 tok/s | 342 tok/s | 345 tok/s | 99% |
| Qwen3-VL 2B 4bit | 158 tok/s | 276 tok/s | FAIL | - |
| Qwen3-VL 30B-A3B 4bit | 22 tok/s | 35 tok/s | FAIL | - |
| Qwen3-VL 32B 4bit | 18 tok/s | 20 tok/s | FAIL | - |
| Gemma 4 E2B 4bit | 106 tok/s | 216 tok/s | 202 tok/s | 107% |
| Gemma 4 26B-A4B 4bit | 66 tok/s | 129 tok/s | 137 tok/s | 94% |
| Phi 3.5 Vision 4bit | 88 tok/s | 121 tok/s | 160 tok/s | 76% |
| Llama 4 Scout 17B 4bit | 35 tok/s | 48 tok/s | FAIL | - |

The 2026-05-18 M5 Max sweep tested 98 text model directories with 89 pass, 4
partial, and 5 fail results, plus a separate 98-entry VLM prompt pass. VLM runs
used a 224x224 benchmark fixture image and should be read separately because
vision preprocessing, image size, and prompt construction differ by family.
Re-run the benchmark suite on your target hardware before using these numbers
for capacity planning.

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
