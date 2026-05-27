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
- **Differentiated runtime controls.** Default builds expose first-class YAML load-time model surgery through `--surgery` / `MLXCEL_SURGERY`, with operations such as `scale`, `add`, `prune`, `replace`, and `interpolate` for reproducible weight-space changes without retraining or writing converted checkpoints.
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

The quickest path is `mlxcel run`: it resolves the model argument, auto-downloads
on first use, reuses it afterward, and runs from any directory.

```bash
# Interactive chat REPL.
mlxcel run mlx-community/Qwen3.5-0.8B-4bit

# Bare name resolves to mlx-community/<name>.
mlxcel run Qwen3.5-0.8B-4bit

# One-shot generation with -p, then exit.
mlxcel run Qwen3.5-0.8B-4bit -p "Hello, world!" -n 100

# No model argument falls back to the default
# mlx-community/Llama-3.2-3B-Instruct-4bit (mlx-lm parity).
mlxcel run
```

`generate`, `serve`, and `inspect` take the same model argument via `-m` — a HuggingFace `owner/name` repo-id (auto-downloaded into the store and reused after), a bare name (resolved as `mlx-community/<name>`), or an existing local path. `mlxcel run` is a thin wrapper over `mlxcel generate` and shares its sampling and generation flags.

```bash
# One-off generation.
mlxcel generate -m Qwen3.5-0.8B-4bit -p "Hello, world!" -n 100

# OpenAI-compatible server (mlxcel serve is the subcommand equivalent).
mlxcel-server -m Qwen3.5-0.8B-4bit --port 8080

# Read-only memory budget: weights + KV cache vs. available unified memory.
mlxcel inspect -m Qwen3.5-0.8B-4bit --max-tokens 32768

# Preflight that aborts if the model + 32K KV cache will not fit
# (--force, alias --no-memory-check, overrides the abort).
mlxcel generate -m Qwen3.5-0.8B-4bit -p "Hello, world!" -n 32768 --estimate-memory
```

Downloaded models land in a location-independent global store at `${MLXCEL_CACHE_DIR:-$HOME/.cache/mlxcel}/models/<owner>/<name>`, shared across every working directory. To relocate the store, write a snapshot to an exact path, change the default org, or tune the memory preflight, see [Environment variables](docs/environment-variables.md) — `MLXCEL_MODELS_DIR` / `--models-dir`, `--local-dir`, `MLXCEL_DEFAULT_ORG`, and `MLXCEL_MEMORY_LIMIT` / `MLXCEL_HEADROOM_FACTOR`.

If you build from source instead, use `./target/release/mlxcel` and
`./target/release/mlxcel-server` in place of the installed commands above.

### Manage downloaded models

List and prune the global store from any directory:

```bash
# List downloaded models with their on-disk size and path.
mlxcel list --local

# Remove a model from the global store (prompts for confirmation).
mlxcel rm mlx-community/Qwen3.5-0.8B-4bit

# Remove without the prompt (for scripts / non-interactive shells).
mlxcel rm mlx-community/Qwen3.5-0.8B-4bit --yes
```

`mlxcel list` without `--local` prints the supported model architectures
instead. `mlxcel rm <repo-id>` deletes only inside the mlxcel store and honors
the same `--models-dir` override; a model that exists solely in the read-only
HuggingFace cache (`HF_HUB_CACHE` / `HF_HOME`) is reported but never deleted.

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

mlxcel targets near-`mlx-lm` / `mlx-vlm` decode throughput for MLX-format
checkpoints while keeping a native Rust runtime. In the mlxcel 0.1.0 M5 Max
128GB benchmark set, the headline result has two parts: faster short-prompt
text prefill and near-reference decode throughput.

### Prefill: prompt ingestion before the first generated token

Short-prompt text prefill is the standout result. mlxcel measured **2.70x**
the `mlx-lm` median on M5 Max across 66 comparable text pairs, and **1.76x**
on M1 Ultra across 73 comparable text pairs. VLM prefill is listed separately
because image preprocessing, vision encoder, and projector work can be included
in the prefill path.

| Mode | Baseline | M5 Max pairs | M5 Max median vs baseline | M1 Ultra pairs | M1 Ultra median vs baseline |
|------|----------|-------------:|--------------------------:|---------------:|----------------------------:|
| Text | `mlx-lm` | 66 | **2.70x** | 73 | **1.76x** |
| VLM | `mlx-vlm` | 20 | 0.94x | 17 | **1.33x** |

### Decode: steady-state token generation

Decode stays close to the Python MLX references on the same host. For M5 Max,
text decode averaged **98%** of `mlx-lm` with a **99%** median, while VLM decode
averaged **101%** of `mlx-vlm` with a **100%** median.

| Mode | Baseline | Comparable pairs | Average vs baseline | Median vs baseline | >=90% parity | >= baseline | Range |
|------|----------|-----------------:|--------------------:|-------------------:|-------------:|------------:|------:|
| Text | `mlx-lm` | 66 | 98% | **99%** | 62 / 66 (94%) | 27 / 66 (41%) | 72%-127% |
| VLM | `mlx-vlm` | 20 | 101% | **100%** | 17 / 20 (85%) | 10 / 20 (50%) | 74%-123% |

Representative decode throughput is shown below in tokens per second. M5 Max
reference columns are same-host `mlx-lm` or `mlx-vlm` runs; M1 Ultra values are
included as mlxcel-only capacity references. Absolute results depend on model
family, quantization, prompt shape, decode length, and hardware. See
[Benchmark results](docs/benchmark_results/benchmark-report.md) and
[Benchmarks](docs/benchmarks.md) for methodology and caveats.

| Text model | M1 Ultra mlxcel | M5 Max mlxcel | M5 Max mlx-lm | mlxcel / mlx-lm |
|------------|----------------:|--------------:|--------------:|----------------:|
| SmolLM-135M 4bit | 407 tok/s | 905 tok/s | 712 tok/s | 127% |
| Llama 3.1 8B 4bit | 107 tok/s | 117 tok/s | 117 tok/s | 99% |
| Qwen2.5 7B 4bit | 110 tok/s | 126 tok/s | 124 tok/s | 102% |
| Gemma 2B 4bit | 190 tok/s | 217 tok/s | 223 tok/s | 97% |
| Gemma 3 4B 4bit | 114 tok/s | 182 tok/s | 182 tok/s | 100% |
| Gemma 4 26B-A4B 4bit | 73 tok/s | 137 tok/s | 141 tok/s | 97% |
| Qwen3 MoE 30B 4bit | 71 tok/s | 156 tok/s | 147 tok/s | 106% |
| GLM-4 Flash 4bit | 47 tok/s | 104 tok/s | 104 tok/s | 100% |
| Nemotron-H 30B 4bit | 90 tok/s | 177 tok/s | 179 tok/s | 99% |
| Mixtral 8x7B 4bit | 54 tok/s | 65 tok/s | 66 tok/s | 99% |
| StarCoder2 3B 4bit | 171 tok/s | 216 tok/s | 215 tok/s | 101% |
| Qwen3.5 0.8B 4bit | 243 tok/s | 517 tok/s | 545 tok/s | 95% |
| Qwen3-VL 30B-A3B 4bit, text path | 70 tok/s | 151 tok/s | 147 tok/s | 103% |
| Qwen3-VL 32B 4bit, text path | 21 tok/s | 28 tok/s | 29 tok/s | 96% |
| GPT-OSS 120B 4bit | 59 tok/s | 114 tok/s | 110 tok/s | 103% |
| Solar Open 100B 4bit | 36 tok/s | 65 tok/s | 66 tok/s | 99% |

| VLM model | M1 Ultra mlxcel | M5 Max mlxcel | M5 Max mlx-vlm | mlxcel / mlx-vlm |
|-----------|----------------:|--------------:|---------------:|-----------------:|
| LLaVA Interleave Qwen 0.5B bf16 | 270 tok/s | 344 tok/s | 345 tok/s | 100% |
| Qwen3.5 0.8B 4bit | 202 tok/s | 506 tok/s | 411 tok/s | 123% |
| Qwen3.5 35B-A3B 4bit | 71 tok/s | 151 tok/s | 129 tok/s | 117% |
| Gemma 4 E2B 4bit | 107 tok/s | 217 tok/s | 202 tok/s | 108% |
| Gemma 3n E2B 4bit | 72 tok/s | 151 tok/s | 125 tok/s | 121% |
| Gemma 4 26B-A4B 4bit | 63 tok/s | 134 tok/s | 137 tok/s | 98% |
| Molmo2 4B | 59 tok/s | 64 tok/s | 67 tok/s | 96% |
| Phi 3.5 Vision 4bit | 94 tok/s | 123 tok/s | 160 tok/s | 77% |

The M5 Max sweep covers 98 text model directories and a matching 98-entry VLM
mode pass. Ratio summaries include only rows where both mlxcel and the Python
reference produced comparable decode measurements; unsupported checkpoints and
benchmark-configuration failures are tracked in the benchmark notes. VLM rows
should be read separately because vision preprocessing, processor setup, and
prompt construction differ by family. Re-run the benchmark suite on your target
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
