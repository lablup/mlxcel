# mlxcel

[![License: Apache 2.0](https://img.shields.io/github/license/lablup/mlxcel)](LICENSE)
[![Latest Release](https://img.shields.io/github/v/release/lablup/mlxcel)](https://github.com/lablup/mlxcel/releases/latest)
[![CI](https://github.com/lablup/mlxcel/actions/workflows/ci.yml/badge.svg)](https://github.com/lablup/mlxcel/actions/workflows/ci.yml)

High-performance LLM/VLM inference runtime and server for Apple Silicon. The CLI and server are implemented in Rust and execute models through native MLX C++ bindings. Linux/CUDA builds are supported as a secondary target.

## New in v0.4.2

- **Three more model families.** MiniMax-M3, a hybrid dense/MoE text model with block-sparse attention (#799); MiniMax-M3-VL multimodal (#800); and Unlimited-OCR, whose decode runs against a ring sliding KV cache so a long document does not grow the cache unbounded past the window (#801).
- **XTC sampling.** Exclude Top Choices sampling (`xtc_probability`, `xtc_threshold`) is read and applied end to end on the OpenAI-compatible chat, completions, and responses routes; out-of-range values return a 400 (#802).
- **DeepSeek-V2 is correct on CUDA again.** An upstream MLX 0.32.1 RMSNorm kernel regression turned DeepSeek-V2-Lite generation into repeated tokens on GB10. The affected kernel is overlaid with its last-good version and CUDA graph capture is disabled for the family (as it already is for Gemma 4), so output is coherent again (#829).
- **Empty-input requests are rejected up front.** A request whose effective input is empty or whitespace-only returns a 400 before model dispatch, on `/v1/chat/completions`, `/v1/completions`, and `/v1/messages` (#803, #813, #814).
- **The server survives more backend faults.** An MLX evaluation throw in the decode loop fails the affected request instead of aborting the worker (#825), and CUDA builds raise the MLX graph-cache default so long-lived speculative serving no longer hits the fatal "Cache thrashing" abort (#818).
- **Gemma 4 audio transcription fix.** The CLI renders the audio placeholder after the prompt text, so the 12B unified model transcribes acoustically hard clips instead of answering their perceived content (#798).

## New in v0.4

- **Experimental OpenXLA / IREE backend (opt-in).** A second forward-execution engine built on a Rust-native StableHLO emitter and the IREE runtime, selectable with `MLXCEL_BACKEND=xla` behind the `xla-backend` / `xla-iree` build features. It runs on Metal and CUDA and serves through a continuous-batching engine. Default builds do not compile it, and the MLX path is unchanged.
- **Over 20 new vision-language and OCR families.** Qwen3-Omni (with talker speech output), Llama 3.2 Vision, GLM-4V and GLM-4V MoE, Hunyuan-VL, ERNIE-4.5 MoE VL, DeepSeek-VL2, Kimi-VL, FastVLM, Moondream2, Idefics2, SmolVLM, LFM2-VL, and Granite Vision, plus the OCR set DeepSeek-OCR and DeepSeek-OCR 2, dots.ocr, GLM-OCR, and PaddleOCR-VL. The set kept growing after 0.4.0: Kimi-VL video, Step-3, and Command MoE (Cohere2 MoE) in 0.4.1, then MiniMax-M3, MiniMax-M3-VL, and Unlimited-OCR in 0.4.2.
- **Tool calling across model families.** Server tool-call parsers cover Kimi K2, the pythonic `[func(arg=value)]` form, function-calling Gemma, MiniMax-M3, GLM-4.7, LongCat, and the bracketed Mistral format, added in 0.4.1.
- **Batching on by default.** `mlxcel-server` and `mlxcel serve` default to batched decode (`--parallel 4`), batched prefill (`--max-batch-prefill 4`), and the prompt-prefix cache, guarded by an automatic KV-cache budget. On M1 Ultra, 4 concurrent clients reach 1.90x the single-client aggregate throughput and about 17x lower time-to-first-token, with single-client speed unchanged. Restore the old behavior with `--parallel 1 --no-batch --no-prompt-cache`.
- **CUDA / GB10 kernel parity.** Native paged-attention decode, fused SSM decode, and MoE prefill (sorted grouped GEMM) kernels are ported to CUDA, alongside a Blackwell (sm_120/121) quantized-matmul tile and a single-dtype decode graph.
- **Speculative decoding overhaul.** Tick-cooperative scheduling removes the burst head-of-line block, and the MTP accept/decline policy is set from measured round cost.
- **NVFP4 (Blackwell).** ModelOpt NVFP4 checkpoints transcode directly to the native MLX layout, with Metal defaulting to the native path.
- **New attention paths.** DeepSeek-V3.2 / GLM-MoE DSA lightning indexer, phi3-small blocksparse attention, and qwen3-next pipeline-parallel stages.
- **Server hardening.** In 0.4.2, requests with no effective input are rejected before dispatch, a decode-loop MLX throw fails the request instead of the worker, and long-lived CUDA speculative serving no longer aborts on the MLX graph-cache limit.
- **Interrupted downloads recover.** A partial model snapshot is detected against its own weight index and re-fetched at load instead of failing with a bare `Weight not found`.

See the [changelog](CHANGELOG.md) for the full list.

## Overview

`mlxcel` provides a Rust command-line runtime and an OpenAI-compatible model server for MLX-format checkpoints. Loading, scheduling, and inference stay in one native process while model execution goes through MLX C++ bindings. It runs a broad range of text and vision-language model families directly from [mlx-community](https://huggingface.co/mlx-community) checkpoints, with no conversion step.

The project started as work on structural model fine-tuning and has grown into a general-purpose serving runtime for local and small-cluster inference.

## Why mlxcel

- **Smaller runtime surface.** Model loading, scheduling, and inference stay in a single native server process. Deployments do not need to provision a Python environment, keep package versions in sync, or route requests through an interpreter layer.
- **Simple deployment artifact.** `mlxcel` and `mlxcel-server` build as native executables, which makes packaging, service supervision, and upgrades straightforward. Platform runtime libraries are still required: for example macOS frameworks on Apple Silicon, and CUDA/OpenBLAS/LAPACK components for Linux builds.
- **`llama-server`-style operation.** `mlxcel-server` accepts many `llama-server`-compatible flags and `LLAMA_ARG_*` environment variables, which makes migration from llama.cpp-based scripts simpler. Treat this as compatibility-oriented, not a guarantee that every llama.cpp option has identical behavior.
- **OpenAI-compatible HTTP API subset.** The server supports SSE streaming and the `/v1/chat/completions`, `/v1/completions`, and `/v1/responses` endpoints.
- **Serving features for real deployments.** Continuous batching, prompt-prefix caching, and automatic prefix caching are on by default; speculative decoding and KV-cache compression are available for supported model/runtime combinations.
- **Differentiated runtime controls.** Default builds expose first-class YAML load-time model surgery through `--surgery` / `MLXCEL_SURGERY`, with operations such as `scale`, `add`, `prune`, `replace`, and `interpolate` for reproducible weight-space changes without retraining or writing converted checkpoints.
- **Multi-device and distributed modes.** Tensor parallelism and pipeline parallelism are implemented for selected model families, including zero-config pipeline startup with static or mDNS-based discovery.
- **Broad model-family coverage.** The runtime includes loaders for Llama, Qwen, Gemma, Phi, Mistral/Mixtral, DeepSeek, Cohere, InternLM, GLM, ExaOne, OLMo, ERNIE, Hunyuan, Mamba/RWKV/Jamba, Nemotron, MiniMax, Step, and Kimi, plus a broad vision-language and OCR set (Qwen3-Omni, GLM-4V, Llama 3.2 Vision, Hunyuan-VL, ERNIE-4.5 VL, DeepSeek-VL2, DeepSeek-OCR, PaddleOCR-VL, and more). See [Supported models](docs/supported-models.md) for the maintained list.

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
# mlx-community/gemma-4-e2b-it-4bit.
mlxcel run
```

`generate`, `serve`, and `inspect` take the same model argument via `-m`, a HuggingFace `owner/name` repo-id (auto-downloaded into the store and reused after), a bare name (resolved as `mlx-community/<name>`), or an existing local path. `mlxcel run` is a thin wrapper over `mlxcel generate` and shares its sampling and generation flags.

The default model, and other thinking-capable checkpoints such as Qwen-style `<think>` models, write a chain-of-thought before the final answer. `generate` and `run` hide it from the terminal by default so only the answer prints; pass `--show-reasoning` to also print the reasoning, dimmed on a terminal. The raw `<|channel>thought` / `<channel|>` or `<think>` / `</think>` markers never print either way.

Output length follows llama.cpp: with no `-n/--max-tokens` (default `-1`), `generate` / `run` keep generating until the model emits an end-of-sequence token or fills its context window. The server's `--n-predict` default (`-1`) behaves the same per request. Pass an explicit `-n N` (or `--n-predict N`) to cap output at exactly `N` tokens.

```bash
# One-off generation (omit -n to run until EOS / context window; -n N caps it).
mlxcel generate -m Qwen3.5-0.8B-4bit -p "Hello, world!" -n 100

# OpenAI-compatible server (mlxcel serve is the subcommand equivalent).
mlxcel-server -m Qwen3.5-0.8B-4bit --port 8080

# Restrict browser CORS to specific origins (default reflects any origin).
mlxcel-server -m Qwen3.5-0.8B-4bit --port 8080 --allowed-origins https://app.example.com,https://admin.example.com

# Read-only memory budget: weights + KV cache vs. available unified memory.
mlxcel inspect -m Qwen3.5-0.8B-4bit --max-tokens 32768

# Preflight that aborts if the model + 32K KV cache will not fit
# (--force, alias --no-memory-check, overrides the abort).
mlxcel generate -m Qwen3.5-0.8B-4bit -p "Hello, world!" -n 32768 --estimate-memory
```

`mlxcel-server` mirrors `mlxcel serve` flag for flag, including the two speculative-decoding flags whose primary spelling differs by convention: the drafter checkpoint path (`--draft-model` on `mlxcel serve`, mlx-lm style; `--model-draft` on `mlxcel-server`, llama-server style) and the per-step draft-token budget (`--draft-max` on `mlxcel serve`; `--draft` on `mlxcel-server`). Both binaries accept both spellings as aliases, so a speculative-decoding command line built for one runs unchanged on the other, for example `--draft-model <path> --draft-kind mtp` also works as `--model-draft <path> --draft-kind mtp` on `mlxcel-server`. `--draft-kind` and `--draft-block-size` already share one spelling everywhere.

Downloaded models land in a location-independent global store at `${MLXCEL_CACHE_DIR:-$HOME/.cache/mlxcel}/models/<owner>/<name>`, shared across every working directory. To relocate the store, write a snapshot to an exact path, change the default org, or tune the memory preflight, see [Environment variables](docs/environment-variables.md), `MLXCEL_MODELS_DIR` / `--models-dir`, `--local-dir`, `MLXCEL_DEFAULT_ORG`, and `MLXCEL_MEMORY_LIMIT` / `MLXCEL_HEADROOM_FACTOR`.

If you build from source instead, use `./target/release/mlxcel` and
`./target/release/mlxcel-server` in place of the installed commands above.

### Manage downloaded models

List and prune the global store from any directory:

```bash
# List downloaded models with name, size, and last-modified time.
mlxcel list

# Machine-readable output (stable JSON array: repo_id, size_bytes, path, modified).
mlxcel list --json

# Repo-ids only, pipe-friendly for scripting (e.g. xargs mlxcel rm).
mlxcel list -q

# Restore the absolute path column.
mlxcel list -v

# Remove a model from the global store (prompts for confirmation).
mlxcel rm mlx-community/Qwen3.5-0.8B-4bit

# Remove without the prompt (for scripts / non-interactive shells).
mlxcel rm mlx-community/Qwen3.5-0.8B-4bit --yes
```

`mlxcel arch` prints the supported model-architecture catalog instead. `mlxcel
rm <repo-id>` deletes only inside the mlxcel store and honors the same
`--models-dir` override; a model that exists solely in the read-only HuggingFace
cache (`HF_HUB_CACHE` / `HF_HOME`) is reported but never deleted.

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

Linux/CUDA builds use the `cuda` feature and require the CUDA toolkit plus the system libraries used by MLX. A plain `cargo build --release` on Linux omits the `cuda` feature and produces a CPU-only binary that still runs but silently executes MLX on the CPU at a fraction of GPU throughput, so always pass `--features cuda` on an NVIDIA host. See [Installation](docs/installation.md) for the detailed prerequisite matrix.

## Performance

mlxcel targets near-`mlx-lm` / `mlx-vlm` decode throughput for MLX-format
checkpoints while keeping a native Rust runtime. In the M5 Max 128GB benchmark
campaign, the headline result has two parts: faster short-prompt text prefill
and near-reference decode throughput.

### Prefill: prompt ingestion before the first generated token

Short-prompt text prefill is the standout result. mlxcel measured **2.78x**
the `mlx-lm` median on M5 Max across 67 comparable text pairs, and **1.79x**
on M1 Ultra across 74 comparable text pairs. VLM prefill is listed separately
because image preprocessing, vision encoder, and projector work can be included
in the prefill path.

| Mode | Baseline | M5 Max pairs | M5 Max median vs baseline | M1 Ultra pairs | M1 Ultra median vs baseline |
|------|----------|-------------:|--------------------------:|---------------:|----------------------------:|
| Text | `mlx-lm` | 67 | **2.78x** | 74 | **1.79x** |
| VLM | `mlx-vlm` | 25 | **1.01x** | 20 | **1.05x** |

### Decode: steady-state token generation

Decode stays close to the Python MLX references on the same host. For M5 Max,
text decode averaged **99%** of `mlx-lm` with a **100%** median, while VLM decode
averaged **98%** of `mlx-vlm` with a **98%** median.

| Mode | Baseline | Comparable pairs | Average vs baseline | Median vs baseline | >=90% parity | >= baseline | Range |
|------|----------|-----------------:|--------------------:|-------------------:|-------------:|------------:|------:|
| Text | `mlx-lm` | 67 | 99% | **100%** | 62 / 67 (93%) | 31 / 67 (46%) | 45%-129% |
| VLM | `mlx-vlm` | 24 | 98% | **98%** | 18 / 24 (75%) | 10 / 24 (42%) | 59%-121% |

Representative decode throughput is shown below in tokens per second. The
mlxcel columns are the 2026-06-15 v0.3.0 sweep on each host. The 0.4.0 sweep on
2026-07-12 (MLX 0.32.1 pin `57c66cac`, `--cooldown 30`) re-measured every model
on M1 Ultra, M5 Max, and GB10 and closely tracks these figures: MoE families are
faster after the fused decode-MoE wiring, and a few small fast models read a few
percent lower under the cooldown-30 thermal protocol. The M5 Max `mlx-lm` /
`mlx-vlm` reference columns are retained from the same-host campaign and were not
re-run at 0.4.0, so each ratio is mlxcel over that retained reference and the
prefill and decode parity summaries above are the last full same-host comparison.
M1 Ultra values are mlxcel-only capacity references. Per-model 0.4.0 numbers for
all three hardware targets, including GB10 (DGX Spark) CUDA decode, are in
[Benchmark results](docs/benchmark_results/model_tests.md). Mixtral 8x7B stays on
the gather path via the expert-size guard, so its figures are unchanged. Absolute
results depend on model family, quantization, prompt shape, decode length, and
hardware. See
[Benchmark report](docs/benchmark_results/benchmark-report.md) and
[Benchmarks](docs/benchmarks.md) for methodology and caveats.

| Text model | M1 Ultra mlxcel | M5 Max mlxcel | M5 Max mlx-lm | mlxcel / mlx-lm |
|------------|----------------:|--------------:|--------------:|----------------:|
| SmolLM-135M 4bit | 375 tok/s | 917 tok/s | 712 tok/s | 129% |
| Llama 3.1 8B 4bit | 108 tok/s | 117 tok/s | 117 tok/s | 100% |
| Qwen2.5 7B 4bit | 113 tok/s | 126 tok/s | 124 tok/s | 102% |
| Gemma 2B 4bit | 196 tok/s | 215 tok/s | 223 tok/s | 96% |
| Gemma 3 4B 4bit | 117 tok/s | 183 tok/s | 182 tok/s | 101% |
| Gemma 2 2B 4bit | 166 tok/s | 241 tok/s | 242 tok/s | 100% |
| Phi-3.5-mini 4bit | 164 tok/s | 203 tok/s | 208 tok/s | 98% |
| Jamba v0.1 4bit (hybrid SSM) | 122 tok/s | 216 tok/s | 219 tok/s | 99% |
| Gemma 4 26B-A4B 4bit | 80 tok/s | 151 tok/s | 141 tok/s | 107% |
| Qwen3 MoE 30B 4bit | 84 tok/s | 176 tok/s | 147 tok/s | 120% |
| GLM-4 Flash 4bit | 46 tok/s | 104 tok/s | 104 tok/s | 100% |
| Nemotron-H 30B 4bit | 92 tok/s | 176 tok/s | 179 tok/s | 98% |
| Mixtral 8x7B 4bit | 54 tok/s | 65 tok/s | 66 tok/s | 98% |
| StarCoder2 3B 4bit | 166 tok/s | 216 tok/s | 215 tok/s | 100% |
| Qwen3.5 0.8B 4bit | 230 tok/s | 504 tok/s | 545 tok/s | 92% |
| Qwen3-VL 30B-A3B 4bit, text path | 82 tok/s | 151 tok/s | 147 tok/s | 103% |
| Qwen3-VL 32B 4bit, text path | 21 tok/s | 27 tok/s | 29 tok/s | 93% |
| GPT-OSS 120B 4bit | 58 tok/s | 114 tok/s | 110 tok/s | 104% |
| Solar Open 100B 4bit | 33 tok/s | 65 tok/s | 66 tok/s | 98% |

| VLM model | M1 Ultra mlxcel | M5 Max mlxcel | M5 Max mlx-vlm | mlxcel / mlx-vlm |
|-----------|----------------:|--------------:|---------------:|-----------------:|
| LLaVA Interleave Qwen 0.5B bf16 | 265 tok/s | 341 tok/s | 345 tok/s | 99% |
| Qwen3.5 0.8B 4bit | 232 tok/s | 454 tok/s | 411 tok/s | 110% |
| Qwen3.5 35B-A3B 4bit | 75 tok/s | 149 tok/s | 129 tok/s | 116% |
| Gemma 4 E2B 4bit | 106 tok/s | 220 tok/s | 202 tok/s | 109% |
| Gemma 3n E2B 4bit | 73 tok/s | 151 tok/s | 125 tok/s | 121% |
| InternVL3 1B | 238 tok/s | 575 tok/s | 529 tok/s | 109% |
| Gemma 4 26B-A4B 4bit | 70 tok/s | 144 tok/s | 137 tok/s | 105% |
| Molmo2 4B | 60 tok/s | 64 tok/s | 67 tok/s | 96% |
| Phi 3.5 Vision 4bit | 122 tok/s | 168 tok/s | 160 tok/s | 105% |

### DiffusionGemma (block diffusion)

DiffusionGemma generates a canvas block at a time through iterative denoising
rather than left-to-right autoregression. The decode harness above measures
inter-token timing, which does not apply to diffusion's burst output, so the
automated sweep records this checkpoint as a benchmark failure. The numbers
below are a manual same-host comparison (192-token generation, chat template,
seed 42, `max_denoising_steps=48`, median of 3 runs):

| Diffusion model | M1 Ultra mlxcel | M1 Ultra mlx-vlm | mlxcel / mlx-vlm |
|-----------------|----------------:|-----------------:|-----------------:|
| DiffusionGemma 26B-A4B 4bit | 32 tok/s | 29 tok/s | 110% |

Released `mlx-vlm` (0.4.4) does not include `diffusion_gemma`, so the reference
column is `mlx-vlm` upstream `main`. The reported tok/s amortizes the per-block
denoising passes and is not directly comparable to the autoregressive decode
rows above. No M5 Max figure is listed because that comparison was not run on
the same-host campaign.

The 0.4.0 M5 Max sweep covers 175 text model directories (160 with decode
numbers) and a 75-row VLM-mode pass. The Linux/CUDA GB10 (DGX Spark) sweep
covers 159 directories, 142 measured with no code-level failures and 7
memory-gated skips. Ratio summaries include only rows where both mlxcel and the
Python reference produced comparable decode measurements; unsupported checkpoints
and benchmark-configuration failures are tracked in the benchmark notes. VLM rows
should be read separately because vision preprocessing, processor setup, and
prompt construction differ by family. Re-run the benchmark suite on your target
hardware before using these numbers for capacity planning.

## Supported models

Model support is architecture- and checkpoint-dependent. Run:

```bash
mlxcel arch
```

for the CLI summary, and see [Supported models](docs/supported-models.md) for the maintained architecture table, known limitations, and VLM coverage notes.

## Python

`mlxcel` ships a pure-Python client that drives the OpenAI-compatible server from Python. It spawns and manages a local `mlxcel serve` process (managed mode) or connects to a running one (connect mode), auto-discovers the served model id, and exposes the raw `openai` client for the full API surface.

```python
import mlxcel

with mlxcel.LLM("mlx-community/Qwen3-4B-4bit") as llm:
    print(llm.generate("def fib(n):", max_tokens=128))
    for delta in llm.stream("Write a haiku about autumn"):
        print(delta, end="", flush=True)
```

Install with `pip install ./python`. See [Python client](docs/python-client.md) for managed and connect modes, streaming, structured output, async usage, and troubleshooting. The client lives in [`python/`](python) and builds entirely on the existing server (no native extension).

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
- [Audio input preprocessing](docs/audio-preprocessing.md)
- [Python client](docs/python-client.md)
- [Adding a new model](docs/adding-models.md)

## Contributing

Issues and pull requests are welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the contributor workflow, local quality gates (`cargo fmt`, `clippy`, `cargo test`, `cargo deny check`), and commit conventions. New model architectures, performance work, bug fixes, and documentation improvements are all useful. For larger changes, please open an issue first so the scope and validation plan can be discussed.

For security vulnerabilities, see [`SECURITY.md`](SECURITY.md), do **not** file these as public issues.

## License

Apache License 2.0 unless otherwise noted, see [LICENSE](LICENSE). Third-party attributions carried forward under Apache-2.0 Section 4(d) are listed in [NOTICE](NOTICE).

## Acknowledgments

- [MLX](https://github.com/ml-explore/mlx), Apple's machine learning framework
- [mlx-lm](https://github.com/ml-explore/mlx-lm) (MIT, Copyright 2023 Apple Inc.), [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) (MIT, Copyright 2025 Prince Canuma), and [mlx-audio](https://github.com/Blaizzy/mlx-audio) (MIT, Copyright 2024 Prince Canuma): Python projects whose model coverage and behavior mlxcel ports and mirrors. See [NOTICE](NOTICE).
- [MLX Community](https://huggingface.co/mlx-community), pre-converted MLX model checkpoints
- [turboquant_plus](https://github.com/TheTom/turboquant_plus): TurboQuant KV cache compression algorithms ported in `src/lib/mlxcel-core/src/cache/turbo/` (Apache-2.0, Copyright 2026 Tom Turney). See [NOTICE](NOTICE).
