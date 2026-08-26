# mlxcel

[![License: Apache 2.0](https://img.shields.io/github/license/lablup/mlxcel)](LICENSE)
[![Latest Release](https://img.shields.io/github/v/release/lablup/mlxcel)](https://github.com/lablup/mlxcel/releases/latest)
[![CI](https://github.com/lablup/mlxcel/actions/workflows/ci.yml/badge.svg)](https://github.com/lablup/mlxcel/actions/workflows/ci.yml)

High-performance LLM/VLM inference runtime and server for Apple Silicon / NVIDIA CUDA-compatible / (experimental) OpenXLA-compatible devices. The CLI and server are implemented in Rust and execute models through native MLX C++ bindings. Linux/CUDA builds are supported as a secondary target.

## Overview

`mlxcel` provides a Rust command-line runtime and an OpenAI-compatible model server for generation, embeddings, and reranking with MLX-format checkpoints. Loading, scheduling, and inference stay in one native process while model execution goes through MLX C++ bindings. It runs a broad range of text, vision-language, embedding, and reranker model families directly from HuggingFace checkpoints, with no conversion step.

The project started as work on structural model fine-tuning and has grown into a general-purpose serving runtime for local and small-cluster inference.

## New in v0.6.0

- **Gemma 4 MTP now measures its exactness instead of assuming it.** The gate used to return an unconditional yes for Gemma 4, advertising temperature-0 byte-identity that generation 15+ hardware does not provide under the default kernel. It now probes, and where the probe fails it buys the contract back by selecting the exact kernel: 2.13x classic decode on M5 Max with byte-identity, where the fast kernel would read 2.76x without it.
- **The MTP verify width is chosen by measured throughput.** The old controller waited for the configured prefix to be fully accepted, a bar the Gemma 4 12B pairing never clears, so it never widened. Measuring instead finds the pairing's own optimum: 94.9 tok/s at width 5 against 89.2 at 4, and it still refuses width 8, which is slower.
- **`GET /v1/internal/mtp-policy` says whether MTP is running and why.** Including the case that used to be invisible: a pairing the exactness probe vetoes now reports `exactness_declined` with the probe's reason, instead of claiming a measurement is still in progress for the life of the process.
- **Speculative rollback stops rewriting the whole KV cache every round.** A dense tail trim moves an offset instead of copying the live window, which is +5.3% at 24k context and grows with the conversation.
- **Draft trees, behind `MLXCEL_MTP_TREE` and off by default.** The measurement says a linear tree costs 1.9% to 4.1% and branching 8.1% to 10.1%, so the flag ships off and the numbers ship with it.
- **The OpenXLA backend is alpha.** Still off by default behind `xla-backend` / `xla-iree`, but 23 pull requests of multimodal execution, session plumbing and operator numeric contracts that never reached a release note are collected in the changelog, and the crate now ships on the workspace version line.
- **Five silent nondeterminism fixes** where a `HashMap` iteration order reached a decision: language-bias priority, RT-DETRv2 sanitization, distributed registry ordering, and two eviction paths.

## New in v0.5.2

- **Two silent correctness fixes on M5-class hardware.** A VLM wrapper and three server prefill paths padded prompts to a 32-token tile on models whose recurrent state cannot survive it, so Qwen 3.5, Mamba, Jamba, RWKV, Nemotron-H, Falcon-H1 and the other hybrid families produced changed greedy output whenever a prompt was not already tile-aligned. Nothing failed; the text just differed.
- **MTP speculative decoding works on Apple GPU generation 15 and later.** The exactness gate used to decline there every time, so MTP was off on the hardware it pays on. It now recovers byte-identity by selecting the kernel that has it, at about 1.04x classic decode where declining is 1.00x.
- **The MTP drafter is quantized at load, 810 MiB to 228 MiB.** The drafter step drops from 10.5 to 2.7 ms per round and throughput goes to roughly 1.5x classic decode, with acceptance unmoved and output byte-identical.
- **MLX moves to upstream `9a795735`,** 168 commits on, with the four drifted in-tree overlays rebased against it.
- **`MLXCEL_METAL4_ATTENTION=0`** turns off the M5 neural-accelerator attention route, so its effect can be measured instead of patched out and rebuilt.

## New in v0.5.0

- **Meta Muse Glimmer support.** The 52-layer mixed-cache decoder and the 50-layer vision and fusion path run single-image and multi-image prompts through both the CLI and the continuous-batching server, with ATEM reasoning channels parsed across the Chat Completions, Responses, and Anthropic-compatible routes. Both the bf16 checkpoint and `mlx-community/Muse-Glimmer-30B-4bit` are supported; the 4-bit decodes at about 3.1x the bf16 rate on NVIDIA GB10.
- **Florence-2 end to end.** DaViT vision tower plus BART seq2seq decoder, fifteen task markers, 3/4/6/8-bit checkpoints, and HTTP serving through a dedicated seq2seq worker. Parsed coordinates ride the response as `message.florence2_result`.
- **3 more new vision-language families:** LocateAnything grounding, Falcon-OCR early fusion, and Jina VLM.
- **8 new text model families**, including Ling/Bailing, OpenELM, TeleChat3, DBRX, Phixtral, AFMoE, and Klear.
- **Paged-attention decode v2 is now the production path.** Multi-CTA CSR-page-table decode scales with context length; batched decode is up to 1.47x faster on M1 Ultra. Set `MLXCEL_PAGED_ATTENTION_NATIVE=0` to restore the gather path.
- **Unified sparse/shared-prefix decode.** Sparse decode uses page tables without gathering; MiniMax-M3 sees up to 2.06x speedup. Shared-prefix (cascade) decode is available but disabled by default.
- **Faster sampling.** Sort-free top-p and Gumbel-max sampling improve performance, but may change fixed-seed outputs.
- **CUDA JIT kernels are keyed on their input dtypes.** Running one geometry at two dtypes in a single process no longer reuses the first compiled kernel and returns numbers unrelated to its inputs.
- **Chunked GLA prefill is the `bailing_moe_linear` default:** 2.1x to 2.4x faster prefill and lower perplexity at every window measured.
- **Generation-mask fix** for `deepseek_v2`, `internlm3`, `hunyuan`, and `gemma2`, plus earlier quantization validation at load time and Jamba MoE checkpoint fixes.
- **Workspace-wide verification:** `make verify` runs all workspace tests with a faster test profile, and gates crate versions and CUDA kernel dtype keys.

See the [changelog](CHANGELOG.md) for the full list.

## Why mlxcel

- **Smaller runtime surface.** Model loading, scheduling, and inference stay in a single native server process. Deployments do not need to provision a Python environment, keep package versions in sync, or route requests through an interpreter layer.
- **Simple deployment artifact.** `mlxcel` and `mlxcel-server` build as native executables, which makes packaging, service supervision, and upgrades straightforward. Platform runtime libraries are still required: for example macOS frameworks on Apple Silicon, and CUDA/OpenBLAS/LAPACK components for Linux builds.
- **`llama-server`-style operation.** `mlxcel-server` accepts many `llama-server`-compatible flags and `LLAMA_ARG_*` environment variables, which makes migration from llama.cpp-based scripts simpler. Treat this as compatibility-oriented, not a guarantee that every llama.cpp option has identical behavior.
- **OpenAI-compatible HTTP API subset.** The server supports SSE streaming and the `/v1/chat/completions`, `/v1/completions`, and `/v1/responses` endpoints.
- **Serving features for real deployments.** Continuous batching, prompt-prefix caching, and automatic prefix caching are on by default; speculative decoding and KV-cache compression are available for supported model/runtime combinations.
- **Differentiated runtime controls.** Default builds expose first-class YAML load-time model surgery through `--surgery` / `MLXCEL_SURGERY`, with operations such as `scale`, `add`, `prune`, `replace`, and `interpolate` for reproducible weight-space changes without retraining or writing converted checkpoints.
- **Multi-device and distributed modes.** Tensor parallelism and pipeline parallelism are implemented for selected model families, including zero-config pipeline startup with static or mDNS-based discovery.
- **Broad model-family coverage.** The runtime includes loaders for Llama, Qwen, Gemma, Phi, Mistral/Mixtral, DeepSeek, Cohere, InternLM, GLM, ExaOne, OLMo, ERNIE, Hunyuan, Mamba/RWKV/Jamba, Nemotron, MiniMax, Step, and Kimi, plus a broad vision-language and OCR set (Qwen3-Omni, Muse Glimmer, GLM-4V, Llama 3.2 Vision, Hunyuan-VL, ERNIE-4.5 VL, DeepSeek-VL2, DeepSeek-OCR, PaddleOCR-VL, and more). See [Supported models](docs/supported-models.md) for the maintained list.

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

Output length follows llama.cpp: with no `-n/--max-tokens` (default `-1`), `generate` / `run` keep generating until the model emits an end-of-sequence token or fills its context window. The server's `--n-predict` default (`-1`) behaves the same per request. Pass an explicit `-n N` (or `--n-predict N`) to cap output at exactly `N` tokens. HTTP `max_tokens` and native `/completion` `n_predict` overrides are silently clamped to the effective per-slot context window; when `--ctx-size` is unset, the resolved server default is the cap. With the default `--n-predict -1`, that value comes from the checkpoint context window, or 4096 when unavailable.

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

### Embed and rerank

The offline commands use the same loaders and batching paths as the HTTP
endpoints, which makes them useful both for local retrieval workflows and for
validating a checkpoint before serving it.

```bash
# Embed two texts and print their vectors plus cosine similarity.
mlxcel embed -m sentence-transformers/all-MiniLM-L6-v2 \
  -p "The weather is lovely" -p "It is sunny"

# Score and rank documents against one query.
mlxcel rerank -m BAAI/bge-reranker-v2-m3 \
  -q "what is panda?" -d "hi" -d "The giant panda is a bear species."

# Serve an embedding checkpoint on POST /v1/embeddings.
mlxcel-server -m sentence-transformers/all-MiniLM-L6-v2 --port 8080

# Serve chat and POST /v1/rerank from separate checkpoints.
mlxcel-server -m Qwen3.5-0.8B-4bit \
  --reranker-model mlx-community/Qwen3-Reranker-0.6B-4bit --port 8080
```

Embedding and reranking also support multimodal checkpoints and side-model
serving. See [Embeddings and reranking](docs/embeddings.md) for request schemas,
model-specific input formats, server flags, and supported families.

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

### Speculative decoding (MTP)

A small drafter proposes a block of tokens and the target verifies the whole
block in one forward, so a round emits everything the target would have chosen
itself plus one more. The gain is therefore a property of the output, not of
the model: predictable continuations accept long runs of drafts, open prose
accepts few. The three Gemma rows per host differ in nothing but the prompt,
and the ratio moves by two thirds across them on M5 Max, by half on M3 Ultra,
and from a gain to a loss on M1 Ultra. Measured at `temperature 0`, warm, with
the arms alternated, a warm-up discarded and the host's background indexers
suspended.

| Host | Pairing | Output | Classic | MTP | Speedup |
|------|---------|--------|--------:|----:|--------:|
| M5 Max (128 GB) | Gemma 4 12B + 4-bit assistant | enumeration | 43.1 tok/s | 135.4 tok/s | **3.14x** |
| M5 Max (128 GB) | Gemma 4 12B + 4-bit assistant | source code | 43.5 tok/s | 121.0 tok/s | **2.79x** |
| M5 Max (128 GB) | Gemma 4 12B + 4-bit assistant | prose | 43.3 tok/s | 82.4 tok/s | **1.90x** |
| M5 Max (128 GB) | Qwen 3.8 27B + its 4-bit MTP head | source code | 32.7 tok/s | 53.4 tok/s | **1.63x** |
| M3 Ultra (512 GB) | Gemma 4 12B + 4-bit assistant | enumeration | 63.4 tok/s | 165.5 tok/s | **2.61x** |
| M3 Ultra (512 GB) | Gemma 4 12B + 4-bit assistant | source code | 64.2 tok/s | 138.5 tok/s | **2.16x** |
| M3 Ultra (512 GB) | Gemma 4 12B + 4-bit assistant | prose | 64.0 tok/s | 111.2 tok/s | **1.74x** |
| M3 Ultra (512 GB) | Qwen 3.8 27B + its 4-bit MTP head | source code | 35.7 tok/s | 59.5 tok/s | **1.67x** |
| M1 Ultra (128 GB) | Gemma 4 12B + 4-bit assistant | enumeration | 34.2 tok/s | 50.4 tok/s | **1.48x** |
| M1 Ultra (128 GB) | Gemma 4 12B + 4-bit assistant | source code | 34.9 tok/s | 43.5 tok/s | **1.25x** |
| M1 Ultra (128 GB) | Gemma 4 12B + 4-bit assistant | prose | 34.5 tok/s | 32.7 tok/s | **0.95x** |
| M1 Ultra (128 GB) | Qwen 3.8 27B + its 4-bit MTP head | source code | 23.8 tok/s | 23.4 tok/s | **0.98x** |

The host matters as much as the prompt. Whether a verify block pays for itself
depends on how the GPU generation dispatches the quantized projections it runs,
which is why the runtime profiles each pairing rather than assuming: the
batch-capable Gemma 4 31B pair measures 1.2 to 1.4x on M5 Max and a consistent
regression on M1 Ultra, so the policy enables it on one and declines it on the
other. The M3 Ultra rows show the same point from the other direction: all
three Gemma ratios fall against their M5 Max twins while *both* arms are
faster in absolute terms, because the classic baseline a speedup divides by
rose further than the MTP arm did. A ratio falling is therefore not by itself
a regression, which is why a row without its host cannot be compared to one.
The Qwen pairing barely moves across the two hosts (1.63x against 1.67x),
so how much the host matters is itself a property of the pairing.

The M1 Ultra rows are where it becomes one. The quantity behind all three
hosts is what a verify round costs in units of that host's own classic decode
step: 1.28 on M5 Max, 1.51 on M3 Ultra, 2.71 on M1 Ultra at the same block
width, measured to within 1% by two prompts that have nothing else in common.
A round has to emit more tokens than that to pay for itself, so the prose
prompt, which emits 2.463 tokens per verify on M5 Max and between 2.46 and
2.63 across all three, clears the bar on the first two hosts and misses it on
the third, where MTP is a 5% loss. Acceptance is not what separates them: the
enumeration rows accept at 0.997 on both M5 Max and M1 Ultra, to three digits.

Enable it with `--draft-model <drafter> --draft-kind mtp`. Widening the verify
block stops paying past a point, because the tokens a round emits saturate near
`1 / (1 - acceptance)` while the verify keeps getting more expensive. Where that
point sits is a per-host fact, and the shipped default does not track it: with
no width passed the block settles at 4, which is the peak on M3 Ultra, 5.8%
short of it on M5 Max, where an explicit `--draft-block-size 5` measures 121.2
tok/s against the default's 114.6, and on M1 Ultra one of three tied widths
that the samples cannot separate. Three hosts, three answers. What does carry
across them is the cost side: a round is affine in the block width, at
`1.14 + 0.090 K` classic decode steps on M3 Ultra against `1.35 + 0.346 K` on
M1 Ultra, so the usable band narrows on older silicon and a block of 12 on
M1 Ultra turns the code prompt into a 0.97x loss. See
[`docs/benchmarks.md`](docs/benchmarks.md) before assuming either way.

Qwen's lower ratio is not a worse implementation. On the same code prompt it
accepts more of its drafts than the Gemma pairing does (0.753 against 0.733 on
M3 Ultra), and it pays for a byte-identity guarantee the Gemma arms do not:
a startup probe compares a verify block against the single-token chain and, on
Apple GPU generation 15 and newer, drops to the kernel that has byte-identity
at about 17 to 20% of the verify forward. Gemma 4 is not probed yet
([#1188](https://github.com/lablup/mlxcel/issues/1188)); keeping the same
guarantee there measures 93.2 tok/s on the code row, 2.14x, and 117.5 tok/s on
M3 Ultra, 1.83x. The gap the missing probe leaves is not theoretical: the
unprobed Gemma pairing reproducibly diverges from classic decode on the prose
prompt on M1 Ultra, and on the prose and source-code prompts both on M3 Ultra,
while the probed Qwen pairing diverges on neither host.

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

for the CLI summary, and see [Supported models](docs/supported-models.md) for the maintained architecture table, known limitations, real-checkpoint qualification status, and VLM coverage notes.

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
- [Speculative-decoding acceptance](docs/speculative-acceptance.md)
- [Adaptive MTP policy API](docs/mtp-policy-api.md)
- [OpenAI Responses API](docs/responses-api.md)
- [Embeddings and reranking APIs and CLI](docs/embeddings.md)
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
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer) (Apache-2.0, Copyright 2023-2026 FlashInfer community, Copyright 2025-2026 NVIDIA): LLM serving kernel library whose paged-attention, split-KV and cascade state-merge, and sampling algorithm designs inform mlxcel's serving-performance kernels.
