# Environment variables

This page documents the `MLXCEL_*` environment variables that affect mlxcel
runtime, server, downloader, build, and diagnostic behavior.

Prefer CLI flags for settings that have a flag equivalent. Environment
variables are useful for containers, service units, and repeatable benchmark
runs, but they are process-wide and several of the low-level knobs are read once
and cached on first use. Set them before starting `mlxcel` or `mlxcel-server`.

## Precedence and value conventions

- If a CLI flag and an environment variable control the same option, the CLI
  flag wins unless the flag help states otherwise.
- `LLAMA_ARG_*` aliases exist for a subset of llama-server-compatible flags.
  This page focuses on `MLXCEL_*`; use `--help` for the full flag/env surface.
- Boolean parsing is not completely uniform across all internal knobs:
  - documented server options generally accept `true/false`, `1/0`, `yes/no`,
    and `on/off`;
  - many diagnostic switches are presence-based, so any set value enables the
    behavior;
  - variables whose row says "falsy disables" treat `0`, `false`, `off`, or
    `no` as disabled.
- Variables marked **advanced** or **diagnostic** are not a stable public API.
  They exist for benchmarking, rollback, or kernel-development work and may
  change between releases.

## Common runtime variables

| Variable | Values | Default | Notes |
|----------|--------|---------|-------|
| `MLXCEL_DEVICE` | `gpu`, `metal`, `cpu` | `gpu` hint | `cpu` requests CPU execution. Invalid values are ignored with a warning and treated as `gpu`; if no GPU backend is available, runtime falls back to CPU. |
| `MLXCEL_WIRED_LIMIT` | `max`, `0`, `none`, bytes, `NGB`, `NMB` | `max` | Apple Silicon GPU wired-memory limit. Unset/empty/`max` sets MLX's reported GPU max memory size; `0`/`none` disables the limit; numeric values set an explicit limit. |
| `MLXCEL_MEMORY_LIMIT` | `0`, `none`, bytes, `NGB`, `NMB` | unset | Soft MLX allocator memory cap. Unset/`0`/`none` lets MLX use its backend default; numeric values cap the allocator and make MLX raise an exception once allocations would push the working set past this value. Also feeds the `mlxcel inspect` / `--estimate-memory` preflight as the authoritative "available unified memory" figure when nonzero. |
| `MLXCEL_HEADROOM_FACTOR` | positive `f64` | `1.20` | Runtime/activation headroom multiplier used by the unified memory estimator (`mlxcel inspect`, `--estimate-memory`, `--recommend-quant`). Positive values `<= 1.0` disable the headroom term; invalid or non-positive values warn and fall back to the default. Override only for calibration runs — see the in-code recipe in `src/execution/memory_estimate.rs`. |
| `MLXCEL_CACHE_DIR` | directory path | `$HOME/.cache/mlxcel` | Root for mlxcel's on-disk caches. The tokenizer language-analysis disk cache (language-bias features) lives under `tokenizer-scripts/`, and the location-independent global model store lives under `models/<owner>/<name>` when `MLXCEL_MODELS_DIR` and `--models-dir` are both unset. |
| `MLXCEL_MODELS_DIR` | directory path | unset (falls back to `${MLXCEL_CACHE_DIR:-$HOME/.cache/mlxcel}/models`) | Dedicated model-store root. Snapshots live directly at `$MLXCEL_MODELS_DIR/<owner>/<name>` with no `models/` subdir, so the whole store can sit on a separate volume without dragging the tokenizer-script cache along. Read by `mlxcel download`, the `-m/--model` resolver (`generate` / `serve` / `inspect` / `run`), the `mlxcel-server -m/--model` resolver, and `list --local` / `rm`. Resolution precedence for the models root: the `--models-dir <PATH>` CLI flag, then `MLXCEL_MODELS_DIR`, then `${MLXCEL_CACHE_DIR:-$HOME/.cache/mlxcel}/models`. (`download --local-dir <PATH>` is separate: it writes the snapshot verbatim at that exact path.) |
| `MLXCEL_DEFAULT_ORG` | HuggingFace org/user name | `mlx-community` | Org prepended to a bare, prefix-less model name (no `/`) by the `-m/--model` resolver (`generate` / `serve` / `inspect` / `run`) and the `mlxcel-server -m` resolver, so `mlxcel run Qwen3-4B-4bit` resolves to `mlx-community/Qwen3-4B-4bit`. An explicit `owner/name` repo-id and an existing local path are unaffected. An empty/whitespace value falls back to `mlx-community`. |
| `MLXCEL_SERVER_DECODE_STORAGE` | `auto`, `dense`, `paged` | `auto` | Server continuous-batching decode storage. `--decode-storage-backend` takes precedence. Invalid values warn and fall back to `auto`. |
| `MLXCEL_SURGERY` | YAML file path | unset | Feature-gated weight-load surgery configuration. `--surgery` takes precedence when the `surgery` feature is built. |

## Server context sizing

`mlxcel serve` and `mlxcel-server` follow llama.cpp server semantics for the
llama-compatible flags `--ctx-size` / `LLAMA_ARG_CTX_SIZE` and `--parallel` /
`LLAMA_ARG_N_PARALLEL`: an explicit `--ctx-size C` is a total context budget
shared by the active request slots, so each slot receives `floor(C / N)` tokens
when `--parallel N` is used. If `--max-batch-size M` is set, `M` is the divisor
because it controls the maximum number of concurrent decode sequences. With
`--no-batch`, the divisor is `1`.

Startup fails when the effective per-slot context window is below 512 tokens.
The `/slots` endpoint and `/health.context_size` report the effective per-slot
window, not the total `--ctx-size` budget. The `--estimate-memory` preflight uses
the same per-slot window and active-sequence count so increasing `--parallel`
does not multiply KV memory for a fixed explicit `--ctx-size`.

## Build-time variables

These are read by the `mlxcel-core` build script.

| Variable | Values | Default | Notes |
|----------|--------|---------|-------|
| `MLXCEL_BUILD_METAL` | `1/0`, `on/off`, `true/false`, `yes/no` | `on` on macOS | Overrides the CMake `MLX_BUILD_METAL` setting for local builds. Invalid values fail the build. |
| `MLXCEL_BUILD_ACCELERATE` | `1/0`, `on/off`, `true/false`, `yes/no` | `on` on macOS | Overrides the CMake `MLX_BUILD_ACCELERATE` setting for local builds. Invalid values fail the build. |

CUDA builds also use non-`MLXCEL_*` variables such as `CUDA_HOME` and
`MLX_CUDA_ARCHITECTURES`; see [Installation](installation.md#linux-with-cuda).

## Downloader variables

| Variable | Values | Default | Notes |
|----------|--------|---------|-------|
| `MLXCEL_NO_PROGRESS` | any non-empty value | unset | Suppresses interactive download progress bars. `NO_COLOR` and `CI=true` also suppress bars. |
| `MLXCEL_ALLOW_INSECURE_ENDPOINT` | any non-empty value | unset | Allows sending a Hugging Face token to a non-HTTPS `HF_ENDPOINT`. Leave unset outside audited internal mirrors. |
| `HF_HUB_CACHE` | directory path | unset | Probed read-only for an already-downloaded snapshot before `mlxcel download` fetches anything (the existing copy is reused, never re-fetched). Used verbatim as the HuggingFace Hub cache directory. mlxcel never writes into the HF content-addressed layout. |
| `HF_HOME` | directory path | `$HOME/.cache/huggingface` | Fallback HuggingFace cache root when `HF_HUB_CACHE` is unset; the hub lives under `HF_HOME/hub`. Same read-only reuse semantics as `HF_HUB_CACHE`. |

## Server prompt-cache variables

These variables are applied when the corresponding CLI flag is absent.

| Variable | Values | Default | Flag equivalent |
|----------|--------|---------|-----------------|
| `MLXCEL_PROMPT_CACHE_ENABLED` | boolean | `true` | `--prompt-cache-enabled` |
| `MLXCEL_PROMPT_CACHE_CAPACITY_BYTES` | unsigned integer bytes | `2147483648` | `--prompt-cache-capacity-bytes` |
| `MLXCEL_PROMPT_CACHE_MAX_ENTRIES` | unsigned integer | `1024` | `--prompt-cache-max-entries` |
| `MLXCEL_PROMPT_CACHE_TTL` | unsigned integer seconds | `3600` | `--prompt-cache-ttl` |
| `MLXCEL_PROMPT_CACHE_MIN_PREFIX` | unsigned integer tokens | `32` | `--prompt-cache-min-prefix` |

`MLXCEL_PROMPT_CACHE_ENABLED` has higher precedence than the llama.cpp
compatibility alias `LLAMA_ARG_CACHE_REUSE` when both are set and no CLI flag is
provided.

## Speculative-decoding variables

| Variable | Values | Default | Notes |
|----------|--------|---------|-------|
| `MLXCEL_DRAFT_KIND` | `dflash`, `mtp` | auto/none | Alias for `--draft-kind` when the CLI flag and `LLAMA_ARG_DRAFT_KIND` are absent. |
| `MLXCEL_DRAFT_BLOCK_SIZE` | unsigned integer | per drafter (`4` for MTP, `16` for DFlash) | Alias for `--draft-block-size` when the CLI flag and `LLAMA_ARG_DRAFT_BLOCK_SIZE` are absent. |
| `MLXCEL_ENABLE_MTP_B1` | truthy value | off | **Advanced.** Forces the singleton Gemma 4 MTP burst path for parity/debug testing. |
| `MLXCEL_ENABLE_MTP_BATCH` | truthy value | off | **Advanced.** Forces the batched Gemma 4 MTP burst path for parity/debug testing. |
| `MLXCEL_ENABLE_MTP_DEFERRED` | `1` | off | **Advanced.** Enables the deferred greedy verifier path for Gemma 4 MTP when sampling settings allow it. |

## KV cache and TurboQuant variables

Use CLI flags such as `--cache-type-k`, `--cache-type-v`, `--kv-cache-mode`,
`--turbo-boundary-v`, and the batch KV quantization flags when possible. The
variables below are useful for service-level defaults and A/B experiments. See
[TurboQuant KV cache](turbo-kv-cache.md) for the user-facing mode descriptions.

| Variable | Values | Default | Notes |
|----------|--------|---------|-------|
| `MLXCEL_KV_BOUNDARY_V_LAYERS` | integer count | `2` | Number of first/last layers kept at higher precision for Turbo4-family modes. `0` disables. `--turbo-boundary-v` writes this value before cache construction and takes precedence. |
| `MLXCEL_TURBO_BOUNDARY_V` | integer count | fallback alias | Compatibility alias for `MLXCEL_KV_BOUNDARY_V_LAYERS`; the primary name wins when both are set. |
| `MLXCEL_KV_SKIP_LAST_LAYER` | boolean | `true` | Fallback for `--kv-skip-last-layer` in continuous-batching KV quantization. |
| `MLXCEL_SPARSE_V_THRESHOLD` | non-negative float | `1e-6` | Sparse-V alive threshold. `0` disables sparse-V; invalid values warn and use the default. |
| `MLXCEL_SPARSE_V_KERNEL` | falsy disables | enabled on macOS | Allows the fused Sparse-V/dequant Metal kernels. Set `0`, `false`, `off`, or `no` to force graph fallback. |
| `MLXCEL_TURBO4_DEQUANT_SDPA` | falsy disables | on | Controls the dequant-first SDPA path for symmetric `Turbo4`. |
| `MLXCEL_TURBO4_DELEGATED_DEQUANT_SDPA` | falsy disables | on | Controls the default dequant-first SDPA path for `Turbo4Delegated`. |
| `MLXCEL_TURBO4_DELEGATED_FUSED` | truthy enables | off | **Advanced.** Enables the older custom fused delegated-kernel route, mainly for comparison when dequant-first SDPA is disabled. |
| `MLXCEL_TURBO4_DELEGATED_FP16_FAST_PATH` | truthy enables | off | **Advanced.** Keeps a unified FP16 V working set in delegated mode for speed experiments while maintaining packed sidecars. |
| `MLXCEL_TURBO4_DELEGATED_FP16_SIDECARS` | `predecode`, `eager`, `lazy`, `on-demand` | `predecode` | Sidecar maintenance policy for the delegated FP16 fast path. |
| `MLXCEL_ENABLE_DIRECT_PREFILL_CACHE_STORE` | presence enables | off | **Advanced.** Installs the incoming prefill tensor directly as the initial KV cache buffer when applicable. |

## Video and local-media variables

These apply to video-capable VLM request handling.

| Variable | Values | Default | Notes |
|----------|--------|---------|-------|
| `MLXCEL_VIDEO_DIR_ALLOWLIST` | comma-separated directories | unset | Local `video_url` file paths are rejected unless they resolve under one of these canonicalized directories. Keep directories owner-writable only; group/world-writable entries warn at startup. |
| `MLXCEL_VIDEO_MAX_PIXELS` | unsigned integer | `16777216` | Rejects source videos whose `width × height` exceeds the cap. |
| `MLXCEL_VIDEO_MAX_DURATION_SEC` | float seconds | `600` | Rejects source videos longer than the cap. |
| `MLXCEL_VIDEO_MAX_PNG_FRAME_BYTES` | unsigned integer bytes | `268435456` | Per-frame cap for the ffmpeg PNG stream splitter. |

## Hardware and kernel diagnostic variables

These variables are for profiling, rollback, or experiments. They are not
recommended as normal deployment settings.

| Variable | Values | Default | Purpose |
|----------|--------|---------|---------|
| `MLXCEL_NO_PADDED_PREFILL` | presence disables | auto | Disables M5+/Neural-Accelerator prefill tile alignment. |
| `MLXCEL_FORCE_PADDED_PREFILL_MASK` | presence enables | off | Forces an explicit padded prefill mask path for debugging. |
| `MLXCEL_LOG_NA_ATTENTION` | `sampled`, `all`, truthy | off | Logs Neural Accelerator attention dispatch decisions. |
| `MLXCEL_ENABLE_FUSED_CAUSAL_PREFILL_ATTENTION` | presence enables | off | Enables an experimental Llama-family fused causal prefill path when supported. |
| `MLXCEL_ENABLE_FUSED_QKV_SPLIT_ROPE` | presence enables | off | Enables an experimental fused QKV projection/split/RoPE path. |
| `MLXCEL_GEMMA4_ENABLE_FUSED_QKV` | presence enables | off | Enables a Gemma 4 fused-QKV projection experiment. |
| `MLXCEL_DISABLE_COMPILED_SWITCH_QGEGLU` | presence disables | compiled path on when supported | Rolls back Gemma 4 compiled Switch-QGeGLU decode path. |
| `MLXCEL_ENABLE_SOFTCAP_GQA_DECODE_GROUPED` | any value except `0` enables | off | Enables grouped softcap-GQA decode optimization. |
| `MLXCEL_DISABLE_SOFTCAP_GQA_DECODE_GROUPED` | `1` disables, `0` enables | unset | Legacy rollback/override for grouped softcap-GQA decode. |
| `MLXCEL_DISABLE_SINGLE_QUERY_MASKLESS` | truthy disables | maskless path on | Disables the single-query maskless attention path. |
| `MLXCEL_EXPERIMENTAL_BOOL_CAUSAL_MASK` | truthy enables | off | Enables an experimental boolean causal-mask path. |
| `MLXCEL_PIPELINE_GRANULARITY` | `off`, `layer`, `block:N` | `off` | Inserts layer-boundary async-eval hints for pipeline experiments. |

## Logging, profiling, and capture variables

Most of these switches force synchronization or extra graph work and will change
throughput measurements. Use them for diagnosis, not capacity planning.

| Variable | Values | Default | Purpose |
|----------|--------|---------|---------|
| `MLXCEL_TRACE_DTYPE` | presence enables | off | Prints selected tensor dtypes/shapes during generation. |
| `MLXCEL_FORCE_SYNC` | presence enables | off | Forces synchronous decode evaluation. |
| `MLXCEL_PROFILE_PIPELINE` | presence enables | off | Emits high-level generation pipeline timing. |
| `MLXCEL_PROFILE_PIPELINE_DETAIL` | presence enables | off | Adds per-step pipeline timing detail. |
| `MLXCEL_PROFILE_BLOCKS` | presence enables | off | Emits per-block/model-family timing where implemented. |
| `MLXCEL_PROFILE_FORWARD` | presence enables | off | Enables model-specific forward profiling where implemented. |
| `MLXCEL_PROFILE_QWEN3_MOE_DETAIL` | presence enables | off | Profiles Qwen3 MoE internals. |
| `MLXCEL_PROFILE_MOE_INNER` | presence enables | off | Profiles Gemma 4 MoE sub-operations. |
| `MLXCEL_PROFILE_PER_LAYER` | presence enables | off | Prints per-layer Gemma 4 timing. |
| `MLXCEL_PROFILE_LAYER_BUILD` | presence enables | off | Adds Gemma 4 layer-build timing. |
| `MLXCEL_PROFILE_LAYER_SUBOPS` | presence enables | off | Adds Gemma 4 per-suboperation timing. |
| `MLXCEL_EXPORT_DECODE_DOT` | file path | unset | Exports the first decode graph pair to DOT. |
| `MLXCEL_CAPTURE_DECODE` | path | unset | Captures one warmed decode token to a Metal GPU trace and exits; requires `MTL_CAPTURE_ENABLED=1`. |
| `MLXCEL_METAL_CAPTURE_PATH` | file path | unset | Starts a Metal capture around steady-state generation; requires `MTL_CAPTURE_ENABLED=1`. |
| `MLXCEL_DEBUG_GEMMA4_LOAD` | presence enables | off | Emits Gemma 4 safetensors loading diagnostics. |
| `MLXCEL_NO_PRECISION_WARNING` | presence suppresses | warning on | Suppresses the bf16-on-Apple-Silicon precision/performance note. |

## Test and CI variables

These are intended for the repository's own tests and automation rather than
normal end-user operation.

| Variable | Purpose |
|----------|---------|
| `MLXCEL_CI_PP_MODEL` | Model path used by the pipeline-parallel CI integration test. |
| `MLXCEL_SKIP_HEAVY_TESTS` | Skips selected heavy tests. |
| `MLXCEL_BENCH_DATE` | Metadata override for Turbo KV benchmark tests. |
| `MLXCEL_BENCH_MACHINE` | Metadata override for Turbo KV benchmark tests. |
