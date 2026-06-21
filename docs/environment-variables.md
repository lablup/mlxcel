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
| `MLXCEL_MODELS_DIR` | directory path | unset (falls back to `${MLXCEL_CACHE_DIR:-$HOME/.cache/mlxcel}/models`) | Dedicated model-store root. Snapshots live directly at `$MLXCEL_MODELS_DIR/<owner>/<name>` with no `models/` subdir, so the whole store can sit on a separate volume without dragging the tokenizer-script cache along. Read by `mlxcel download`, the `-m/--model` resolver (`generate` / `serve` / `inspect` / `run`), the `mlxcel-server -m/--model` resolver, and `list` / `rm`. Resolution precedence for the models root: the `--models-dir <PATH>` CLI flag, then `MLXCEL_MODELS_DIR`, then `${MLXCEL_CACHE_DIR:-$HOME/.cache/mlxcel}/models`. (`download --local-dir <PATH>` is separate: it writes the snapshot verbatim at that exact path.) |
| `MLXCEL_DEFAULT_ORG` | HuggingFace org/user name | `mlx-community` | Org prepended to a bare, prefix-less model name (no `/`) by the `-m/--model` resolver (`generate` / `serve` / `inspect` / `run`), the `mlxcel-server -m` resolver, and the `download` verb (`mlxcel download` / `mlx-server download`), so `mlxcel run Qwen3-4B-4bit` resolves to `mlx-community/Qwen3-4B-4bit` and `mlxcel download Qwen3-4B-4bit` downloads that same repo. An explicit `owner/name` repo-id and an existing local path are unaffected. An empty/whitespace value falls back to `mlx-community`. |
| `MLXCEL_SERVER_DECODE_STORAGE` | `auto`, `dense`, `paged` | `auto` | Server continuous-batching decode storage. `--decode-storage-backend` takes precedence. Invalid values warn and fall back to `auto`. |
| `MLXCEL_KV_CACHE_BUDGET` | `auto` or unsigned integer bytes | unset | Paged KV block-pool budget for continuous batching. `--kv-cache-budget` takes precedence. Applies when `--decode-storage-backend paged` uses pool-backed Fp16 caches. |
| `MLXCEL_ALLOWED_ORIGINS` | comma-separated origin list (e.g. `https://app.example.com,https://admin.example.com`) | unset (permissive) | Restricts CORS to the listed origins. `--allowed-origins` takes precedence. Unset keeps the permissive default that reflects any origin. Each value must be a bare `scheme://host[:port]` origin (`http`/`https`, no path or query); a malformed value fails server startup with a clear message instead of being silently dropped. Only affects the browser-reachable TCP HTTP listener: the Unix-socket transport sends no `Origin` header and is unaffected. |
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
| `MLXCEL_CXX_MARCH` | a `-march=` value, or `none` | `native` | ISA baseline for the C++ bridge in release builds. Set a portable baseline (e.g. `x86-64-v3`) for binaries that run on machines other than the build host; `none` omits the flag. See [Installation](installation.md#c-isa-baseline-mlxcel_cxx_march). |

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
| `MLXCEL_ENABLE_VLM_PREFIX_CACHE` | boolean | `false` | `--enable-vlm-prefix-cache` |
| `APC_ENABLED` | boolean | `true` | `--apc-enabled` |
| `APC_BLOCK_SIZE` | unsigned integer tokens | `16` | `--apc-block-size` |
| `APC_NUM_BLOCKS` | unsigned integer | derived from max entries | `--apc-num-blocks` |
| `APC_HASH` | `sha256` or `blake3` | `sha256` | `--apc-hash` |

`MLXCEL_PROMPT_CACHE_ENABLED` has higher precedence than the llama.cpp
compatibility alias `LLAMA_ARG_CACHE_REUSE` when both are set and no CLI flag is
provided.

Automatic Prefix Caching is on by default; pass `--apc-enabled=false` or set
`APC_ENABLED=false` to fall back to whole-prefix matching only (a stored prefix
is then reusable only when it is fully contained in the new request). The
`APC_*` names mirror the upstream `mlx-vlm` env surface.

`MLXCEL_ENABLE_VLM_PREFIX_CACHE` opts same-image multimodal follow-up turns into
prompt-prefix sharing while leaving text-only prompt-cache behavior unchanged.

## Server audio admission variables

The OpenAI audio endpoints (`/v1/audio/speech`, `/v1/audio/transcriptions`, `/v1/audio/translations`) dispatch work to a single dedicated worker thread over a bounded command queue. These knobs bound that queue and the per-request reply wait, so a burst of requests cannot grow memory without bound (each queued speech-to-text command holds up to the 25 MiB per-request payload) and a stuck request does not block its caller forever.

| Variable | Values | Default | CLI flag | Notes |
|----------|--------|---------|----------|-------|
| `MLXCEL_AUDIO_QUEUE_DEPTH` | unsigned integer | `8` | `--audio-queue-depth` | Bound on the audio worker command queue. When the queue is full, new audio requests get a structured `503` ("All slots are busy") instead of queueing without bound. A depth of `8` caps queued payload at roughly 200 MiB plus the one request in flight. A `0` is clamped to at least one queued command. |
| `MLXCEL_AUDIO_REQUEST_TIMEOUT_SECS` | unsigned integer seconds | `120` | `--audio-request-timeout-secs` | Per-request reply timeout. A stuck or pathologically slow audio request frees its blocking thread and returns a structured `504` after this, instead of hanging. The timeout does not cancel the in-flight model work on the worker; it only frees the caller. A `0` falls back to the default rather than timing out instantly. |

## Speculative-decoding variables

| Variable | Values | Default | Notes |
|----------|--------|---------|-------|
| `MLXCEL_DRAFT_KIND` | `dflash`, `mtp` | auto/none | Alias for `--draft-kind` when the CLI flag and `LLAMA_ARG_DRAFT_KIND` are absent. |
| `MLXCEL_DRAFT_BLOCK_SIZE` | unsigned integer | per drafter (`4` for MTP, `16` for DFlash) | Alias for `--draft-block-size` when the CLI flag and `LLAMA_ARG_DRAFT_BLOCK_SIZE` are absent. |
| `MLXCEL_MTP_ADAPTIVE` | `0`/`false`/`no`/`off` to disable, any other value (or unset) to enable | on | Adaptive B=1 MTP policy (issue #333). When on, the server profiles the first few B=1 MTP bursts of each (target, drafter, hardware, block_size) pairing (acceptance length, verify latency, drafter latency, batch size, prompt shape) and settles to a data-driven enable/decline verdict that overrides the static per-hardware gate when the measured profile is clearly favorable or unfavorable, falling back to the static default otherwise. The verdict (enable/decline plus the coarse acceptance rate, no prompt data) is persisted at `${MLXCEL_CACHE_DIR:-$HOME/.cache/mlxcel}/mtp-policy/<key-hash>.json` (hint format v2), so profiling runs once per pairing and a restart reuses the verdict. Changing `MLXCEL_DRAFT_BLOCK_SIZE` changes the block_size dimension of the key, so the old hint is discarded and profiling restarts for the new K. Set to an off value to disable profiling and use the pre-#333 static per-hardware gates. `MLXCEL_ENABLE_MTP_B1` still pins the decision and, when set, suppresses profiling. The experimental batched (B>1) path is unaffected and stays behind `MLXCEL_ENABLE_MTP_BATCH`. |
| `MLXCEL_ENABLE_MTP_B1` | `0`/`false`/`no`/`off` to disable, any other value to force on | adaptive (per hardware) | Manual override for the singleton (B=1) MTP burst, in both directions. When set it pins the decision and disables adaptive profiling (issue #333). When unset, the adaptive policy decides (see `MLXCEL_MTP_ADAPTIVE`); with `MLXCEL_MTP_ADAPTIVE=0` the decision is the static per-hardware default (issue #165): non-batchable targets (`gemma4_unified` 12B pairs, whose only decode path is B=1) default **on** everywhere (measured ~1.87× on M5 Max and ~1.1 to 1.4× on M1 Ultra); batch-capable targets (the 31B + bf16 assistant) default **on only on M5+** (Neural Accelerator generation): M5 Max measured ~1.2 to 1.4×, while M1 Ultra measured a consistent ~0.75 to 0.96× regression, so pre-M5 chips fall back to classic decode. |
| `MLXCEL_ENABLE_MTP_BATCH` | truthy value | off | **Advanced.** Forces the batched Gemma 4 MTP burst path for parity/debug testing. Not governed by the adaptive policy (issue #333), which scopes to the validated, byte-identical B=1 path. |
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
| `MLXCEL_SPARSE_V_COUNT` | output file path | unset (off) | **Diagnostic.** When set to a non-empty path, every single-token `Turbo4Asym` decode appends `call_idx,kv_tokens,skipped,total` to that CSV, counting post-softmax attention weights below `MLXCEL_SPARSE_V_THRESHOLD` across all heads. Aggregate per layer offline by grouping on `kv_tokens` (constant within a decode step) and using each call's position within the step as the layer index. A graph-only side computation; zero cost when unset. See `scripts/measure_sparse_v_skip_rate.sh`. |
| `MLXCEL_TURBO4_DEQUANT_SDPA` | falsy disables | on | Controls the dequant-first SDPA path for symmetric `Turbo4`. |
| `MLXCEL_TURBO4_ASYM_DEQUANT_SDPA` | falsy disables | on | Controls the dequant-first SDPA path for asymmetric `Turbo4Asym` (FP16 K + 4-bit V). Falsy values fall back to the lossy sparse-V approximation. |
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
| `MLXCEL_FUSED_MOE` | `0`/`false`/`off`/`no` disable; any other value or unset enables | on | Fused single-token decode-MoE kernel (#268), on by default since #282 (Metal) and #319 (CUDA, via `mx.fast.cuda_kernel`); validated on M1 Ultra, M5, and GB10. Set to `0` to force the proven `gather_qmm`/`SwitchGLU` path. Active for qwen3_moe, qwen3_next, dots.llm1, gemma4, qwen2_moe, mixtral, phimoe, lfm2, qwen3_vl_moe, and olmoe decode. |
| `MLXCEL_FUSED_MOE_SGY` | `1`-`32` | `8` | Simdgroups (Metal) / warps-per-block (CUDA) per threadgroup for the fused decode-MoE kernel; tune per hardware. |
| `MLXCEL_FUSED_MOE_MAX_DFF` | positive int | `4096` | Expert-intermediate (Dff) upper bound for the fused path; above it the caller falls back to `gather_qmm`. The fused path wins only while `gather_qmm` underutilizes the GPU (small experts). The break-even is hardware-dependent: ~4096 on M1 Ultra, ~13-14k on GB10/CUDA. 4096 is the conservative shared default; raise it on CUDA for mid-size experts. |
| `MLXCEL_FUSED_MOE_RELU2` | presence enables | off | Enables the squared-ReLU fused MoE path for nemotron-class experts; performance-neutral on nemotron-h, kept for a future MoE-dominated squared-ReLU model. |
| `MLXCEL_FUSED_QK_NORM` | `1`/`true`/`on`/`yes` enable; any other value or unset disables | off (opt-in) | Fused single-token QKV projection + Q/K RMSNorm + RoPE kernel (#326) for Qwen3 and Qwen3-MoE decode. Opt-in: set to `1` to enable. Decode output is byte-identical to the graph path (the RMSNorm reduces over the transpose-invariant head_dim axis), but the kernel cuts Rust/C++ FFI crossings rather than MLX op count, so on M1 Ultra (fast FFI) it measured about 1 to 3.4% slower than the graph path (qwen3-0.6b 275 vs 284, qwen3-8b 82.3 vs 83.2 tok/s). It ships as a reusable shared primitive for the deferred QK-norm families and is gated off pending a per-backend win (for example CUDA, where op-dispatch and FFI cost differ), mirroring the opt-in `MLXCEL_FUSED_MOE_RELU2`. Active only when `l == 1` (decode) and weights are quantized. |

## Block-diffusion diagnostic variables

| Variable | Values | Default | Purpose |
|----------|--------|---------|---------|
| `MLXCEL_DIFFUSION_DEBUG_CANVAS=1` | `1` enables | off | **Diagnostic.** Replaces all DiffusionGemma canvas random-noise initialization with a fixed deterministic pattern (`(i+1)*7919 + k*104729) % vocab_size`) and prints `DIFFUSION_COMMIT block=<n> ids=...` per committed block. Intended for cross-implementation parity testing against the mlx-vlm Python reference at temperature 0. Output is not suitable for normal generation. |

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
