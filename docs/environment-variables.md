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
| `MLXCEL_CACHE_LIMIT` | `0`, `none`, bytes, `NG`/`NGB`, `NM`/`NMB` | unset | Bound on MLX's buffer cache (issue #627). Unset/`0`/`none` leaves MLX's default cache behavior; numeric values cap cached-buffer bytes via `set_cache_limit`. On CUDA this is the intended way to bound cache growth now that the periodic decode-loop clear is disabled by default (see `MLXCEL_CACHE_CLEAR_INTERVAL`): it keeps the memory pool bounded without the per-step churn that defeats CUDA-graph reuse (ml-explore/mlx#2358). |
| `MLXCEL_CACHE_CLEAR_INTERVAL` | `0` (disable), positive integer (token cadence) | `0` on CUDA, `256` on Metal/CPU | Cadence of the periodic `clear_memory_cache` in the decode loops and batch scheduler, in generated tokens (issue #627). `0` disables the periodic clear. The default is backend-aware: on CUDA the clear churns the memory pool and defeats CUDA-graph reuse (ml-explore/mlx#2358) so it is off by default (bound the cache with `MLXCEL_CACHE_LIMIT` instead); on Metal/CPU the cheap 256-token trim used by Python mlx-lm is kept. |
| `MLXCEL_HEADROOM_FACTOR` | positive `f64` | `1.20` | Runtime/activation headroom multiplier used by the unified memory estimator (`mlxcel inspect`, `--estimate-memory`, `--recommend-quant`). Positive values `<= 1.0` disable the headroom term; invalid or non-positive values warn and fall back to the default. Override only for calibration runs — see the in-code recipe in `src/execution/memory_estimate.rs`. |
| `MLXCEL_CACHE_DIR` | directory path | `$HOME/.cache/mlxcel` | Root for mlxcel's on-disk caches. The tokenizer language-analysis disk cache (language-bias features) lives under `tokenizer-scripts/`, and the location-independent global model store lives under `models/<owner>/<name>` when `MLXCEL_MODELS_DIR` and `--models-dir` are both unset. |
| `MLXCEL_MODELS_DIR` | directory path | unset (falls back to `${MLXCEL_CACHE_DIR:-$HOME/.cache/mlxcel}/models`) | Dedicated model-store root. Snapshots live directly at `$MLXCEL_MODELS_DIR/<owner>/<name>` with no `models/` subdir, so the whole store can sit on a separate volume without dragging the tokenizer-script cache along. Read by `mlxcel download`, the `-m/--model` resolver (`generate` / `serve` / `inspect` / `run`), the `mlxcel-server -m/--model` resolver, and `list` / `rm`. Resolution precedence for the models root: the `--models-dir <PATH>` CLI flag, then `MLXCEL_MODELS_DIR`, then `${MLXCEL_CACHE_DIR:-$HOME/.cache/mlxcel}/models`. (`download --local-dir <PATH>` is separate: it writes the snapshot verbatim at that exact path.) |
| `MLXCEL_DEFAULT_ORG` | HuggingFace org/user name | `mlx-community` | Org prepended to a bare, prefix-less model name (no `/`) by the `-m/--model` resolver (`generate` / `serve` / `inspect` / `run`), the `mlxcel-server -m` resolver, and the `download` verb (`mlxcel download` / `mlx-server download`), so `mlxcel run Qwen3-4B-4bit` resolves to `mlx-community/Qwen3-4B-4bit` and `mlxcel download Qwen3-4B-4bit` downloads that same repo. An explicit `owner/name` repo-id and an existing local path are unaffected. An empty/whitespace value falls back to `mlx-community`. |
| `MLXCEL_SERVER_DECODE_STORAGE` | `auto`, `dense`, `paged` | `auto` | Server continuous-batching decode storage. `--decode-storage-backend` takes precedence. Invalid values warn and fall back to `auto`. |
| `MLXCEL_KV_CACHE_BUDGET` | `auto`, unsigned integer bytes, or `none`/`0` | `auto` | Paged KV block-pool budget for continuous batching. `--kv-cache-budget` takes precedence. Defaults to `auto` (#628): it pairs with the batched-decode default so admission caps KV for the concurrent batch and returns backpressure instead of an OOM abort. Applies to pool-backed Fp16 caches under the paged decode backend (the `--parallel > 1` default); inert on the dense backend. `none` / `0` leaves the pool unbounded. |
| `MLXCEL_PREFILL_CHUNK` | unsigned integer tokens, `0` disables | `2048` | Cache-level chunked prefill for the single-sequence `mlxcel generate` / `mlxcel-bench-decode` path: the prompt is fed through the model in chunks of this many tokens, so sliding-window KV caches trim between chunks and prefill peak memory stays near the chunk size instead of scaling with the whole prompt (issues #672/#674). `0` forces the previous single-pass prefill. Models can opt out of multi-call prefill via `LanguageModel::supports_chunked_prefill`. The server path is controlled by `--prefill-chunk-size` instead. |
| `MLXCEL_ATTENTION_CHUNK_BUDGET_MB` | unsigned integer MiB, `0` disables | `1024` | **Advanced.** Score-matrix byte budget for the CUDA attention fallback. When a configuration cannot reach a fused SDPA kernel (head_dim > 128 families such as gemma-3/gemma-4, or softcap composites) and one attention call would materialize more scores than this budget, the query axis is processed in budget-sized chunks (issue #672). Larger values trade memory for fewer kernel launches; `0` restores the unchunked fallback. |
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

`--parallel` defaults to `4` (serving-throughput default, #628), so an explicit
`--ctx-size C` is divided across 4 slots by default; set `--parallel 1` to give a
single slot the full budget. The default `--ctx-size 0` (use the model's context
window per slot) is not divided. Non-batching families (SSM / hybrid) run a
single decode slot regardless of `--parallel`.

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

## Generation loop detection (issue #432)

N-gram tail repetition detection ends a generation early when the raw generated token stream collapses into a short repeated pattern (a single token such as `様様様様`, or a short block such as `abcdabcd...`). It runs on the raw stream, so it also catches loops inside the reasoning/thought channel and tool-call JSON, not just the final answer. The wire `finish_reason` is `stop`, the same as vLLM. Sampling penalties (`repeat_penalty`, DRY) cannot recover once the logits collapse, which is why this is a stop condition rather than a logit reshaper.

The detector mirrors vLLM's `SamplingParams` fields, with the same JSON names on the OpenAI chat surface:

| Field | Meaning |
|-------|---------|
| `max_pattern_size` | Largest N-gram pattern size to scan. `0` (default) disables detection. |
| `min_pattern_size` | Smallest N-gram pattern size to scan. `0` (default) is treated as `1`; clamped to `<= max_pattern_size`. |
| `min_count` | Minimum consecutive repeats of a pattern that ends generation. Must be `>= 2`; any smaller value disables detection. |

The preferred activation surface is engine-level: detection is **default-on for the Gemma 4 family with no configuration required**, so a downstream serving app needs no setup and end users see no toggle. The per-request fields and the global env var below are additional tuning surfaces, not the only way to turn it on.

| Variable | Values | Default | Notes |
|----------|--------|---------|-------|
| `MLXCEL_LOOP_DETECTION` | `off`/`0`/`none`/`false`/`disabled`, `on`/`default`/`true`/`enabled`, or `MIN,MAX,COUNT` (also `MIN:MAX:COUNT`) | unset | Global operator override for any model. Unset lets the Gemma 4 family default-on apply (non-Gemma models stay disabled). `off` force-disables for every request, including the Gemma 4 family; `on` forces the recommended threshold `1,20,4` for every model; an explicit triple (e.g. `1,20,4`) sets exact values. A malformed value warns and is ignored. |

Resolution precedence, highest first:

1. **Explicit per-request fields.** If a chat request sets any of `max_pattern_size` / `min_pattern_size` / `min_count`, those values are used verbatim, including an explicit disable (`max_pattern_size=0`). A client never has to send anything; the fields are only for tuning or opting out.
2. **Global override.** `MLXCEL_LOOP_DETECTION`, which an operator can use to force-enable, tune, or force-disable for any model.
3. **Gemma 4 family default-on.** When the loaded model is in the Gemma 4 family (`Gemma4`, `Gemma4VLM`, `Gemma4Unified`), the conservative threshold `min_pattern_size=1, max_pattern_size=20, min_count=4` is applied unconditionally. This does not require tools or a structured-output request, so plain Gemma 4 chat is covered too. Detection only ends generation when a real repetition loop is present, so a conservative default-on for this family is low risk.
4. **Disabled.** The default for every non-Gemma-4 model, preserving the exact baseline output.

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
| `MLXCEL_NVFP4_DENSE_REPACK` | `1`/`true`/`on`/`yes` (matched case-insensitively) forces the dense fallback; unset or any other value keeps the default | off (direct transcode) | **Any build (route only).** Forces the older dense f16 repack route instead of the direct ModelOpt-triplet transcode (issues #693/#705) when loading ModelOpt NVFP4 checkpoints. On CUDA, and on non-CUDA when `MLXCEL_NVFP4_NATIVE_REPACK=1` is also set, the dense route targets MLX native NVFP4; on plain non-CUDA it targets the affine 4-bit fallback, preserving the pre-#705 comparison/rollback path. Debug/parity fallback only: the direct transcode is bit-exact to the checkpoint, while the dense repack re-derives block scales and drifts by roughly one FP8/FP4 rounding step. Wins over the direct route when set. |
| `MLXCEL_NVFP4_NATIVE_REPACK` | `1`/`true`/`on`/`yes` (matched case-insensitively) selects native NVFP4 inside the dense fallback; direct native is already the default | default native on every build | **Compatibility/rollback selector (issues #694/#705).** Direct ModelOpt-triplet transcode is now the default on Metal/CPU as well as CUDA. This variable is retained as a compatibility no-op for the direct route and as the way to steer `MLXCEL_NVFP4_DENSE_REPACK=1` toward dense f16 -> native NVFP4 instead of the plain non-CUDA dense-affine rollback. |
| `MLXCEL_ENABLE_SOFTCAP_GQA_DECODE_GROUPED` | any value except `0` enables | off | Enables grouped softcap-GQA decode optimization. |
| `MLXCEL_DISABLE_SOFTCAP_GQA_DECODE_GROUPED` | `1` disables, `0` enables | unset | Legacy rollback/override for grouped softcap-GQA decode. |
| `MLXCEL_DISABLE_SINGLE_QUERY_MASKLESS` | truthy disables | maskless path on | Disables the single-query maskless attention path. |
| `MLXCEL_EXPERIMENTAL_BOOL_CAUSAL_MASK` | truthy enables | off | Enables an experimental boolean causal-mask path. |
| `MLXCEL_PAGED_ATTENTION_NATIVE` | `1`/`true`/`on`/`yes` force the native kernel; `0`/`false`/`off`/`no` force gather (case-insensitive); unset or any other value defers to the adaptive selector | selector-governed (native only for Metal + batch>=4 + ctx<=4096 + single-slab, gather otherwise) | Overrides the fused split-K Metal paged-attention decode kernel behind the library-only `paged_decode_attention_pooled` entry point (epic #116 Phase 6, #123). Since #331 an unset value no longer means "always gather": `select_pooled_paged_dispatch` picks the kernel only inside the regime ADR 0001 measured it winning, and this variable still force-pins either arm for A/B testing. #710 retired this entry point to a library-only API: it is not on the `mlxcel serve` decode path (which dispatches through the separate `DecodeBatchContext::use_native_paged_kernel` block-table kernel), so this variable is a control for external mlxcel-core consumers and `examples/paged_attention_kernel_bench.rs`, not a server knob. See [ADR 0001](adr/0001-paged-attention-gather-vs-fused-kernel.md) and #710. |
| `MLXCEL_SDPA_VECTOR_LARGE_D` | `0`/`false`/`off`/`no` disable; any other value or unset enables | on | **CUDA only.** Gates whether the CUDA `supports_sdpa_vector` check accepts head_dim 256/288 (gemma family, qwen3.5/3.6, baichuan-m1, paligemma2), routing their decode to the fused `sdpa_vector` kernels instead of the materializing SDPA fallback (issue #675). Disabling restores the prior fallback with no rebuild; used for the A/B in `benchmarks/cuda_gb10_sdpav_675_2026-07-06.csv`. |
| `MLXCEL_PIPELINE_GRANULARITY` | `off`, `layer`, `block:N` | `off` | Inserts layer-boundary async-eval hints for pipeline experiments. |
| `MLXCEL_FUSED_MOE` | `0`/`false`/`off`/`no` disable; any other value or unset enables | on | Fused single-token decode-MoE kernel (#268), on by default since #282 (Metal) and #319 (CUDA, via `mx.fast.cuda_kernel`); validated on M1 Ultra, M5, and GB10. Set to `0` to force the proven `gather_qmm`/`SwitchGLU` path. Active for qwen3_moe, qwen3_next, dots.llm1, gemma4, qwen2_moe, mixtral, phimoe, lfm2, qwen3_vl_moe, and olmoe decode. |
| `MLXCEL_FUSED_MOE_SGY` | `1`-`32` | `8` | Simdgroups (Metal) / warps-per-block (CUDA) per threadgroup for the fused decode-MoE kernel; tune per hardware. |
| `MLXCEL_FUSED_MOE_MAX_DFF` | positive int | `4096` (Metal) / `8192` (CUDA) | Expert-intermediate (Dff) upper bound for the fused path; above it the caller falls back to `gather_qmm`. The fused path wins only while `gather_qmm` underutilizes the GPU (small experts), so the break-even is backend-dependent and the default is chosen from the live backend: `4096` on Metal (M1 Ultra tuning) and `8192` on CUDA (GB10 re-measured under MLX pin e9463bb, #626; fused wins through Dff 6400 and is break-even at 8192). An explicit value overrides the default on both backends: lower it to force `gather_qmm` sooner, raise it (e.g. `20000`) to force the fused kernel on larger experts such as mixtral (Dff 14336, where it is a slight net loss). |
| `MLXCEL_FUSED_MOE_RELU2` | presence enables | off | Enables the squared-ReLU fused MoE path for nemotron-class experts; performance-neutral on nemotron-h, kept for a future MoE-dominated squared-ReLU model. |
| `MLXCEL_GATHER_QMM_GROUPED` | `0` disables; any other value or unset enables | on | **CUDA only.** Gates the sorted MoE prefill fast path in `GatherQMM::eval_gpu` (issue #629). When the sorted-indices `M == 1` prefill contract holds (right-sorted, transpose, one activation row pre-gathered per (token, expert) pair, non-nvfp4, float activation dtype, `E <= 1024`) and the batch is large enough to amortize (see `MLXCEL_GATHER_QMM_GROUPED_MIN_ROWS`), the expert weight stack is dequantized once and routed through `cutlass_grouped_gemm_unaligned` instead of one 1-row `qmm_sm80` GEMM per (token, expert) pair. Fixes a 5-10x CUDA MoE prefill collapse relative to Metal M1 Ultra; measured 3.6-40x prefill speedup across mixtral-8x7b, phi-3.5-moe, llama-4-scout, minimax-m2, gpt-oss-20b, and solar-open-100b on GB10, with no decode-path regression. Set to `0` to force the legacy per-row dispatch (A/B, rollback). See `docs/benchmark_results/moe-prefill-grouped-gemm-gb10-2026-07-10.md`. |
| `MLXCEL_GATHER_QMM_GROUPED_MIN_ROWS` | positive integer | `8` | **CUDA only.** Amortization threshold for `MLXCEL_GATHER_QMM_GROUPED`: the fast path activates only once the sorted batch size `B` reaches `min_rows * E` (E = expert count), the point past which the one-time expert dequant traffic is cheaper than the legacy per-row re-reads. Lower it to trigger the fast path on smaller sorted batches (e.g. high-concurrency batched decode crossing the same sort gate); raise it to keep more traffic on the legacy path for tuning or rollback. |
| `MLXCEL_FUSED_QK_NORM` | `1`/`true`/`on`/`yes` enable; any other value or unset disables | off (opt-in) | Fused single-token QKV projection + Q/K RMSNorm + RoPE kernel (#326) for Qwen3 and Qwen3-MoE decode. Opt-in: set to `1` to enable. Matches the graph path within RMS < 5e-3 (the reduction is over the transpose-invariant head_dim axis), but greedy temp-0 is not byte-identical over long generation; on CUDA the graph path is itself non-deterministic run-to-run from GPU FP-reduction order, while the fused path is deterministic, so its output stays inside the graph baseline's own envelope. The kernel cuts Rust/C++ FFI crossings rather than MLX op count, so it does not speed up the GPU/bandwidth-bound decode loop: on M1 Ultra it measured 1 to 3.4% slower (qwen3-0.6b 275 vs 284, qwen3-8b 82.3 vs 83.2 tok/s); on GB10/CUDA (SM 12.1) it is also slower (qwen3-0.6b 0.96x, qwen3-8b ~1.0x, qwen3-30b-a3b 0.92x fused/graph; see `docs/benchmark_results/fused-qk-norm-decode-gb10.md`), so there is no per-backend win. Ships as a reusable shared primitive for the deferred QK-norm families and stays opt-in (default off) on every measured backend (M1 Ultra, M5 Max, GB10/CUDA), mirroring the opt-in `MLXCEL_FUSED_MOE_RELU2`. Active only when `l == 1` (decode) and weights are quantized. |
| `MLXCEL_COMPILED_QGELU_MLP` | `0`/`false`/`off`/`no` disable; any other value or unset enables | on | Compiles the affine-quantized GeGLU MLP (gate/up/gelu/down) into one `mx::compile` graph so the tanh-approx GELU's ~14 element-wise ops per layer collapse into a single fused `Compiled` primitive. Covers the Gemma family (gemma/gemma2/gemma3/gemma4). The group_size=64/bits=4 case was already compiled on every shape and is unchanged; other affine quantizations (notably the group_size=64/bits=8 MLP weights in Gemma 4 mixed-precision checkpoints, issue #680) are compiled only on the single-token decode call, because compiling the 8-bit prefill GEMM measured 8-9% slower and +0.2-0.7 GB peak on GB10 (the shapeless fused graph forces a decode-oriented qmm kernel onto the large prefill matmul). GB10 gemma-4-12b decode is weight-bandwidth-bound (~94% GPU-busy), so this primitive-count cut is throughput-neutral there (measured <1% from a ~530-primitive/step reduction); it still removes real CPU dispatch and helps op-count-bound backends. Set to `0` to force the op-at-a-time fallback (A/B + rollback). It also gates the NVFP4 scaled fused MLP variant (`MLXCEL_DISABLE_FUSED_GLOBAL_SCALE`), forcing its eager fold when set to `0`. |
| `MLXCEL_DISABLE_FUSED_GLOBAL_SCALE` | `1`/`true`/`on`/`yes` (case-insensitive) disable; unset or any other value keeps the fold | off (fold on) | Rolls back the NVFP4 global-scale fold for Gemma 4 (issues #698/#705). By default the fused MLP and per-layer-input-gate C++ paths fold each per-projection `weight_scale_2` sidecar (from the direct ModelOpt transcode, issue #693/#697) into the fused kernel at the mathematically correct points: the gate scale before the GeGLU activation, the up scale on the up product, and the down scale on the fused output, each reproducing `apply_global_scale` byte-for-byte. Native NVFP4 prefill now uses a shape-specific scaled MLP graph, and standalone `UnifiedLinear` sidecar projections can apply qmm + global scale + dense bias through one C++ helper. When set, sidecar-carrying paths fall back to the op-at-a-time `UnifiedLinear::forward` scalar application (the pre-#698 bypass). Greedy temp-0 decode is token-identical across the two paths on `gemma-4-31b-it-nvfp4`; the fold removes element-wise dispatches on the gemma-4 path where CUDA graphs are disabled (#688). See `docs/benchmark_results/nvfp4-direct-transcode-gb10-2026-07-08.md` and `docs/benchmark_results/nvfp4-native-prefill-m1ultra-2026-07-09.md`. |
| `MLXCEL_FUSED_XIELU` | `0`/`false`/`off`/`no` disable; any other value or unset enables | on | Fused single-launch Metal xIELU kernel for the Apertus MLP activation (#409), on by default since the M5 Max validation. `MLP::forward` routes through one Metal dispatch covering the ~11 elementwise ops in `apertus_xielu` (square, minimum, expm1, where, and neighbors) instead of the per-op graph. Greedy temp-0 decode is byte-identical to the elementwise path on Apple Silicon: every intermediate stays in the input dtype (bf16) and the kernel reproduces MLX's `expm1f` exactly. Measured decode speedup on M1 Ultra (+2.7%, Apertus-8B 83.4 to 85.7 tok/s) and M5 Max (+1.9%, 112.0 to 114.2 tok/s), with no regression. Set to `0` to force the elementwise path. On non-Metal back-ends the FFI falls back to an equivalent elementwise graph, so the flag is safe to set everywhere. Apertus only; no other model family is affected. |

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
