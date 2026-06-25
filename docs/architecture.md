# Architecture overview

`mlxcel` is a Rust inference runtime that calls MLX through a C++ bridge. The
public entry points are intentionally thin: CLI parsing happens at the edge, and
model loading, request preparation, scheduling, and MLX operations live in
focused modules.

## Top-level layout

```text
src/
├── main.rs                      # `mlxcel` CLI schema and subcommand routing
├── bin/mlx_server.rs            # standalone `mlxcel-server` binary
├── commands/                    # CLI subcommand handlers
├── execution/                   # runtime/device and sampling helpers
├── model_metadata.rs            # model-kind and loading-policy descriptors
├── backend/                     # ComputeBackend seam: which engine executes forward()
├── loading/                     # model loading routers and family registries
├── loaded_model.rs              # LoadedModel enum and LanguageModel dispatch
├── loaded_model_capabilities.rs # multimodal capability routing
├── models/                      # text model implementations and detection
├── multimodal/                  # shared multimodal prompt/runtime helpers
├── vision/                      # vision encoders, processors, connectors
├── audio/                       # audio encoder support
├── server/                      # HTTP server, request translation, scheduler
├── distributed/                 # TP/PP/DI config, transports, registries
├── tokenizer/                   # tokenizer loading helpers
├── lora/                        # LoRA adapter loading
└── lib/mlxcel-core/             # MLX C++ FFI crate and low-level generation primitives
```

## `mlxcel-core`

`src/lib/mlxcel-core/` owns the direct MLX bridge and low-level runtime pieces:

- `src/lib/mlxcel-core/src/lib.rs` — `cxx::bridge` definitions and crate exports.
- `src/lib/mlxcel-core/src/cache.rs` and `src/lib/mlxcel-core/src/cache/` — FP16/INT8/TurboQuant KV cache variants,
  paged cache layout, detach/adopt helpers, and cache tests.
- `src/lib/mlxcel-core/src/ops.rs`, `src/lib/mlxcel-core/src/dtype.rs`, `src/lib/mlxcel-core/src/streams.rs` — wrappers around common MLX
  operations and runtime concepts.
- `src/lib/mlxcel-core/src/sampling.rs` — penalties and token sampling shared by CLI/server paths.
- `src/lib/mlxcel-core/src/generate.rs` — `LanguageModel` trait and generation loops.
- `src/lib/mlxcel-core/src/drafter/` and `src/lib/mlxcel-core/src/speculative/` — speculative decoding support.
- `src/lib/mlxcel-core/src/layers.rs`, `src/lib/mlxcel-core/src/weights.rs`, `src/lib/mlxcel-core/src/utils.rs` — model building blocks,
  SafeTensors loading, masks, and helper operations.

The in-tree MLX source is under `src/lib/mlx-cpp/`; `src/lib/mlxcel-core/build.rs` builds the pinned
MLX commit and compiles the bridge code.

## Loading pipeline

A normal text generation request follows this path:

```text
model path
  → src/models/detection.rs reads config.json and returns ModelType
  → src/model_metadata.rs selects loading policy
  → src/loading/ dispatches to config-backed, non-standard, special, or VLM loader
  → tokenizer is loaded
  → LoadedModel + tokenizer are returned to CLI/server
```

Important control surfaces:

- `src/models/detection.rs` maps `config.json::model_type` and related config
  hints to `ModelType`.
- `src/model_metadata.rs` records whether a family is text or VLM, how it is
  loaded, and whether adapters are supported.
- `src/loading/config_backed.rs`, `src/loading/nonstandard.rs`,
  `src/loading/special.rs`, and `src/loading/vlm*.rs` contain the loading
  implementation.
- `src/loaded_model.rs` and `src/loaded_model_capabilities.rs` keep downstream
  CLI/server code from matching on every concrete model type.
- `src/backend/` is the compute-backend seam. CLI and server load sites call
  `select_backend().load_model(...)` rather than `loading::load_model` directly,
  so the engine that runs `LanguageModel::forward` is chosen at the load
  boundary. Under default features the seam folds to the MLX backend at compile
  time and adds no runtime dispatch; the optional `experimental-backend` feature
  reserves a slot for a future non-MLX engine (issue #338).

## Request paths

### `mlxcel generate`

1. `src/main.rs` parses CLI arguments.
2. `src/commands/generate.rs` prepares prompt/media inputs and sampling options.
3. The loading pipeline constructs a `LoadedModel`.
4. `mlxcel-core` runs the decode loop and writes output to stdout.

### `mlxcel serve` / `mlxcel-server`

1. `src/main.rs` or `src/bin/mlx_server.rs` parses CLI flags and `LLAMA_ARG_*`
   environment-backed options.
2. `src/server/startup.rs` resolves startup configuration, loads the model, and
   builds the Axum application.
3. `src/server/app.rs` mounts routes such as `/v1/chat/completions`,
   `/v1/completions`, `/v1/responses`, `/health`, and `/v1/models`. The OpenAI
   audio surface is also mounted (both `/v1`-prefixed and unversioned):
   `/v1/audio/speech` (text-to-speech), `/v1/audio/transcriptions`, and
   `/v1/audio/translations` (speech-to-text). These return a structured
   `501 Not Implemented` until a speech model is wired into the audio-model
   slot on `AppState`.
4. Route handlers translate requests into internal generation work.
5. `src/server/batch/` schedules batched decode when enabled.
6. Streaming responses are emitted as SSE frames.

#### Panic and threading posture

Release builds use `panic = "unwind"` (issue #375), so the deliberate audio worker `catch_unwind` works in production: a synthesis or transcription panic on the audio worker (`src/server/audio_worker.rs` `run_guarded`) is contained as a per-request error. That audio worker is the only deliberately contained boundary. Every core inference worker thread takes the opposite posture on purpose: `run_core_thread_or_abort` in `src/worker_failfast.rs` wraps the batched and legacy server workers (`src/server/model_worker.rs`) and the remote pipeline stage service thread (`src/distributed/pipeline/remote_service.rs`) so a panic, which signals a broken invariant, logs and aborts the process for a supervised restart rather than silently unwinding and leaving the server unable to generate. The distributed pipeline stage has no `catch_unwind` of its own; stage faults are handled at the coordinator by `Result` propagation plus stage timeout and health probing, which surface a dead or failed stage as a per-request error. There is no global abort panic hook, which would run before unwinding and defeat the audio worker backstop. An MLX C++ FFI exception still becomes `std::terminate` rather than a Rust panic and terminates the process (tracked as issue #382). See [ADR 0003](adr/0003-release-panic-unwind-with-core-thread-abort.md).

## Platform-specific behavior

- macOS/Metal and Linux/CUDA behavior is primarily determined by the pinned MLX
  build under `src/lib/mlx-cpp/` and the feature flags passed to Cargo.
- Apple Silicon runtime/device helpers live in `src/lib/mlxcel-core/src/hardware.rs`
  and `src/execution/runtime.rs`.
- Custom TurboQuant Metal kernels live under `src/lib/mlx-cpp/turbo/` and are
  called through the C++ bridge.
- CUDA kernel behavior is mostly inherited from MLX; `mlxcel` passes the CUDA
  architecture list through `MLX_CUDA_ARCHITECTURES` at build time.

## Distributed and multi-device

`src/distributed/` contains the shared cluster configuration, transport,
registry, metrics, and scheduler infrastructure used by tensor parallelism,
pipeline parallelism, and disaggregated inference experiments. See
[distributed inference](distributed.md) for the operator-facing summary.

## Further reading

- [Adding a new model](adding-models.md)
- [Supported models](supported-models.md)
- [TurboQuant KV cache](turbo-kv-cache.md)
- [OpenAI Responses API](responses-api.md)
- [Distributed inference](distributed.md)
