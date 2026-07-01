# mlxcel-xla

OpenXLA / StableHLO compiler-family inference backend for mlxcel (issue #449,
ADR 0004 Track B). **Default-off.** The root crate compiles this only under the
`xla-backend` feature, so Apple-Silicon and CUDA shipping builds never touch it.

It hosts [`XlaInferenceSession`], which fills in the engine-neutral
`InferenceSession` contract from `mlxcel-core` (token-level `prefill` /
`decode_step` with on-device sampling). A model is authored once as a StableHLO
graph (the Rust emitter, issue #451), `iree-compile` lowers `prefill` and a
single-token `decode_step` to a vmfb, and the IREE runtime executes them with the
weights resident on the device and the next-token argmax computed on-device.

## Two feature gates

| Build | What compiles | Needs IREE dist? |
|-------|---------------|------------------|
| default / `--features xla-backend` (root) | Pure Rust: the crate + seam. `prefill` / `decode_step` return a clear "built without `iree`" error. | No (so CI builds it) |
| `--features xla-iree` (root) = `xla-backend` + `mlxcel-xla/iree` | Real execution: the C shim (`csrc/xla_iree.c`) is compiled against the prebuilt IREE runtime and the session drives the bundled graphs. | Yes (`IREE_DIST`) |

Why the split: `--features xla-backend` must stay buildable in CI, which has no
IREE distribution. The native execution path is behind the extra `iree` feature.

## Running it (Phase 3 M2)

```bash
# 1. Get the prebuilt IREE dist (runtime static libs + headers + iree-compile).
#    Pin the version used to author the bundled vmfbs (see spike/iree-ffi).
export IREE_DIST=/path/to/extracted/iree-dist-<ver>-linux-<arch>

# 2. Build with real execution on.
cargo build --release --features xla-iree

# 3. Select the backend at runtime and generate.
MLXCEL_BACKEND=xla ./target/release/mlxcel generate \
  -m <Llama-3.2-1B-Instruct dir> -p "..." -n 48
```

CPU (`local-task`) is the proven path, token-exact (48/48) vs the HF temp-0
reference. `MLXCEL_XLA_DEVICE` selects the HAL device (default `local-task`).

### CUDA (GPU) build

The prebuilt dist is CPU/Vulkan only (no CUDA driver, and its `iree-compile` has
no CUDA codegen). The CUDA path therefore uses a **source-built cuda-enabled IREE
runtime** plus a **cuda-capable `iree-compile`**, version-matched to each other.
It is a separate, mutually-exclusive build mode (set `IREE_CUDA_HOME` instead of
`IREE_DIST`).

```bash
# 1. Build the IREE *runtime* from source at the version your iree-compile uses
#    (runtime only -> no LLVM; skip the third_party/llvm-project submodule):
git clone --depth 1 --branch <iree-tag> https://github.com/iree-org/iree.git src
git -C src -c submodule."third_party/llvm-project".update=none \
    submodule update --init --recursive --depth 1
cmake -S src -B build -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release \
  -DIREE_BUILD_COMPILER=OFF -DIREE_HAL_DRIVER_DEFAULTS=OFF \
  -DIREE_HAL_DRIVER_LOCAL_TASK=ON -DIREE_HAL_DRIVER_LOCAL_SYNC=ON \
  -DIREE_HAL_DRIVER_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda
make -C build -j"$(nproc)" iree_runtime_unified

# 2. Point the build at it; provide a cuda-capable iree-compile (matching version).
export IREE_CUDA_HOME=/abs/path/to/that/iree   # the dir holding src/ and build/
export IREE_CUDA_COMPILE=/abs/path/to/iree-compile   # cuda codegen, version-matched
cargo build --release --features xla-iree

# 3. Run on the GPU.
MLXCEL_BACKEND=xla MLXCEL_XLA_DEVICE=cuda ./target/release/mlxcel generate \
  -m <Llama-3.2-1B-Instruct dir> -p "..." -n 48
```

Validated on a GB10 (Grace-Blackwell, sm_121): token-exact 48/48, ~5 tok/s
(~2.6x the CPU path). Vulkan via the prebuilt dist does **not** work on the GB10
(IREE's Vulkan allocator vs NVIDIA's unified memory), so CUDA is the GPU path.

### macOS (Apple Silicon, Metal) build

On Apple Silicon, **MLX is the default and primary backend**; this OpenXLA path
is a development / parity path, opt-in exactly like CUDA. IREE publishes no macOS
`iree-dist` (only linux dists + python wheels), so the runtime is **source-built**
like the CUDA path. `scripts/iree/setup-macos.sh` automates it: it installs the
pinned macOS `iree-compile` (metal-spirv codegen) from the universal2 wheel into a
private venv, source-builds the IREE runtime (`local-task`/`local-sync`/`metal`
drivers), and prints the env.

```bash
# 1. One-time setup (clones + builds the IREE runtime; idempotent).
eval "$(scripts/iree/setup-macos.sh)"        # sets IREE_MACOS_HOME, MLXCEL_XLA_IREE_COMPILE
# (later shells: eval "$(scripts/iree/setup-macos.sh --env)")

# 2. Build with real execution on (alongside the usual MLX features).
cargo build --release --features metal,accelerate,xla-iree

# 3. Opt into XLA at runtime; MLX stays default when MLXCEL_BACKEND is unset.
#    On Apple Silicon the XLA device defaults to `metal`, so just:
MLXCEL_BACKEND=xla ./target/release/mlxcel generate \
  -m <Llama-3.2-1B-Instruct dir> -p "The capital of France is" -n 48
#    Force the CPU path (token-exact vs the HF temp-0 reference) if you need it:
MLXCEL_BACKEND=xla MLXCEL_XLA_DEVICE=local-task ./target/release/mlxcel generate \
  -m <Llama-3.2-1B-Instruct dir> -p "The capital of France is" -n 48
```

On Apple Silicon `MLXCEL_XLA_DEVICE` defaults to `metal` (the GPU); set it to
`local-task` to force the CPU path. Validated on an M1 Ultra: `metal` is
token-exact with the `local-task` path (which is itself token-exact vs the HF
temp-0 reference) and ~1.9x its tok/s.

This is a correctness / parity path, **not** a performance path: on the same
M1 Ultra and model (Llama-3.2-1B-Instruct, greedy, 64 tokens), MLX runs at
~186 tok/s versus ~1.6 tok/s for XLA-on-Metal (~117x), because the bundled XLA
graphs are f32 and use generic metal-spirv codegen with a per-step host logits
readback, while MLX uses bf16 hand-tuned Metal kernels. MLX remains the
production Apple-Silicon backend; XLA-on-Metal exists to run the same StableHLO
graphs that target CUDA/Linux on a Mac for development and parity.

The root `build.rs` macOS arm uses Apple `ld` (`-force_load`, no
`--whole-archive` / `--start-group` / `-lgcc`) and links the
Metal/Foundation/QuartzCore frameworks; the C shim is unchanged (the `metal`
driver registers via `use_all_available_drivers`).

### Precision (low-precision matmul)

The contraction (matmul) input precision is authored in the emitted StableHLO
graph, so it applies on every IREE target (CPU, CUDA, Metal, and future NPUs),
not just one backend:

- `f16` / `bf16`: demote the f32 inputs of every `dot_general` to the narrow
  type while keeping the f32 accumulate and output, so only the matmuls change
  and the sensitive elementwise ops (norm, softmax, RoPE) stay f32. A blanket
  program-wide f32 to f16 is deliberately not done (it regressed accuracy and was
  slower).
- `f32`: no demotion.

**Default is per device:** `f16` on the GPU devices (`metal`, `cuda`), `f32` on
the CPU (`local-task` / `local-sync`). `MLXCEL_XLA_PRECISION` (`f16` | `bf16` |
`f32`) overrides the default on any device. So on Apple Silicon the Metal path
runs f16 by default:

```bash
# f16 by default on metal (the device default); no env needed:
MLXCEL_BACKEND=xla ./target/release/mlxcel generate \
  -m <Llama-3.2-1B-Instruct dir> -p "..." -n 48
# Force f32 for a token-exact reference:
MLXCEL_XLA_PRECISION=f32 MLXCEL_BACKEND=xla ./target/release/mlxcel generate \
  -m <Llama-3.2-1B-Instruct dir> -p "..." -n 48
```

On an M1 Ultra (Llama-3.2-1B-Instruct, greedy) `f16` is ~1.9x the `f32` tok/s and
token-exact with it. The same graph change also speeds up the CPU path. Weights
are still uploaded f32 and demoted in the graph; keeping the resident weights in
the narrow type (a bandwidth win that needs the f16 weight FFI) lands with the
quantized-weight path. This is a transferable, correctness-first lever; it does
not close the gap to MLX (see the perf note above).

### Scope / limits

- The bundled graphs are authored for **Llama-3.2-1B-Instruct** specifically;
  `load` verifies `config.json` matches and errors otherwise.
- Prompt length is capped at the prefill bucket (`MAX_SEQ = 256` tokens).
- Greedy sampling only; text-only (no VLM / draft).
- `XlaInferenceSession` is single-sequence; `XlaBatchEngine` (below) adds
  multi-sequence throughput.

## Batched continuous batching (Stage 2b)

`XlaBatchEngine` runs many sequences at once: `B_max` slots share one rank-5 KV
cache and serve a request stream, so the device stays full. Requests of different
lengths join and leave the batch at different times; a freed slot is recycled by
a new request whose prompt KV is written **device-side** into just that slot
(no host round-trip), so admitting one sequence does not disturb the others. The
ragged decode graph advances every active slot one token per step, each at its
own position. Greedy, fixed `B_max ∈ {4, 8}` (the bundled ragged graphs),
contiguous per-slot KV.

The engine is backend-neutral at the request level (`submit` a prompt + budget +
[`SampleParams`](src/sampler.rs), `pump` a step, read per-request `EngineEvent`s,
`cancel`); it holds no server types, so the server adapter wraps it unchanged.

### Sampling (Stage 2d)

The engine reads the per-row logits back to the host (the logits graph variant)
and samples there, so it honors **temperature, top-k, top-p, min-p, and a seed**.
Greedy (`temperature 0`) is host argmax of the logits, token-exact with the
single-sequence argmax path. Reading the full `[B, vocab]` logits per step is a
small fraction of the decode matmuls; on-device sampling is a later optimization.

### Serving (Stage 2c/2d)

On an `xla-iree` build, `mlxcel-server` serves through this engine when
`MLXCEL_BACKEND=xla` is selected: a server-side `XlaServeWorker` adapts the engine
to the server's backend-neutral `BatchEngine` contract (the MLX scheduler
implements the same contract), so the HTTP path is unchanged.

```bash
MLXCEL_BACKEND=xla MLXCEL_XLA_DEVICE=cuda ./target/release/mlxcel-server \
  -m <Llama-3.2-1B-Instruct dir> --port 8080
# then POST /v1/completions (temperature / top_p / top_k / seed are honored).
```

The serve path is text-only: it honors `max_tokens`, the model's EOS ids, and
sampling (temperature / top-k / top-p / min-p / seed). The history-based penalties
(repetition / frequency / presence / DRY) are not applied (logged once). Requests
it cannot serve faithfully are rejected with a clear error rather than served
wrong: logprobs, structured / JSON-schema output, and multimodal inputs. Stop
strings are not enforced yet. `--max-batch-size` maps to the engine's bundled
`B_max` (`>= 8` -> 8, else 4).

Prove the engine without the server with the reference-equivalence + throughput
example (every request's batched stream must equal its independent single-seq
reference):

```bash
# CPU (prebuilt dist):
IREE_DIST=/path/to/iree-dist cargo run --release --features xla-iree \
  --example xla_batch_bench -- --batch 4 --requests 8 --maxcap 24
# CUDA (GB10): source-built runtime + cuda iree-compile (as above), then:
MLXCEL_XLA_DEVICE=cuda cargo run --release --features xla-iree \
  --example xla_batch_bench -- --device cuda --batch 8 --requests 16 --maxcap 48
```

## Per-architecture validation harness (#496)

Validating an architecture has two tiers, split so the cheap one can gate every
change while the expensive one stays opt-in. Both are reusable, so adding a family
is turnkey.

**Structural (byte-exact, pure Rust, no GPU).** The emitter must reproduce a
frozen StableHLO golden for each registered architecture, byte for byte. This is
the fast regression gate that catches a graph which drifted from its validated
form. It is the `src/validation.rs` module, run as a normal test:

```bash
cargo test -p mlxcel-xla --lib validation
```

`validation::REGISTERED` lists the fixtures (currently `llama-3.2-1b`, whose
goldens live in `assets/llama-3.2-1b/`). The check honors `MLXCEL_XLA_PRECISION`:
the goldens are the default `f32` graphs, so a byte-exact run under `f16` / `bf16`
is rejected with a clear message instead of a confusing diff.

**Execution (token-exact + reference-exact).** One command produces the HF fp32
oracle and runs both run gates (needs a real IREE build and a checkpoint):

```bash
export IREE_DIST=/path/to/iree-dist          # or IREE_CUDA_HOME=... for CUDA
scripts/xla/validate_arch.sh --model <checkpoint dir> --device local-task
```

It (0) runs the structural pre-gate, (1) produces the oracle with
`spike/openxla/oracle_continuation.py` (loads the checkpoint in fp32,
dequantizing an MLX 4-bit / 8-bit checkpoint offline first with the same affine
formula as `src/weights.rs`), (2) runs `xla_oracle_check` (single-seq greedy ==
HF oracle), and (3) runs `xla_batch_bench` (every batched request == its
single-seq reference). It exits non-zero if any run gate fails.
`--structural-only` runs just the pure-Rust pre-gate (no IREE, no GPU).

### Adding a family is turnkey

1. Emit correctly: extend `Config::from_json` and the emitter for the
   architecture (the structural invariant tests in `emitter/mod.rs` cover the new
   switches: RoPE kind, q/k/v bias, tied / untied head, soft-caps).
2. Prove it: `scripts/xla/validate_arch.sh --model <checkpoint>` must report a
   clean token-exact + reference-exact pass on a real checkpoint.
3. Freeze goldens (optional, for a byte-exact CI guard): emit each graph with
   `validation::emit_graphs(config_json, kinds)` and write them to
   `assets/<arch>/*.mlir`; an already-registered family re-freezes in place with
   `MLXCEL_FREEZE_GOLDENS=1 cargo test -p mlxcel-xla --lib validation::tests::freeze_goldens`.
4. Register: add an `ArchFixture` to `validation::REGISTERED`; the byte-exact test
   then guards it forever.

Not every family bundles goldens: Qwen2.5 (`assets/qwen2.5-0.5b/`) is emitted at
load and covered by the emitter's structural tests plus the execution tier, with
no committed `.mlir`. The harness still drives its emit through `emit_graphs`.

## Performance and the low-precision decision

This backend is a portability / parity path, not a performance path. On Apple
Silicon, MLX is and remains the production backend; XLA runs the same StableHLO
graphs that target CUDA/Linux on a Mac, for development and cross-checking.

### Measured (M1 Ultra, Llama-3.2-1B-Instruct, greedy)

| | Metal | CPU (`local-task`) |
|---|---|---|
| one decode step, f32 | ~600 ms | ~233 ms |
| one decode step, f16 | ~291 ms (~2.1x) | ~187 ms (~1.25x) |
| end-to-end, f32 | ~1.5 tok/s | ~0.75 tok/s |
| end-to-end, f16 | ~3.0 tok/s | - |
| MLX (reference) | ~186 tok/s | - |

The decode-step figures are `iree-benchmark-module` (pure runtime, no host glue).

### Where the time goes

The Metal decode step is ~600 ms with the GPU busy the whole time (its host-side
`process_time` is ~50 ms), so the time is in the GPU kernels, not invoke overhead
or host round-trips. On the **same** StableHLO graph a 13-thread CPU (~233 ms)
beats the Metal GPU (~600 ms), which is the tell: the bottleneck is IREE's
`metal-spirv` kernel codegen (generic, unfused MSL via SPIRV-Cross), not
bandwidth. ~600 ms is ~110x the ~5 ms/token bandwidth floor MLX runs at.

### What is and is not worth optimizing

- **Graph-level (precision, quantization, op selection): in scope, transferable.**
  Authored once in the portable graph, it helps every IREE target (CPU, CUDA,
  Metal, future NPUs). f16 (landed) is ~1.9x and token-exact, and speeds up the
  CPU path too. For NPUs, low precision / quantization is not a 2x optimization
  but the entry ticket (they are int8 / fp16 native). This is where investment
  pays off.
- **Per-backend kernel codegen (the remaining ~50x to MLX): out of scope.** That
  is upstream IREE's job, is Metal-specific (does not transfer to non-SPIR-V
  NPUs), and MLX already owns Apple-Silicon performance.

So Metal's absolute tok/s is a *pessimistic* proxy for an NPU, which brings its
own optimized kernels; what transfers is the graph, not the Metal tuning.

### Decision (2026-07)

- **In scope:** graph-level low precision. f16 / bf16 is landed. int8 / int4
  weight quantization is the NPU lever, but its payoff is memory bandwidth, which
  a compute-bound Metal decode cannot demonstrate and which needs an actual NPU to
  measure; it is **deferred to a hardware-gated follow-up** (on Metal only its
  token-exactness would be verifiable, not the speedup).
- **Out of scope:** hand-writing Metal kernels or tuning IREE's `metal-spirv`
  codegen to chase MLX.

## File map

| Path | Purpose |
|------|---------|
| `src/lib.rs` | `XlaInferenceSession`: the single-sequence `InferenceSession` impl + greedy drive loop. |
| `src/iree.rs` | (feature `iree`) FFI to the shim; `IreeLlama` (single-seq) and `IreeRaggedLlama` (batched) load weights, compile + run the graphs. |
| `src/batch.rs` | (feature `iree`) `XlaBatchEngine`: the continuous-batching engine (slots + queue + admit/decode/evict) and `XlaReferenceEngine` (single-seq reference for validation). The backend-neutral `Scheduler` bookkeeping is unit-tested without IREE. |
| `src/validation.rs` | (issue #496) Reusable per-architecture structural harness: the `ArchFixture` registry + `check_arch` byte-exact golden gate + `emit_graphs` freeze primitive. Pure Rust; runs under `cargo test`. The execution tier lives in `scripts/xla/validate_arch.sh` + `spike/openxla/oracle_continuation.py`. |
| `csrc/xla_iree.c` | C shim over the IREE runtime C API (one session, resident weights, threaded KV; single-seq `prefill`/`decode` plus the ragged `prefill_slot`/`decode_ragged` with a device-side per-slot KV write). |
| `build.rs` | (feature `iree`) compiles the shim against `IREE_DIST` headers. The runtime link recipe lives in the **root** `mlxcel/build.rs` (a dependency's link-args do not propagate to the binary). |
| `assets/llama-3.2-1b/` | The #451-emitted `prefill` / `decode_step` StableHLO graphs (on-device-argmax variant) plus the ragged `decode_ragged_b{4,8}` graphs, compiled to vmfbs at load. |
