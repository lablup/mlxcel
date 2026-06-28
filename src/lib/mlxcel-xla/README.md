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

The engine is backend-neutral at the request level (`submit` a prompt + budget,
`pump` a step, read per-request `EngineEvent`s, `cancel`); it holds no server
types, so the Stage 2c `BatchEngine` trait + server adapter wrap it unchanged.
`XlaBackend::supports_batched_serving()` stays `false` until 2c wires it into the
server.

Prove it without the server with the reference-equivalence + throughput example
(every request's batched stream must equal its independent single-seq reference):

```bash
# CPU (prebuilt dist):
IREE_DIST=/path/to/iree-dist cargo run --release --features xla-iree \
  --example xla_batch_bench -- --batch 4 --requests 8 --maxcap 24
# CUDA (GB10): source-built runtime + cuda iree-compile (as above), then:
MLXCEL_XLA_DEVICE=cuda cargo run --release --features xla-iree \
  --example xla_batch_bench -- --device cuda --batch 8 --requests 16 --maxcap 48
```

## File map

| Path | Purpose |
|------|---------|
| `src/lib.rs` | `XlaInferenceSession`: the single-sequence `InferenceSession` impl + greedy drive loop. |
| `src/iree.rs` | (feature `iree`) FFI to the shim; `IreeLlama` (single-seq) and `IreeRaggedLlama` (batched) load weights, compile + run the graphs. |
| `src/batch.rs` | (feature `iree`) `XlaBatchEngine`: the continuous-batching engine (slots + queue + admit/decode/evict) and `XlaReferenceEngine` (single-seq reference for validation). The backend-neutral `Scheduler` bookkeeping is unit-tested without IREE. |
| `csrc/xla_iree.c` | C shim over the IREE runtime C API (one session, resident weights, threaded KV; single-seq `prefill`/`decode` plus the ragged `prefill_slot`/`decode_ragged` with a device-side per-slot KV write). |
| `build.rs` | (feature `iree`) compiles the shim against `IREE_DIST` headers. The runtime link recipe lives in the **root** `mlxcel/build.rs` (a dependency's link-args do not propagate to the binary). |
| `assets/llama-3.2-1b/` | The #451-emitted `prefill` / `decode_step` StableHLO graphs (on-device-argmax variant) plus the ragged `decode_ragged_b{4,8}` graphs, compiled to vmfbs at load. |
