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

CPU (`local-task`) is the proven M2 path, token-exact (48/48) vs the HF temp-0
reference. `MLXCEL_XLA_DEVICE` selects the HAL device (default `local-task`).

### Scope / limits (M2)

- The bundled graphs are authored for **Llama-3.2-1B-Instruct** specifically;
  `load` verifies `config.json` matches and errors otherwise.
- Prompt length is capped at the prefill bucket (`MAX_SEQ = 256` tokens).
- Greedy sampling only; text-only (no VLM / draft).
- **GPU:** the prebuilt aarch64 dist registers `local-sync` / `local-task` /
  `vulkan` drivers but **no CUDA**, so a GPU device needs a CUDA/Vulkan-enabled
  runtime; M2 runs on CPU.

## File map

| Path | Purpose |
|------|---------|
| `src/lib.rs` | `XlaInferenceSession`: the `InferenceSession` impl + greedy drive loop. |
| `src/iree.rs` | (feature `iree`) FFI to the shim; `IreeLlama` loads weights, compiles + runs the graphs. |
| `csrc/xla_iree.c` | C shim over the IREE runtime C API (one session, two modules, resident weights, threaded KV). |
| `build.rs` | (feature `iree`) compiles the shim against `IREE_DIST` headers. The runtime link recipe lives in the **root** `mlxcel/build.rs` (a dependency's link-args do not propagate to the binary). |
| `assets/llama-3.2-1b/` | The #451-emitted `prefill` / `decode_step` StableHLO graphs (on-device-argmax variant), compiled to vmfbs at session load. |
