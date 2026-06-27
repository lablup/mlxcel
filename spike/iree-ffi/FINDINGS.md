# FFI execution findings: driving Llama-3.2-1B through IREE from Rust (issue #449 Phase 3 M2)

## Verdict

Rust drives the real Llama-3.2-1B-Instruct through IREE on this aarch64 box
(`spark-101`, GB10), **token-exact (48/48)** against the HF temp-0 reference in
`spike/openxla/artifacts/results.json`, using only the **prebuilt IREE
distribution, with no IREE source build**. Both the FFI gate (a trivial `add`)
and the full model run pass:

```
IREE via Rust FFI: a + b = [11.0, 22.0, 33.0, 44.0]
FFI GATE: PASS

prompt tokens = 46 (bucket Lp = 64), device = local-task
loaded 146 weight tensors (4.9 GB f32) in 1.4s
prefill: 722 ms (46 tok, bucket 64) | decode: 447 ms/tok
token match vs HF temp-0: 48/48  (EXACT)
RESULT: TOKEN-EXACT PASS
```

So the substrate the `mlxcel-xla` backend needs is proven end to end: download one
tarball, build a thin C shim, FFI to it, load the #451-emitted graphs and the
weights, run the token loop. This is the shape ported into `mlxcel-xla` (the
`iree` feature).

## The substrate

IREE publishes `iree-dist-<ver>-linux-aarch64.tar.xz` on GitHub releases (85 MB).
Pinned: **`iree-3.12.0rc20260626`**. It contains everything needed:

- `lib/` static libs, including the single bundled `libiree_runtime_unified.a`
  (plus `libflatcc_*.a`),
- `include/iree/runtime/*.h` and the HAL headers (the C API),
- `bin/iree-compile` (compiles StableHLO to a vmfb matching this runtime).

Use the dist's own `iree-compile` for the emitter output so the vmfb and the
linked runtime match (the pip `iree-base-compiler` in `spike/openxla/.venv` is a
different version, `3.11.0rc20260316`, and its vmfbs need not load in this
runtime). The pip `iree-base-runtime` is Python-only (`.abi3.so`, no linkable C
lib/headers), so it is not the path; the release dist is.

## Architecture (the M2 shape, now in mlxcel-xla)

A thin C shim (`iree_gate.c`) calls the IREE runtime C API and exposes a flat C
ABI; Rust FFIs to that, so Rust never binds the IREE runtime structs directly.
The grown shim (`xla_llama_*`):

- **one session, two modules.** The #451 emitter names its modules `@prefill`
  and `@decode_step`, so both load into one `iree_runtime_session_t` and the
  calls are `prefill.main` / `decode_step.main`. (Distinct names are why one
  session can hold both.)
- **resident weights.** The 146 weights (bf16 -> f32, the emitter's exact arg
  order: embed, final_norm, then per layer down, gate, in_ln, post_ln, up, wk,
  wo, wq, wv) upload once as device buffer views and are pushed to every call.
  Read in Rust via the `safetensors` crate; the shim copies them into device
  buffers, so the host copy frees right after create.
- **threaded KV.** `prefill` returns the K/V cache; the shim keeps it and feeds
  it to each `decode_step`, replacing it with the step's output. No host
  round-trip for the cache.
- **auto-detected sampling.** The output is read by element type: a scalar `i32`
  is an on-device-argmax token (4-byte readback, the Phase 2b pattern); a `[V]`
  `f32` vector is raw logits, argmaxed on the host. The same shim drives both the
  logits-returning and the on-device-argmax vmfbs. Both validate 48/48.

The runtime call sequence:

```
instance_create -> try_create_default_device("local-task")
  -> session_create_with_device
  -> session_append_bytecode_module_from_file(prefill.vmfb)
  -> session_append_bytecode_module_from_file(decode.vmfb)
  -> hal_buffer_view_allocate_buffer_copy x146 (resident weights)
  -> per step: call_initialize_by_name("prefill.main" | "decode_step.main")
       ; push weights + step inputs (+ resident KV for decode)
       ; call_invoke ; pop (token|logits, kcache, vcache)
       ; read token (4 B scalar, or [V] d2h + host argmax)
```

## On-device argmax (the emitter addition, Phase 2b)

The #451 emitter gained `Builder::argmax`, emitting the JAX/IREE argmax reducer
(a two-operand `stablehlo.reduce` over (values, iota indices): keep the larger
value or NaN, tie-break to the lower index, so it returns the first index of the
max). `emit_decode` / `emit_prefill` take a `sample` flag; with it the graph ends
in that argmax and returns a `tensor<i32>` token instead of `[V]` logits (`emit
decode-argmax` / `prefill-argmax`). The argmax variant is **48/48 token-exact**,
so a decode step ships 4 bytes back instead of a 513 KB logits copy.

## Two non-obvious link requirements (recorded for the port)

1. **The dist leaves the system allocator to the application.** Its
   `iree_allocator_system()` is compiled out unless `IREE_ALLOCATOR_SYSTEM_CTL`
   is defined, so the shim defines a libc malloc / calloc / realloc / free
   control function and points `IREE_ALLOCATOR_SYSTEM_CTL` at it before the IREE
   headers.

2. **Linking the static runtime needs a specific order and set** (GNU ld is
   single-pass, left to right): the shim object first, then
   `--whole-archive libiree_runtime_unified.a --no-whole-archive` (keeps the
   `local-task` HAL driver registration objects), then a `--start-group` of
   `libflatcc_runtime.a` + `libflatcc_parsing.a` + `-lgcc` (aarch64
   outline-atomics) + `-lm` (CPU-kernel math) + pthread/dl.

## GPU: the prebuilt aarch64 dist has no CUDA driver

Correcting the earlier note that "the dist also has the cuda driver": it does
**not**. The dist's registered HAL drivers are `local-sync`, `local-task`, and
**vulkan** (`libiree_hal_drivers_vulkan_*`), but no CUDA. Compiling the graphs
with `--iree-hal-target-device=cuda` succeeds, but at runtime
`try_create_default_device("cuda")` fails with `no driver 'cuda' registered`.
Phase 2b's CUDA run worked because it used the **pip** `iree.runtime`, which
bundles the CUDA driver; the prebuilt static runtime we link does not. So GPU
through this static path is **Vulkan** (untried here), or it needs a
CUDA-enabled IREE build / the pip runtime. M2 lands the **CPU (local-task)**
path, token-exact, which is the correctness gate; a GPU device is a follow-up.

## Reproduce

```bash
export IREE_DIST=/home/inureyes/Development/mlxcel/spike/iree-ffi/iree-dist   # extracted dist
# emit the on-device-argmax graphs and compile with the dist iree-compile
(cd ../rust-emitter && cargo run --release -- prefill-argmax out/prefill_argmax.mlir \
                    && cargo run --release -- decode-argmax  out/decode_argmax.mlir)
IC="$IREE_DIST/bin/iree-compile"; F="--iree-input-type=stablehlo --iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu"
$IC $F ../rust-emitter/out/prefill_argmax.mlir -o prefill_argmax.vmfb
$IC $F ../rust-emitter/out/decode_argmax.mlir  -o decode_argmax.vmfb
cargo run --release --bin llama -- --prefill prefill_argmax.vmfb --decode decode_argmax.vmfb
# -> RESULT: TOKEN-EXACT PASS

cargo run --release -- add.vmfb     # the original FFI gate still passes
```

See `README.md` for the file map.
