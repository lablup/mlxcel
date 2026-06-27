# FFI gate findings: running IREE from Rust on aarch64 (issue #449 Phase 3 M2)

## Verdict

The gate passes. A Rust program drives IREE execution of a compiled vmfb on this
aarch64 box (`spark-101`, GB10) and gets the correct result, using only the
**prebuilt IREE distribution, with no IREE source build**. So the substrate the
`mlxcel-xla` backend needs is feasible and cheap: download one tarball, build a
thin C shim, FFI to it. The earlier worry (build libIREE from source for aarch64)
is avoided.

```
IREE via Rust FFI: a + b = [11.0, 22.0, 33.0, 44.0]
FFI GATE: PASS
```

## The substrate

IREE publishes `iree-dist-<ver>-linux-aarch64.tar.xz` on GitHub releases (85 MB).
It contains everything needed:

- `lib/` static libs, including the single bundled `libiree_runtime_unified.a`
  (plus `libflatcc_*.a`),
- `include/iree/runtime/*.h` and the HAL headers (the C API),
- `bin/iree-compile` (compiles StableHLO to a vmfb matching this runtime).

So the pip `iree-base-runtime` (Python-only `.abi3.so`, no linkable C lib or
headers) is not the path; the release dist is.

## Architecture (this is the M2 shape)

A thin C shim (`iree_gate.c`) calls the IREE runtime C API and exposes a flat C
ABI; Rust FFIs to that, so Rust never binds the IREE runtime structs directly.
`mlxcel-xla` will use the same split: a C shim over the runtime, a Rust FFI, with
the shim loading the prefill / decode_step vmfbs (the #451 emitter output) and the
weights and running the token loop.

The runtime call sequence used:

```
instance_options_initialize / use_all_available_drivers / instance_create
  -> instance_try_create_default_device("local-task")
  -> session_create_with_device
  -> session_append_bytecode_module_from_file(vmfb)
  -> call_initialize_by_name("module.main")
  -> hal_buffer_view_allocate_buffer_copy (inputs)
  -> call_inputs_push_back_buffer_view ; call_invoke
  -> call_outputs_pop_front_buffer_view ; hal_device_transfer_d2h (read output)
```

## Two non-obvious requirements (the time sinks, recorded for M2)

1. **The dist leaves the system allocator to the application.** Its
   `iree_allocator_system()` is compiled out unless `IREE_ALLOCATOR_SYSTEM_CTL`
   is defined, so the shim defines a libc malloc / calloc / realloc / free control
   function and points `IREE_ALLOCATOR_SYSTEM_CTL` at it before the IREE headers.

2. **Linking the static runtime needs a specific order and set** (GNU ld is
   single-pass, left to right):
   - the shim object first (it references IREE),
   - `--whole-archive libiree_runtime_unified.a --no-whole-archive` so the
     `local-task` HAL driver registration objects are not dropped (otherwise
     `try_create_default_device` finds no device),
   - then a `--start-group` of `libflatcc_runtime.a` + `libflatcc_parsing.a`
     (vmfb/flatbuffer parsing), `-lgcc` (the aarch64 outline-atomics
     `__aarch64_ldadd4_acq_rel` etc.), `-lm` (CPU-kernel math: `expf`, `tanh`,
     `erf`, ...), `-lpthread`, `-ldl`.

   `build.rs` emits these as ordered `cargo:rustc-link-arg`s so they land after
   the cc-linked shim.

## What this means for M2 / Phase 3

- `mlxcel-xla` gains a C shim + `build.rs` of this shape, with the IREE dist as a
  build input (download to a known path, or vendor, set via an env var like
  `IREE_DIST`). Pin one dist version and use its `iree-compile` for the emitter
  output so the vmfb and runtime match.
- The shim grows from "run add" to: load the prefill and decode vmfbs, upload the
  (int4 or fp16) weights as resident device buffers once, and run prefill /
  decode_step with on-device argmax (the Phase 2b pattern), returning a token id.
- CPU (`local-task`) is proven here; the dist also has the `cuda` driver, so the
  same shim runs on the GB10 by selecting the cuda device (Phase 2b showed the
  GPU path works through IREE).

## Reproduce

See `README.md`.
