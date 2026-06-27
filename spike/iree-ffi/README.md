# IREE-from-Rust FFI gate (issue #449 Phase 3 M2)

Proves Rust can drive IREE execution of a compiled vmfb on aarch64 via the
prebuilt IREE distribution, with no IREE source build. The substrate the
`mlxcel-xla` backend needs. See `FINDINGS.md` for the verdict and the linking
recipe.

Standalone (own `[workspace]`); not part of the mlxcel build.

## Files

| File | Purpose |
|------|---------|
| `iree_gate.c` | C shim over the IREE runtime C API (libc system allocator + run a vmfb). |
| `build.rs` | Compiles the shim against the dist headers, links `libiree_runtime_unified.a` + flatcc + libgcc + libm. |
| `src/main.rs` | Rust FFI to the shim; runs `add.vmfb` and checks `a + b`. |
| `add.mlir` | Trivial StableHLO (`add`) used as the test graph. |

## Reproduce

```bash
# 1. Get the prebuilt IREE dist for aarch64 (runtime libs + headers + iree-compile).
cd /tmp
curl -sSL -o iree-dist.tar.xz \
  https://github.com/iree-org/iree/releases/download/iree-3.12.0rc20260626/iree-dist-3.12.0rc20260626-linux-aarch64.tar.xz
mkdir -p iree-dist && tar xf iree-dist.tar.xz -C iree-dist
export IREE_DIST=/tmp/iree-dist

# 2. Compile the test graph to a vmfb with the dist's iree-compile.
cd <this dir>
"$IREE_DIST/bin/iree-compile" --iree-input-type=stablehlo \
  --iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu \
  add.mlir -o add.vmfb

# 3. Build + run the Rust FFI gate.
cargo run --release -- add.vmfb
# -> "FFI GATE: PASS (Rust drove an IREE vmfb on aarch64 via the prebuilt runtime)"
```

`IREE_DIST` must point at the extracted dist (it has `include/`, `lib/`, `bin/`).
