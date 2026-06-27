// Build script for mlxcel-xla.
//
// Default / `xla-backend`-only builds do NOTHING here: the crate is pure Rust
// stubs and needs no native toolchain, so CI (which has no IREE distribution)
// builds it unchanged. Only when the `iree` feature is on do we compile the C
// shim (`csrc/xla_iree.c`) against the prebuilt IREE runtime headers.
//
// The compiled shim object propagates to a dependent binary the normal way
// (`cc` emits `rustc-link-lib=static=xla_iree` + a search path, which DO carry to
// the final link). The IREE *runtime* archives, however, are linked with
// `--whole-archive` / `--start-group` ordering that can only be expressed with
// `cargo:rustc-link-arg`, and a dependency's link-args do NOT propagate to the
// binary that links it. So that recipe lives in the consuming binary's build
// script (the root `mlxcel` crate's `build.rs`, gated on `xla-iree`), not here.
//
// Set `IREE_DIST` to the extracted `iree-dist-<ver>-linux-<arch>` tree (it has
// `include/`, `lib/`, `bin/`). Use that same dist's `bin/iree-compile` to build
// the vmfbs at runtime so the bytecode and the linked runtime match.
use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=csrc/xla_iree.c");
    println!("cargo:rerun-if-env-changed=IREE_DIST");

    // Only build the native shim under the `iree` feature.
    if env::var_os("CARGO_FEATURE_IREE").is_none() {
        return;
    }

    let dist = env::var("IREE_DIST").unwrap_or_else(|_| {
        panic!(
            "the `iree` feature is enabled but IREE_DIST is not set; point it at \
             the extracted iree-dist-<ver>-linux-<arch> tree (include/, lib/, bin/)"
        )
    });
    let dist = PathBuf::from(dist);
    let include = dist.join("include");
    assert!(
        include.join("iree/runtime/api.h").exists(),
        "IREE_DIST={} does not look like an iree-dist tree (missing include/iree/runtime/api.h)",
        dist.display()
    );

    cc::Build::new()
        .file("csrc/xla_iree.c")
        .include(&include)
        .compile("xla_iree");

    // Bake the dist path so the session can find `bin/iree-compile` at runtime
    // (a runtime `IREE_DIST` env var still takes precedence). The vmfb must be
    // compiled by the same dist whose runtime is linked into the binary.
    println!("cargo:rustc-env=MLXCEL_XLA_IREE_DIST={}", dist.display());
}
