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
    println!("cargo:rerun-if-env-changed=IREE_CUDA_HOME");
    println!("cargo:rerun-if-env-changed=IREE_CUDA_COMPILE");
    // cfg used by src/iree.rs to select the cuda iree-compile / device behavior.
    println!("cargo:rustc-check-cfg=cfg(xla_iree_cuda)");

    // Only build the native shim under the `iree` feature.
    if env::var_os("CARGO_FEATURE_IREE").is_none() {
        return;
    }

    // CUDA mode (GB10): build against a source-built cuda-enabled IREE runtime
    // (IREE_CUDA_HOME, the iree-cuda dir with src/ and build/) instead of the
    // prebuilt CPU/vulkan dist. Mutually exclusive with the IREE_DIST path: the
    // runtime version differs and the dist's iree-compile has no cuda codegen, so
    // vmfbs are compiled by a separate cuda-capable iree-compile
    // (IREE_CUDA_COMPILE, baked here / overridable at runtime). The runtime link
    // recipe lives in the root build.rs (link-args don't propagate); it reads
    // IREE_CUDA_HOME the same way. Headers are split across the source tree
    // (src/runtime/src) and the build tree (build/runtime/src, generated).
    if let Ok(home) = env::var("IREE_CUDA_HOME") {
        let home = PathBuf::from(home);
        let src_inc = home.join("src/runtime/src");
        let bld_inc = home.join("build/runtime/src");
        assert!(
            src_inc.join("iree/runtime/api.h").exists(),
            "IREE_CUDA_HOME={} is not an iree source+build tree (missing src/runtime/src/iree/runtime/api.h)",
            home.display()
        );
        cc::Build::new()
            .file("csrc/xla_iree.c")
            .include(&src_inc)
            .include(&bld_inc)
            .define("XLA_GATE_CUDA", None)
            .compile("xla_iree");
        println!("cargo:rustc-cfg=xla_iree_cuda");
        if let Ok(ic) = env::var("IREE_CUDA_COMPILE") {
            println!("cargo:rustc-env=MLXCEL_XLA_IREE_COMPILE={ic}");
        }
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
