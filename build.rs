// Root build script for the `mlxcel` binary.
//
// It does NOTHING for default / `xla-backend` builds, so Apple-Silicon, CUDA, and
// CI builds are unaffected. Only under the `xla-iree` feature (real OpenXLA
// execution, issue #449 Phase 3) does it emit the IREE *runtime* link recipe.
//
// Why here and not in `mlxcel-xla`: the C shim (`mlxcel-xla/csrc/xla_iree.c`) is
// compiled by that crate's build script and its object links via the normal
// `rustc-link-lib` path. But the runtime static libs need `--whole-archive`
// (to keep the local-task HAL driver registration) and a `--start-group`, which
// can only be expressed with `cargo:rustc-link-arg`, and a *dependency's*
// link-args do not propagate to the binary that links it. The binary's own build
// script is the one place those args reach the final link, so the recipe lives
// here. (Proven in spike/iree-ffi; see its FINDINGS.md.)
use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=IREE_DIST");

    // The feature env is set for this crate's own enabled features; `xla-iree`
    // becomes `CARGO_FEATURE_XLA_IREE`.
    if env::var_os("CARGO_FEATURE_XLA_IREE").is_none() {
        return;
    }

    let dist = env::var("IREE_DIST").expect(
        "the `xla-iree` feature is enabled but IREE_DIST is not set; point it at \
         the extracted iree-dist-<ver>-linux-<arch> tree (include/, lib/, bin/)",
    );
    let lib = PathBuf::from(dist).join("lib");
    assert!(
        lib.join("libiree_runtime_unified.a").exists(),
        "IREE_DIST lib dir {} is missing libiree_runtime_unified.a",
        lib.display()
    );
    println!("cargo:rustc-link-search=native={}", lib.display());

    // GNU ld is single-pass, left to right. The shim object (linked via
    // mlxcel-xla's rustc-link-lib) references the runtime; `--whole-archive` on
    // the unified runtime archive forces in all its objects (including the
    // local-task HAL driver registration that try_create_default_device needs),
    // so the shim's references resolve regardless of order. flatcc (vmfb
    // parsing), libgcc (aarch64 outline atomics), and libm (CPU-kernel math)
    // sit after the runtime in a group so ld re-scans cross-references.
    for arg in [
        "-Wl,--whole-archive",
        "-l:libiree_runtime_unified.a",
        "-Wl,--no-whole-archive",
        "-Wl,--start-group",
        "-l:libflatcc_runtime.a",
        "-l:libflatcc_parsing.a",
        "-lgcc",
        "-lm",
        "-lpthread",
        "-ldl",
        "-Wl,--end-group",
    ] {
        println!("cargo:rustc-link-arg={arg}");
    }
}
