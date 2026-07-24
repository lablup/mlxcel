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
use std::process::Command;

fn cuda_nvcc() -> PathBuf {
    if let Some(path) = env::var_os("NVCC") {
        return PathBuf::from(path);
    }
    let conventional = PathBuf::from("/usr/local/cuda/bin/nvcc");
    if conventional.is_file() {
        return conventional;
    }
    let available_on_path = Command::new("nvcc")
        .arg("--version")
        .output()
        .is_ok_and(|output| output.status.success());
    if available_on_path {
        return PathBuf::from("nvcc");
    }
    panic!(
        "the CUDA IREE Gemma3n QMV build requires nvcc; set NVCC, install \
         /usr/local/cuda/bin/nvcc, or put nvcc on PATH"
    );
}

fn compile_gemma3n_qmv_ptx() {
    let source = PathBuf::from("csrc/gemma3n_qmv.cu");
    let output = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo sets OUT_DIR"))
        .join("gemma3n_qmv_sm80.ptx");
    let nvcc = cuda_nvcc();
    let mut command = Command::new(&nvcc);
    command
        .arg("-ptx")
        .arg("--std=c++17")
        .arg("-O3")
        .arg("-arch=compute_80");
    let result = command
        .arg("-o")
        .arg(&output)
        .arg(&source)
        .output()
        .unwrap_or_else(|error| panic!("failed to run {}: {error}", nvcc.display()));
    if !result.status.success() {
        panic!(
            "failed to compile Gemma3n QMV PTX with {}:\n{}",
            nvcc.display(),
            String::from_utf8_lossy(&result.stderr)
        );
    }
}

fn main() {
    println!("cargo:rerun-if-changed=csrc/xla_iree.c");
    println!("cargo:rerun-if-changed=csrc/xla_aux.c");
    println!("cargo:rerun-if-changed=csrc/gemma3n_qmv.cu");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=IREE_DIST");
    println!("cargo:rerun-if-env-changed=IREE_CUDA_HOME");
    println!("cargo:rerun-if-env-changed=IREE_CUDA_COMPILE");
    println!("cargo:rerun-if-env-changed=IREE_MACOS_HOME");
    println!("cargo:rerun-if-env-changed=IREE_MACOS_COMPILE");
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
            .file("csrc/xla_aux.c")
            .include(&src_inc)
            .include(&bld_inc)
            .define("XLA_GATE_CUDA", None)
            .compile("xla_iree");
        compile_gemma3n_qmv_ptx();
        println!("cargo:rustc-cfg=xla_iree_cuda");
        if let Ok(ic) = env::var("IREE_CUDA_COMPILE") {
            println!("cargo:rustc-env=MLXCEL_XLA_IREE_COMPILE={ic}");
        }
        return;
    }

    // macOS (Apple Silicon dev path). IREE publishes no macOS iree-dist (only
    // linux dists + python wheels), so the runtime is source-built like the CUDA
    // path and IREE_MACOS_HOME points at that iree source+build tree (src/ and
    // build/). The shim compiles against the in-tree runtime headers. The vmfbs
    // are lowered by the pinned macOS universal2 iree-compile (it has metal-spirv
    // codegen), set via IREE_MACOS_COMPILE (baked here) or MLXCEL_XLA_IREE_COMPILE
    // at runtime. The runtime link recipe (Apple ld -force_load + Metal
    // frameworks) lives in the root build.rs. scripts/iree/setup-macos.sh
    // produces this tree and prints the matching env.
    #[cfg(target_os = "macos")]
    if let Ok(home) = env::var("IREE_MACOS_HOME") {
        let home = PathBuf::from(home);
        let src_inc = home.join("src/runtime/src");
        let bld_inc = home.join("build/runtime/src");
        assert!(
            src_inc.join("iree/runtime/api.h").exists(),
            "IREE_MACOS_HOME={} is not an iree source+build tree (missing \
             src/runtime/src/iree/runtime/api.h); run scripts/iree/setup-macos.sh",
            home.display()
        );
        cc::Build::new()
            .file("csrc/xla_iree.c")
            .file("csrc/xla_aux.c")
            .include(&src_inc)
            .include(&bld_inc)
            .compile("xla_iree");
        if let Ok(ic) = env::var("IREE_MACOS_COMPILE") {
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
        .file("csrc/xla_aux.c")
        .include(&include)
        .compile("xla_iree");

    // Bake the dist path so the session can find `bin/iree-compile` at runtime
    // (a runtime `IREE_DIST` env var still takes precedence). The vmfb must be
    // compiled by the same dist whose runtime is linked into the binary.
    println!("cargo:rustc-env=MLXCEL_XLA_IREE_DIST={}", dist.display());
}
