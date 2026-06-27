// Compile the C shim against the prebuilt IREE dist headers and link the single
// bundled runtime static lib. Point IREE_DIST at the extracted
// iree-dist-<ver>-linux-aarch64 tree (it has include/, lib/, bin/).
use std::env;
use std::path::PathBuf;

fn main() {
    let dist = env::var("IREE_DIST").expect("set IREE_DIST to the extracted iree-dist tree");
    let dist = PathBuf::from(dist);
    let include = dist.join("include");
    let lib = dist.join("lib");

    println!("cargo:rerun-if-changed=iree_gate.c");
    println!("cargo:rerun-if-env-changed=IREE_DIST");

    cc::Build::new()
        .file("iree_gate.c")
        .include(&include)
        .compile("iree_gate");

    println!("cargo:rustc-link-search=native={}", lib.display());
    // Emit all of the IREE link bits as ordered link-args so they land AFTER the
    // shim (which the cc crate links first and which references IREE symbols),
    // and so libgcc lands AFTER the runtime archive: the aarch64 outline-atomics
    // helpers (__aarch64_ldadd4_acq_rel, __aarch64_cas8_rel, ...) the runtime
    // uses are defined in libgcc, and ld is single-pass left-to-right.
    //
    // --whole-archive keeps the HAL driver registration objects (local-task)
    // that try_create_default_device needs from being dropped as unreferenced.
    println!("cargo:rustc-link-arg=-Wl,--whole-archive");
    println!("cargo:rustc-link-arg=-l:libiree_runtime_unified.a");
    println!("cargo:rustc-link-arg=-Wl,--no-whole-archive");
    // The unified archive references flatcc (vmfb/flatbuffer parsing) and the
    // aarch64 outline-atomics in libgcc. Wrap them in a group so ld re-scans to
    // resolve any cross-references regardless of order.
    // flatcc (vmfb parsing), libgcc (aarch64 outline atomics), libm (CPU kernel
    // math: expf/tanh/erf/...), and pthread/dl all sit after the runtime archive
    // because the unified archive references them; the group lets ld re-scan.
    println!("cargo:rustc-link-arg=-Wl,--start-group");
    println!("cargo:rustc-link-arg=-l:libflatcc_runtime.a");
    println!("cargo:rustc-link-arg=-l:libflatcc_parsing.a");
    println!("cargo:rustc-link-arg=-lgcc");
    println!("cargo:rustc-link-arg=-lm");
    println!("cargo:rustc-link-arg=-lpthread");
    println!("cargo:rustc-link-arg=-ldl");
    println!("cargo:rustc-link-arg=-Wl,--end-group");
}
