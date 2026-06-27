// Compile the C shim against the prebuilt IREE dist headers and link the single
// bundled runtime static lib. Point IREE_DIST at the extracted
// iree-dist-<ver>-linux-aarch64 tree (it has include/, lib/, bin/).
use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=iree_gate.c");
    println!("cargo:rerun-if-env-changed=IREE_DIST");
    println!("cargo:rerun-if-env-changed=IREE_CUDA_HOME");

    // CUDA experiment (GB10): build against a source-built cuda-enabled IREE
    // runtime instead of the CPU/vulkan prebuilt dist. Set IREE_CUDA_HOME to the
    // iree-cuda dir (it has src/ and build/). Headers are split across the source
    // (src/runtime/src) and the build (build/runtime/src, generated flatcc/config),
    // and the cuda driver is separate libs registered explicitly in the shim.
    if let Ok(home) = env::var("IREE_CUDA_HOME") {
        let home = PathBuf::from(home);
        let src_inc = home.join("src/runtime/src");
        let bld_inc = home.join("build/runtime/src");
        cc::Build::new()
            .file("iree_gate.c")
            .include(&src_inc)
            .include(&bld_inc)
            .define("XLA_GATE_CUDA", None)
            .compile("iree_gate");

        let b = home.join("build");
        for d in [
            "runtime/src/iree/runtime",
            "runtime/src/iree/hal/drivers/cuda/registration",
            "build_tools/third_party/flatcc",
            "build_tools/third_party/printf",
        ] {
            println!("cargo:rustc-link-search=native={}", b.join(d).display());
        }
        // The unified archive already bundles the cuda driver impl + local-task,
        // so whole-archive only it (re-linking the cuda impl libs would multiply-
        // define their objects). The cuda registration wrapper (just
        // driver_module.c.o, pulled by the explicit register call), IREE's
        // vendored printf (vsnprintf_/vfctprintf the unified printf.c.o needs),
        // and flatcc go in a group. The cuda driver dlopens libcuda at runtime,
        // so no link-time -lcuda.
        println!("cargo:rustc-link-arg=-Wl,--whole-archive");
        println!("cargo:rustc-link-arg=-l:libiree_runtime_unified.a");
        println!("cargo:rustc-link-arg=-Wl,--no-whole-archive");
        println!("cargo:rustc-link-arg=-Wl,--start-group");
        println!("cargo:rustc-link-arg=-l:libiree_hal_drivers_cuda_registration_registration.a");
        println!("cargo:rustc-link-arg=-l:libprintf_printf.a");
        println!("cargo:rustc-link-arg=-l:libflatcc_parsing.a");
        println!("cargo:rustc-link-arg=-lgcc");
        println!("cargo:rustc-link-arg=-lm");
        println!("cargo:rustc-link-arg=-lpthread");
        println!("cargo:rustc-link-arg=-ldl");
        println!("cargo:rustc-link-arg=-Wl,--end-group");
        return;
    }

    let dist = env::var("IREE_DIST").expect("set IREE_DIST to the extracted iree-dist tree");
    let dist = PathBuf::from(dist);
    let include = dist.join("include");
    let lib = dist.join("lib");

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
