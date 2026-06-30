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
    println!("cargo:rerun-if-env-changed=IREE_CUDA_HOME");
    println!("cargo:rerun-if-env-changed=IREE_MACOS_HOME");

    // The feature env is set for this crate's own enabled features; `xla-iree`
    // becomes `CARGO_FEATURE_XLA_IREE`.
    if env::var_os("CARGO_FEATURE_XLA_IREE").is_none() {
        return;
    }

    // macOS (Apple Silicon dev path): Apple ld, not GNU ld. IREE ships no macOS
    // iree-dist, so the runtime is the source-built libiree_runtime_unified.a
    // under IREE_MACOS_HOME. Apple ld uses -force_load (not --whole-archive), is
    // multi-pass (no --start-group), and has no libgcc. force_load the unified
    // runtime so use_all_available_drivers + the VM keep all their objects; link
    // the per-driver and flatcc archives normally so only the members unified
    // lacks (the driver registration objects) are pulled, without duplicating the
    // impl objects unified already bundles. The Metal HAL driver needs the system
    // frameworks.
    #[cfg(target_os = "macos")]
    {
        let home = env::var("IREE_MACOS_HOME").unwrap_or_else(|_| {
            panic!(
                "the `xla-iree` feature is enabled on macOS but IREE_MACOS_HOME is \
                 not set (there is no prebuilt macOS iree-dist); run \
                 scripts/iree/setup-macos.sh and export the env it prints"
            )
        });
        let build_root = PathBuf::from(home).join("build");
        let runtime = build_root.join("runtime/src");
        let unified = runtime.join("iree/runtime/libiree_runtime_unified.a");
        assert!(
            unified.exists(),
            "IREE_MACOS_HOME build is missing {} (run scripts/iree/setup-macos.sh first)",
            unified.display()
        );
        // force_load ONLY the unified runtime: all of its objects must be present
        // for use_all_available_drivers + the VM, and it already bundles the
        // local-task / metal driver *impl* objects. The per-driver archives and
        // flatcc are linked normally (no -force_load) so the linker pulls only the
        // members unified lacks -- chiefly each driver's *registration* object,
        // which unified's register-all references. force_loading those archives
        // too would duplicate the impl objects unified already carries.
        println!("cargo:rustc-link-arg=-Wl,-force_load,{}", unified.display());
        let mut extra = Vec::new();
        collect_driver_archives(&runtime.join("iree/hal/drivers"), &mut extra);
        // flatcc (vmfb FlatBuffer verify/parse) is referenced by the runtime and
        // the metal driver but lives outside the runtime tree and is not bundled
        // into the unified archive, so link it explicitly.
        let flatcc = build_root.join("build_tools/third_party/flatcc/libflatcc_parsing.a");
        assert!(
            flatcc.exists(),
            "IREE_MACOS_HOME build is missing {} (run scripts/iree/setup-macos.sh first)",
            flatcc.display()
        );
        extra.push(flatcc);
        for a in &extra {
            println!("cargo:rustc-link-arg={}", a.display());
        }
        // The Metal HAL driver is Objective-C against the system frameworks.
        for fw in ["Metal", "Foundation", "QuartzCore"] {
            println!("cargo:rustc-link-arg=-framework");
            println!("cargo:rustc-link-arg={fw}");
        }
        println!("cargo:rustc-link-lib=c++");
        return;
    }

    // The CUDA and Linux-dist recipes below are GNU-ld only and never apply on
    // macOS (handled by the arm above), so they are gated out there to keep the
    // macOS build free of dead code.
    #[cfg(not(target_os = "macos"))]
    {
        // CUDA mode (GB10): link the source-built cuda-enabled IREE runtime instead
        // of the prebuilt dist (mutually exclusive; IREE_CUDA_HOME wins). The
        // source-built `libiree_runtime_unified.a` already bundles the cuda driver
        // impl + local-task, so whole-archive only it (re-linking the separate cuda
        // impl libs would multiply-define their objects); the cuda registration
        // wrapper (driver_module.c.o, pulled by the shim's explicit register call),
        // IREE's vendored printf (the unified printf.c.o needs vsnprintf_), and flatcc
        // go in a group. The cuda driver dlopens libcuda at runtime (no -lcuda).
        if let Ok(home) = env::var("IREE_CUDA_HOME") {
            let b = PathBuf::from(home).join("build");
            let unified = b.join("runtime/src/iree/runtime/libiree_runtime_unified.a");
            assert!(
                unified.exists(),
                "IREE_CUDA_HOME build is missing {} (run the runtime build first)",
                unified.display()
            );
            for d in [
                "runtime/src/iree/runtime",
                "runtime/src/iree/hal/drivers/cuda/registration",
                "build_tools/third_party/flatcc",
                "build_tools/third_party/printf",
            ] {
                println!("cargo:rustc-link-search=native={}", b.join(d).display());
            }
            for arg in [
                "-Wl,--whole-archive",
                "-l:libiree_runtime_unified.a",
                "-Wl,--no-whole-archive",
                "-Wl,--start-group",
                "-l:libiree_hal_drivers_cuda_registration_registration.a",
                "-l:libprintf_printf.a",
                "-l:libflatcc_parsing.a",
                "-lgcc",
                "-lm",
                "-lpthread",
                "-ldl",
                "-Wl,--end-group",
            ] {
                println!("cargo:rustc-link-arg={arg}");
            }
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
}

/// Recursively collect the enabled HAL driver static archives under
/// `<build>/runtime/src/iree/hal/drivers` for the macOS link recipe (linked
/// normally, so only the registration members unified lacks are pulled). The
/// top-level `libiree_hal_drivers_drivers.a` (the register-all init) is already
/// bundled into `libiree_runtime_unified.a`, so it is skipped.
#[cfg(target_os = "macos")]
fn collect_driver_archives(dir: &std::path::Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_driver_archives(&path, out);
        } else if path.extension().is_some_and(|e| e == "a")
            && path.file_name().and_then(|n| n.to_str()) != Some("libiree_hal_drivers_drivers.a")
        {
            out.push(path);
        }
    }
}
