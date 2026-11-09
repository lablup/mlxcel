// Copyright 2025-2026 Lablup Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Native HIP GEMV for packed 4-bit decode projections (issue #2889).
//!
//! The single-sequence decode projections otherwise stream f16-expanded
//! weights. This module compiles a small HIP kernel to an HSACO for the active
//! `gfx` target and hands the graph a `hal.executable.source` block that
//! embeds it, so IREE dispatches the kernel directly while the loader keeps the
//! checkpoint's 4-bit tensors resident.

use std::path::PathBuf;
use std::process::Command;

/// The kernel source, compiled at model load with `hipcc --genco`.
pub(crate) const QMV_KERNEL: &str = include_str!("../csrc/qmv4.hip");

/// Whether `MLXCEL_XLA_QUANT=native` asked for the native packed projection.
#[must_use]
pub(crate) fn requested() -> bool {
    matches!(std::env::var("MLXCEL_XLA_QUANT").as_deref(), Ok("native"))
}

fn cache_dir() -> PathBuf {
    std::env::temp_dir().join("mlxcel-xla-qmv4")
}

fn content_hash() -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    QMV_KERNEL.hash(&mut h);
    h.finish()
}

/// Compile [`QMV_KERNEL`] to an HSACO for `arch`, caching the result by kernel
/// content and architecture.
///
/// Returns the HSACO bytes, or an error the caller turns into a fallback to the
/// f16 path (a missing `hipcc`, an unsupported arch, or a compile failure must
/// never fail the model load).
pub(crate) fn hsaco(arch: &str) -> Result<Vec<u8>, String> {
    let dir = cache_dir();
    std::fs::create_dir_all(&dir).map_err(|e| format!("mkdir {}: {e}", dir.display()))?;
    let hash = content_hash();
    let source: PathBuf = dir.join(format!("qmv4-{hash:016x}.hip"));
    let output: PathBuf = dir.join(format!("qmv4-{arch}-{hash:016x}.hsaco"));
    if output.is_file() {
        return std::fs::read(&output).map_err(|e| format!("read {}: {e}", output.display()));
    }
    // Compile to a private path and rename it into place, so a concurrent load
    // in another process never reads a half-written HSACO.
    let pid = std::process::id();
    let source_tmp: PathBuf = dir.join(format!("qmv4-{hash:016x}.{pid}.hip"));
    let output_tmp: PathBuf = dir.join(format!("qmv4-{arch}-{hash:016x}.{pid}.hsaco"));
    std::fs::write(&source_tmp, QMV_KERNEL)
        .map_err(|e| format!("write {}: {e}", source_tmp.display()))?;
    let hipcc = std::env::var("HIPCC").unwrap_or_else(|_| "hipcc".to_string());
    let out = Command::new(&hipcc)
        .args(["--genco", "-O3", &format!("--offload-arch={arch}")])
        .arg(&source_tmp)
        .arg("-o")
        .arg(&output_tmp)
        .output()
        .map_err(|e| format!("run {hipcc}: {e}"))?;
    if !out.status.success() {
        let _ = std::fs::remove_file(&output_tmp);
        return Err(format!(
            "{hipcc} --genco failed for {arch}: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    publish(&source_tmp, &source)?;
    publish(&output_tmp, &output)?;
    std::fs::read(&output).map_err(|e| format!("read {}: {e}", output.display()))
}

/// Rename `tmp` onto `final_path`, tolerating a concurrent load that published
/// the same content first (`rename` replaces an existing file on Unix but not on
/// Windows).
fn publish(tmp: &std::path::Path, final_path: &std::path::Path) -> Result<(), String> {
    match std::fs::rename(tmp, final_path) {
        Ok(()) => Ok(()),
        Err(_) if final_path.is_file() => {
            let _ = std::fs::remove_file(tmp);
            Ok(())
        }
        Err(error) => Err(format!("rename {}: {error}", final_path.display())),
    }
}

/// The `#hal.executable.target` alias the executable objects are keyed by.
fn target_alias() -> &'static str {
    "#mlxcel_qmv_target = #hal.executable.target<\"rocm\", \"rocm-hsaco-fb\">\n"
}

fn hex(bytes: &[u8], out: &mut String) {
    use std::fmt::Write as _;
    for byte in bytes {
        let _ = write!(out, "{byte:02X}");
    }
}

/// The module-level additions for a native decode graph: the target alias, then
/// the `hal.executable.source` block carrying the HSACO and its single export.
///
/// The ABI mirrors `Builder::native_qmv`: two `i32` constants (K, N), four
/// read-only bindings (activations, packed weights, scales, biases) and the
/// `[N] f32` output, 256 threads, one workgroup per output row.
pub(crate) fn executable_source(hsaco: &[u8], arch: &str) -> String {
    let mut dense = String::with_capacity(hsaco.len() * 2);
    hex(hsaco, &mut dense);
    format!(
        "  hal.executable.source private @custom_qmv attributes {{\n\
         \x20   objects = #hal.executable.objects<{{\n\
         \x20     #mlxcel_qmv_target = [#hal.executable.object<{{\n\
         \x20       path = \"qmv4-{arch}.hsaco\",\n\
         \x20       data = dense<\"0x{dense}\"> : vector<{len}xi8>\n\
         \x20     }}>]\n\
         \x20   }}>\n\
         \x20 }} {{\n\
         \x20   hal.executable.export public @qmv4_kernel ordinal(0)\n\
         \x20       layout(#hal.pipeline.layout<constants = 2, bindings = [\n\
         \x20         #hal.pipeline.binding<storage_buffer, ReadOnly>,\n\
         \x20         #hal.pipeline.binding<storage_buffer, ReadOnly>,\n\
         \x20         #hal.pipeline.binding<storage_buffer, ReadOnly>,\n\
         \x20         #hal.pipeline.binding<storage_buffer, ReadOnly>,\n\
         \x20         #hal.pipeline.binding<storage_buffer>\n\
         \x20       ]>) count(%device: !hal.device, %n: index) -> (index, index, index) {{\n\
         \x20     %c1 = arith.constant 1 : index\n\
         \x20     hal.return %n, %c1, %c1 : index, index, index\n\
         \x20   }} attributes {{ workgroup_size = [256 : index, 1 : index, 1 : index] }}\n\
         \x20 }}\n",
        len = hsaco.len(),
    )
}

/// The alias and the source block together, for tests that assemble a whole
/// module in one string.
#[cfg(test)]
#[must_use]
pub(crate) fn module_prefix(hsaco: &[u8], arch: &str) -> String {
    format!("{}{}", target_alias(), executable_source(hsaco, arch))
}

/// The target alias alone, for assembling the module around the source block.
#[must_use]
pub(crate) fn alias_text() -> &'static str {
    target_alias()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The compiled-in kernel is the one described by the module ABI.
    #[test]
    fn kernel_source_matches_the_dispatch_contract() {
        assert!(QMV_KERNEL.contains("extern \"C\" __global__ void qmv4_kernel"));
        assert!(QMV_KERNEL.contains("const uint32_t*"), "packed weights");
        assert!(QMV_KERNEL.contains("__half2float(scales"), "affine scales");
    }

    /// The module prefix names the export and the five-binding ABI the builder
    /// emits, and embeds the bytes it was given.
    #[test]
    fn module_prefix_embeds_the_object_and_the_abi() {
        let prefix = module_prefix(&[0x00, 0xFF], "gfx1151");
        assert!(prefix.contains("#mlxcel_qmv_target"));
        assert!(prefix.contains("hal.executable.export public @qmv4_kernel ordinal(0)"));
        assert!(prefix.contains("constants = 2"));
        assert!(prefix.contains("dense<\"0x00FF\"> : vector<2xi8>"));
        assert!(prefix.contains("workgroup_size = [256 : index, 1 : index, 1 : index]"));
        assert!(prefix.contains("hal.return %n, %c1, %c1"));
    }
}
