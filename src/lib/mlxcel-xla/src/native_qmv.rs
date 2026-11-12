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
//!
//! On `MLXCEL_XLA_DEVICE=cuda` the same exports come from `csrc/qmv4.cu`, which
//! `build.rs` compiles to PTX when the crate is built against a CUDA IREE
//! runtime (issue #2905). The dispatch ABI, the entry names and the workgroup
//! geometry are shared; only the embedded object and its executable target
//! differ. The ragged multi-row exports are HIP-only.

use std::path::PathBuf;
use std::process::Command;

/// The kernel source, compiled at model load with `hipcc --genco`.
pub(crate) const QMV_KERNEL: &str = include_str!("../csrc/qmv4.hip");

/// The CUDA port of the single-sequence exports of [`QMV_KERNEL`] (issue
/// #2905). `build.rs` compiles this file to the PTX behind [`cuda_ptx`]; the
/// source text itself is only read by the tests that pin it to the HIP kernel.
#[cfg(test)]
pub(crate) const QMV_KERNEL_CUDA: &str = include_str!("../csrc/qmv4.cu");

/// Lowest compute capability the embedded PTX targets, matching the
/// `--iree-cuda-target=sm_80` the loader compiles the graph with.
pub(crate) const CUDA_MIN_SM: u32 = 80;

#[cfg(xla_iree_cuda_ptx)]
const QMV4_PTX: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/qmv4_sm80.ptx"));

/// The PTX compiled from `csrc/qmv4.cu`, or an error the caller turns into a
/// fallback to the f16 path.
///
/// Present only in a build against a CUDA IREE runtime (`IREE_CUDA_HOME`),
/// and an available `nvcc`. Without either, the loader warns once and keeps
/// the f16 weight path, even when the CUDA IREE runtime itself is available.
pub(crate) fn cuda_ptx() -> Result<&'static [u8], String> {
    #[cfg(xla_iree_cuda_ptx)]
    {
        if QMV4_PTX.is_empty() {
            return Err("the embedded native GEMV PTX is empty".to_string());
        }
        Ok(QMV4_PTX)
    }
    #[cfg(not(xla_iree_cuda_ptx))]
    {
        Err("this build has no native GEMV PTX (CUDA IREE runtime and nvcc are required at build time)".to_string())
    }
}

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

/// The batch bound one bucket compiles with: `None` reuses the default
/// compilation, whose bound already equals [`MAX_BATCH`].
fn bucket_bound(b_max: usize) -> Result<Option<usize>, String> {
    if !(1..=MAX_BATCH).contains(&b_max) {
        return Err(format!(
            "the multi-row kernel is compiled for 1..={MAX_BATCH} activation rows; {b_max} is out of range"
        ));
    }
    Ok((b_max != MAX_BATCH).then_some(b_max))
}

/// The `hipcc` arguments for one kernel compilation.
fn genco_args(arch: &str, max_b: Option<usize>) -> Vec<String> {
    let mut args = vec![
        "--genco".to_string(),
        "-O3".to_string(),
        format!("--offload-arch={arch}"),
    ];
    if let Some(max_b) = max_b {
        args.push(format!("-DQMV4_MAX_B={max_b}"));
    }
    args
}

/// The cache stem for one compilation. The default bound keeps the unsuffixed
/// form, so a narrower bucket never displaces the default entry.
fn cache_stem(arch: &str, hash: u64, max_b: Option<usize>) -> String {
    match max_b {
        Some(max_b) => format!("qmv4-{arch}-{hash:016x}-b{max_b}"),
        None => format!("qmv4-{arch}-{hash:016x}"),
    }
}

/// Compile [`QMV_KERNEL`] to an HSACO for `arch`, caching the result by kernel
/// content, architecture and, when present, the batch bound.
fn compile(arch: &str, hash: u64, max_b: Option<usize>) -> Result<Vec<u8>, String> {
    let dir = cache_dir();
    std::fs::create_dir_all(&dir).map_err(|e| format!("mkdir {}: {e}", dir.display()))?;
    let stem = cache_stem(arch, hash, max_b);
    let source: PathBuf = dir.join(format!("qmv4-{hash:016x}.hip"));
    let output: PathBuf = dir.join(format!("{stem}.hsaco"));
    if output.is_file() {
        return std::fs::read(&output).map_err(|e| format!("read {}: {e}", output.display()));
    }
    // Compile to a private path and rename it into place, so a concurrent load
    // in another process never reads a half-written HSACO.
    let pid = std::process::id();
    let source_tmp: PathBuf = dir.join(format!("qmv4-{hash:016x}.{pid}.hip"));
    let output_tmp: PathBuf = dir.join(format!("{stem}.{pid}.hsaco"));
    std::fs::write(&source_tmp, QMV_KERNEL)
        .map_err(|e| format!("write {}: {e}", source_tmp.display()))?;
    let hipcc = std::env::var("HIPCC").unwrap_or_else(|_| "hipcc".to_string());
    let out = Command::new(&hipcc)
        .args(genco_args(arch, max_b))
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

/// Compile [`QMV_KERNEL`] to an HSACO for `arch`, caching the result by kernel
/// content and architecture.
///
/// Returns the HSACO bytes, or an error the caller turns into a fallback to the
/// f16 path (a missing `hipcc`, an unsupported arch, or a compile failure must
/// never fail the model load).
pub(crate) fn hsaco(arch: &str) -> Result<Vec<u8>, String> {
    compile(arch, content_hash(), None)
}

/// Compile [`QMV_KERNEL`] for a ragged bucket that carries at most `b_max`
/// activation rows (issue #2955).
///
/// The multi-row exports unroll over `QMV4_MAX_B` and predicate the live rows.
/// Compiling the widest bound for every bucket makes a `B=4` step carry the
/// register pressure, shared-memory footprint and predication branches of
/// `B=8` while half its accumulators are dead, which measured slower than the
/// f16 path at `B=4`. Each bucket now gets an artifact compiled with its own
/// bound; the widest bucket reuses [`hsaco`] because the default bound already
/// equals [`MAX_BATCH`].
///
/// Errors follow [`hsaco`]'s fallback contract.
pub(crate) fn hsaco_multi(arch: &str, b_max: usize) -> Result<Vec<u8>, String> {
    let max_b = bucket_bound(b_max)?;
    compile(arch, content_hash(), max_b)
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

/// The CUDA counterpart of [`target_alias`]. The alias name is the same, so the
/// source blocks below key their object by it on either device.
fn cuda_target_alias() -> &'static str {
    "#mlxcel_qmv_target = #hal.executable.target<\"cuda\", \"cuda-nvptx-fb\">\n"
}

fn hex(bytes: &[u8], out: &mut String) {
    use std::fmt::Write as _;
    for byte in bytes {
        let _ = write!(out, "{byte:02X}");
    }
}

/// Output rows one workgroup reduces (issue #2891): one warp per row in a
/// 256-thread block. The kernel derives the same value from `blockDim.x`, and
/// the emitter must divide `N` by it for the workgroup count.
pub(crate) const ROWS_PER_BLOCK: usize = 8;

/// Threads in a workgroup; one warp per output row.
pub(crate) const THREADS_PER_BLOCK: usize = 256;

/// From this contraction width on, the emitter dispatches
/// [`ROW_KERNEL_ENTRY`] instead: with a 4K-wide row one warp per row leaves too
/// little memory parallelism in flight, and the block-per-row layout measured
/// about 1.6x faster on `gfx1151` (issue #2891). The threshold sits between the
/// hidden-width projections (K = hidden) and the down projection (K = 4 * hidden
/// on Llama-shaped models).
pub(crate) const ROW_KERNEL_MIN_K: usize = 4096;

/// The export that reduces one row per workgroup.
pub(crate) const ROW_KERNEL_ENTRY: &str = "@custom_qmv::@qmv4_row_kernel";

/// The export that reduces one row per warp, `ROWS_PER_BLOCK` rows per
/// workgroup.
pub(crate) const WARP_KERNEL_ENTRY: &str = "@custom_qmv::@qmv4_kernel";

/// The dispatch entry point and workgroup count for a `[N]` output with
/// contraction width `K`.
#[must_use]
pub(crate) fn dispatch(k: usize, n: usize) -> (&'static str, usize) {
    if k >= ROW_KERNEL_MIN_K {
        (ROW_KERNEL_ENTRY, n)
    } else {
        (WARP_KERNEL_ENTRY, n.div_ceil(ROWS_PER_BLOCK))
    }
}

/// The export that multiplies three weight sets (q, k, v) in one launch.
pub(crate) const QKV_WARP_ENTRY: &str = "@custom_qmv_qkv::@qmv4_qkv_kernel";
pub(crate) const QKV_ROW_ENTRY: &str = "@custom_qmv_qkv::@qmv4_qkv_row_kernel";

/// The export that multiplies two weight sets (gate, up) in one launch.
pub(crate) const GATEUP_WARP_ENTRY: &str = "@custom_qmv_gateup::@qmv4_gateup_kernel";
pub(crate) const GATEUP_ROW_ENTRY: &str = "@custom_qmv_gateup::@qmv4_gateup_row_kernel";

/// The entry point and workgroup count for a fused multi-output GEMV that
/// reduces `total` output rows with contraction width `K` (issue #2895).
#[must_use]
pub(crate) fn fused_dispatch(
    k: usize,
    total: usize,
    warp: &'static str,
    row: &'static str,
) -> (&'static str, usize) {
    if k >= ROW_KERNEL_MIN_K {
        (row, total)
    } else {
        (warp, total.div_ceil(ROWS_PER_BLOCK))
    }
}

/// The dispatch entry point and workgroup count for fused q/k/v.
#[must_use]
pub(crate) fn qkv_dispatch(k: usize, total: usize) -> (&'static str, usize) {
    fused_dispatch(k, total, QKV_WARP_ENTRY, QKV_ROW_ENTRY)
}

/// The dispatch entry point and workgroup count for fused gate/up.
#[must_use]
pub(crate) fn gateup_dispatch(k: usize, total: usize) -> (&'static str, usize) {
    fused_dispatch(k, total, GATEUP_WARP_ENTRY, GATEUP_ROW_ENTRY)
}

/// Largest batch the multi-row exports accept (issue #2922), matching
/// `QMV4_MAX_B` in `csrc/qmv4.hip`. The ragged engine only emits the bundled
/// `RAGGED_B_VALUES` graphs (4 and 8), so this bound is never approached.
pub(crate) const MAX_BATCH: usize = 8;

/// The exports that reduce `B` activation rows at once (issue #2922).
pub(crate) const MULTI_WARP_ENTRY: &str = "@custom_qmv_multi::@qmv4_multi_kernel";
pub(crate) const MULTI_ROW_ENTRY: &str = "@custom_qmv_multi::@qmv4_multi_row_kernel";
pub(crate) const QKV_MULTI_WARP_ENTRY: &str = "@custom_qmv_qkv_multi::@qmv4_qkv_multi_kernel";
pub(crate) const QKV_MULTI_ROW_ENTRY: &str = "@custom_qmv_qkv_multi::@qmv4_qkv_multi_row_kernel";
pub(crate) const GATEUP_MULTI_WARP_ENTRY: &str =
    "@custom_qmv_gateup_multi::@qmv4_gateup_multi_kernel";
pub(crate) const GATEUP_MULTI_ROW_ENTRY: &str =
    "@custom_qmv_gateup_multi::@qmv4_gateup_multi_row_kernel";

/// The entry point and workgroup count for the multi-row GEMV: `[B, K] x
/// [N, K/8] -> [B, N]`.
///
/// The export follows the same width rule as the single-row form: rows with
/// `K >= ROW_KERNEL_MIN_K` take the block-per-row export (`N` workgroups), and
/// the rest the warp export (`ceil(N / ROWS_PER_BLOCK)` workgroups). Supported
/// shapes are `B <= MAX_BATCH`, `K % 64 == 0` (8 nibbles per u32 and one
/// scale/bias per 64 inputs), and any `N` (the grid rounds up and the kernel
/// bounds its rows by `N`).
#[must_use]
pub(crate) fn multi_dispatch(k: usize, n: usize) -> (&'static str, usize) {
    fused_dispatch(k, n, MULTI_WARP_ENTRY, MULTI_ROW_ENTRY)
}

/// The entry point and workgroup count for the fused multi-row q/k/v dispatch
/// over `total` output rows.
#[must_use]
pub(crate) fn qkv_multi_dispatch(k: usize, total: usize) -> (&'static str, usize) {
    fused_dispatch(k, total, QKV_MULTI_WARP_ENTRY, QKV_MULTI_ROW_ENTRY)
}

/// The entry point and workgroup count for the fused multi-row gate/up dispatch
/// over `total` output rows.
#[must_use]
pub(crate) fn gateup_multi_dispatch(k: usize, total: usize) -> (&'static str, usize) {
    fused_dispatch(k, total, GATEUP_MULTI_WARP_ENTRY, GATEUP_MULTI_ROW_ENTRY)
}

/// A pipeline layout with `readonly` input buffers followed by `writable`
/// output buffers and `constants` push constants.
fn pipeline_layout(constants: usize, readonly: usize, writable: usize) -> String {
    let mut layout = format!("#hal.pipeline.layout<constants = {constants}, bindings = [\n");
    for _ in 0..readonly {
        layout.push_str("          #hal.pipeline.binding<storage_buffer, ReadOnly>,\n");
    }
    for index in 0..writable {
        layout.push_str("          #hal.pipeline.binding<storage_buffer>");
        layout.push_str(if index + 1 < writable { ",\n" } else { "\n" });
    }
    layout.push_str("        ]>");
    layout
}

fn export_block(
    name: &str,
    ordinal: usize,
    constants: usize,
    readonly: usize,
    writable: usize,
) -> String {
    format!(
        "    hal.executable.export public @{name} ordinal({ordinal})\n\
         \x20       layout({layout}) count(%device: !hal.device, %n: index) -> (index, index, index) {{\n\
         \x20     %c1 = arith.constant 1 : index\n\
         \x20     hal.return %n, %c1, %c1 : index, index, index\n\
         \x20   }} attributes {{ workgroup_size = [{threads} : index, 1 : index, 1 : index] }}\n",
        layout = pipeline_layout(constants, readonly, writable),
        threads = THREADS_PER_BLOCK,
    )
}

/// `(name, ordinal, constants, read-only bindings, writable bindings)` per
/// export of one source block.
type ExportTable = &'static [(&'static str, usize, usize, usize, usize)];

/// The single-output exports: two constants (K, N), the activation row, one
/// packed/scales/biases set and the `[N]` output.
const SINGLE_EXPORTS: ExportTable = &[("qmv4_kernel", 0, 2, 4, 1), ("qmv4_row_kernel", 1, 2, 4, 1)];

/// The fused q/k/v exports (issue #2895): four constants (K, N0, N1, N2), the
/// activation row, three weight sets and three outputs.
const QKV_EXPORTS: ExportTable = &[
    ("qmv4_qkv_kernel", 0, 4, 10, 3),
    ("qmv4_qkv_row_kernel", 1, 4, 10, 3),
];

/// The fused gate/up exports (issue #2895): three constants (K, N0, N1), the
/// activation row, two weight sets and two outputs.
const GATEUP_EXPORTS: ExportTable = &[
    ("qmv4_gateup_kernel", 0, 3, 7, 2),
    ("qmv4_gateup_row_kernel", 1, 3, 7, 2),
];

/// One `hal.executable.source` block: the embedded HSACO plus one export per
/// entry point, each with its own pipeline layout.
fn executable_source_named(
    hsaco: &[u8],
    arch: &str,
    symbol: &str,
    exports: &[(&str, usize, usize, usize, usize)],
) -> String {
    executable_source_object(hsaco, &format!("qmv4-{arch}.hsaco"), symbol, exports)
}

/// [`executable_source_named`] for an object of any target: `object` is the
/// device binary (an HSACO or a PTX module) and `path` the name it is embedded
/// under.
fn executable_source_object(
    object: &[u8],
    path: &str,
    symbol: &str,
    exports: &[(&str, usize, usize, usize, usize)],
) -> String {
    let mut dense = String::with_capacity(object.len() * 2);
    hex(object, &mut dense);
    let mut out = format!(
        "  hal.executable.source private @{symbol} attributes {{\n\
         \x20   objects = #hal.executable.objects<{{\n\
         \x20     #mlxcel_qmv_target = [#hal.executable.object<{{\n\
         \x20       path = \"{path}\",\n\
         \x20       data = dense<\"0x{dense}\"> : vector<{len}xi8>\n\
         \x20     }}>]\n\
         \x20   }}>\n\
         \x20 }} {{\n",
        len = object.len(),
    );
    for (name, ordinal, constants, readonly, writable) in exports {
        out.push_str(&export_block(
            name, *ordinal, *constants, *readonly, *writable,
        ));
    }
    out.push_str("  }\n");
    out
}

/// The `hal.executable.source` block carrying the HSACO and the two
/// single-output exports.
///
/// The ABI mirrors `Builder::native_qmv`: two `i32` constants (K, N), four
/// read-only bindings (activations, packed weights, scales, biases) and the
/// `[N] f32` output. `@qmv4_kernel` gives each output row one warp and each
/// workgroup [`ROWS_PER_BLOCK`] rows, so `N` rows take
/// `ceil(N / ROWS_PER_BLOCK)` workgroups; `@qmv4_row_kernel` puts a whole
/// workgroup on one row and takes `N` workgroups.
pub(crate) fn executable_source(hsaco: &[u8], arch: &str) -> String {
    executable_source_named(hsaco, arch, "custom_qmv", SINGLE_EXPORTS)
}

/// The `hal.executable.source` block for the fused q/k/v export (issue #2895):
/// four constants (K, N0, N1, N2), the activation row, three packed/scales/biases
/// sets and three outputs.
pub(crate) fn qkv_source(hsaco: &[u8], arch: &str) -> String {
    executable_source_named(hsaco, arch, "custom_qmv_qkv", QKV_EXPORTS)
}

/// The `hal.executable.source` block for the fused gate/up export (issue #2895):
/// three constants (K, N0, N1), the activation row, two packed/scales/biases
/// sets and two outputs.
pub(crate) fn gateup_source(hsaco: &[u8], arch: &str) -> String {
    executable_source_named(hsaco, arch, "custom_qmv_gateup", GATEUP_EXPORTS)
}

/// The `hal.executable.source` block for the multi-row export (issue #2922):
/// three constants (K, N, B), the `[B, K]` activations, the packed/scales/biases
/// set and the `[B, N]` output.
pub(crate) fn multi_source(hsaco: &[u8], arch: &str) -> String {
    executable_source_named(
        hsaco,
        arch,
        "custom_qmv_multi",
        &[
            ("qmv4_multi_kernel", 0, 3, 4, 1),
            ("qmv4_multi_row_kernel", 1, 3, 4, 1),
        ],
    )
}

/// The `hal.executable.source` block for the fused multi-row q/k/v export:
/// five constants (K, N0, N1, N2, B), the `[B, K]` activations, three
/// packed/scales/biases sets and three `[B, Ni]` outputs.
pub(crate) fn qkv_multi_source(hsaco: &[u8], arch: &str) -> String {
    executable_source_named(
        hsaco,
        arch,
        "custom_qmv_qkv_multi",
        &[
            ("qmv4_qkv_multi_kernel", 0, 5, 10, 3),
            ("qmv4_qkv_multi_row_kernel", 1, 5, 10, 3),
        ],
    )
}

/// The `hal.executable.source` block for the fused multi-row gate/up export:
/// four constants (K, N0, N1, B), the `[B, K]` activations, two
/// packed/scales/biases sets and two `[B, Ni]` outputs.
pub(crate) fn gateup_multi_source(hsaco: &[u8], arch: &str) -> String {
    executable_source_named(
        hsaco,
        arch,
        "custom_qmv_gateup_multi",
        &[
            ("qmv4_gateup_multi_kernel", 0, 4, 7, 2),
            ("qmv4_gateup_multi_row_kernel", 1, 4, 7, 2),
        ],
    )
}

/// The target alias and the executable source blocks the single-sequence decode
/// graph dispatches, in module order.
///
/// Only the three rank-1 export groups are embedded: that graph never
/// dispatches a multi-row export, and every embedded source carries a full copy
/// of the HSACO, so including the ragged blocks here would grow the
/// single-sequence module and vmfb for nothing (issue #2922).
#[must_use]
pub(crate) fn module_parts(hsaco: &[u8], arch: &str) -> (String, String) {
    (
        target_alias().to_string(),
        format!(
            "{}{}{}",
            executable_source(hsaco, arch),
            qkv_source(hsaco, arch),
            gateup_source(hsaco, arch),
        ),
    )
}

/// Name the PTX module is embedded under.
const CUDA_OBJECT_PATH: &str = "qmv4_compute80.ptx";

/// [`module_parts`] for `MLXCEL_XLA_DEVICE=cuda` (issue #2905): the same three
/// source blocks, exports and pipeline layouts, embedding the PTX compiled from
/// `csrc/qmv4.cu` under the CUDA executable target.
#[must_use]
pub(crate) fn cuda_module_parts(ptx: &[u8]) -> (String, String) {
    let block = |symbol: &str, exports: ExportTable| {
        executable_source_object(ptx, CUDA_OBJECT_PATH, symbol, exports)
    };
    (
        cuda_target_alias().to_string(),
        format!(
            "{}{}{}",
            block("custom_qmv", SINGLE_EXPORTS),
            block("custom_qmv_qkv", QKV_EXPORTS),
            block("custom_qmv_gateup", GATEUP_EXPORTS),
        ),
    )
}

/// The target alias and the executable source blocks the ragged decode graph
/// dispatches (issue #2922), in module order.
///
/// The ragged graph routes every projection through the multi-row exports, so
/// it embeds only those three groups; the rank-1 blocks would be dead weight in
/// the module and artifact.
#[must_use]
pub(crate) fn ragged_module_parts(hsaco: &[u8], arch: &str) -> (String, String) {
    (
        target_alias().to_string(),
        format!(
            "{}{}{}",
            multi_source(hsaco, arch),
            qkv_multi_source(hsaco, arch),
            gateup_multi_source(hsaco, arch),
        ),
    )
}

/// The alias and both kinds' source blocks together, for tests that assemble
/// a whole module in one string.
#[cfg(test)]
#[must_use]
pub(crate) fn module_prefix(hsaco: &[u8], arch: &str) -> String {
    let (alias, single) = module_parts(hsaco, arch);
    let (_, ragged) = ragged_module_parts(hsaco, arch);
    format!("{alias}{single}{ragged}")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The compiled-in kernel is the one described by the module ABI.
    #[test]
    fn kernel_source_matches_the_dispatch_contract() {
        for kernel in [
            "qmv4_kernel",
            "qmv4_row_kernel",
            "qmv4_qkv_kernel",
            "qmv4_qkv_row_kernel",
            "qmv4_gateup_kernel",
            "qmv4_gateup_row_kernel",
            "qmv4_multi_kernel",
            "qmv4_multi_row_kernel",
            "qmv4_qkv_multi_kernel",
            "qmv4_qkv_multi_row_kernel",
            "qmv4_gateup_multi_kernel",
            "qmv4_gateup_multi_row_kernel",
        ] {
            assert!(
                QMV_KERNEL.contains(&format!("extern \"C\" __global__ void {kernel}")),
                "missing {kernel}"
            );
        }
        assert!(QMV_KERNEL.contains("const uint32_t*"), "packed weights");
        assert!(
            QMV_KERNEL.contains("__half2float(srow[g])"),
            "affine scales"
        );
        assert!(QMV_KERNEL.contains("uint4"), "16-byte packed loads");
        assert!(
            QMV_KERNEL.contains("blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)"),
            "one warp per output row"
        );
        assert!(
            QMV_KERNEL.contains("QmvSet"),
            "the fused kernels select a weight set per output row"
        );
        assert!(
            QMV_KERNEL.contains("float acc[QMV4_MAX_B]"),
            "the multi-row kernels keep one accumulator per activation row"
        );
        assert!(
            QMV_KERNEL.contains("x[(size_t)bb * K"),
            "multi-row activations are indexed per row by K"
        );
        assert!(
            QMV_KERNEL.contains("y[(size_t)bb * N + row]"),
            "multi-row outputs are written at b * N + row"
        );
        assert!(
            QMV_KERNEL.contains("set.y[(size_t)bb * set.n + r]"),
            "fused multi-row outputs are written per set at b * Ni + local row"
        );
        assert_eq!(MAX_BATCH, 8, "QMV4_MAX_B and MAX_BATCH must agree");
        assert!(
            QMV_KERNEL.contains(&format!("#define QMV4_MAX_B {MAX_BATCH}")),
            "the kernel's batch bound matches MAX_BATCH"
        );
        assert_eq!(
            ROWS_PER_BLOCK * 32,
            THREADS_PER_BLOCK,
            "one warp per row must fill the workgroup"
        );
    }

    /// The kernel's bound is a default the loader overrides per batch bucket
    /// (issue #2955).
    #[test]
    fn kernel_bound_is_overridable_at_compile_time() {
        assert!(
            QMV_KERNEL.contains(&format!(
                "#ifndef QMV4_MAX_B\n#define QMV4_MAX_B {MAX_BATCH}\n#endif"
            )),
            "a guarded define lets `-DQMV4_MAX_B=<bucket>` reach the kernel"
        );
    }

    /// Each ragged bucket compiles with its own bound; the widest bucket reuses
    /// the default artifact and the out-of-range buckets are rejected before
    /// `hipcc` runs (issue #2955).
    #[test]
    fn ragged_buckets_compile_with_their_own_bound() {
        assert_eq!(bucket_bound(4).unwrap(), Some(4));
        assert_eq!(
            bucket_bound(MAX_BATCH).unwrap(),
            None,
            "the default artifact already carries the widest bound"
        );
        assert!(bucket_bound(0).is_err());
        assert!(bucket_bound(MAX_BATCH + 1).is_err());

        let narrow_args = genco_args("gfx1151", Some(4));
        let args: Vec<&str> = narrow_args.iter().map(String::as_str).collect();
        assert_eq!(
            args,
            ["--genco", "-O3", "--offload-arch=gfx1151", "-DQMV4_MAX_B=4"]
        );
        assert!(
            !genco_args("gfx1151", None)
                .iter()
                .any(|arg| arg.starts_with("-DQMV4_MAX_B")),
            "the default bound comes from the source"
        );

        let hash = content_hash();
        assert_eq!(
            cache_stem("gfx1151", hash, Some(4)),
            format!("qmv4-gfx1151-{hash:016x}-b4"),
            "a narrow bucket must not share the default cache entry"
        );
        assert_eq!(
            cache_stem("gfx1151", hash, None),
            format!("qmv4-gfx1151-{hash:016x}"),
            "the default stem keeps its pre-fix form"
        );
    }

    /// Long rows take the block-per-row export, where one workgroup is one row.
    #[test]
    fn long_rows_use_the_block_per_row_export() {
        assert_eq!(dispatch(2048, 8192), (WARP_KERNEL_ENTRY, 1024));
        assert_eq!(
            dispatch(ROW_KERNEL_MIN_K - 1, 2048),
            (WARP_KERNEL_ENTRY, 256)
        );
        assert_eq!(dispatch(ROW_KERNEL_MIN_K, 2048), (ROW_KERNEL_ENTRY, 2048));
        assert_eq!(dispatch(8192, 2048), (ROW_KERNEL_ENTRY, 2048));
    }

    /// The fused groups follow the same width rule and grid rounding.
    #[test]
    fn fused_groups_pick_the_same_row_layout() {
        assert_eq!(qkv_dispatch(2048, 3072), (QKV_WARP_ENTRY, 384));
        assert_eq!(gateup_dispatch(2048, 16384), (GATEUP_WARP_ENTRY, 2048));
        assert_eq!(qkv_dispatch(8192, 3072), (QKV_ROW_ENTRY, 3072));
        assert_eq!(gateup_dispatch(8192, 16384), (GATEUP_ROW_ENTRY, 16384));
    }

    /// The multi-row exports reuse the single-row width rule and grid rounding
    /// (issue #2922); the batch size never changes the workgroup count because
    /// one weight read serves every row.
    #[test]
    fn multi_row_dispatch_matches_the_single_row_grid() {
        assert_eq!(multi_dispatch(2048, 8192), (MULTI_WARP_ENTRY, 1024));
        assert_eq!(multi_dispatch(4096, 2048), (MULTI_ROW_ENTRY, 2048));
        assert_eq!(qkv_multi_dispatch(2048, 3072), (QKV_MULTI_WARP_ENTRY, 384));
        assert_eq!(
            gateup_multi_dispatch(2048, 16384),
            (GATEUP_MULTI_WARP_ENTRY, 2048)
        );
        assert_eq!(qkv_multi_dispatch(8192, 3072), (QKV_MULTI_ROW_ENTRY, 3072));
        assert_eq!(
            gateup_multi_dispatch(8192, 16384),
            (GATEUP_MULTI_ROW_ENTRY, 16384)
        );
        // N need not be a multiple of ROWS_PER_BLOCK: the warp grid rounds up and
        // the kernel bounds its rows, so the extra workgroups exit early.
        assert_eq!(multi_dispatch(2048, 100), (MULTI_WARP_ENTRY, 13));
        assert_eq!(qkv_multi_dispatch(2048, 100), (QKV_MULTI_WARP_ENTRY, 13));
        assert_eq!(multi_dispatch(2048, 8), (MULTI_WARP_ENTRY, 1));
    }

    /// The module prefix names every export and the per-group binding counts the
    /// builder emits, and embeds the bytes it was given.
    #[test]
    fn module_prefix_embeds_the_object_and_the_abi() {
        let prefix = module_prefix(&[0x00, 0xFF], "gfx1151");
        assert!(prefix.contains("#mlxcel_qmv_target"));
        for source in [
            "@custom_qmv ",
            "@custom_qmv_qkv ",
            "@custom_qmv_gateup ",
            "@custom_qmv_multi ",
            "@custom_qmv_qkv_multi ",
            "@custom_qmv_gateup_multi ",
        ] {
            assert!(
                prefix.contains(&format!("hal.executable.source private {source}")),
                "missing source {source}"
            );
        }
        for export in [
            "hal.executable.export public @qmv4_kernel ordinal(0)",
            "hal.executable.export public @qmv4_row_kernel ordinal(1)",
            "hal.executable.export public @qmv4_qkv_kernel ordinal(0)",
            "hal.executable.export public @qmv4_qkv_row_kernel ordinal(1)",
            "hal.executable.export public @qmv4_gateup_kernel ordinal(0)",
            "hal.executable.export public @qmv4_gateup_row_kernel ordinal(1)",
            "hal.executable.export public @qmv4_multi_kernel ordinal(0)",
            "hal.executable.export public @qmv4_multi_row_kernel ordinal(1)",
            "hal.executable.export public @qmv4_qkv_multi_kernel ordinal(0)",
            "hal.executable.export public @qmv4_qkv_multi_row_kernel ordinal(1)",
            "hal.executable.export public @qmv4_gateup_multi_kernel ordinal(0)",
            "hal.executable.export public @qmv4_gateup_multi_row_kernel ordinal(1)",
        ] {
            assert!(prefix.contains(export), "missing {export}");
        }
        assert!(prefix.contains("constants = 2"));
        assert!(
            prefix.contains("constants = 3"),
            "gate/up: K, N0, N1; multi: K, N, B"
        );
        assert!(
            prefix.contains("constants = 4"),
            "q/k/v: K, N0, N1, N2; multi gate/up: K, N0, N1, B"
        );
        assert!(
            prefix.contains("constants = 5"),
            "multi q/k/v: K, N0, N1, N2, B"
        );
        assert!(prefix.contains("dense<\"0x00FF\"> : vector<2xi8>"));
        assert!(prefix.contains(&format!(
            "workgroup_size = [{THREADS_PER_BLOCK} : index, 1 : index, 1 : index]"
        )));
        assert!(prefix.contains("hal.return %n, %c1, %c1"));
    }

    /// Each decode graph embeds only the exports it dispatches (issue #2922).
    /// Every source block carries a full HSACO copy, so embedding the other
    /// kind's blocks would grow that graph's module and vmfb without a dispatch
    /// ever reaching them.
    #[test]
    fn decode_kinds_embed_only_their_own_exports() {
        let (single_alias, single) = module_parts(&[0x00, 0xFF], "gfx1151");
        let (ragged_alias, ragged) = ragged_module_parts(&[0x00, 0xFF], "gfx1151");
        assert!(single_alias.contains("#mlxcel_qmv_target"));
        assert_eq!(single_alias, ragged_alias);
        for source in ["@custom_qmv ", "@custom_qmv_qkv ", "@custom_qmv_gateup "] {
            assert!(single.contains(source), "single decode missing {source}");
            assert!(
                !ragged.contains(source),
                "ragged decode embeds unused {source}"
            );
        }
        for source in [
            "@custom_qmv_multi ",
            "@custom_qmv_qkv_multi ",
            "@custom_qmv_gateup_multi ",
        ] {
            assert!(ragged.contains(source), "ragged decode missing {source}");
            assert!(
                !single.contains(source),
                "single decode embeds unused {source}"
            );
        }
        for (export, in_single) in [
            ("@qmv4_kernel ", true),
            ("@qmv4_qkv_kernel ", true),
            ("@qmv4_gateup_kernel ", true),
            ("@qmv4_multi_kernel ", false),
            ("@qmv4_qkv_multi_kernel ", false),
            ("@qmv4_gateup_multi_kernel ", false),
        ] {
            let (own, other) = if in_single {
                (&single, &ragged)
            } else {
                (&ragged, &single)
            };
            assert!(own.contains(export), "missing export {export}");
            assert!(!other.contains(export), "export {export} embedded twice");
        }
    }
    /// The CUDA file carries the six single-sequence exports and nothing of the
    /// ragged form.
    #[test]
    fn cuda_kernel_source_matches_the_dispatch_contract() {
        for kernel in [
            "qmv4_kernel",
            "qmv4_row_kernel",
            "qmv4_qkv_kernel",
            "qmv4_qkv_row_kernel",
            "qmv4_gateup_kernel",
            "qmv4_gateup_row_kernel",
        ] {
            assert!(
                QMV_KERNEL_CUDA.contains(&format!("extern \"C\" __global__ void {kernel}(")),
                "missing {kernel}"
            );
        }
        assert_eq!(
            QMV_KERNEL_CUDA.matches("extern \"C\" __global__").count(),
            6,
            "the ragged multi-row exports are HIP-only"
        );
        assert!(!QMV_KERNEL_CUDA.contains("void qmv4_multi_kernel"));
        assert!(QMV_KERNEL_CUDA.contains("#include <cuda_fp16.h>"));
        assert!(!QMV_KERNEL_CUDA.contains("#include <hip/"));
        assert!(
            QMV_KERNEL_CUDA.contains("__shfl_down_sync(0xffffffffu, acc, off)"),
            "CUDA has no maskless warp shuffle"
        );
    }

    /// The CUDA kernels are the HIP single-sequence kernels, line for line, with
    /// one intrinsic renamed. A change to either file that does not reach the
    /// other fails here, so the two devices cannot drift in arithmetic.
    #[test]
    fn cuda_kernels_are_the_hip_single_sequence_kernels() {
        let slice = |source: &'static str, from: &str, to: &str| -> &'static str {
            let start = source
                .find(from)
                .unwrap_or_else(|| panic!("marker {from:?} not found"));
            let end = source[start..]
                .find(to)
                .map(|offset| start + offset)
                .unwrap_or_else(|| panic!("marker {to:?} not found"));
            &source[start..end]
        };
        // HIP: the shared helpers, then (past the multi-row helpers) the six
        // single-sequence kernels, up to the first ragged kernel.
        let helpers = slice(QMV_KERNEL, "struct QmvSet {", "// B-row warp dot:");
        let kernels = slice(
            QMV_KERNEL,
            "// One warp per output row, eight rows per workgroup.\nextern",
            "// B activation rows in one launch",
        );
        let expected = format!("{}\n{}", helpers.trim_end(), kernels.trim_end()).replace(
            "__shfl_down(acc, off)",
            "__shfl_down_sync(0xffffffffu, acc, off)",
        );
        let cuda_start = QMV_KERNEL_CUDA
            .find("struct QmvSet {")
            .expect("the CUDA file defines QmvSet");
        let actual = QMV_KERNEL_CUDA[cuda_start..].trim_end();
        // Compare modulo blank lines, which separate the two HIP slices
        // differently from the CUDA file.
        let lines = |text: &str| -> Vec<String> {
            text.lines()
                .filter(|line| !line.trim().is_empty())
                .map(str::to_string)
                .collect()
        };
        assert_eq!(lines(actual), lines(&expected));
    }

    /// The CUDA module is the HIP single-sequence module with the object and
    /// the executable target swapped: same symbols, exports, ordinals and
    /// pipeline layouts, so the emitter's dispatches bind on either device.
    #[test]
    fn cuda_module_parts_embed_the_ptx_under_the_cuda_target() {
        let object = b"\x01\xABptx";
        let (alias, source) = cuda_module_parts(object);
        assert_eq!(
            alias,
            "#mlxcel_qmv_target = #hal.executable.target<\"cuda\", \"cuda-nvptx-fb\">\n"
        );
        assert!(source.contains("path = \"qmv4_compute80.ptx\""));
        assert!(source.contains("data = dense<\"0x01AB707478\"> : vector<5xi8>"));
        assert!(!source.contains("hsaco"));
        for block in ["@custom_qmv ", "@custom_qmv_qkv ", "@custom_qmv_gateup "] {
            assert!(source.contains(block), "missing source block {block}");
        }
        assert!(!source.contains("_multi"), "the ragged blocks are HIP-only");

        let (hip_alias, hip_source) = module_parts(object, "gfx1151");
        assert_ne!(alias, hip_alias);
        assert_eq!(
            source,
            hip_source.replace("qmv4-gfx1151.hsaco", "qmv4_compute80.ptx")
        );
    }

    /// A build against a CUDA IREE runtime carries the PTX for every
    /// single-sequence entry point, compiled for the minimum architecture the
    /// graph is compiled for.
    #[cfg(xla_iree_cuda_ptx)]
    #[test]
    fn cuda_ptx_carries_every_single_sequence_entry() {
        let ptx = cuda_ptx().expect("a CUDA IREE build embeds the PTX");
        let text = std::str::from_utf8(ptx).expect("PTX is text");
        assert!(text.contains(&format!(".target sm_{CUDA_MIN_SM}")));
        for exports in [SINGLE_EXPORTS, QKV_EXPORTS, GATEUP_EXPORTS] {
            for (entry, ..) in exports {
                assert!(
                    text.contains(&format!(".entry {entry}(")),
                    "missing {entry}"
                );
            }
        }
    }

    /// Any other build has no PTX; the accessor reports why and the loader
    /// keeps the f16 path.
    #[cfg(not(xla_iree_cuda_ptx))]
    #[test]
    fn cuda_ptx_is_absent_without_a_compiled_artifact() {
        let reason = cuda_ptx().expect_err("no compiled artifact, no PTX");
        assert!(reason.contains("CUDA IREE runtime"), "{reason}");
    }
}
