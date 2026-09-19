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

//! Runtime Apple Silicon generation detection.
//!
//! Detects chip generation, GPU core count, memory bandwidth, and Neural
//! Accelerator availability (M5+) at runtime via `sysctlbyname`. Results are
//! cached in a `OnceLock` so detection runs exactly once per process.

use std::sync::OnceLock;

// ── CUDA compute capability ───────────────────────────────────────────────────

// The CUDA side of "what machine is this?" lives in [`crate::cuda_arch`] and is
// re-exported here so there is a single hardware-facing module to ask, whether
// the answer comes from `sysctlbyname` on Apple Silicon or from MLX's cached
// CUDA device attributes. It is a separate file because the architecture-list
// parser and its coverage rules carry their own unit tests and would otherwise
// push this module well past the size where it stays readable.
//
// Nothing here feeds [`HardwareCapabilities`]: that struct is built by
// [`detect_hardware`], which runs from [`apply_metal_ops_per_buffer_default`]
// at the very top of `main`, and probing a CUDA device from there would move
// device initialisation ahead of the environment defaults that must be set
// before MLX touches the GPU. The capability probe stays lazy and separate.
pub use crate::cuda_arch::{
    ArchCoverage, ArchVariant, CudaArchEntry, CudaArchMismatch, TRACE_ARCH_ENV, arch_list_coverage,
    compiled_cuda_architectures, cuda_arch_mismatch, cuda_arch_startup_summary,
    cuda_compute_capability, enforce_cuda_arch_compatibility, entry_coverage,
    parse_cuda_arch_entry, parse_cuda_arch_list, trace_arch_enabled, trace_arch_once,
};
pub use crate::cuda_graph_budget::{
    AppliedCudaGraphBudget, CudaGraphBudget, GB10_GRAPH_BUDGET, GB10_RAISED_BUDGET_MODEL_TYPES,
    ModelGraphShape, applied_cuda_graph_budget, apply_cuda_graph_budget_default,
    cuda_graph_budget_default, cuda_graph_budget_startup_summary, model_graph_shape_from_config,
    model_graph_shape_from_dir,
};

// ── Public types ──────────────────────────────────────────────────────────────

/// Apple Silicon chip generation detected at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum AppleSiliconGen {
    M1,
    M2,
    M3,
    M4,
    M5,
    Unknown,
}

impl AppleSiliconGen {
    /// Returns true for chips that have a Neural Accelerator (M5+).
    #[inline]
    pub fn has_neural_accelerator(self) -> bool {
        matches!(self, AppleSiliconGen::M5)
    }

    /// Whether this generation runs an affine-quantized projection at `M >= 2`
    /// as one wide pass rather than as narrow per-position passes.
    ///
    /// This is MLX's `use_qmv_wide` predicate (`mode != "affine" || arch_gen
    /// >= 15` in `mlx/backend/metal/quantized.cpp`) reduced to the part that
    /// depends only on the chip: Apple GPU generation 15 and newer, which is
    /// M3, M4 and M5. Generation 13 (M1, M2) has no such path and runs a
    /// verify block as `K` narrow passes whose cost grows with the block.
    ///
    /// That difference is what decides whether a speculative verify round pays
    /// for itself, so it is the discriminator behind the B=1 MTP gate
    /// (`mtp_b1_default` in the `mlxcel` crate); see `docs/benchmarks.md` for
    /// the round-cost measurements that order the generations by it.
    ///
    /// `Unknown` reads false, which is deliberate in both directions it
    /// covers: a non-Apple GPU (CUDA / GB10), where a K-wide verify does not
    /// amortize at all (issue #638), and an Apple generation newer than the
    /// enumerated ones, which `parse_silicon_gen` also maps to `Unknown`. Both
    /// decline rather than assume. Extend the enum when a new Apple generation
    /// ships; this carries the same staleness contract as
    /// [`Self::has_neural_accelerator`].
    #[inline]
    pub fn wide_quantized_projections(self) -> bool {
        matches!(
            self,
            AppleSiliconGen::M3 | AppleSiliconGen::M4 | AppleSiliconGen::M5
        )
    }

    /// Returns the expected Metal GPU family version (3 for M1–M4, 4 for M5+).
    #[inline]
    pub fn metal_version(self) -> u32 {
        match self {
            AppleSiliconGen::M5 => 4,
            _ => 3,
        }
    }
}

impl std::fmt::Display for AppleSiliconGen {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AppleSiliconGen::M1 => write!(f, "M1"),
            AppleSiliconGen::M2 => write!(f, "M2"),
            AppleSiliconGen::M3 => write!(f, "M3"),
            AppleSiliconGen::M4 => write!(f, "M4"),
            AppleSiliconGen::M5 => write!(f, "M5"),
            AppleSiliconGen::Unknown => write!(f, "Unknown"),
        }
    }
}

/// Which vendor made the GPU this process will run on.
///
/// Kept separate from [`AppleSiliconGen`] on purpose. That enum answers "which
/// Apple Silicon generation", and twelve call sites across the VLM loaders,
/// `models/sanitize.rs` and `drafter/dflash/drafter.rs` read its `Unknown`
/// variant as "this is not Apple Silicon" in order to gate a bf16 to f16 weight
/// conversion. Adding a vendor variant there would flip all eleven to true on
/// that vendor and silently enable the conversion, which is the same shape of
/// defect issue #1803 removed one layer down, where `!metal::is_available()`
/// was read as "CUDA" because only two backends existed.
///
/// `#[non_exhaustive]` so a future vendor does not break a downstream `match`.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuVendor {
    Apple,
    Nvidia,
    Amd,
    /// No GPU, or one this build does not identify.
    Unknown,
}

/// Which GPU backend MLX resolved for this process.
///
/// The Rust view of `mlxcel::GpuKernelBackend` (issue #1803). Read this rather
/// than inferring the backend from which `device_info()` keys are present: the
/// ROCm backend publishes `compute_capability_major`/`minor` from the `gfx`
/// target, so key presence says "AMD looks like CUDA", which is the misreport
/// this type exists to stop.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBackendKind {
    /// No GPU backend, or one this build does not know.
    None,
    Metal,
    Cuda,
    Rocm,
}

impl GpuBackendKind {
    /// Every variant, in declaration order.
    ///
    /// Crates downstream of this one cannot `match` exhaustively on a
    /// `#[non_exhaustive]` enum, so adding a variant gives them no compile
    /// error. They walk this list in a test instead: the WebUI catalog checks
    /// that every kind reads its own registry column (issue #1886). The const
    /// block below fails the build when a variant is missing from the list.
    pub const ALL: [GpuBackendKind; 4] = [
        GpuBackendKind::None,
        GpuBackendKind::Metal,
        GpuBackendKind::Cuda,
        GpuBackendKind::Rocm,
    ];
}

// Compile-time guard for `GpuBackendKind::ALL`. The match has no wildcard, so a
// new variant fails to compile here until it has an arm, and each arm reads its
// own variant's slot of `ALL`. The arm for a fifth variant reads `ALL[4]`,
// which fails to compile (`unconditional_panic`) until `ALL` gains that entry.
// The loop then rejects a list that is out of order, repeats a variant, or has
// an arm reading another variant's slot.
const _: () = {
    const fn slot(kind: GpuBackendKind) -> GpuBackendKind {
        match kind {
            GpuBackendKind::None => GpuBackendKind::ALL[0],
            GpuBackendKind::Metal => GpuBackendKind::ALL[1],
            GpuBackendKind::Cuda => GpuBackendKind::ALL[2],
            GpuBackendKind::Rocm => GpuBackendKind::ALL[3],
        }
    }
    let mut position = 0;
    while position < GpuBackendKind::ALL.len() {
        let kind = GpuBackendKind::ALL[position];
        assert!(
            kind as usize == position && slot(kind) as usize == position,
            "GpuBackendKind::ALL must list every variant once, in declaration order"
        );
        position += 1;
    }
};

/// The GPU backend MLX resolved, probed once per process.
#[must_use]
pub fn gpu_backend_kind() -> GpuBackendKind {
    static KIND: OnceLock<GpuBackendKind> = OnceLock::new();
    *KIND.get_or_init(|| match crate::ffi::gpu_backend_kind() {
        1 => GpuBackendKind::Metal,
        2 => GpuBackendKind::Cuda,
        3 => GpuBackendKind::Rocm,
        // 0 is `None`; anything else is a bridge the Rust side does not know
        // yet, which is also "no backend I can report on".
        _ => GpuBackendKind::None,
    })
}

/// Hardware capabilities detected at runtime.
#[derive(Debug, Clone)]
pub struct HardwareCapabilities {
    /// Which vendor made the GPU. Ask [`HardwareCapabilities::is_apple_silicon`]
    /// rather than comparing `silicon_gen` when the question is "is this Apple".
    pub vendor: GpuVendor,
    /// Apple Silicon chip generation. `Unknown` on every non-Apple vendor.
    pub silicon_gen: AppleSiliconGen,
    /// Number of GPU cores (performance cluster logical CPUs as a proxy).
    pub gpu_core_count: u32,
    /// True for M5+ chips which have a dedicated Neural Accelerator.
    pub has_neural_accelerator: bool,
    /// Metal GPU family version (3 for M1–M4, 4 for M5+).
    pub metal_version: u32,
    /// True when running on macOS 26.2+ (required to use the Neural Accelerator).
    pub macos_supports_na: bool,
    /// Approximate memory bandwidth in GB/s (estimated from chip generation).
    pub memory_bandwidth_gbps: f64,
    /// Unified memory size in GB.
    ///
    /// Apple only. Left at 0 on every other vendor on purpose: it is the third
    /// step of `resolve_available_memory`, and filling it from device memory
    /// would change what that walk returns on CUDA and ROCm. Read
    /// [`HardwareCapabilities::device_memory_bytes`] for device memory.
    pub unified_memory_gb: u32,
    /// Device name as the backend reports it ("AMD Radeon Graphics", "NVIDIA
    /// GB10"), or empty when the backend publishes none.
    pub device_name: String,
    /// Architecture string as the running backend spells it, or `None` when
    /// this build did not probe one.
    ///
    /// `None` on macOS even though Metal publishes an architecture: detection
    /// runs before MLX's environment defaults are applied there, so it reads
    /// sysctl rather than initialising the device. Read `silicon_gen` for the
    /// Apple generation.
    ///
    /// Each backend has its own vocabulary and they are not comparable:
    /// `gfx1151` on ROCm, `sm_89` on CUDA, an Apple GPU family string on
    /// Metal. Treating this as a `gfx` target is exactly the cross-vendor
    /// confusion #1805 exists to remove, so code that means the HIP target
    /// reads [`crate::rocm_arch::device_gfx_target`], which is `None` off
    /// ROCm and normalized. This field is for reporting.
    pub device_architecture: Option<String>,
    /// Total device memory in bytes, or 0 when the backend does not publish
    /// it. On a UMA host, Apple or an AMD carve-out, this is the unified pool
    /// rather than a separate VRAM figure.
    pub device_memory_bytes: u64,
}

impl Default for HardwareCapabilities {
    fn default() -> Self {
        Self {
            vendor: GpuVendor::Unknown,
            silicon_gen: AppleSiliconGen::Unknown,
            gpu_core_count: 0,
            has_neural_accelerator: false,
            metal_version: 3,
            macos_supports_na: false,
            memory_bandwidth_gbps: 0.0,
            unified_memory_gb: 0,
            device_name: String::new(),
            device_architecture: None,
            device_memory_bytes: 0,
        }
    }
}

impl HardwareCapabilities {
    /// True when this is Apple Silicon.
    ///
    /// This is the predicate twelve call sites use to decide whether to convert
    /// bf16 weights to f16 (the VLM loaders, `models/sanitize.rs` and
    /// `drafter/dflash/drafter.rs`). It reads `silicon_gen` rather than
    /// `vendor` so that adding a vendor cannot change what those sites do:
    /// a new vendor leaves `silicon_gen` at `Unknown` and the answer stays
    /// false. `gpu_vendor_does_not_imply_apple_silicon` pins that.
    #[must_use]
    pub fn is_apple_silicon(&self) -> bool {
        self.silicon_gen != AppleSiliconGen::Unknown
    }
}

// ── Process-level singleton ───────────────────────────────────────────────────

static HARDWARE_CAPABILITIES: OnceLock<HardwareCapabilities> = OnceLock::new();

/// Return a reference to the cached `HardwareCapabilities`.
///
/// Detection runs at most once per process; subsequent calls return the cached
/// result immediately.
#[inline]
pub fn get_hardware() -> &'static HardwareCapabilities {
    HARDWARE_CAPABILITIES.get_or_init(detect_hardware)
}

/// True only on M5-class Apple Silicon whose Neural Accelerator is driven by
/// the running macOS (Metal GPU Family 4).
///
/// This is the gate for M5-Max-specific numerical workarounds. The M5 Max NAx
/// GEMM kernels can fuse a lazy float32/float16 graph into NaN within a single
/// Metal command buffer; code that builds such graphs (the Mamba2/SSM mixers)
/// forces an intermediate `eval` boundary, but only when this returns true. On
/// every other chip that eval boundary is pure throughput loss, so it is
/// skipped. See CLAUDE.md "Apple Silicon precision".
#[inline]
pub fn is_m5_neural_accelerator() -> bool {
    let hw = get_hardware();
    hw.has_neural_accelerator && hw.macos_supports_na
}

/// The `MLX_MAX_OPS_PER_BUFFER` default to apply for an Apple Silicon class, or
/// `None` to leave MLX's built-in default in place.
///
/// `MLX_MAX_OPS_PER_BUFFER` caps how many ops MLX batches into one Metal command
/// buffer before committing it. On pre-M5 Apple Silicon (Metal GPU family 3) the
/// default cap is small enough that high-op-density decode (Gemma3n: AltUp
/// 4-plane, LAUREL, per-layer input gating, dual norms) stalls the GPU at
/// command-buffer boundaries; raising the cap to 1000 closes that dispatch-gap
/// idle and recovers +11 to 13% decode throughput on M1 Ultra (see
/// docs/benchmark_results/gemma3n-decode-profile.md, #329/#345).
///
/// The lever is hardware-specific, so it is gated:
/// - M1 through M4 (no Neural Accelerator, Metal family 3): apply 1000.
/// - M5+ (has Neural Accelerator): leave MLX's default. The same sweep is flat
///   on M5 Max (within run-to-run noise, no plateau) and an earlier M5 study
///   recorded larger buffers as slower, so raising it offers no gain and risks a
///   regression (docs/benchmark_results/gemma3n-decode-profile-m5max.md, #358).
/// - Unknown (non-Apple, e.g. CUDA): leave untouched; this is a Metal command
///   buffer scheduling knob, irrelevant off Apple GPUs.
///
/// The gain is most pronounced on Gemma3n's dense high-op-density stack; the MoE
/// decode sweep was flat (#268), so other families are expected neutral, not
/// regressed. This is a default only: an explicit `MLX_MAX_OPS_PER_BUFFER` in the
/// environment always wins (see [`apply_metal_ops_per_buffer_default`]).
#[must_use]
pub fn metal_ops_per_buffer_default(
    r#gen: AppleSiliconGen,
    has_neural_accelerator: bool,
) -> Option<u32> {
    if r#gen != AppleSiliconGen::Unknown && !has_neural_accelerator {
        Some(1000)
    } else {
        None
    }
}

/// Apply the hardware-gated [`metal_ops_per_buffer_default`] to the process
/// environment, unless `MLX_MAX_OPS_PER_BUFFER` is already set.
///
/// Call this once, early in `main()`, before any MLX op runs and before spawning
/// threads: MLX reads `MLX_MAX_OPS_PER_BUFFER` when it commits command buffers,
/// and setting an environment variable is only sound while the process is
/// effectively single-threaded. An operator-set `MLX_MAX_OPS_PER_BUFFER` (any
/// value) is always respected.
pub fn apply_metal_ops_per_buffer_default() {
    if std::env::var_os("MLX_MAX_OPS_PER_BUFFER").is_some() {
        return;
    }
    let hw = get_hardware();
    if let Some(value) = metal_ops_per_buffer_default(hw.silicon_gen, hw.has_neural_accelerator) {
        // SAFETY: set_var mutates the process-global environment and is unsound
        // only if another thread reads or writes the environment concurrently.
        // Per this function's documented contract, all in-tree callers invoke it
        // once at the top of `main` right after CLI parsing (src/main.rs,
        // src/bin/mlx_server.rs, src/bin/bench_decode.rs), before any model load,
        // MLX op, or worker thread touches the environment, so no other thread is
        // accessing it here.
        unsafe { std::env::set_var("MLX_MAX_OPS_PER_BUFFER", value.to_string()) };
    }
}

/// The `MLX_CUDA_GRAPH_CACHE_SIZE` default to apply on a CUDA build, or `None`
/// off CUDA.
///
/// MLX's CUDA backend keeps an LRU cache of captured CUDA graphs keyed by graph
/// shape, and its built-in capacity default is 400
/// (`mlx/backend/cuda/device.cpp`). `mlx/backend/cuda/lru_cache.h` also keeps a
/// lifetime miss counter that is never reset (not on hits, not on trim) and
/// throws a fatal `std::runtime_error("Cache thrashing is happening ...")` once
/// that counter passes `2 * capacity`, i.e. 800 lifetime misses at the default.
/// A long-lived, shape-diverse CUDA server crosses that threshold over its
/// lifetime, and speculative or batched decode reaches it fastest: the draft and
/// verify phases, multiplied across varying batch sizes and sequence-length
/// buckets, produce many distinct graph shapes. The throw is a process death,
/// not a request-level error, so it drops every in-flight request (issue #818).
///
/// Raising the cap to 2000 was validated to eliminate the crash (13/13 requests
/// across multiple bursts on GB10). 2000 is an LRU capacity cap, not a
/// preallocation, so it only costs memory as distinct graph shapes actually
/// accumulate. This is a default only: an operator-set `MLX_CUDA_GRAPH_CACHE_SIZE`
/// (any value) always wins (see [`apply_cuda_graph_cache_default`]). The variable
/// is read only by MLX's CUDA backend, so it is a harmless no-op when the runtime
/// device is CPU.
///
/// Tracking (issue #821): the upstream behavior is written up in
/// `docs/upstream/mlx-cuda-graph-cache-lifetime-miss-abort.md`. Raising the cap
/// only buys a proportionally larger lifetime budget, so it is a delay, not a
/// fix. Removal condition: once an upstream fix lands that stops punishing a
/// large lifetime working set (a windowed miss rate, or a non-fatal report) and
/// the MLX pin in `src/lib/mlx-cpp/CMakeLists.txt` moves past it, this default
/// can be lowered back toward MLX's own 400 or dropped entirely.
#[must_use]
pub fn cuda_graph_cache_default() -> Option<u32> {
    #[cfg(feature = "cuda")]
    {
        Some(2000)
    }
    #[cfg(not(feature = "cuda"))]
    {
        None
    }
}

/// Apply the [`cuda_graph_cache_default`] to the process environment, unless
/// `MLX_CUDA_GRAPH_CACHE_SIZE` is already set.
///
/// Call this once, early in `main()`, before any MLX op runs and before spawning
/// threads: MLX reads `MLX_CUDA_GRAPH_CACHE_SIZE` when it first sizes the CUDA
/// graph cache, and setting an environment variable is only sound while the
/// process is effectively single-threaded. An operator-set
/// `MLX_CUDA_GRAPH_CACHE_SIZE` (any value) is always respected.
pub fn apply_cuda_graph_cache_default() {
    if std::env::var_os("MLX_CUDA_GRAPH_CACHE_SIZE").is_some() {
        return;
    }
    if let Some(value) = cuda_graph_cache_default() {
        // SAFETY: set_var mutates the process-global environment and is unsound
        // only if another thread reads or writes the environment concurrently.
        // Per this function's documented contract, all in-tree callers invoke it
        // once at the top of `main` right after CLI parsing (src/main.rs,
        // src/bin/mlx_server.rs, src/bin/bench_decode.rs,
        // src/bin/speculative_bench.rs), before any model load, MLX op, or worker
        // thread touches the environment, so no other thread is accessing it here.
        unsafe { std::env::set_var("MLX_CUDA_GRAPH_CACHE_SIZE", value.to_string()) };
    }
}

// ── Detection ─────────────────────────────────────────────────────────────────

/// The `MLX_CUDA_SDPA_CACHE_SIZE` default to apply on a CUDA build, or `None`
/// off CUDA.
///
/// MLX's CUDA backend caches cuDNN SDPA execution plans in an LRU keyed by the
/// exact query, key, value and mask shapes and strides
/// (`mlx/backend/cuda/scaled_dot_product_attention.cpp`, capacity 256 by
/// default), built on the same `lru_cache.h` as the graph cache, so it carries
/// the same lifetime miss counter and the same fatal `Cache thrashing` throw
/// once lifetime misses pass `2 * capacity` (512 at MLX's default). Every
/// distinct prompt length that prefills through cuDNN is one miss per attention
/// layer class, so a long-lived server crosses 512 on prompt diversity alone;
/// a multi-row speculative verify used to cross it in a few hundred rounds
/// before its small-query calls were routed away from cuDNN (issue #1799,
/// where a 400-token Laguna DFlash run at block 2 aborted on it). Raised to
/// 2000 for the same reasons and with the same caveats as
/// [`cuda_graph_cache_default`]: an LRU cap, not a preallocation; a larger
/// lifetime budget, not a fix; an operator-set value always wins.
#[must_use]
pub fn cuda_sdpa_cache_default() -> Option<u32> {
    #[cfg(feature = "cuda")]
    {
        Some(2000)
    }
    #[cfg(not(feature = "cuda"))]
    {
        None
    }
}

/// Apply the [`cuda_sdpa_cache_default`] to the process environment, unless
/// `MLX_CUDA_SDPA_CACHE_SIZE` is already set. Same contract as
/// [`apply_cuda_graph_cache_default`]: once, early in `main()`, before any MLX
/// op and before spawning threads.
pub fn apply_cuda_sdpa_cache_default() {
    if std::env::var_os("MLX_CUDA_SDPA_CACHE_SIZE").is_some() {
        return;
    }
    if let Some(value) = cuda_sdpa_cache_default() {
        // SAFETY: same argument as `apply_cuda_graph_cache_default`: every
        // in-tree caller invokes this once at the top of `main` right after
        // CLI parsing (src/main.rs, src/bin/mlx_server.rs,
        // src/bin/bench_decode.rs, src/bin/speculative_bench.rs), before any
        // model load, MLX op, or worker thread touches the environment, so no
        // other thread is accessing it here.
        unsafe { std::env::set_var("MLX_CUDA_SDPA_CACHE_SIZE", value.to_string()) };
    }
}

/// Detect hardware capabilities by querying the OS at runtime.
///
/// On non-macOS platforms this always returns [`HardwareCapabilities::default`].
pub fn detect_hardware() -> HardwareCapabilities {
    #[cfg(target_os = "macos")]
    {
        detect_hardware_macos()
    }

    #[cfg(not(target_os = "macos"))]
    {
        detect_hardware_gpu()
    }
}

/// Detection off macOS, where the vendor comes from the resolved MLX backend
/// and the device strings come from `device_info()`.
///
/// Everything Apple-specific stays at its default: `silicon_gen` `Unknown`, no
/// Neural Accelerator, and `unified_memory_gb` 0 so `resolve_available_memory`
/// keeps the order it had on CUDA before this function existed. What it adds
/// is the vendor tag plus the three device strings the ROCm backend publishes
/// and nothing previously read (issue #1805).
#[cfg(not(target_os = "macos"))]
fn detect_hardware_gpu() -> HardwareCapabilities {
    let backend = gpu_backend_kind();
    let vendor = match backend {
        GpuBackendKind::Metal => GpuVendor::Apple,
        GpuBackendKind::Cuda => GpuVendor::Nvidia,
        GpuBackendKind::Rocm => GpuVendor::Amd,
        GpuBackendKind::None => GpuVendor::Unknown,
    };
    if vendor == GpuVendor::Unknown {
        // No device to describe; asking for its name would just return "".
        return HardwareCapabilities::default();
    }
    let architecture = crate::ffi::gpu_architecture(0);
    HardwareCapabilities {
        vendor,
        device_name: crate::ffi::gpu_device_name(0),
        // Empty means no device at index 0. `None` says that; "" would read as
        // an architecture named nothing at every consumer.
        device_architecture: (!architecture.is_empty()).then_some(architecture),
        device_memory_bytes: crate::ffi::gpu_total_memory(0) as u64,
        ..HardwareCapabilities::default()
    }
}

// ── macOS implementation ──────────────────────────────────────────────────────

#[cfg(target_os = "macos")]
mod platform {
    use std::os::raw::{c_char, c_int, c_void};

    // sysctlbyname(3) is in <sys/sysctl.h> — link against libSystem automatically.
    unsafe extern "C" {
        fn sysctlbyname(
            name: *const c_char,
            oldp: *mut c_void,
            oldlenp: *mut usize,
            newp: *mut c_void,
            newlen: usize,
        ) -> c_int;
    }

    /// Read a NUL-terminated string sysctl value.
    pub fn sysctl_string(name: &str) -> Option<String> {
        let c_name = std::ffi::CString::new(name).ok()?;
        // First call: query required buffer length.
        let mut len: usize = 0;
        // SAFETY: `c_name` is a valid NUL-terminated C string. We pass null for
        // `oldp` with a valid `&mut len` to query the required buffer size.
        // `newp` is null and `newlen` is 0 (read-only).
        let rc = unsafe {
            sysctlbyname(
                c_name.as_ptr(),
                std::ptr::null_mut(),
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        if rc != 0 || len == 0 {
            return None;
        }
        // Second call: fill buffer.
        let mut buf = vec![0u8; len];
        // SAFETY: `buf` is a valid, writable buffer of `len` bytes. `len` is
        // updated by the kernel to reflect the actual bytes written (at most
        // the original buffer size).
        let rc = unsafe {
            sysctlbyname(
                c_name.as_ptr(),
                buf.as_mut_ptr() as *mut c_void,
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        if rc != 0 {
            return None;
        }
        // Truncate to the actual length returned by the kernel (may be shorter
        // than the original allocation if the value shrank between calls).
        buf.truncate(len);
        // Trim trailing NUL bytes.
        while buf.last() == Some(&0) {
            buf.pop();
        }
        String::from_utf8(buf).ok()
    }

    /// Read a `u64` sysctl value.
    pub fn sysctl_u64(name: &str) -> Option<u64> {
        let c_name = std::ffi::CString::new(name).ok()?;
        let mut value: u64 = 0;
        let mut len = std::mem::size_of::<u64>();
        // SAFETY: `c_name` is a valid NUL-terminated C string. `value` is a
        // properly aligned `u64` with `len` set to its exact size. The kernel
        // writes at most `len` bytes into `value`.
        let rc = unsafe {
            sysctlbyname(
                c_name.as_ptr(),
                &mut value as *mut u64 as *mut c_void,
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        if rc != 0 || len != std::mem::size_of::<u64>() {
            return None;
        }
        Some(value)
    }

    /// Read a `u32` sysctl value.
    pub fn sysctl_u32(name: &str) -> Option<u32> {
        let c_name = std::ffi::CString::new(name).ok()?;
        let mut value: u32 = 0;
        let mut len = std::mem::size_of::<u32>();
        // SAFETY: `c_name` is a valid NUL-terminated C string. `value` is a
        // properly aligned `u32` with `len` set to its exact size. The kernel
        // writes at most `len` bytes into `value`.
        let rc = unsafe {
            sysctlbyname(
                c_name.as_ptr(),
                &mut value as *mut u32 as *mut c_void,
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        if rc != 0 || len != std::mem::size_of::<u32>() {
            return None;
        }
        Some(value)
    }
}

#[cfg(target_os = "macos")]
fn detect_hardware_macos() -> HardwareCapabilities {
    use platform::{sysctl_string, sysctl_u32, sysctl_u64};

    // ── Chip generation ───────────────────────────────────────────────────────
    let brand = sysctl_string("machdep.cpu.brand_string").unwrap_or_default();
    let silicon_gen = parse_silicon_gen(&brand);

    // ── GPU core count ────────────────────────────────────────────────────────
    // `hw.perflevel0.logicalcpu` is the performance-cluster CPU count, which
    // closely correlates with GPU core count on Apple Silicon.
    let gpu_core_count = sysctl_u32("hw.perflevel0.logicalcpu").unwrap_or(0);

    // ── Unified memory ────────────────────────────────────────────────────────
    let mem_bytes = sysctl_u64("hw.memsize").unwrap_or(0);
    let unified_memory_gb = (mem_bytes / (1024 * 1024 * 1024)) as u32;

    // ── macOS version ─────────────────────────────────────────────────────────
    let macos_version_str = sysctl_string("kern.osproductversion").unwrap_or_default();
    let (macos_major, macos_minor) = parse_macos_version(&macos_version_str);
    // Neural Accelerator API requires macOS 26.2+
    let macos_supports_na = (macos_major > 26) || (macos_major == 26 && macos_minor >= 2);

    // ── Derived fields ────────────────────────────────────────────────────────
    let has_neural_accelerator = silicon_gen.has_neural_accelerator();
    let metal_version = silicon_gen.metal_version();
    let memory_bandwidth_gbps = estimate_bandwidth(silicon_gen, unified_memory_gb);

    HardwareCapabilities {
        // macOS only ever runs the Metal backend here, so the vendor is not a
        // probe result. `silicon_gen` still carries the generation, and
        // `is_apple_silicon()` still reads that field rather than this one.
        vendor: GpuVendor::Apple,
        silicon_gen,
        gpu_core_count,
        has_neural_accelerator,
        metal_version,
        macos_supports_na,
        memory_bandwidth_gbps,
        unified_memory_gb,
        // Both of these come from sysctl rather than from MLX `device_info()`,
        // because this function runs at the top of `main`, before the
        // environment defaults MLX reads are applied, and probing the device
        // here would move device initialisation ahead of them. On a SoC the
        // chip brand is the device name a reader wants anyway, and the
        // architecture is left unreported rather than guessed.
        device_name: brand,
        device_architecture: None,
        // Unified memory: on Apple Silicon the GPU pool is host RAM.
        device_memory_bytes: mem_bytes,
    }
}

/// Parse Apple Silicon generation from the CPU brand string.
///
/// Examples:
/// - `"Apple M1"` → `M1`
/// - `"Apple M4 Max"` → `M4`
/// - `"Apple M5 Pro"` → `M5`
#[cfg(any(target_os = "macos", test))]
fn parse_silicon_gen(brand: &str) -> AppleSiliconGen {
    // Look for "Apple M<n>" pattern anywhere in the string.
    if let Some(pos) = brand.find("Apple M") {
        let rest = &brand[pos + "Apple M".len()..];
        // Extract the full generation number (handles multi-digit like M10+).
        let gen_str: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        match gen_str.as_str() {
            "1" => AppleSiliconGen::M1,
            "2" => AppleSiliconGen::M2,
            "3" => AppleSiliconGen::M3,
            "4" => AppleSiliconGen::M4,
            "5" => AppleSiliconGen::M5,
            _ => AppleSiliconGen::Unknown,
        }
    } else {
        AppleSiliconGen::Unknown
    }
}

/// Parse `"major.minor[.patch]"` version strings.
///
/// Returns `(major, minor)`. Returns `(0, 0)` on parse failure.
#[cfg(any(target_os = "macos", test))]
fn parse_macos_version(version: &str) -> (u32, u32) {
    let mut parts = version.split('.');
    let major = parts
        .next()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(0);
    let minor = parts
        .next()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(0);
    (major, minor)
}

/// Return approximate memory bandwidth in GB/s based on chip generation and
/// memory configuration.  Values are midpoints from Apple's published specs
/// for the base chip variant at the given memory size.
#[cfg(target_os = "macos")]
fn estimate_bandwidth(r#gen: AppleSiliconGen, memory_gb: u32) -> f64 {
    match r#gen {
        AppleSiliconGen::M1 => {
            if memory_gb > 16 {
                400.0 // M1 Max / Ultra
            } else {
                68.25 // M1 base / Pro
            }
        }
        AppleSiliconGen::M2 => {
            if memory_gb > 24 {
                800.0 // M2 Ultra
            } else if memory_gb > 16 {
                400.0 // M2 Max
            } else {
                100.0 // M2 base / Pro
            }
        }
        AppleSiliconGen::M3 => {
            if memory_gb > 36 {
                800.0 // M3 Ultra
            } else if memory_gb > 18 {
                400.0 // M3 Max
            } else {
                100.0 // M3 base / Pro
            }
        }
        AppleSiliconGen::M4 => {
            if memory_gb > 64 {
                800.0 // M4 Ultra
            } else if memory_gb > 32 {
                546.0 // M4 Max
            } else {
                120.0 // M4 base / Pro
            }
        }
        AppleSiliconGen::M5 => {
            // Estimated; update once Apple publishes official specs.
            if memory_gb > 64 { 1000.0 } else { 150.0 }
        }
        AppleSiliconGen::Unknown => 0.0,
    }
}

// ── KV cache memory estimation ────────────────────────────────────────────────

/// The pre-allocation step used by [`KVCache`](crate::cache::KVCache).
///
/// All buffer allocations are rounded up to the next multiple of this value so
/// that the estimated reservation matches what the runtime actually allocates.
pub const KV_CACHE_ALLOC_STEP: u64 = 256;

/// Estimate the KV-cache memory reservation in bytes.
///
/// The formula is:
/// ```text
/// num_layers × 2 (K + V) × num_kv_heads × head_dim × elem_bytes
///     × round_up(ctx_len, 256) × batch
/// ```
///
/// `ctx_len` is rounded up to the next multiple of [`KV_CACHE_ALLOC_STEP`]
/// (256) to match the actual buffer pre-allocation performed by
/// [`KVCache`](crate::cache::KVCache).
///
/// # Arguments
/// * `num_layers`  — number of transformer layers.
/// * `num_kv_heads` — number of KV attention heads (may be < `num_heads` for GQA/MQA).
/// * `head_dim`    — per-head dimension (usually `hidden_size / num_heads`).
/// * `elem_bytes`  — bytes per element: 2 for FP16/BF16, 1 for INT8 KV.
/// * `ctx_len`     — requested context length in tokens.
/// * `batch`       — batch size (typically 1 for interactive generation).
///
/// # Examples
/// ```
/// use mlxcel_core::hardware::kv_cache_bytes;
///
/// // 32-layer, 8-head GQA model, 128-dim heads, FP16, 8K context, batch 1.
/// let bytes = kv_cache_bytes(32, 8, 128, 2, 8192, 1);
/// // ctx_len rounded to 8192 (already a multiple of 256).
/// // 32 × 2 × 8 × 128 × 2 × 8192 × 1 = 1_073_741_824 (1 GiB)
/// assert_eq!(bytes, 1_073_741_824);
/// ```
#[must_use]
pub fn kv_cache_bytes(
    num_layers: u64,
    num_kv_heads: u64,
    head_dim: u64,
    elem_bytes: u64,
    ctx_len: u64,
    batch: u64,
) -> u64 {
    // Round ctx_len up to the next multiple of KV_CACHE_ALLOC_STEP so the
    // estimate matches the actual buffer reservation.
    let rounded_ctx =
        ctx_len.saturating_add(KV_CACHE_ALLOC_STEP - 1) / KV_CACHE_ALLOC_STEP * KV_CACHE_ALLOC_STEP;

    num_layers
        .saturating_mul(2) // K and V
        .saturating_mul(num_kv_heads)
        .saturating_mul(head_dim)
        .saturating_mul(elem_bytes)
        .saturating_mul(rounded_ctx)
        .saturating_mul(batch)
}

/// Model architecture parameters needed to compute KV cache memory.
///
/// Passed to [`kv_cache_bytes_from_params`] to avoid long argument lists and
/// to give the unified estimator a single stable entry point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvCacheParams {
    /// Number of transformer layers.
    pub num_layers: u64,
    /// Number of KV attention heads (may be < `num_heads` for GQA/MQA).
    pub num_kv_heads: u64,
    /// Per-head dimension (`hidden_size / num_heads`).
    pub head_dim: u64,
    /// `true` when INT8 KV cache is active (`--cache-type-k int8` /
    /// `--cache-type-v int8` / `--kv-cache-mode int8`).  INT8 storage halves
    /// `elem_bytes` relative to FP16.
    pub int8_kv: bool,
    /// Requested context length in tokens.
    pub ctx_len: u64,
    /// Batch size (typically 1 for interactive generation).
    pub batch: u64,
}

impl KvCacheParams {
    /// Create params with `int8_kv = false` and `batch = 1`.
    pub fn new(num_layers: u64, num_kv_heads: u64, head_dim: u64, ctx_len: u64) -> Self {
        Self {
            num_layers,
            num_kv_heads,
            head_dim,
            int8_kv: false,
            ctx_len,
            batch: 1,
        }
    }
}

/// Compute KV-cache memory reservation from a [`KvCacheParams`] struct.
///
/// `elem_bytes` is derived from `params.int8_kv`: `1` for INT8, `2` for FP16/BF16.
///
/// # Examples
/// ```
/// use mlxcel_core::hardware::{KvCacheParams, kv_cache_bytes_from_params};
///
/// let params = KvCacheParams {
///     num_layers: 32,
///     num_kv_heads: 8,
///     head_dim: 128,
///     int8_kv: false,
///     ctx_len: 8192,
///     batch: 1,
/// };
/// let bytes = kv_cache_bytes_from_params(&params);
/// assert_eq!(bytes, 1_073_741_824); // 1 GiB
/// ```
#[must_use]
pub fn kv_cache_bytes_from_params(params: &KvCacheParams) -> u64 {
    let elem_bytes = if params.int8_kv { 1 } else { 2 };
    kv_cache_bytes(
        params.num_layers,
        params.num_kv_heads,
        params.head_dim,
        elem_bytes,
        params.ctx_len,
        params.batch,
    )
}

// ── Quantization recommendation ───────────────────────────────────────────────

/// Recommended quantization mode for a given hardware + model combination.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum QuantRecommendation {
    /// 8-bit integer quantization — best throughput on M5 Neural Accelerator.
    Int8 { reason: &'static str },
    /// 4-bit affine quantization — best balance of speed and memory footprint.
    Int4Affine { reason: &'static str },
    /// FP16 (no quantization) — for small models that fit comfortably in memory.
    Fp16 { reason: &'static str },
}

impl QuantRecommendation {
    /// Short label used in CLI output (e.g. `"int8"`, `"int4"`, `"fp16"`).
    pub fn label(&self) -> &'static str {
        match self {
            QuantRecommendation::Int8 { .. } => "int8",
            QuantRecommendation::Int4Affine { .. } => "int4",
            QuantRecommendation::Fp16 { .. } => "fp16",
        }
    }

    /// Human-readable rationale returned by the recommendation engine.
    pub fn reason(&self) -> &'static str {
        match self {
            QuantRecommendation::Int8 { reason }
            | QuantRecommendation::Int4Affine { reason }
            | QuantRecommendation::Fp16 { reason } => reason,
        }
    }
}

/// Recommend the optimal quantization mode for a given model and hardware.
///
/// The decision tree is:
/// 1. **M5 with Neural Accelerator + enough memory for 8-bit**: prefer INT8.
///    The M5 NA delivers ~2x compute throughput for INT8 vs FP16, making 8-bit
///    quantized models strictly faster when they fit in unified memory.
/// 2. **4-bit headroom**: prefer INT4 affine — best latency-per-memory trade-off
///    on all other Apple Silicon generations.
/// 3. **Fallback**: INT4 is also recommended when memory is tight (8-bit would
///    not fit), so we never recommend FP16 unless the model is tiny enough that
///    no quantization is needed.
///
/// # Arguments
/// * `model_params_billions` — approximate model parameter count in billions.
/// * `available_memory_gb` — total unified memory in GB (`unified_memory_gb`
///   from [`HardwareCapabilities`]).
/// * `hw` — hardware capabilities from [`get_hardware`].
/// * `kv_cache_headroom_bytes` — KV cache memory reservation in bytes. When
///   `None`, falls back to a conservative 2 GiB default for backward
///   compatibility. Pass the result of [`kv_cache_bytes`] or
///   [`kv_cache_bytes_from_params`] when model architecture info is available.
pub fn recommend_quantization(
    model_params_billions: f64,
    available_memory_gb: u32,
    hw: &HardwareCapabilities,
    kv_cache_headroom_bytes: Option<u64>,
) -> QuantRecommendation {
    // Rough memory footprints (parameters only — add KV cache headroom):
    //   FP16: ~2 bytes/param  →  model_params_billions * 2 GB
    //   INT8: ~1 byte/param   →  model_params_billions * 1 GB
    //   INT4: ~0.5 bytes/param →  model_params_billions * 0.5 GB
    //
    // KV headroom is computed from model architecture when available; fall back
    // to a conservative 2 GiB constant for callers that do not supply it.
    const FALLBACK_KV_HEADROOM_GB: u32 = 2;
    let kv_headroom_gb = match kv_cache_headroom_bytes {
        Some(bytes) => {
            // Convert bytes → GiB, rounding up so the headroom is never
            // under-estimated (1 GiB = 1_073_741_824 bytes).
            bytes.div_ceil(1_073_741_824) as u32
        }
        None => FALLBACK_KV_HEADROOM_GB,
    };

    let mem_fp16_gb = (model_params_billions * 2.0).ceil() as u32 + kv_headroom_gb;
    let mem_8bit_gb = (model_params_billions * 1.0).ceil() as u32 + kv_headroom_gb;
    let mem_4bit_gb = (model_params_billions * 0.5).ceil() as u32 + kv_headroom_gb;

    // M5 Neural Accelerator path: 2x INT8 throughput over FP16.
    if hw.has_neural_accelerator && hw.macos_supports_na {
        if mem_8bit_gb <= available_memory_gb {
            return QuantRecommendation::Int8 {
                reason: "M5 NA delivers 2x throughput for INT8 vs FP16",
            };
        }
        // NA still helps with INT4 on M5 for models too large for 8-bit.
        if mem_4bit_gb <= available_memory_gb {
            return QuantRecommendation::Int4Affine {
                reason: "M5 NA available but 8-bit exceeds memory; 4-bit recommended",
            };
        }
    }

    // Non-M5 or macOS < 26.2: FP16 if small enough, else INT4.
    if mem_fp16_gb <= available_memory_gb {
        return QuantRecommendation::Fp16 {
            reason: "Model fits in memory as FP16; no quantization needed",
        };
    }

    if mem_4bit_gb <= available_memory_gb {
        return QuantRecommendation::Int4Affine {
            reason: "Best balance of speed and memory on this hardware",
        };
    }

    // Even 4-bit is tight — still recommend it as the only viable option.
    QuantRecommendation::Int4Affine {
        reason: "Memory constrained; 4-bit required to fit model",
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_brand_strings() {
        assert_eq!(parse_silicon_gen("Apple M1"), AppleSiliconGen::M1);
        assert_eq!(parse_silicon_gen("Apple M2 Pro"), AppleSiliconGen::M2);
        assert_eq!(parse_silicon_gen("Apple M3 Max"), AppleSiliconGen::M3);
        assert_eq!(parse_silicon_gen("Apple M4"), AppleSiliconGen::M4);
        assert_eq!(parse_silicon_gen("Apple M5 Pro"), AppleSiliconGen::M5);
        assert_eq!(parse_silicon_gen("Intel Core i9"), AppleSiliconGen::Unknown);
        assert_eq!(parse_silicon_gen(""), AppleSiliconGen::Unknown);
        // Multi-digit generation should not mis-parse as single digit.
        assert_eq!(parse_silicon_gen("Apple M10 Pro"), AppleSiliconGen::Unknown);
    }

    #[test]
    fn parse_macos_versions() {
        assert_eq!(parse_macos_version("14.5"), (14, 5));
        assert_eq!(parse_macos_version("26.2.0"), (26, 2));
        assert_eq!(parse_macos_version("26.2"), (26, 2));
        assert_eq!(parse_macos_version("27.0"), (27, 0));
        assert_eq!(parse_macos_version(""), (0, 0));
    }

    #[test]
    fn neural_accelerator_flag() {
        assert!(!AppleSiliconGen::M4.has_neural_accelerator());
        assert!(AppleSiliconGen::M5.has_neural_accelerator());
        assert!(!AppleSiliconGen::Unknown.has_neural_accelerator());
    }

    #[test]
    fn wide_quantized_projection_flag_splits_at_generation_15() {
        // Generation 13 runs a verify block as narrow per-position passes.
        assert!(!AppleSiliconGen::M1.wide_quantized_projections());
        assert!(!AppleSiliconGen::M2.wide_quantized_projections());
        // Generation 15+ takes the one wide pass at `M >= 2`.
        assert!(AppleSiliconGen::M3.wide_quantized_projections());
        assert!(AppleSiliconGen::M4.wide_quantized_projections());
        assert!(AppleSiliconGen::M5.wide_quantized_projections());
        // Non-Apple and not-yet-enumerated Apple parts decline rather than
        // assume; on CUDA a K-wide verify does not amortize at all (#638).
        assert!(!AppleSiliconGen::Unknown.wide_quantized_projections());
    }

    #[test]
    fn wide_quantized_projections_is_weaker_than_neural_accelerator() {
        // The B=1 MTP gate moved from `has_neural_accelerator` to this
        // predicate (issue #1217). That is only safe if it is strictly more
        // permissive: no generation may lose a capability the old gate
        // granted, so every NA generation must also read wide here. This is
        // the property that lets the gate change on M3 Ultra evidence alone
        // without touching what M1/M2 hosts do.
        for chip in [
            AppleSiliconGen::M1,
            AppleSiliconGen::M2,
            AppleSiliconGen::M3,
            AppleSiliconGen::M4,
            AppleSiliconGen::M5,
            AppleSiliconGen::Unknown,
        ] {
            assert!(
                !chip.has_neural_accelerator() || chip.wide_quantized_projections(),
                "{chip} has a Neural Accelerator but does not read as wide-projection",
            );
        }
    }

    #[test]
    fn ops_per_buffer_default_gates_on_hardware_class() {
        // Pre-M5 Apple Silicon (Metal family 3) raises the command-buffer op cap
        // to close dispatch-gap idle (#329/#345). M5+ is flat (#358) and an
        // earlier study saw it slower, so it stays on MLX's default; non-Apple
        // (Unknown) is untouched.
        assert_eq!(
            metal_ops_per_buffer_default(AppleSiliconGen::M1, false),
            Some(1000)
        );
        assert_eq!(
            metal_ops_per_buffer_default(AppleSiliconGen::M2, false),
            Some(1000)
        );
        assert_eq!(
            metal_ops_per_buffer_default(AppleSiliconGen::M3, false),
            Some(1000)
        );
        assert_eq!(
            metal_ops_per_buffer_default(AppleSiliconGen::M4, false),
            Some(1000)
        );
        assert_eq!(
            metal_ops_per_buffer_default(AppleSiliconGen::M5, true),
            None
        );
        assert_eq!(
            metal_ops_per_buffer_default(AppleSiliconGen::Unknown, false),
            None
        );
    }

    #[test]
    fn cuda_sdpa_cache_default_matches_build_feature() {
        #[cfg(feature = "cuda")]
        assert_eq!(cuda_sdpa_cache_default(), Some(2000));
        #[cfg(not(feature = "cuda"))]
        assert_eq!(cuda_sdpa_cache_default(), None);
    }

    #[test]
    fn cuda_graph_cache_default_matches_build_feature() {
        // The CUDA graph-cache default is a compile-time gate: a `cuda`-feature
        // build raises the LRU capacity to 2000 so long-lived, shape-diverse
        // decode does not hit MLX's fatal "Cache thrashing" throw at the default
        // capacity 400 (issue #818); every other build leaves MLX's default in
        // place and the CUDA-only variable is never touched.
        #[cfg(feature = "cuda")]
        assert_eq!(cuda_graph_cache_default(), Some(2000));
        #[cfg(not(feature = "cuda"))]
        assert_eq!(cuda_graph_cache_default(), None);
    }

    #[test]
    fn metal_version_by_gen() {
        assert_eq!(AppleSiliconGen::M1.metal_version(), 3);
        assert_eq!(AppleSiliconGen::M4.metal_version(), 3);
        assert_eq!(AppleSiliconGen::M5.metal_version(), 4);
    }

    #[test]
    fn macos_na_threshold() {
        // Boundary conditions for "macOS 26.2+" check.
        assert!(!{
            let (maj, min) = parse_macos_version("26.1");
            (maj > 26) || (maj == 26 && min >= 2)
        });
        assert!({
            let (maj, min) = parse_macos_version("26.2");
            (maj > 26) || (maj == 26 && min >= 2)
        });
        assert!({
            let (maj, min) = parse_macos_version("27.0");
            (maj > 26) || (maj == 26 && min >= 2)
        });
    }

    #[test]
    fn detect_hardware_does_not_panic() {
        // Just verify detection runs without panicking on the current machine.
        let caps = detect_hardware();
        // We cannot assert specific values (depends on the test runner's hardware),
        // but the enum must be one of the valid variants.
        let _ = format!("{}", caps.silicon_gen);
        let _ = caps.has_neural_accelerator;
    }

    #[test]
    fn get_hardware_returns_same_instance() {
        let a = get_hardware();
        let b = get_hardware();
        // Pointer equality — same cached allocation.
        assert!(std::ptr::eq(a, b));
    }

    // ── recommend_quantization tests ──────────────────────────────────────────

    fn make_hw(has_na: bool, macos_supports_na: bool, memory_gb: u32) -> HardwareCapabilities {
        HardwareCapabilities {
            silicon_gen: if has_na {
                AppleSiliconGen::M5
            } else {
                AppleSiliconGen::M4
            },
            gpu_core_count: 10,
            has_neural_accelerator: has_na,
            metal_version: if has_na { 4 } else { 3 },
            macos_supports_na,
            memory_bandwidth_gbps: 150.0,
            unified_memory_gb: memory_gb,
            ..Default::default()
        }
    }

    #[test]
    fn recommends_int8_on_m5_with_sufficient_memory() {
        // 7B model needs ~7 GB for INT8 + 2 GB headroom = 9 GB.
        // 32 GB memory gives ample headroom → INT8.
        let hw = make_hw(true, true, 32);
        let rec = recommend_quantization(7.0, 32, &hw, None);
        assert_eq!(
            rec,
            QuantRecommendation::Int8 {
                reason: "M5 NA delivers 2x throughput for INT8 vs FP16",
            }
        );
        assert_eq!(rec.label(), "int8");
    }

    #[test]
    fn recommends_int4_on_m5_when_8bit_too_large() {
        // 70B model: INT8 needs 70 + 2 = 72 GB, exceeds 64 GB.
        // INT4 needs 35 + 2 = 37 GB — fits in 64 GB → INT4.
        let hw = make_hw(true, true, 64);
        let rec = recommend_quantization(70.0, 64, &hw, None);
        assert_eq!(
            rec,
            QuantRecommendation::Int4Affine {
                reason: "M5 NA available but 8-bit exceeds memory; 4-bit recommended",
            }
        );
    }

    #[test]
    fn recommends_fp16_on_m4_with_small_model() {
        // 1B model: FP16 needs 2 + 2 = 4 GB, fits in 24 GB → FP16.
        let hw = make_hw(false, false, 24);
        let rec = recommend_quantization(1.0, 24, &hw, None);
        assert_eq!(
            rec,
            QuantRecommendation::Fp16 {
                reason: "Model fits in memory as FP16; no quantization needed",
            }
        );
        assert_eq!(rec.label(), "fp16");
    }

    #[test]
    fn recommends_int4_on_m4_with_large_model() {
        // 8B model: FP16 needs 16 + 2 = 18 GB, exceeds 16 GB.
        // INT4 needs 4 + 2 = 6 GB, fits → INT4.
        let hw = make_hw(false, false, 16);
        let rec = recommend_quantization(8.0, 16, &hw, None);
        assert_eq!(
            rec,
            QuantRecommendation::Int4Affine {
                reason: "Best balance of speed and memory on this hardware",
            }
        );
    }

    #[test]
    fn recommends_int4_on_m5_without_na_os_support() {
        // M5 hardware but macOS < 26.2: NA path skipped, falls through to FP16/INT4.
        let hw = make_hw(true, false, 32);
        let rec = recommend_quantization(7.0, 32, &hw, None);
        // 7B FP16 = 14 + 2 = 16 GB, fits in 32 GB → FP16 (no NA).
        assert_eq!(
            rec,
            QuantRecommendation::Fp16 {
                reason: "Model fits in memory as FP16; no quantization needed",
            }
        );
    }

    #[test]
    fn recommends_int4_on_memory_constrained_m5() {
        // 30B model on 16 GB M5: INT8 = 30 + 2 = 32 GB (too big), INT4 = 15 + 2 = 17 GB (too big).
        let hw = make_hw(true, true, 16);
        let rec = recommend_quantization(30.0, 16, &hw, None);
        assert_eq!(
            rec,
            QuantRecommendation::Int4Affine {
                reason: "Memory constrained; 4-bit required to fit model",
            }
        );
        assert_eq!(rec.label(), "int4");
    }

    #[test]
    fn reason_accessor_works() {
        let rec = QuantRecommendation::Int8 {
            reason: "test reason",
        };
        assert_eq!(rec.reason(), "test reason");
    }

    // ── kv_cache_bytes tests ──────────────────────────────────────────────────

    #[test]
    fn kv_cache_dense_mha() {
        // Dense MHA: 32 layers, 32 kv_heads, 128 head_dim, FP16, 8K ctx, batch 1.
        // ctx_len 8192 is already a multiple of 256 — no rounding needed.
        // 32 × 2 × 32 × 128 × 2 × 8192 × 1 = 4_294_967_296 bytes (4 GiB)
        let bytes = kv_cache_bytes(32, 32, 128, 2, 8192, 1);
        assert_eq!(bytes, 4_294_967_296);
    }

    #[test]
    fn kv_cache_gqa_fewer_kv_heads() {
        // GQA: 32 layers, 8 kv_heads (e.g. Llama-3 8B), 128 head_dim, FP16, 8K ctx, batch 1.
        // 32 × 2 × 8 × 128 × 2 × 8192 × 1 = 1_073_741_824 bytes (1 GiB)
        let bytes = kv_cache_bytes(32, 8, 128, 2, 8192, 1);
        assert_eq!(bytes, 1_073_741_824);
    }

    #[test]
    fn kv_cache_long_context_128k() {
        // 32 layers, 8 kv_heads, 128 head_dim, FP16, 128K ctx, batch 1.
        // ctx_len 131072 is a multiple of 256 — no rounding.
        // 32 × 2 × 8 × 128 × 2 × 131_072 × 1 = 17_179_869_184 bytes (16 GiB)
        let bytes = kv_cache_bytes(32, 8, 128, 2, 131_072, 1);
        assert_eq!(bytes, 17_179_869_184);
    }

    #[test]
    fn kv_cache_int8_half_memory() {
        // INT8 KV (elem_bytes = 1) should be exactly half of FP16 (elem_bytes = 2).
        let fp16 = kv_cache_bytes(32, 8, 128, 2, 8192, 1);
        let int8 = kv_cache_bytes(32, 8, 128, 1, 8192, 1);
        assert_eq!(int8 * 2, fp16);
    }

    #[test]
    fn kv_cache_256_token_rounding() {
        // ctx_len not a multiple of 256 must be rounded up.
        // ctx_len = 257 → rounded = 512.
        let bytes_257 = kv_cache_bytes(1, 1, 1, 1, 257, 1);
        let bytes_512 = kv_cache_bytes(1, 1, 1, 1, 512, 1);
        // rounded_ctx for 257 should be 512 → same result as passing 512 directly.
        assert_eq!(bytes_257, bytes_512);

        // ctx_len = 256 → no rounding (already aligned).
        let bytes_256 = kv_cache_bytes(1, 1, 1, 1, 256, 1);
        assert_eq!(bytes_256, 512_u64); // num_layers=1 * K+V=2 * kv_heads=1 * head_dim=1 * elem=1 * 256 * batch=1

        // ctx_len = 255 → rounds up to 256.
        let bytes_255 = kv_cache_bytes(1, 1, 1, 1, 255, 1);
        assert_eq!(bytes_255, bytes_256);

        // ctx_len = 1 → rounds up to 256.
        let bytes_1 = kv_cache_bytes(1, 1, 1, 1, 1, 1);
        assert_eq!(bytes_1, bytes_256);
    }

    #[test]
    fn kv_cache_from_params_fp16() {
        let params = KvCacheParams {
            num_layers: 32,
            num_kv_heads: 8,
            head_dim: 128,
            int8_kv: false,
            ctx_len: 8192,
            batch: 1,
        };
        // elem_bytes = 2 for FP16.
        assert_eq!(
            kv_cache_bytes_from_params(&params),
            kv_cache_bytes(32, 8, 128, 2, 8192, 1)
        );
    }

    #[test]
    fn kv_cache_from_params_int8() {
        let params = KvCacheParams {
            num_layers: 32,
            num_kv_heads: 8,
            head_dim: 128,
            int8_kv: true,
            ctx_len: 8192,
            batch: 1,
        };
        // elem_bytes = 1 for INT8 → half of FP16.
        let expected = kv_cache_bytes(32, 8, 128, 1, 8192, 1);
        assert_eq!(kv_cache_bytes_from_params(&params), expected);
    }

    #[test]
    fn kv_cache_new_constructor() {
        // KvCacheParams::new sets int8_kv=false, batch=1.
        let params = KvCacheParams::new(32, 8, 128, 8192);
        assert!(!params.int8_kv);
        assert_eq!(params.batch, 1);
        assert_eq!(
            kv_cache_bytes_from_params(&params),
            kv_cache_bytes(32, 8, 128, 2, 8192, 1)
        );
    }

    // ── recommend_quantization with computed KV headroom ─────────────────────

    #[test]
    fn recommend_quant_uses_computed_kv_headroom() {
        // A 7B FP16 model needs 14 GB for weights.
        // With a 2 GB flat headroom (None), it fits in 16 GB: 14 + 2 = 16 GB.
        // With a 4 GB computed headroom, it does NOT fit in 16 GB: 14 + 4 = 18 GB → INT4.
        let hw = make_hw(false, false, 16);

        // Flat headroom (None = 2 GB) → FP16 fits.
        let rec_flat = recommend_quantization(7.0, 16, &hw, None);
        assert_eq!(
            rec_flat,
            QuantRecommendation::Fp16 {
                reason: "Model fits in memory as FP16; no quantization needed",
            }
        );

        // Computed headroom: 4 GiB (passes as bytes) → FP16 no longer fits → INT4.
        let kv_headroom_bytes: u64 = 4 * 1_073_741_824; // 4 GiB
        let rec_computed = recommend_quantization(7.0, 16, &hw, Some(kv_headroom_bytes));
        assert_eq!(
            rec_computed,
            QuantRecommendation::Int4Affine {
                reason: "Best balance of speed and memory on this hardware",
            }
        );
    }

    #[test]
    fn recommend_quant_long_context_tightens_headroom() {
        // 8B model (FP16 = 16 GB) on 24 GB M4.
        // Short context (8K): KV headroom = 1 GiB → FP16 total = 17 GB, fits.
        // Long context (128K): KV headroom = 16 GiB → FP16 total = 32 GB, does NOT fit
        //   → decision flips from FP16 to INT4.
        let hw = make_hw(false, false, 24);

        // 8K ctx, 32 layers, 8 kv_heads, 128 head_dim, FP16 KV.
        // kv_cache_bytes(32, 8, 128, 2, 8192, 1) = 1_073_741_824 bytes = 1 GiB.
        // headroom_gb = ceil(1 GiB / 1 GiB) = 1 GB.
        // mem_fp16_gb = ceil(8 * 2) + 1 = 17 GB ≤ 24 GB → FP16.
        let kv_8k = kv_cache_bytes(32, 8, 128, 2, 8_192, 1);
        let rec_8k = recommend_quantization(8.0, 24, &hw, Some(kv_8k));
        assert_eq!(
            rec_8k,
            QuantRecommendation::Fp16 {
                reason: "Model fits in memory as FP16; no quantization needed",
            }
        );

        // 128K ctx — KV headroom balloons to 16 GiB.
        // kv_cache_bytes(32, 8, 128, 2, 131_072, 1) = 17_179_869_184 bytes = 16 GiB.
        // headroom_gb = ceil(16 GiB / 1 GiB) = 16 GB.
        // mem_fp16_gb = 16 + 16 = 32 GB > 24 GB → can't use FP16.
        // mem_4bit_gb = ceil(8 * 0.5) + 16 = 20 GB ≤ 24 GB → INT4.
        let kv_128k = kv_cache_bytes(32, 8, 128, 2, 131_072, 1);
        let rec_128k = recommend_quantization(8.0, 24, &hw, Some(kv_128k));
        assert!(
            matches!(rec_128k, QuantRecommendation::Int4Affine { .. }),
            "Expected INT4 for 128K context on tight memory, got: {:?}",
            rec_128k
        );
    }

    /// No vendor other than Apple may make `is_apple_silicon()` true.
    ///
    /// That predicate gates a bf16 to f16 weight conversion at eleven call
    /// sites (the VLM loaders, `models/sanitize.rs`,
    /// `drafter/dflash/drafter.rs`). If a vendor ever answers true here, those
    /// sites start converting weights on hardware where bf16 is native, the
    /// output changes, and nothing else fails: the code compiles, and the gate
    /// stays green wherever that vendor's checkpoints are absent. Hence a test
    /// rather than a comment.
    #[test]
    fn gpu_vendor_does_not_imply_apple_silicon() {
        for vendor in [GpuVendor::Nvidia, GpuVendor::Amd, GpuVendor::Unknown] {
            let hw = HardwareCapabilities {
                vendor,
                ..Default::default()
            };
            assert!(
                !hw.is_apple_silicon(),
                "{vendor:?} must not be reported as Apple Silicon; \
                 is_apple_silicon() gates the bf16 to f16 weight conversion"
            );
        }
        // The Apple arm is decided by `silicon_gen`, not by the vendor tag, so
        // an Apple tag with no generation is still not Apple Silicon.
        let tagged_but_ungenerationed = HardwareCapabilities {
            vendor: GpuVendor::Apple,
            ..Default::default()
        };
        assert!(!tagged_but_ungenerationed.is_apple_silicon());
        let real_apple = HardwareCapabilities {
            vendor: GpuVendor::Apple,
            silicon_gen: AppleSiliconGen::M1,
            ..Default::default()
        };
        assert!(real_apple.is_apple_silicon());
    }

    #[test]
    fn the_detected_vendor_agrees_with_the_resolved_backend() {
        // The vendor is read off the backend MLX resolved, not guessed from
        // which `device_info()` keys are present. Those two can disagree: the
        // ROCm backend publishes `compute_capability_major`/`minor`, so key
        // presence alone says "AMD is CUDA" (issue #1805).
        let hw = get_hardware();
        let expected = match gpu_backend_kind() {
            GpuBackendKind::Metal => GpuVendor::Apple,
            GpuBackendKind::Cuda => GpuVendor::Nvidia,
            GpuBackendKind::Rocm => GpuVendor::Amd,
            GpuBackendKind::None => GpuVendor::Unknown,
        };
        assert_eq!(
            hw.vendor,
            expected,
            "backend {:?} must report vendor {expected:?}",
            gpu_backend_kind()
        );
    }

    #[test]
    fn the_architecture_string_is_never_empty_and_never_a_gfx_claim() {
        // Every backend publishes an `architecture`, each in its own
        // vocabulary: `gfx1151`, `sm_89`, an Apple GPU family string. The
        // field carries whichever one this backend speaks, and "" is never a
        // value, only `None`.
        let hw = get_hardware();
        if let Some(arch) = hw.device_architecture.as_deref() {
            assert!(!arch.is_empty(), "an empty architecture must be None");
        }
        // Only ROCm has a gfx target. A CUDA host reporting `sm_89` here must
        // not make `device_gfx_target()` answer, which is the misreading that
        // sent AMD down the CUDA path in the first place.
        if gpu_backend_kind() != GpuBackendKind::Rocm {
            assert_eq!(
                crate::rocm_arch::device_gfx_target(),
                None,
                "only ROCm has a gfx target, whatever `architecture` says"
            );
        }
    }

    #[test]
    fn unified_memory_stays_apple_only() {
        // `resolve_available_memory` reads this field at step 3. Filling it
        // from device memory off Apple would change what that walk returns on
        // CUDA and ROCm, which #1805 put out of scope; `device_memory_bytes`
        // is the field that carries device memory instead.
        let hw = get_hardware();
        if hw.vendor != GpuVendor::Apple {
            assert_eq!(
                hw.unified_memory_gb, 0,
                "unified_memory_gb must stay 0 off Apple; read device_memory_bytes instead"
            );
        }
    }
}
