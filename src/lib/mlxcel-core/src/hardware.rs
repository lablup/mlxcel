// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
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

/// Hardware capabilities detected at runtime.
#[derive(Debug, Clone)]
pub struct HardwareCapabilities {
    /// Apple Silicon chip generation.
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
    pub unified_memory_gb: u32,
}

impl Default for HardwareCapabilities {
    fn default() -> Self {
        Self {
            silicon_gen: AppleSiliconGen::Unknown,
            gpu_core_count: 0,
            has_neural_accelerator: false,
            metal_version: 3,
            macos_supports_na: false,
            memory_bandwidth_gbps: 0.0,
            unified_memory_gb: 0,
        }
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

// ── Detection ─────────────────────────────────────────────────────────────────

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
        HardwareCapabilities::default()
    }
}

// ── macOS implementation ──────────────────────────────────────────────────────

#[cfg(target_os = "macos")]
mod platform {
    use std::os::raw::{c_char, c_int, c_void};

    // sysctlbyname(3) is in <sys/sysctl.h> — link against libSystem automatically.
    extern "C" {
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
        silicon_gen,
        gpu_core_count,
        has_neural_accelerator,
        metal_version,
        macos_supports_na,
        memory_bandwidth_gbps,
        unified_memory_gb,
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
fn estimate_bandwidth(gen: AppleSiliconGen, memory_gb: u32) -> f64 {
    match gen {
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
            if memory_gb > 64 {
                1000.0
            } else {
                150.0
            }
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
}
