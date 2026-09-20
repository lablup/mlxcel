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

//! Hardware-dependent default for the speculative draft block width
//! (issue #1797).
//!
//! The flat per-kind default in [`super::speculative_args`] resolves with no
//! device or quantization input, so on GB10 a Qwen 3.5 DFlash pairing served
//! at the effective default of 16 and lost 17% against classic decode, while
//! the same pairing at width 4 won 32%
//! (`docs/benchmark_results/dflash-verify-fixed-cost-gb10-2026-09-11.md`).
//! The throughput was there, it was just unreachable without the operator
//! passing `--draft-block-size` by hand.
//!
//! This module owns the narrow fix: a **default only**, keyed on the pair
//! (running compute capability, target quantization path) that actually
//! selects the CUDA kernel the verify block runs on. It never overrides
//! anything. `--draft-block-size`, `MLXCEL_DRAFT_BLOCK_SIZE` and
//! `LLAMA_ARG_DRAFT_BLOCK_SIZE` all resolve ahead of it, and so does a
//! drafter checkpoint that declares its own width (DSpark, Muse Glimmer).
//!
//! ## Why the width is a function of the quantization mode
//!
//! Two kernel boundaries in the pinned MLX tree, at different row counts,
//! and only one of them is on every path.
//!
//! 1. **Accumulator width.** `dispatch_multirow_width`
//!    (`src/lib/mlx-cpp/patches/mlx/backend/cuda/quantized/qmm/qmv.cu`)
//!    instantiates exactly three compile-time widths, 2, 4 and `Cap = 8`, so
//!    a 5, 6 or 7 row verify takes the 8-wide instantiation and pays its
//!    register cost without using the extra rows. This boundary exists only
//!    on the affine `qmv` path: `fp_qmv.cu` has no multirow accumulator
//!    dispatch at all, and its `rows_per_block = 8` tiles output rows rather
//!    than input rows.
//! 2. **Kernel family switch.** `if (can_use_qmv && (M * B < 8))`
//!    (`quantized.cpp`) hands 8 rows and up to `qmm_sm80`, which costs about
//!    3.5x the single-row `qmv` and is flat from 8 to 16. This boundary is on
//!    both paths.
//!
//! An NVFP4 target routes through `supports_fp_qmv` to `fp_qmv` rather than
//! `qmv`, so it sees boundary 2 without boundary 1 and can have a genuinely
//! different optimum. That is why this is a table rather than a constant, and
//! why an entry is only ever added from a committed measurement.
//!
//! ## Re-measurement condition
//!
//! Every entry in [`measured_default_block_size`] was measured against MLX
//! pin `81ba1c6a` (`src/lib/mlx-cpp/CMakeLists.txt`). If a later pin adds a
//! multirow instantiation between 4 and 8, widens `supports_fp_qmv`, or moves
//! the `M * B < 8` family switch, boundary 1 or 2 moves and the seeded widths
//! have to be re-measured against the new pin. This mirrors the way
//! `mlxcel_core::hardware::cuda_graph_cache_default` records the condition
//! under which its own workaround can be withdrawn.
//!
//! Used by: [`super::speculative_args::resolve_draft_block_size`].

use std::path::Path;

use mlxcel_core::drafter::DrafterKind;

/// The target checkpoint's quantization path, at the granularity that
/// decides which CUDA quantized-matmul kernel family a verify block runs on.
///
/// This is deliberately coarser than
/// [`mlxcel_core::layers::SUPPORTED_QUANTIZATION_MODES`]: the only
/// distinction the block-width policy can act on is whether the target
/// reaches `fp_qmv` (NVFP4) or `qmv` / `qmm_sm80` (affine), because those two
/// have different row boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum TargetQuantization {
    /// MLX affine quantization: a `.scales` plus `.biases` plane, declared as
    /// `mode: "affine"` or left implicit next to `bits` and `group_size`.
    /// Routes to `qmv` (below 8 rows) and `qmm_sm80` (8 rows and up).
    Affine,
    /// NVFP4, either MLX-native (`mode: "nvfp4"`) or a ModelOpt /
    /// compressed-tensors `nvfp4-pack-quantized` export. On compute
    /// capability 10 and later `supports_fp_qmv` accepts it and `call_qmv`
    /// routes to `fp_qmv`, which has no multirow accumulator dispatch.
    Nvfp4,
    /// A checkpoint mlxcel can read whose quantization is neither of the
    /// above: `mxfp4`, `mxfp8`, block-scaled fp8, or no quantization at all.
    /// Distinct from [`Self::Unknown`] so a future entry can be seeded for
    /// one of these without re-deriving why it was skipped.
    Other,
    /// The target's `config.json` was missing, unreadable, or not JSON. The
    /// policy has nothing to key on and the caller takes the flat fallback.
    #[default]
    Unknown,
}

/// Where a resolved draft block width came from.
///
/// Carried alongside the width so the worker's startup log can say whether
/// an operator override, a drafter checkpoint, a measured hardware default,
/// or the flat per-kind constant produced it. Without this an operator
/// reading `block_size = 4` cannot tell a measured default from a flag they
/// forgot they had exported.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum BlockSizeSource {
    /// `--draft-block-size`, `MLXCEL_DRAFT_BLOCK_SIZE` or
    /// `LLAMA_ARG_DRAFT_BLOCK_SIZE`. Always wins.
    Override,
    /// The drafter checkpoint declared its own width; the `&'static str`
    /// names which peek matched.
    DrafterCheckpoint(&'static str),
    /// A measured per-device, per-quantization default from
    /// [`measured_default_block_size`].
    MeasuredHardwareDefault,
    /// The flat per-kind constant, which is what every unmeasured host and
    /// quantization path takes.
    KindDefault,
}

impl BlockSizeSource {
    /// One-line reason for the startup log.
    #[must_use]
    pub fn reason(self) -> &'static str {
        match self {
            Self::Override => "operator override",
            Self::DrafterCheckpoint(which) => which,
            Self::MeasuredHardwareDefault => "measured default for this device and quantization",
            Self::KindDefault => "per-kind default (no measurement for this host)",
        }
    }
}

/// A resolved draft block width together with what produced it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolvedBlockSize {
    /// The effective `--draft-block-size` in tokens.
    pub width: u32,
    /// What produced [`Self::width`].
    pub source: BlockSizeSource,
}

impl ResolvedBlockSize {
    /// Log-ready summary, e.g. `4 (measured default for this device and
    /// quantization)`.
    #[must_use]
    pub fn describe(&self) -> String {
        format!("{} ({})", self.width, self.source.reason())
    }
}

/// The measured draft block width for a (drafter kind, device, target
/// quantization) combination, or `None` when nothing has been measured.
///
/// Pure: every input is passed in, nothing is read from the environment or
/// the device here, so every arm is unit-testable without CUDA. `None` is
/// not an error; it is the normal answer on every host and quantization path
/// this project has not swept, and the caller then takes the unchanged flat
/// per-kind default.
///
/// The table:
///
/// | drafter kind | compute capability | target quantization | width | record |
/// |---|---|---|---|---|
/// | DFlash | 12.1 (GB10) | affine | 4 | `draft-block-width-default-gb10-2026-09-20.md` |
///
/// Widths below 2 are unreachable by construction: the DFlash round loop
/// breaks out at `bs <= 1` (`round_loop.rs`), which would silently disable
/// speculation rather than narrow it, so no entry may be 0 or 1. The
/// `measured_defaults_never_disable_speculation` test enforces that over the
/// whole table rather than trusting review.
///
/// Used by: [`resolve_measured_block_size`].
#[must_use]
pub fn measured_default_block_size(
    kind: DrafterKind,
    cuda_compute_capability: Option<(u32, u32)>,
    target_quantization: TargetQuantization,
) -> Option<u32> {
    // The MTP default is a separate constant with its own upstream provenance
    // and was not swept here, so MTP and the auto-detected InternalMtp keep
    // the flat per-kind value.
    if kind != DrafterKind::Dflash {
        return None;
    }
    let (major, minor) = cuda_compute_capability?;
    match (major, minor, target_quantization) {
        // GB10 (sm_121, DGX Spark), affine target: width 4 beat both the
        // classic arm and the flat 16 default with disjoint n = 3 ranges.
        // NVFP4 on the same host was measured in the same session and has no
        // entry; see the record for what it found.
        (12, 1, TargetQuantization::Affine) => Some(4),
        _ => None,
    }
}

/// Apply [`measured_default_block_size`], returning a [`ResolvedBlockSize`]
/// that records whether the measured table or the flat fallback answered.
///
/// `fallback` is [`super::speculative_args::default_block_size_for_kind`],
/// passed in rather than called here so this module stays free of the flag
/// group it serves.
#[must_use]
pub fn resolve_measured_block_size(
    kind: DrafterKind,
    cuda_compute_capability: Option<(u32, u32)>,
    target_quantization: TargetQuantization,
    fallback: u32,
) -> ResolvedBlockSize {
    match measured_default_block_size(kind, cuda_compute_capability, target_quantization) {
        Some(width) => ResolvedBlockSize {
            width,
            source: BlockSizeSource::MeasuredHardwareDefault,
        },
        None => ResolvedBlockSize {
            width: fallback,
            source: BlockSizeSource::KindDefault,
        },
    }
}

/// Read the target checkpoint's quantization path from its `config.json`.
///
/// Never fails: anything that cannot be read or understood is
/// [`TargetQuantization::Unknown`], which takes the flat fallback. This runs
/// once at startup, before the target is loaded, so it cannot consult the
/// loaded weights and reads only what the config declares.
///
/// Used by: the server dispatch resolution and the offline `generate` path.
#[must_use]
pub fn peek_target_quantization(model_path: &Path) -> TargetQuantization {
    let Ok(text) = std::fs::read_to_string(model_path.join("config.json")) else {
        return TargetQuantization::Unknown;
    };
    let Ok(config) = serde_json::from_str::<serde_json::Value>(&text) else {
        return TargetQuantization::Unknown;
    };
    classify_quantization(&config)
}

/// Pure core of [`peek_target_quantization`], over an already-parsed
/// `config.json`.
///
/// Three declaration shapes reach this, and a VLM nests any of them under
/// `text_config`:
///
/// - MLX's own: a root `quantization` object carrying `bits` and
///   `group_size`, with `mode` present on the block-float exports and absent
///   (meaning affine) on the classic ones.
/// - ModelOpt / compressed-tensors: a root `quantization_config` whose
///   `config_groups.*.format` is `nvfp4-pack-quantized` or
///   `mxfp4-pack-quantized`, with no root `quantization` at all. This is the
///   shape the Laguna XS 2.1 NVFP4 release ships.
/// - Neither, which is an unquantized checkpoint and reports
///   [`TargetQuantization::Other`] rather than `Unknown`: the config was read
///   and understood, there is simply no quantization in it.
#[must_use]
pub fn classify_quantization(config: &serde_json::Value) -> TargetQuantization {
    for scope in [Some(config), config.get("text_config")].into_iter().flatten() {
        if let Some(found) = classify_one_scope(scope) {
            return found;
        }
    }
    // A readable config with no quantization declaration anywhere is an
    // unquantized checkpoint: understood, just not a path this policy has an
    // entry for.
    TargetQuantization::Other
}

/// Classify one config scope (the root, or a `text_config` sub-object),
/// returning `None` when that scope declares no quantization at all so the
/// caller can try the next one.
fn classify_one_scope(scope: &serde_json::Value) -> Option<TargetQuantization> {
    if let Some(quantization) = scope.get("quantization").filter(|v| v.is_object()) {
        return Some(match quantization.get("mode").and_then(|m| m.as_str()) {
            Some(mode) => mode_name_to_quantization(mode),
            // MLX's classic quantized exports omit `mode` entirely and are
            // affine; `infer_quantization_mode` reaches the same conclusion
            // from the weights, which are not loaded yet here.
            None => TargetQuantization::Affine,
        });
    }
    let quantization_config = scope.get("quantization_config").filter(|v| v.is_object())?;
    let format = quantization_config
        .get("config_groups")
        .and_then(|groups| groups.as_object())
        .and_then(|groups| {
            groups
                .values()
                .find_map(|group| group.get("format").and_then(|f| f.as_str()))
        });
    Some(match format {
        Some(format) => mode_name_to_quantization(format),
        // A `quantization_config` with no per-group format is a vendor fp8 or
        // bitsandbytes style declaration; readable, but not one of the two
        // kernel paths this policy distinguishes.
        None => TargetQuantization::Other,
    })
}

/// Map a declared mode or compressed-tensors format string onto the coarse
/// kernel-path classification. Matched by substring so
/// `nvfp4-pack-quantized` and a bare `nvfp4` land together, and
/// case-insensitively because at least one vendor export capitalizes it.
fn mode_name_to_quantization(name: &str) -> TargetQuantization {
    let name = name.to_ascii_lowercase();
    if name.contains("nvfp4") {
        TargetQuantization::Nvfp4
    } else if name.contains("affine") {
        TargetQuantization::Affine
    } else {
        TargetQuantization::Other
    }
}

#[cfg(test)]
#[path = "draft_block_policy_tests.rs"]
mod tests;
