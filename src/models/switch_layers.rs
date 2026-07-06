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

//! Shared SwitchLinear / SwitchGLU for MoE models
//!
//! Used by: KimiLinear, LongcatFlashNgram, DeepSeekV3, DeepSeekV32, GLM4Moe,
//!          GLM4MoeLite, ExaOneMoe, Mixtral, Qwen2Moe, Qwen3Moe, PhiMoE, OLMoE, etc.
//!
//! SwitchLinear: per-expert 3D matmul (quantized via gather_qmm, regular via gather_mm)
//! SwitchGLU: SwiGLU MLP routing through SwitchLinear
//! group_mask_scores: group-based expert masking for MoE gates with n_group > 1

use mlxcel_core::utils::slice_axis;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

/// Whether the fused single-token decode-MoE kernel (#268) is enabled.
///
/// Default-on as of #282: across the validated MoE set the kernel is
/// byte-identical or within the documented f16 jitter class, never regresses
/// decode, and gives a measured speedup on both M1 Ultra and M5 (Neural
/// Accelerator) hardware. Set `MLXCEL_FUSED_MOE=0` (also `false`/`off`/`no`,
/// case-insensitive) to force the proven `gather_qmm` / `SwitchGLU` path; any
/// other value, or leaving it unset, keeps the kernel on.
///
/// This only chooses whether to *attempt* the kernel. Callers still fall back to
/// `gather_qmm` automatically for any config the kernel does not support
/// (non-affine, unsupported bit widths, mismatched gate/up bits, prefill).
///
/// The variable is read once and cached for the process lifetime.
pub fn fused_moe_enabled() -> bool {
    // Cache the launch-time flag, mirroring the `OnceLock` env-gate convention
    // used by other hot-path flags (e.g. gemma4's `mtp_divergent_fix_disabled`).
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED
        .get_or_init(|| fused_moe_enabled_from(std::env::var("MLXCEL_FUSED_MOE").ok().as_deref()))
}

/// Pure decision behind [`fused_moe_enabled`], split out so it can be unit-tested
/// without mutating process-global environment state. `None` (unset) is on; an
/// explicit `0`/`false`/`off`/`no` (case-insensitive, trimmed) is off; any other
/// value is on.
pub(crate) fn fused_moe_enabled_from(value: Option<&str>) -> bool {
    match value {
        Some(v) => !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "off" | "no"
        ),
        None => true,
    }
}

/// Metal default for the fused decode-MoE expert-intermediate (Dff) upper bound.
///
/// The fused two-kernel path wins only while `gather_qmm` underutilizes the GPU
/// (small experts); above this bound the caller falls back to `gather_qmm`.
/// Tuned on M1 Ultra, where phi-3.5-moe (Dff 6400) already regresses. See
/// `docs/benchmark_results/fused-moe-decode-kernel-design.md`.
const FUSED_MOE_MAX_DFF_METAL: i32 = 4096;

/// CUDA default for the same bound. At batch=1 `gather_qmm` leaves far more of
/// the GPU idle than on Metal, so the fused path keeps winning to a larger
/// expert size. Re-measured on GB10 (DGX Spark, sm_121) under MLX pin e9463bb
/// (#626, after the #625 pin bump moved `gather_gemm` to JIT and sped the
/// fallback): fused wins through Dff 6400 (phi-3.5-moe +5%, lfm2 +12%), is
/// break-even at 8192 (llama-4-scout, within run-to-run noise) and loses at
/// 14336 (mixtral -1.2%), so the crossover interpolates to ~8000. Set to the
/// 8192 break-even boundary, which stays well below the 14336 regression while
/// capturing the phi-3.5-moe / llama-4-scout mid-size experts that the old 4096
/// default silently declined. The env var overrides this on both backends.
const FUSED_MOE_MAX_DFF_CUDA: i32 = 8192;

/// Resolve the fused-MoE Dff upper bound. An explicit positive
/// `MLXCEL_FUSED_MOE_MAX_DFF` wins on both backends; otherwise the default is
/// backend-specific (Metal keeps the conservative 4096, CUDA uses the higher
/// measured crossover). Split out as a pure function so the backend-aware
/// default is unit-testable without mutating process-global env or querying the
/// live device; `metal_available` mirrors `mlx::core::metal::is_available()`.
pub(crate) fn fused_moe_max_dff_from(env: Option<&str>, metal_available: bool) -> i32 {
    env.and_then(|s| s.parse::<i32>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(if metal_available {
            FUSED_MOE_MAX_DFF_METAL
        } else {
            FUSED_MOE_MAX_DFF_CUDA
        })
}

/// Per-expert 3D linear layer (falls back to gather_mm for non-quantized models)
/// Supports affine, mxfp4, nvfp4, and mxfp8 quantization modes.
pub enum SwitchLinear {
    /// Quantized path: uses gather_qmm
    Quantized {
        weight: UniquePtr<MlxArray>,
        scales: UniquePtr<MlxArray>,
        biases: Option<UniquePtr<MlxArray>>,
        group_size: i32,
        bits: i32,
        mode: String,
    },
    /// Non-quantized path: uses gather_mm
    Regular { weight: UniquePtr<MlxArray> },
}

impl SwitchLinear {
    pub fn forward(&self, x: &MlxArray, indices: &MlxArray, sorted: bool) -> UniquePtr<MlxArray> {
        match self {
            Self::Quantized {
                weight,
                scales,
                biases,
                group_size,
                bits,
                mode,
            } => {
                let biases_ptr: *const MlxArray = match biases {
                    Some(b) => b.as_ref().unwrap() as *const MlxArray,
                    None => std::ptr::null(),
                };
                unsafe {
                    mlxcel_core::gather_qmm(
                        x,
                        weight,
                        scales,
                        biases_ptr,
                        std::ptr::null(),
                        indices as *const _,
                        true,
                        *group_size,
                        *bits,
                        sorted,
                        mode,
                    )
                }
            }
            Self::Regular { weight } => {
                // Python: gather_mm(x, weight.swapaxes(-1, -2), rhs_indices=indices)
                let wt = mlxcel_core::swap_axes(weight, -1, -2);
                unsafe {
                    mlxcel_core::gather_mm(x, &wt, std::ptr::null(), indices as *const _, sorted)
                }
            }
        }
    }

    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        Self::from_weights_with_mode(weights, prefix, group_size, bits, "affine")
    }

    /// Expose the quantized triple (weight, scales, biases, group_size,
    /// bits, mode) for callers that want to drive a fused compile path
    /// over several `SwitchLinear`s (e.g. SwitchGeGLU gate/up/down
    /// compiled into one graph). Returns `None` for the non-quantized
    /// `Regular` variant or when biases are absent (the fused compile
    /// currently requires affine biases).
    pub fn quantized_parts(&self) -> Option<QuantizedSwitchLinearRef<'_>> {
        match self {
            Self::Quantized {
                weight,
                scales,
                biases: Some(biases),
                group_size,
                bits,
                mode,
            } => Some(QuantizedSwitchLinearRef {
                weight,
                scales,
                biases,
                group_size: *group_size,
                bits: *bits,
                mode: mode.as_str(),
            }),
            _ => None,
        }
    }
}

/// Borrowed view over a `SwitchLinear::Quantized` with biases present,
/// used to drive fused compile paths over several quantized switch
/// linears without moving the originals.
pub struct QuantizedSwitchLinearRef<'a> {
    pub weight: &'a UniquePtr<MlxArray>,
    pub scales: &'a UniquePtr<MlxArray>,
    pub biases: &'a UniquePtr<MlxArray>,
    pub group_size: i32,
    pub bits: i32,
    pub mode: &'a str,
}

impl SwitchLinear {
    pub fn from_weights_with_mode(
        weights: &WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
        mode: &str,
    ) -> Result<Self, String> {
        // Pre-stacked layout: `{prefix}.weight` (plus optional `.scales`/`.biases`)
        // already holds every expert in one `[num_experts, ...]` tensor.
        if let Some(weight) = weights.get(&format!("{}.weight", prefix)) {
            let weight = mlxcel_core::copy(weight);
            let scales = weights
                .get(&format!("{}.scales", prefix))
                .map(|w| mlxcel_core::copy(w));
            // biases may not exist for mxfp4/nvfp4/mxfp8 modes
            let biases = weights
                .get(&format!("{}.biases", prefix))
                .map(|w| mlxcel_core::copy(w));
            return Ok(Self::from_stacked_parts(
                weight, scales, biases, group_size, bits, mode,
            ));
        }

        // Per-expert layout: `{root}.experts.{idx}.{proj}.{weight,scales,biases}`.
        // Some mlx-lm checkpoints (e.g. Qwen2-MoE / Qwen1.5-MoE) ship the experts
        // unstacked under `experts.{idx}` rather than as a single `switch_mlp`
        // tensor. Stack them here so the gather paths see the same `[num_experts,
        // ...]` layout as the pre-stacked checkpoints. This branch only runs when
        // the pre-stacked tensor is absent, so it never changes behavior for an
        // already-loadable checkpoint.
        if let Some((weight, scales, biases)) = stack_individual_experts(weights, prefix) {
            return Ok(Self::from_stacked_parts(
                weight, scales, biases, group_size, bits, mode,
            ));
        }

        Err(format!("Missing weight: {}", prefix))
    }

    /// Build a `SwitchLinear` from a stacked `[num_experts, ...]` weight (plus
    /// optional scales/biases). Present scales select the quantized path and the
    /// per-tensor bit width is inferred from the packed-weight and scales shapes;
    /// absent scales select the non-quantized `Regular` path.
    fn from_stacked_parts(
        weight: UniquePtr<MlxArray>,
        scales: Option<UniquePtr<MlxArray>>,
        biases: Option<UniquePtr<MlxArray>>,
        group_size: i32,
        bits: i32,
        mode: &str,
    ) -> Self {
        match scales {
            Some(scales) => {
                // Infer the actual bit width from the packed weight and scales
                // shapes (group_size fixed): mixed-precision checkpoints such as
                // dots.llm1 quantize some expert projections at 6-bit while the
                // model default is 4-bit, so the passed `bits` is only the
                // default. The invariant is `packed_in * 32 == bits * num_groups *
                // group_size`.
                let w_shape = mlxcel_core::array_shape(&weight);
                let s_shape = mlxcel_core::array_shape(&scales);
                let packed_in = *w_shape.last().unwrap_or(&0);
                let num_groups = *s_shape.last().unwrap_or(&0);
                let denom = num_groups * group_size;
                let effective_bits = if denom > 0 && (packed_in * 32) % denom == 0 {
                    let inferred = (packed_in * 32) / denom;
                    if (2..=8).contains(&inferred) {
                        inferred
                    } else {
                        bits
                    }
                } else {
                    bits
                };

                Self::Quantized {
                    weight,
                    scales,
                    biases,
                    group_size,
                    bits: effective_bits,
                    mode: mode.to_string(),
                }
            }
            None => Self::Regular { weight },
        }
    }
}

/// Stack per-expert projection tensors (`{root}.experts.{idx}.{proj}.{weight,
/// scales,biases}`) into single `[num_experts, ...]` tensors, given a
/// stacked-style `prefix` of the form `{root}.switch_mlp.{proj}`. Returns `None`
/// when the prefix is not in that form or no `experts.0` weight exists, so the
/// caller falls through to its own missing-weight error.
///
/// `scales`/`biases` are stacked only when expert 0 carries them, matching the
/// quantized/regular split the stacked loader applies. Experts are gathered
/// contiguously from index 0 until the first gap.
///
/// The expert leaf name comes from the `{proj}` segment of the prefix, so a
/// caller that passes overridden leaf names (e.g. Mixtral's `w1`/`w2`/`w3` via
/// `SwitchGLU::from_weights_with_proj_names`) loads from the matching expert
/// keys without any name baked in here.
///
/// Used by: Qwen2Moe (Qwen1.5-MoE / Qwen2-MoE individual-expert checkpoints),
///          Mixtral (`block_sparse_moe.experts.{idx}.{w1,w2,w3}` checkpoints)
fn stack_individual_experts(
    weights: &WeightMap,
    prefix: &str,
) -> Option<(
    UniquePtr<MlxArray>,
    Option<UniquePtr<MlxArray>>,
    Option<UniquePtr<MlxArray>>,
)> {
    // prefix: "{root}.switch_mlp.{proj}"  ->  experts at "{root}.experts.{idx}.{proj}"
    let proj = prefix.rsplit('.').next()?;
    let root = prefix.strip_suffix(&format!(".switch_mlp.{}", proj))?;
    let expert_key = |idx: usize, leaf: &str| format!("{}.experts.{}.{}.{}", root, idx, proj, leaf);

    if !weights.contains_key(&expert_key(0, "weight")) {
        return None;
    }
    let has_scales = weights.contains_key(&expert_key(0, "scales"));
    let has_biases = weights.contains_key(&expert_key(0, "biases"));

    let mut stacked_weight = Vec::new();
    let mut stacked_scales = Vec::new();
    let mut stacked_biases = Vec::new();
    let mut idx = 0;
    while let Some(weight) = weights.get(&expert_key(idx, "weight")) {
        stacked_weight.push(mlxcel_core::copy(weight));
        if has_scales {
            stacked_scales.push(mlxcel_core::copy(weights.get(&expert_key(idx, "scales"))?));
        }
        if has_biases {
            stacked_biases.push(mlxcel_core::copy(weights.get(&expert_key(idx, "biases"))?));
        }
        idx += 1;
    }

    let weight = mlxcel_core::stack_owned(&stacked_weight, 0);
    let scales = has_scales.then(|| mlxcel_core::stack_owned(&stacked_scales, 0));
    let biases = has_biases.then(|| mlxcel_core::stack_owned(&stacked_biases, 0));
    Some((weight, scales, biases))
}

pub struct SwitchGLU {
    gate_proj: SwitchLinear,
    up_proj: SwitchLinear,
    down_proj: SwitchLinear,
}

impl SwitchGLU {
    pub fn forward(&self, x: &MlxArray, indices: &MlxArray) -> UniquePtr<MlxArray> {
        let indices_shape = mlxcel_core::array_shape(indices);
        let n_tokens = indices_shape[0];
        let top_k = indices_shape[1];
        let total = n_tokens * top_k;
        let do_sort = total >= 64;

        let x_exp = mlxcel_core::expand_dims(x, -2);
        let x_exp = mlxcel_core::expand_dims(&x_exp, -3);

        if do_sort {
            let (sorted_x, sorted_idx, inv_order) = gather_sort(&x_exp, indices);
            let x_gate = self.gate_proj.forward(&sorted_x, &sorted_idx, true);
            let x_up = self.up_proj.forward(&sorted_x, &sorted_idx, true);
            let activated = mlxcel_core::compiled_swiglu_activation(&x_gate, &x_up);
            let output = self.down_proj.forward(&activated, &sorted_idx, true);
            scatter_unsort(&output, &inv_order, &indices_shape)
        } else {
            let x_gate = self.gate_proj.forward(&x_exp, indices, false);
            let x_up = self.up_proj.forward(&x_exp, indices, false);
            let activated = mlxcel_core::compiled_swiglu_activation(&x_gate, &x_up);
            let output = self.down_proj.forward(&activated, indices, false);
            mlxcel_core::squeeze_axis(&output, -2)
        }
    }

    /// Single-token decode via the fused MoE expert Metal kernel (#268, step 2b).
    ///
    /// Computes `sum_k scores[k] * down_k(silu(gate_k(x)) * up_k(x))` for the K
    /// selected experts as two all-cores Metal dispatches (gate/up+swiglu, then
    /// down+score), beating gather_qmm by ~3.5% on qwen3-30b-a3b with
    /// byte-identical greedy output. gate/up are 4/8-bit; down also handles
    /// 6-bit, so mixed widths like dots.llm1 (gate/up 4-bit, down 6-bit) are
    /// supported. Returns `None` (caller falls back to `forward` +
    /// `moe_weighted_sum`) for any unsupported config: non-affine, gate/up not
    /// 4/8-bit or down not 4/6/8-bit, gate/up bits mismatch, group_size mismatch
    /// across gate/up/down, missing biases, or a non-single-token `x`. Gated by
    /// the caller (`MLXCEL_FUSED_MOE`).
    pub fn forward_fused_kernel(
        &self,
        x: &MlxArray,
        indices: &MlxArray,
        scores: &MlxArray,
    ) -> Option<UniquePtr<MlxArray>> {
        let gate = self.gate_proj.quantized_parts()?;
        let up = self.up_proj.quantized_parts()?;
        let down = self.down_proj.quantized_parts()?;
        // Kernel A (gate/up) is power-of-2 only; kernel B (down) also handles
        // 6-bit. gate/up must match each other; down may differ (dots.llm1:
        // gate/up 4-bit, down 6-bit). group_size is shared across all three.
        if gate.bits != 4 && gate.bits != 8 {
            return None;
        }
        if down.bits != 4 && down.bits != 8 && down.bits != 6 {
            return None;
        }
        if gate.bits != up.bits
            || gate.group_size != up.group_size
            || gate.group_size != down.group_size
        {
            return None;
        }
        if gate.mode != "affine" || up.mode != "affine" || down.mode != "affine" {
            return None;
        }
        let gw_shape = mlxcel_core::array_shape(gate.weight.as_ref().unwrap());
        if gw_shape.len() != 3 {
            return None;
        }
        let dff = gw_shape[1];
        let din = gw_shape[2] * (32 / gate.bits);
        // Large experts: gather_qmm already saturates the GPU, so the two-kernel
        // fused path's all-cores advantage disappears and its extra dispatch +
        // global-memory activation staging becomes a net loss. The break-even is
        // backend-dependent (Metal ~4096, CUDA ~8192; see FUSED_MOE_MAX_DFF_*),
        // so the default follows the live backend and the env var overrides both.
        // Decline above the bound so the caller falls back to gather_qmm.
        let max_dff = fused_moe_max_dff_from(
            std::env::var("MLXCEL_FUSED_MOE_MAX_DFF").ok().as_deref(),
            mlxcel_core::metal_is_available(),
        );
        if dff > max_dff {
            return None;
        }
        // 6-bit down packs 4 weights into 3 bytes; the kernel reads the row as
        // bytes and needs Dff divisible by 16 (whole uint32 columns).
        if down.bits == 6 && dff % 16 != 0 {
            return None;
        }
        let k = *mlxcel_core::array_shape(indices).last()?;
        let x_elems: i32 = mlxcel_core::array_shape(x).iter().product();
        if x_elems != din {
            return None;
        }
        let x_flat = mlxcel_core::reshape(x, &[din]);
        let idx_flat = mlxcel_core::reshape(indices, &[k]);
        let sc_flat = mlxcel_core::reshape(scores, &[k]);
        Some(mlxcel_core::fused_moe_expert_kernel(
            &x_flat,
            &idx_flat,
            gate.weight.as_ref().unwrap(),
            gate.scales.as_ref().unwrap(),
            gate.biases.as_ref().unwrap(),
            up.weight.as_ref().unwrap(),
            up.scales.as_ref().unwrap(),
            up.biases.as_ref().unwrap(),
            down.weight.as_ref().unwrap(),
            down.scales.as_ref().unwrap(),
            down.biases.as_ref().unwrap(),
            &sc_flat,
            din,
            dff,
            k,
            gate.bits,
            down.bits,
            gate.group_size,
        ))
    }

    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        Self::from_weights_with_proj_names(
            weights,
            prefix,
            group_size,
            bits,
            ["gate_proj", "up_proj", "down_proj"],
        )
    }

    /// Like [`SwitchGLU::from_weights`], but with overridable projection leaf
    /// names so checkpoints that do not use the `gate_proj`/`up_proj`/`down_proj`
    /// convention can still load through the shared loader. `proj_names` is
    /// `[gate, up, down]`, naming the per-projection leaf under `prefix`
    /// (pre-stacked `{prefix}.{leaf}.weight`) or under the per-expert layout
    /// `{root}.experts.{idx}.{leaf}` when `prefix` ends in `.switch_mlp`.
    ///
    /// Mixtral stores its experts under the `w1`/`w2`/`w3` convention, mapping
    /// gate=`w1`, up=`w3`, down=`w2`; it passes those leaf names here so the
    /// shared code stays generic (no checkpoint-specific names baked in).
    pub fn from_weights_with_proj_names(
        weights: &WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
        proj_names: [&str; 3],
    ) -> Result<Self, String> {
        let [gate_leaf, up_leaf, down_leaf] = proj_names;
        Ok(Self {
            gate_proj: SwitchLinear::from_weights(
                weights,
                &format!("{}.{}", prefix, gate_leaf),
                group_size,
                bits,
            )?,
            up_proj: SwitchLinear::from_weights(
                weights,
                &format!("{}.{}", prefix, up_leaf),
                group_size,
                bits,
            )?,
            down_proj: SwitchLinear::from_weights(
                weights,
                &format!("{}.{}", prefix, down_leaf),
                group_size,
                bits,
            )?,
        })
    }
}

/// Sort tokens by expert index for efficient gather_qmm/gather_mm
/// Used by: SwitchGLU, GptOss
pub fn gather_sort(
    x: &MlxArray,
    indices: &MlxArray,
) -> (
    UniquePtr<MlxArray>,
    UniquePtr<MlxArray>,
    UniquePtr<MlxArray>,
) {
    let indices_shape = mlxcel_core::array_shape(indices);
    let top_k = indices_shape[indices_shape.len() - 1];
    let flat_indices = mlxcel_core::reshape(indices, &[-1]);
    let order = mlxcel_core::argsort(&flat_indices, -1);
    let inv_order = mlxcel_core::argsort(&order, -1);
    let x_shape = mlxcel_core::array_shape(x);
    let x_flat = mlxcel_core::reshape(x, &[x_shape[0], 1, x_shape[3]]);
    let top_k_arr = mlxcel_core::from_slice_i32(&[top_k], &[1]);
    let token_indices = mlxcel_core::divide(&order, &top_k_arr);
    let token_indices = mlxcel_core::astype(&token_indices, dtype::INT32);
    let sorted_x = mlxcel_core::take(&x_flat, &token_indices, 0);
    let sorted_indices = mlxcel_core::take(&flat_indices, &order, 0);
    (sorted_x, sorted_indices, inv_order)
}

/// Unsort tokens back to original order
fn scatter_unsort(x: &MlxArray, inv_order: &MlxArray, orig_shape: &[i32]) -> UniquePtr<MlxArray> {
    let unsorted = mlxcel_core::take(x, inv_order, 0);
    let x_shape = mlxcel_core::array_shape(&unsorted);
    let n_tokens = orig_shape[0];
    let top_k = orig_shape[1];
    let reshaped = mlxcel_core::reshape(&unsorted, &[n_tokens, top_k, x_shape[1], x_shape[2]]);
    mlxcel_core::squeeze_axis(&reshaped, 2)
}

/// Weighted sum over selected expert outputs while preserving the residual dtype.
///
/// Used by: DeepSeek, DeepSeekV3, DeepSeekV32, ExaOneMoe, Ernie4_5Moe,
///          GLM4Moe, GLM4MoeLite, GptOss, HunyuanMoe, KimiLinear, MiniMax,
///          Mistral4, Mixtral, Moondream3, OLMoE, PhiMoE, Qwen2Moe, Qwen3Moe,
///          Qwen3Next, Qwen3VLMoe, SolarOpen, Step3p5
///
/// The old `nkh,nk->nh` einsum contraction promotes the combine to float32
/// on M5 for bf16/f16 activations. Match mlx-lm's `y * scores[..., None]`
/// followed by `sum(axis=-2)`, with scores cast to the expert output dtype and
/// the final result restored to the hidden/residual dtype.
pub fn moe_weighted_sum(
    expert_out: &MlxArray,
    scores: &MlxArray,
    output_dtype: i32,
) -> UniquePtr<MlxArray> {
    let scores_exp = mlxcel_core::expand_dims(scores, -1);
    let scores_exp = mlxcel_core::astype(&scores_exp, mlxcel_core::array_dtype(expert_out));
    let weighted = mlxcel_core::multiply(expert_out, &scores_exp);
    let summed = mlxcel_core::sum_axis(&weighted, -2, false);
    if mlxcel_core::array_dtype(&summed) == output_dtype {
        summed
    } else {
        mlxcel_core::astype(&summed, output_dtype)
    }
}

/// Group-based expert masking for MoE gates with n_group > 1.
///
/// Selects the top `topk_group` expert groups (by sum of top-2 scores per group)
/// and zeros out scores for experts in non-selected groups.
///
/// Used by: DeepSeekV3, DeepSeekV32, GLM4Moe, GLM4MoeLite, ExaOneMoe
///
/// Reference: mlx-lm deepseek_v3.py group_expert_select()
pub fn group_mask_scores(scores: &MlxArray, n_group: i32, topk_group: i32) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(scores);
    let n = shape[0];
    let n_experts = shape[1];
    let experts_per_group = n_experts / n_group;

    // Unflatten: [n, n_experts] -> [n, n_group, experts_per_group]
    let grouped = mlxcel_core::reshape(scores, &[n, n_group, experts_per_group]);

    // Compute group_scores = sum of top-2 expert scores per group
    let neg_grouped = mlxcel_core::negative(&grouped);
    let part_idx = mlxcel_core::argpartition(&neg_grouped, 1, -1); // kth=1 for top-2
    let top2_idx = slice_axis(&part_idx, -1, 0, 2); // [n, n_group, 2]
    let top2_vals = mlxcel_core::take_along_axis(&grouped, &top2_idx, -1); // [n, n_group, 2]
    let group_scores = mlxcel_core::sum_axis(&top2_vals, -1, true); // [n, n_group, 1]

    // Find bottom-k groups to zero out (k = n_group - topk_group)
    let k = n_group - topk_group;
    let group_idx = mlxcel_core::argpartition(&group_scores, k - 1, -2); // [n, n_group, 1]
    let group_idx = slice_axis(&group_idx, -2, 0, k); // [n, k, 1]

    // Zero out experts in non-selected groups
    let zero = mlxcel_core::full_f32(&[1], 0.0, mlxcel_core::array_dtype(&grouped));
    let grouped = mlxcel_core::put_along_axis(&grouped, &group_idx, &zero, -2);

    // Flatten back: [n, n_group, experts_per_group] -> [n, n_experts]
    mlxcel_core::reshape(&grouped, &[n, n_experts])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moe_weighted_sum_preserves_bf16_output_dtype() {
        let expert_f32 = mlxcel_core::from_slice_f32(
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, //
                2.0, 4.0, 6.0, 8.0, 1.0, 3.0, 5.0, 7.0,
            ],
            &[2, 2, 4],
        );
        let expert = mlxcel_core::astype(&expert_f32, dtype::BFLOAT16);
        let scores = mlxcel_core::from_slice_f32(&[0.25, 0.75, 0.5, 0.5], &[2, 2]);

        let out = moe_weighted_sum(&expert, &scores, dtype::BFLOAT16);
        mlxcel_core::eval(&out);

        assert_eq!(mlxcel_core::array_shape(&out), vec![2, 4]);
        assert_eq!(mlxcel_core::array_dtype(&out), dtype::BFLOAT16);
    }

    #[test]
    fn moe_weighted_sum_preserves_f16_output_dtype() {
        let expert_f32 = mlxcel_core::from_slice_f32(
            &[
                1.0, 0.0, 3.0, 0.0, 5.0, 0.0, 7.0, 0.0, //
                0.0, 2.0, 0.0, 4.0, 0.0, 6.0, 0.0, 8.0,
            ],
            &[2, 2, 4],
        );
        let expert = mlxcel_core::astype(&expert_f32, dtype::FLOAT16);
        let scores = mlxcel_core::from_slice_f32(&[0.5, 0.5, 0.25, 0.75], &[2, 2]);

        let out = moe_weighted_sum(&expert, &scores, dtype::FLOAT16);
        mlxcel_core::eval(&out);

        assert_eq!(mlxcel_core::array_shape(&out), vec![2, 4]);
        assert_eq!(mlxcel_core::array_dtype(&out), dtype::FLOAT16);
    }

    #[test]
    fn fused_moe_enabled_defaults_on_and_respects_disable_values() {
        // Unset -> on (default-on as of #282).
        assert!(fused_moe_enabled_from(None));
        // Explicit disable values (case-insensitive, trimmed) -> off.
        for v in ["0", "false", "off", "no", "OFF", "False", " 0 ", "No"] {
            assert!(
                !fused_moe_enabled_from(Some(v)),
                "{v:?} should disable the kernel"
            );
        }
        // Any other value, including empty, -> on.
        for v in ["1", "true", "on", "yes", "", "anything"] {
            assert!(
                fused_moe_enabled_from(Some(v)),
                "{v:?} should keep the kernel on"
            );
        }
    }

    #[test]
    fn fused_moe_max_dff_default_is_backend_aware_and_env_overrides() {
        // Unset: Metal keeps the conservative 4096; CUDA (no Metal) uses the
        // higher measured GB10 crossover (#626).
        assert_eq!(fused_moe_max_dff_from(None, true), FUSED_MOE_MAX_DFF_METAL);
        assert_eq!(fused_moe_max_dff_from(None, false), FUSED_MOE_MAX_DFF_CUDA);
        // The CUDA default must exceed Metal's, which is the whole point of #626.
        const { assert!(FUSED_MOE_MAX_DFF_CUDA > FUSED_MOE_MAX_DFF_METAL) };
        // An explicit positive value overrides the default on both backends.
        for metal in [true, false] {
            assert_eq!(fused_moe_max_dff_from(Some("16384"), metal), 16384);
            assert_eq!(fused_moe_max_dff_from(Some("1"), metal), 1);
        }
        // Non-positive or unparseable values fall back to the backend default.
        for bad in ["0", "-1", "abc", "", " 4096"] {
            assert_eq!(
                fused_moe_max_dff_from(Some(bad), true),
                FUSED_MOE_MAX_DFF_METAL
            );
            assert_eq!(
                fused_moe_max_dff_from(Some(bad), false),
                FUSED_MOE_MAX_DFF_CUDA
            );
        }
    }

    #[test]
    fn switch_linear_stacks_individual_quantized_experts() {
        // Per-expert affine 4-bit layout (Qwen1.5-MoE / Qwen2-MoE checkpoints):
        // weight [out, in/8], scales/biases [out, in/group_size], stored under
        // `experts.{idx}.gate_proj.*` instead of a stacked `switch_mlp` tensor.
        let out = 4i32;
        let group = 64i32;
        let in_dim = 64i32; // a single quantization group
        let packed_in = in_dim / 8; // 4-bit packs 8 weights per uint32 column
        let num_groups = in_dim / group;
        let root = "model.layers.0.mlp";

        let mut weights = WeightMap::new();
        for e in 0..3 {
            weights.insert(
                format!("{root}.experts.{e}.gate_proj.weight"),
                mlxcel_core::from_slice_f32(
                    &vec![0.0; (out * packed_in) as usize],
                    &[out, packed_in],
                ),
            );
            weights.insert(
                format!("{root}.experts.{e}.gate_proj.scales"),
                mlxcel_core::from_slice_f32(
                    &vec![1.0; (out * num_groups) as usize],
                    &[out, num_groups],
                ),
            );
            weights.insert(
                format!("{root}.experts.{e}.gate_proj.biases"),
                mlxcel_core::from_slice_f32(
                    &vec![0.0; (out * num_groups) as usize],
                    &[out, num_groups],
                ),
            );
        }

        let sl =
            SwitchLinear::from_weights(&weights, &format!("{root}.switch_mlp.gate_proj"), group, 4)
                .expect("per-expert stacking should load through the shared loader");
        let parts = sl
            .quantized_parts()
            .expect("stacked per-expert weights should yield a quantized SwitchLinear");
        assert_eq!(
            mlxcel_core::array_shape(parts.weight),
            vec![3, out, packed_in]
        );
        assert_eq!(
            parts.bits, 4,
            "bits inferred from packed weight/scales shapes"
        );
        assert_eq!(parts.group_size, group);
        assert_eq!(parts.mode, "affine");
    }

    #[test]
    fn switch_glu_loads_overridden_proj_leaf_names() {
        // Mixtral stores experts under the w1/w2/w3 convention at
        // `{root}.experts.{idx}.{w1,w2,w3}.{weight,scales,biases}`, with
        // gate=w1, up=w3, down=w2. The shared loader must find them when the
        // leaf names are overridden (the `.switch_mlp` virtual prefix keys the
        // per-expert stacker to the `{root}.experts.{idx}` layout, identical to
        // the Qwen2-MoE path).
        let out = 4i32;
        let group = 64i32;
        let in_dim = 64i32; // a single quantization group
        let packed_in = in_dim / 8; // 4-bit packs 8 weights per uint32 column
        let num_groups = in_dim / group;
        let root = "model.layers.0.block_sparse_moe";

        let mut weights = WeightMap::new();
        for e in 0..3 {
            for leaf in ["w1", "w2", "w3"] {
                weights.insert(
                    format!("{root}.experts.{e}.{leaf}.weight"),
                    mlxcel_core::from_slice_f32(
                        &vec![0.0; (out * packed_in) as usize],
                        &[out, packed_in],
                    ),
                );
                weights.insert(
                    format!("{root}.experts.{e}.{leaf}.scales"),
                    mlxcel_core::from_slice_f32(
                        &vec![1.0; (out * num_groups) as usize],
                        &[out, num_groups],
                    ),
                );
                weights.insert(
                    format!("{root}.experts.{e}.{leaf}.biases"),
                    mlxcel_core::from_slice_f32(
                        &vec![0.0; (out * num_groups) as usize],
                        &[out, num_groups],
                    ),
                );
            }
        }

        let glu = SwitchGLU::from_weights_with_proj_names(
            &weights,
            &format!("{root}.switch_mlp"),
            group,
            4,
            ["w1", "w3", "w2"], // gate=w1, up=w3, down=w2
        )
        .expect("overridden leaf names should load the per-expert experts");

        for proj in [&glu.gate_proj, &glu.up_proj, &glu.down_proj] {
            let parts = proj
                .quantized_parts()
                .expect("stacked per-expert weights should yield a quantized SwitchLinear");
            assert_eq!(
                mlxcel_core::array_shape(parts.weight),
                vec![3, out, packed_in]
            );
            assert_eq!(
                parts.bits, 4,
                "bits inferred from packed weight/scales shapes"
            );
            assert_eq!(parts.group_size, group);
            assert_eq!(parts.mode, "affine");
        }
    }

    #[test]
    fn switch_linear_missing_weight_errors_without_stacked_or_individual() {
        // Neither a stacked `switch_mlp` tensor nor `experts.{idx}` tensors exist.
        // (`SwitchLinear` holds non-Debug MlxArray handles, so match on the Result
        // rather than using `expect_err`.)
        let weights = WeightMap::new();
        let err = match SwitchLinear::from_weights(
            &weights,
            "model.layers.0.mlp.switch_mlp.gate_proj",
            64,
            4,
        ) {
            Ok(_) => panic!("absent experts must not load"),
            Err(e) => e,
        };
        assert!(err.contains("Missing weight"), "unexpected error: {err}");
    }
}
