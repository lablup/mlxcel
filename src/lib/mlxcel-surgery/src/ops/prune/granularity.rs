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

//! Per-granularity pruners (layer / attention head / MLP channel) and
//! the tensor-key classifiers that decide which projection role a
//! given matched key plays.
//!
//! Used by: `super::PruneOp::apply` (dispatches to one of the three
//! pruners based on the parsed `PruneSelector`).

use super::model_dims::ModelDims;
use super::tensor_ops::{
    zero_axis0_rows, zero_axis1_columns, zero_axis1_columns_or_packed_only, zero_tensor_inplace,
};
use crate::{SurgeryError, WeightMap};

// ============================================================
// Per-granularity pruners
// ============================================================

/// Zero every matched tensor whose key references one of the listed
/// transformer-block ids (e.g. `model.layers.<id>.`). The match is on
/// the dotted segment so `model.layers.10.x` does not match for id=1.
pub(super) fn prune_layers(
    weights: &mut WeightMap,
    matched: &[String],
    layer_ids: &[usize],
) -> Result<(), SurgeryError> {
    let mut touched = 0usize;
    for key in matched {
        let Some(layer_idx) = extract_layer_index(key) else {
            // Tensor matched the pattern but does not belong to a
            // numbered transformer block (e.g. `model.embed_tokens.*`).
            // For layer-granularity prune we leave it alone.
            continue;
        };
        if layer_ids.contains(&layer_idx) {
            zero_tensor_inplace(weights, key)?;
            touched += 1;
        }
    }
    if touched == 0 {
        return Err(SurgeryError::Other(anyhow::anyhow!(
            "prune (layer): pattern matched tensors but none belonged to the requested layer ids {layer_ids:?}"
        )));
    }
    Ok(())
}

/// Parse the layer index from a tensor key of the form
/// `....layers.<n>.<rest>`. Returns `None` if no such segment exists.
pub(super) fn extract_layer_index(key: &str) -> Option<usize> {
    let mut iter = key.split('.');
    while let Some(seg) = iter.next() {
        if seg != "layers" {
            continue;
        }
        let idx_seg = iter.next()?;
        if let Ok(n) = idx_seg.parse::<usize>() {
            return Some(n);
        }
    }
    None
}

/// Prune one or more attention heads by Q-head id.
///
/// Tensor recognition is by suffix. Per the module-level GQA policy:
/// - `q_proj.weight` / `q_proj.scales` / `q_proj.biases`:
///   zero OUT slice `[h*head_dim, (h+1)*head_dim)` along axis 0.
/// - `q_proj.bias` (1-D): zero `[h*head_dim, (h+1)*head_dim)`.
/// - `o_proj.weight` / `o_proj.scales` / `o_proj.biases`:
///   zero IN slice `[h*head_dim, (h+1)*head_dim)` along axis 1.
/// - `k_proj.*` / `v_proj.*`: skipped with a warning (GQA-shared).
/// - `q_norm.*` / `k_norm.*`: skipped silently (per-dim, not per-head).
/// - anything else: skipped with a warning so the user can refine the
///   glob.
pub(super) fn prune_attention_heads(
    weights: &mut WeightMap,
    matched: &[String],
    model: &ModelDims,
    head_ids: &[usize],
) -> Result<(), SurgeryError> {
    let mut q_touched = 0usize;
    let mut o_touched = 0usize;
    let mut kv_skipped = 0usize;
    let head_dim = model.head_dim;

    for key in matched {
        let role = classify_attention_key(key);
        match role {
            AttentionRole::QProj => {
                for &h in head_ids {
                    let start = (h * head_dim) as i32;
                    let stop = ((h + 1) * head_dim) as i32;
                    zero_axis0_rows(weights, key, start, stop)?;
                }
                q_touched += 1;
            }
            AttentionRole::OProj => {
                for &h in head_ids {
                    let start = (h * head_dim) as i32;
                    let stop = ((h + 1) * head_dim) as i32;
                    // o_proj packs IN axis (= num_heads * head_dim) on
                    // axis 1, so head boundaries are
                    // `head_dim`-aligned along axis 1.
                    zero_axis1_columns(weights, key, start, stop, head_dim)?;
                }
                o_touched += 1;
            }
            AttentionRole::KvProj => {
                kv_skipped += 1;
                if model.is_gqa() {
                    eprintln!(
                        "prune (attention_head): skipping {key} \
                         — GQA model has num_kv_heads={} < num_heads={}; \
                         zeroing KV would silently affect other Q heads. \
                         To prune KV, target `k_proj` / `v_proj` directly \
                         with a layer-granularity surgery and accept the \
                         consequences.",
                        model.num_kv_heads, model.num_heads,
                    );
                } else {
                    eprintln!(
                        "prune (attention_head): skipping {key} — \
                         k_proj/v_proj are skipped under the GQA-safe policy \
                         (see docs in mlxcel-surgery/src/ops/prune/mod.rs)."
                    );
                }
            }
            AttentionRole::Norm => {
                // Per-head-dim norm; not a per-head quantity. Silent
                // skip — common in the matched set.
            }
            AttentionRole::Unknown => {
                eprintln!(
                    "prune (attention_head): skipping {key} — unrecognized \
                     attention-projection suffix. Narrow the glob (e.g. \
                     `*.self_attn.q_proj.*`) if this tensor is unrelated."
                );
            }
        }
    }

    if q_touched == 0 && o_touched == 0 {
        return Err(SurgeryError::Other(anyhow::anyhow!(
            "prune (attention_head): pattern matched {} tensors but none \
             were Q or O projections (skipped {} KV tensors under GQA policy). \
             Adjust the glob to include `*.q_proj.*` / `*.o_proj.*`.",
            matched.len(),
            kv_skipped
        )));
    }
    Ok(())
}

/// Prune intermediate MLP channels.
///
/// - `up_proj.weight` / `up_proj.scales` / `up_proj.biases`:
///   zero OUT slice `[c, c+1)` along axis 0 (intermediate is OUT).
/// - `gate_proj.weight` / `gate_proj.scales` / `gate_proj.biases`:
///   zero OUT slice `[c, c+1)` along axis 0.
/// - `down_proj.weight` / `down_proj.scales` / `down_proj.biases`:
///   zero IN slice `[c, c+1)` along axis 1.
/// - `gate_up_proj.*` (combined gate+up — used by some HF checkpoints):
///   ERROR. Splitting requires extra heuristics; the user can split
///   the checkpoint first.
pub(super) fn prune_mlp_channels(
    weights: &mut WeightMap,
    matched: &[String],
    _model: &ModelDims,
    channel_ids: &[usize],
) -> Result<(), SurgeryError> {
    let mut touched = 0usize;
    for key in matched {
        let role = classify_mlp_key(key);
        match role {
            MlpRole::UpOrGate => {
                for &c in channel_ids {
                    let start = c as i32;
                    let stop = (c + 1) as i32;
                    zero_axis0_rows(weights, key, start, stop)?;
                }
                touched += 1;
            }
            MlpRole::Down => {
                for &c in channel_ids {
                    let start = c as i32;
                    let stop = (c + 1) as i32;
                    // Quantized down_proj single-channel pruning is
                    // refused — see `zero_axis1_columns_or_packed_only`.
                    zero_axis1_columns_or_packed_only(weights, key, start, stop, 1)?;
                }
                touched += 1;
            }
            MlpRole::CombinedGateUp => {
                return Err(SurgeryError::Other(anyhow::anyhow!(
                    "prune (mlp_channel): {key} is a combined gate+up_proj checkpoint; \
                     split it before pruning channels."
                )));
            }
            MlpRole::Unknown => {
                eprintln!(
                    "prune (mlp_channel): skipping {key} — unrecognized MLP \
                     suffix. Narrow the glob (e.g. `*.mlp.gate_proj.*`)."
                );
            }
        }
    }
    if touched == 0 {
        return Err(SurgeryError::Other(anyhow::anyhow!(
            "prune (mlp_channel): pattern matched {} tensors but none \
             were up/gate/down projections.",
            matched.len()
        )));
    }
    Ok(())
}

// ============================================================
// Tensor-suffix classifiers
// ============================================================

pub(super) enum AttentionRole {
    QProj,
    OProj,
    KvProj,
    Norm,
    Unknown,
}

/// Recognize Q / O / KV / norm projection keys by suffix. Handles both
/// raw `.weight` / `.bias` and quantized `.scales` / `.biases` forms.
pub(super) fn classify_attention_key(key: &str) -> AttentionRole {
    // `q_proj.bias` matches `q_proj` — order of checks does not matter
    // because we look for the projection name as a dotted segment.
    if key_has_dotted_segment(key, "q_proj") {
        AttentionRole::QProj
    } else if key_has_dotted_segment(key, "o_proj") {
        AttentionRole::OProj
    } else if key_has_dotted_segment(key, "k_proj") || key_has_dotted_segment(key, "v_proj") {
        AttentionRole::KvProj
    } else if key_has_dotted_segment(key, "q_norm")
        || key_has_dotted_segment(key, "k_norm")
        || key_has_dotted_segment(key, "input_layernorm")
        || key_has_dotted_segment(key, "post_attention_layernorm")
    {
        AttentionRole::Norm
    } else {
        AttentionRole::Unknown
    }
}

pub(super) enum MlpRole {
    UpOrGate,
    Down,
    CombinedGateUp,
    Unknown,
}

/// Recognize MLP projection keys by suffix.
pub(super) fn classify_mlp_key(key: &str) -> MlpRole {
    if key_has_dotted_segment(key, "gate_up_proj") {
        MlpRole::CombinedGateUp
    } else if key_has_dotted_segment(key, "up_proj") || key_has_dotted_segment(key, "gate_proj") {
        MlpRole::UpOrGate
    } else if key_has_dotted_segment(key, "down_proj") {
        MlpRole::Down
    } else {
        MlpRole::Unknown
    }
}

/// `true` when `key` contains `segment` as a complete dotted segment.
/// Avoids spurious matches like `q_projection` for `q_proj`.
pub(super) fn key_has_dotted_segment(key: &str, segment: &str) -> bool {
    key.split('.').any(|seg| seg == segment)
}
