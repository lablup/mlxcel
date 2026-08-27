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

//! DeepSeek-V4 weight sanitization and load-parity validation.
//!
//! Two tensor-name planes exist in the wild and both must load:
//!
//! * The mlx-community plane (the real `DeepSeek-V4-Flash-4bit` ships this):
//!   `model.layers.N.attn.wq_a`, `model.layers.N.ffn.switch_mlp.gate_proj`,
//!   `model.embed_tokens`, `lm_head`, pre-stacked experts, `tid2eid` int64.
//! * The legacy export plane the reference `sanitize` handles: bare `layers.N`
//!   prefixes, `embed` / `head` / `norm` top-level names,
//!   `hc_{attn,ffn}_{fn,base,scale}`, `w1`/`w2`/`w3` shared-expert names,
//!   per-expert `ffn.experts.{e}.w{1,2,3}` planes needing stacking,
//!   `ffn.gate.bias`, singular `.scale` quantization planes with
//!   uint8-packed weights needing a `view(uint32)` reinterpret.
//!
//! [`sanitize_weights`] ports the reference remapping over lazy copies (no
//! tensor data moves), then [`validate_weight_coverage`] enforces strict
//! load parity: every canonical tensor the config describes must exist, and
//! every tensor in the map must be one the model will read. A misnamed or
//! leftover tensor fails the load with named examples instead of silently
//! skewing the forward pass. MTP tensors (`mtp.*`, layers beyond
//! `num_hidden_layers`) are dropped: MTP drafting is out of scope.

use mlxcel_core::weights::WeightMap;
use std::collections::HashSet;

use super::ModelArgs;

/// Remap a checkpoint weight map onto the canonical module names.
pub(crate) fn sanitize_weights(weights: &WeightMap, args: &ModelArgs) -> Result<WeightMap, String> {
    let n_layers = args.num_hidden_layers;

    // Pass 1: drop MTP tensors, extra layers, and rope frequency caches.
    let keep = |k: &str| -> bool {
        if k.starts_with("mtp.") || k.contains("rotary_emb.inv_freq") {
            return false;
        }
        for prefix in ["layers.", "model.layers."] {
            if let Some(rest) = k.strip_prefix(prefix)
                && let Some((idx, _)) = rest.split_once('.')
                && let Ok(layer_idx) = idx.parse::<usize>()
            {
                return layer_idx < n_layers;
            }
        }
        true
    };

    // Pass 2: legacy quantization plane fixups. A key ending in `.scale`
    // with a sibling `.weight` is a legacy singular scale plane; the packed
    // weight ships as uint8 and must be reinterpreted as uint32.
    let mut out = WeightMap::new();
    let mut consumed: HashSet<String> = HashSet::new();
    for (k, v) in weights.iter() {
        if !keep(k) {
            continue;
        }
        let Some(stem) = k.strip_suffix(".scale") else {
            continue;
        };
        let wk = format!("{stem}.weight");
        let Some(weight) = weights.get(&wk) else {
            continue;
        };
        let w_dtype = mlxcel_core::array_dtype(weight);
        let is_packed_u8 =
            w_dtype == mlxcel_core::dtype::UINT8 || w_dtype == mlxcel_core::dtype::INT8;
        let v_shape = mlxcel_core::array_shape(v);
        let w_shape = mlxcel_core::array_shape(weight);
        let scale_last = v_shape.last().copied().unwrap_or(0);
        let weight_last = w_shape.last().copied().unwrap_or(0);

        if k.contains(".ffn.experts.")
            && !k.contains(".shared_experts.")
            && is_packed_u8
            && scale_last * 16 == weight_last
        {
            out.insert(format!("{k}s"), mlxcel_core::copy(v));
            out.insert(
                wk.clone(),
                mlxcel_core::view(weight, mlxcel_core::dtype::UINT32),
            );
            consumed.insert(k.clone());
            consumed.insert(wk);
        } else if w_dtype == mlxcel_core::dtype::UINT8 {
            let scales = mlxcel_core::repeat(&mlxcel_core::repeat(v, 4, -1), 128, 0);
            out.insert(format!("{k}s"), scales);
            out.insert(
                wk.clone(),
                mlxcel_core::view(weight, mlxcel_core::dtype::UINT32),
            );
            consumed.insert(k.clone());
            consumed.insert(wk);
        }
        // Non-u8 sibling: the `.scale` key is kept verbatim below (it is a
        // model parameter such as `attn_hc.scale`, never a quant plane).
    }
    for (k, v) in weights.iter() {
        if !keep(k) || consumed.contains(k) || out.contains_key(k) {
            continue;
        }
        if k.contains("tid2eid") {
            // Shipped int64; index arithmetic wants int32.
            out.insert(k.clone(), mlxcel_core::astype(v, mlxcel_core::dtype::INT32));
        } else {
            out.insert(k.clone(), mlxcel_core::copy(v));
        }
    }
    let weights = out;

    // Pass 3: legacy top-level renames.
    let mut out = WeightMap::new();
    for (k, v) in weights {
        let nk = match k.as_str() {
            "embed.weight" => "model.embed_tokens.weight".to_string(),
            "norm.weight" => "model.norm.weight".to_string(),
            "head.weight" => "lm_head.weight".to_string(),
            "hc_head_fn" => "model.hc_head.fn".to_string(),
            "hc_head_base" => "model.hc_head.base".to_string(),
            "hc_head_scale" => "model.hc_head.scale".to_string(),
            _ => k,
        };
        // Legacy per-layer renames: bare `layers.` prefix, HC parameter
        // names, gate bias, shared-expert w1/w2/w3.
        let mut nk = if nk.starts_with("layers.") {
            format!("model.{nk}")
        } else {
            nk
        };
        nk = nk.replace(".ffn.gate.bias", ".ffn.gate.e_score_correction_bias");
        for sub in ["attn", "ffn"] {
            for param in ["fn", "base", "scale"] {
                nk = nk.replace(&format!(".hc_{sub}_{param}"), &format!(".{sub}_hc.{param}"));
            }
        }
        for (old, new) in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")] {
            nk = nk.replace(
                &format!(".shared_experts.{old}."),
                &format!(".shared_experts.{new}."),
            );
        }
        out.insert(nk, v);
    }
    let mut weights = out;

    // Pass 4: stack legacy per-expert planes into the switch_mlp layout.
    for layer_idx in 0..n_layers {
        let prefix = format!("model.layers.{layer_idx}.ffn.experts");
        for (src, dst) in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")] {
            for suffix in ["weight", "scales"] {
                let key0 = format!("{prefix}.0.{src}.{suffix}");
                if !weights.contains_key(&key0) {
                    continue;
                }
                let mut planes = Vec::with_capacity(args.n_routed_experts);
                for e in 0..args.n_routed_experts {
                    let key = format!("{prefix}.{e}.{src}.{suffix}");
                    let plane = weights.remove(&key).ok_or_else(|| {
                        format!(
                            "DeepSeek-V4 legacy expert plane is incomplete: `{key}` is missing \
                             while expert 0 is present"
                        )
                    })?;
                    planes.push(plane);
                }
                let stacked = mlxcel_core::utils::stack_arrays(&planes, 0);
                weights.insert(
                    format!("model.layers.{layer_idx}.ffn.switch_mlp.{dst}.{suffix}"),
                    stacked,
                );
            }
        }
    }

    // Pass 5: reshape 2-D `wo_a` planes into the grouped MultiLinear layout.
    let o_groups = args.o_groups as i32;
    let o_lora_rank = args.o_lora_rank as i32;
    // The divisor the `-1` axis has to resolve against. Computed in i64 so a
    // config that has not yet been through `ModelArgs::validate` (the
    // sanitize pass runs on whatever the caller hands it) cannot overflow it.
    let plane_rows = args.o_groups as i64 * args.o_lora_rank as i64;
    for layer_idx in 0..n_layers {
        let prefix = format!("model.layers.{layer_idx}.attn.wo_a");
        for suffix in ["weight", "scales", "biases"] {
            let key = format!("{prefix}.{suffix}");
            if let Some(v) = weights.get(&key)
                && mlxcel_core::array_ndim(v) == 2
            {
                // The element count comes from the checkpoint, the divisor
                // from `config.json`, and neither is trustworthy. MLX's
                // `reshape` asserts the `-1` axis divides evenly and throws
                // otherwise, and a throw crossing the cxx bridge is an
                // uncatchable `std::terminate`. This pass runs BEFORE
                // `validate_weight_coverage`, so an unguarded mismatch would
                // abort the process mid-load instead of returning a load
                // error that names the tensor.
                let size = mlxcel_core::array_size(v) as i64;
                if plane_rows <= 0 || size == 0 || size % plane_rows != 0 {
                    return Err(format!(
                        "{key}: 2-D wo_a plane has {size} elements (shape {shape:?}), which is \
                         not a positive multiple of o_groups * o_lora_rank ({plane_rows})",
                        shape = mlxcel_core::array_shape(v)
                    ));
                }
                let reshaped = mlxcel_core::reshape(v, &[o_groups, o_lora_rank, -1]);
                weights.insert(key, reshaped);
            }
        }
    }

    Ok(weights)
}

/// Strict load parity over the sanitized map: every canonical tensor the
/// config describes must be present, and every present tensor must be one
/// the model reads. No silent fallbacks.
pub(crate) fn validate_weight_coverage(
    weights: &WeightMap,
    args: &ModelArgs,
) -> Result<(), String> {
    fn linear(base: &str, required: &mut HashSet<String>, allowed: &mut HashSet<String>) {
        required.insert(format!("{base}.weight"));
        for suffix in ["weight", "scales", "biases", "bias"] {
            allowed.insert(format!("{base}.{suffix}"));
        }
    }

    fn plain(name: String, required: &mut HashSet<String>, allowed: &mut HashSet<String>) {
        required.insert(name.clone());
        allowed.insert(name);
    }

    fn compressor(base: &str, required: &mut HashSet<String>, allowed: &mut HashSet<String>) {
        for name in ["wkv", "wgate"] {
            linear(&format!("{base}.{name}"), required, allowed);
        }
        for name in ["ape", "norm.weight"] {
            plain(format!("{base}.{name}"), required, allowed);
        }
    }

    let mut required: HashSet<String> = HashSet::new();
    let mut allowed: HashSet<String> = HashSet::new();

    linear("model.embed_tokens", &mut required, &mut allowed);
    plain("model.norm.weight".to_string(), &mut required, &mut allowed);
    for param in ["fn", "base", "scale"] {
        plain(
            format!("model.hc_head.{param}"),
            &mut required,
            &mut allowed,
        );
    }
    // A tied-embedding export may still ship (and this model then ignores)
    // an lm_head; only require it when it will be read.
    if args.tie_word_embeddings {
        for suffix in ["weight", "scales", "biases", "bias"] {
            allowed.insert(format!("lm_head.{suffix}"));
        }
    } else {
        linear("lm_head", &mut required, &mut allowed);
    }

    for (layer_idx, &ratio) in args.compress_ratios.iter().enumerate() {
        let p = format!("model.layers.{layer_idx}");
        for name in ["wq_a", "wq_b", "wkv", "wo_a", "wo_b"] {
            linear(&format!("{p}.attn.{name}"), &mut required, &mut allowed);
        }
        for name in [
            "attn.q_norm.weight",
            "attn.kv_norm.weight",
            "attn.attn_sink",
            "attn_norm.weight",
            "ffn_norm.weight",
        ] {
            plain(format!("{p}.{name}"), &mut required, &mut allowed);
        }
        for hc in ["attn_hc", "ffn_hc"] {
            for param in ["fn", "base", "scale"] {
                plain(format!("{p}.{hc}.{param}"), &mut required, &mut allowed);
            }
        }
        if ratio > 0 {
            compressor(&format!("{p}.attn.compressor"), &mut required, &mut allowed);
        }
        if ratio == i64::from(super::OVERLAP_COMPRESS_RATIO) {
            for name in ["wq_b", "weights_proj"] {
                linear(
                    &format!("{p}.attn.indexer.{name}"),
                    &mut required,
                    &mut allowed,
                );
            }
            compressor(
                &format!("{p}.attn.indexer.compressor"),
                &mut required,
                &mut allowed,
            );
        }

        plain(format!("{p}.ffn.gate.weight"), &mut required, &mut allowed);
        let gate_extra = if layer_idx < args.num_hash_layers {
            format!("{p}.ffn.gate.tid2eid")
        } else {
            format!("{p}.ffn.gate.e_score_correction_bias")
        };
        plain(gate_extra, &mut required, &mut allowed);
        for name in ["gate_proj", "up_proj", "down_proj"] {
            linear(
                &format!("{p}.ffn.switch_mlp.{name}"),
                &mut required,
                &mut allowed,
            );
            linear(
                &format!("{p}.ffn.shared_experts.{name}"),
                &mut required,
                &mut allowed,
            );
        }
    }

    let mut missing: Vec<&String> = required
        .iter()
        .filter(|k| !weights.contains_key(k.as_str()))
        .collect();
    let mut unknown: Vec<&String> = weights
        .keys()
        .filter(|k| !allowed.contains(k.as_str()))
        .collect();
    missing.sort();
    unknown.sort();

    if missing.is_empty() && unknown.is_empty() {
        return Ok(());
    }

    let sample = |v: &[&String]| -> String {
        let head: Vec<&str> = v.iter().take(8).map(|s| s.as_str()).collect();
        let more = v.len().saturating_sub(8);
        if more > 0 {
            format!("{head:?} (+{more} more)")
        } else {
            format!("{head:?}")
        }
    };
    let mut msg = String::from("DeepSeek-V4 checkpoint does not match the config it ships:");
    if !missing.is_empty() {
        msg.push_str(&format!(
            " {} required tensors are missing, e.g. {}.",
            missing.len(),
            sample(&missing)
        ));
    }
    if !unknown.is_empty() {
        msg.push_str(&format!(
            " {} tensors map onto no module path, e.g. {}.",
            unknown.len(),
            sample(&unknown)
        ));
    }
    Err(msg)
}
