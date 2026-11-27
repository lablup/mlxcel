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

//! Cross-check a recurrent config against its checkpoint tensors (issue #3338).
//!
//! `SsmConfig::validate` only rules out overflow and non-finite parameters. An
//! untrusted `config.json` could still ask for state and graph shapes far
//! larger than the model it ships with, so every SSM dim is checked against
//! the safetensors headers before any graph is emitted: the checkpoint is the
//! real upper bound on what the config may request.

use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};

use super::config::{Config, ExpertAct, MoeScoring, QuantConfig, SsmKind, WeightScheme};
use crate::shard_path::index_shard_path;

/// Largest safetensors header this check reads (the format caps it at 100 MB).
const MAX_HEADER_BYTES: u64 = 100 * 1024 * 1024;

/// Bits per element of a safetensors dtype, or `None` for one this check does
/// not know (and so cannot size).
fn dtype_bits(dtype: &str) -> Option<u64> {
    Some(match dtype {
        "BOOL" | "U8" | "I8" | "F8_E5M2" | "F8_E4M3" | "F8_E8M0" => 8,
        "F4" => 4,
        "F6_E2M3" | "F6_E3M2" => 6,
        "U16" | "I16" | "F16" | "BF16" => 16,
        "U32" | "I32" | "F32" => 32,
        "U64" | "I64" | "F64" | "C64" => 64,
        _ => return None,
    })
}

/// The byte length a tensor of `dtype` and `shape` occupies, checked.
fn tensor_bytes(dtype: &str, shape: &[usize]) -> Option<u64> {
    let elements = shape
        .iter()
        .try_fold(1u64, |acc, &dim| acc.checked_mul(u64::try_from(dim).ok()?))?;
    let bits = elements.checked_mul(dtype_bits(dtype)?)?;
    bits.is_multiple_of(8).then_some(bits / 8)
}

/// Read the tensor shapes of one safetensors file from its JSON header.
///
/// Every declared tensor must occupy exactly `product(shape) * dtype` bytes
/// inside the file's data section, as the loader's `SafeTensors::deserialize`
/// requires. Without this a header alone could declare shapes far larger than
/// the file, and the cross-check would bound nothing: a few-KB checkpoint
/// would pass with any dims its `config.json` asks for.
fn header_shapes(path: &Path) -> Result<HashMap<String, Vec<usize>>, String> {
    let mut file =
        std::fs::File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let file_len = file
        .metadata()
        .map_err(|e| format!("stat {}: {e}", path.display()))?
        .len();
    let mut size = [0u8; 8];
    file.read_exact(&mut size)
        .map_err(|e| format!("read {} header size: {e}", path.display()))?;
    let size = u64::from_le_bytes(size);
    // The data section follows the 8-byte length and the header.
    let data_len = 8u64
        .checked_add(size)
        .and_then(|start| file_len.checked_sub(start))
        .filter(|_| size != 0 && size <= MAX_HEADER_BYTES)
        .ok_or_else(|| {
            format!(
                "{} has an invalid safetensors header size {size} for a {file_len}-byte file",
                path.display()
            )
        })?;
    let mut bytes = Vec::new();
    file.take(size)
        .read_to_end(&mut bytes)
        .map_err(|e| format!("read {} header: {e}", path.display()))?;
    let header: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|e| format!("parse {} header: {e}", path.display()))?;
    let object = header
        .as_object()
        .ok_or_else(|| format!("{} header is not an object", path.display()))?;
    let mut shapes = HashMap::new();
    for (name, entry) in object {
        if name == "__metadata__" {
            continue;
        }
        let shape = entry
            .get("shape")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| format!("{} tensor {name} has no shape", path.display()))?
            .iter()
            .map(|dim| {
                dim.as_u64()
                    .and_then(|dim| usize::try_from(dim).ok())
                    .ok_or_else(|| format!("{} tensor {name} has a bad dim", path.display()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let dtype = entry
            .get("dtype")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default();
        let offsets = entry
            .get("data_offsets")
            .and_then(serde_json::Value::as_array)
            .and_then(|pair| match pair.as_slice() {
                [begin, end] => Some((begin.as_u64()?, end.as_u64()?)),
                _ => None,
            });
        let fits = match (tensor_bytes(dtype, &shape), offsets) {
            (Some(want), Some((begin, end))) => {
                begin <= end && end <= data_len && end - begin == want
            }
            _ => false,
        };
        if !fits {
            return Err(format!(
                "{} tensor {name} ({dtype} {shape:?}) does not match its data_offsets \
                 {offsets:?} in a {data_len}-byte data section",
                path.display()
            ));
        }
        shapes.insert(name.clone(), shape);
    }
    Ok(shapes)
}

/// The checkpoint files of `model_dir`: the single `model.safetensors`, or the
/// shards an index names, each a plain file name inside `model_dir`
/// ([`index_shard_path`], the loader's rule).
fn checkpoint_files(model_dir: &Path) -> Result<Vec<PathBuf>, String> {
    let single = model_dir.join("model.safetensors");
    if single.is_file() {
        return Ok(vec![single]);
    }
    let index_path = model_dir.join("model.safetensors.index.json");
    let text = std::fs::read_to_string(&index_path).map_err(|e| {
        format!(
            "{}: no model.safetensors and no readable index ({e})",
            model_dir.display()
        )
    })?;
    let index: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| format!("parse {}: {e}", index_path.display()))?;
    let map = index
        .get("weight_map")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| format!("{} has no weight_map", index_path.display()))?;
    // A set, not `Vec::contains`: a hostile index may list a million names.
    let mut seen = std::collections::BTreeSet::new();
    let mut files: Vec<PathBuf> = Vec::new();
    for shard in map.values() {
        let shard = shard
            .as_str()
            .ok_or_else(|| format!("{} maps a tensor to a non-string", index_path.display()))?;
        let path = index_shard_path(model_dir, &index_path, shard)?;
        if seen.insert(path.clone()) {
            files.push(path);
        }
    }
    Ok(files)
}

/// Whether `shape` is a `[out, in]` projection, plain or MLX affine-packed.
fn is_projection(shape: &[usize], out: usize, in_: usize, quant: Option<QuantConfig>) -> bool {
    match shape {
        [o, i] if *o == out && *i == in_ => true,
        // `i` comes from an untrusted header, so its packed width is checked too.
        [o, i] => quant.is_some_and(|q| {
            *o == out
                && q.bits > 0
                && in_
                    .checked_mul(q.bits)
                    .is_some_and(|w| Some(w) == i.checked_mul(32))
        }),
        _ => false,
    }
}

/// Whether a rank-2 or stacked rank-3 projection is stored in MLX affine-packed
/// form. The leading dimensions are empty for `[out, in]` and `[experts]` for
/// `[experts, out, in]`.
fn is_packed_projection(
    shape: &[usize],
    leading: &[usize],
    out: usize,
    in_: usize,
    quant: Option<QuantConfig>,
) -> bool {
    let Some(q) = quant.filter(|q| q.bits > 0) else {
        return false;
    };
    let Some(width_bits) = in_.checked_mul(q.bits) else {
        return false;
    };
    if !width_bits.is_multiple_of(32) {
        return false;
    }
    let mut expected = leading.to_vec();
    expected.extend([out, width_bits / 32]);
    shape == expected
}

/// Validate the affine metadata paired with a packed projection. The loader
/// consumes both companions and dequantizes the whole rank-2 or rank-3 tensor,
/// so accepting only the packed base shape would still defer a malformed expert
/// bank until runtime.
fn validate_affine_metadata(
    shapes: &HashMap<String, Vec<usize>>,
    name: &str,
    leading: &[usize],
    out: usize,
    in_: usize,
    quant: QuantConfig,
) -> Result<(), String> {
    if quant.group_size == 0 || !in_.is_multiple_of(quant.group_size) {
        return Err(format!(
            "checkpoint tensor {name} is affine-packed but config.json group_size={} does not divide input width {in_}",
            quant.group_size
        ));
    }
    let stem = name
        .strip_suffix(".weight")
        .ok_or_else(|| format!("affine-packed checkpoint tensor {name} is not a `.weight`"))?;
    let mut expected = leading.to_vec();
    expected.extend([out, in_ / quant.group_size]);
    for suffix in ["scales", "biases"] {
        let companion = format!("{stem}.{suffix}");
        let got = shapes
            .get(&companion)
            .ok_or_else(|| format!("checkpoint has no tensor {companion}"))?;
        if got != &expected {
            return Err(format!(
                "checkpoint tensor {companion} has shape {got:?}, config.json implies {expected:?}"
            ));
        }
    }
    Ok(())
}

impl Config {
    /// Check every SSM dim against the checkpoint in `model_dir` (a no-op for
    /// a config without recurrent state). Called by [`Config::from_json`].
    pub(crate) fn validate_recurrent_checkpoint(&self, model_dir: &Path) -> Result<(), String> {
        let Some(ssm) = self.ssm.as_ref().filter(|_| self.has_recurrent_state()) else {
            return Ok(());
        };
        let scheme_matches_kind = match ssm.kind {
            SsmKind::Mamba2 => matches!(
                self.weight_scheme,
                WeightScheme::Mamba2 | WeightScheme::NemotronH | WeightScheme::FalconH1
            ),
            SsmKind::Mamba1 { .. } => matches!(
                self.weight_scheme,
                WeightScheme::Mamba1 | WeightScheme::Jamba
            ),
            SsmKind::Plamo2 { .. } => self.weight_scheme == WeightScheme::Plamo2,
        };
        if !scheme_matches_kind {
            return Err(format!(
                "a {:?} recurrent config cannot use the {:?} weight scheme",
                ssm.kind, self.weight_scheme
            ));
        }
        let mut shapes = HashMap::new();
        for file in checkpoint_files(model_dir)? {
            shapes.extend(header_shapes(&file)?);
        }
        if self.weight_scheme == WeightScheme::Plamo2 {
            return super::plamo2_config::validate_checkpoint(self, &shapes);
        }
        let quant = self.quantization;
        let (h, di, cd) = (self.hidden, ssm.d_inner(), ssm.conv_dim);
        let k = ssm.conv_kernel;
        let shape = |name: &str| -> Result<&Vec<usize>, String> {
            shapes
                .get(name)
                .ok_or_else(|| format!("checkpoint has no tensor {name}"))
        };
        let expect = |name: &str, ok: bool, want: String| -> Result<(), String> {
            if ok {
                Ok(())
            } else {
                Err(format!(
                    "checkpoint tensor {name} has shape {:?}, config.json implies {want}",
                    shapes.get(name)
                ))
            }
        };
        let proj = |name: &str, out: usize, in_: usize| -> Result<(), String> {
            let got = shape(name)?;
            let packed = is_packed_projection(got, &[], out, in_, quant);
            expect(
                name,
                is_projection(got, out, in_, quant),
                format!("[{out}, {in_}]"),
            )?;
            if packed {
                validate_affine_metadata(&shapes, name, &[], out, in_, quant.unwrap())?;
            }
            Ok(())
        };
        let stacked_proj =
            |name: &str, experts: usize, out: usize, in_: usize| -> Result<(), String> {
                let got = shape(name)?;
                let packed = is_packed_projection(got, &[experts], out, in_, quant);
                let plain = got.as_slice() == [experts, out, in_];
                expect(name, plain || packed, format!("[{experts}, {out}, {in_}]"))?;
                if packed {
                    validate_affine_metadata(&shapes, name, &[experts], out, in_, quant.unwrap())?;
                }
                Ok(())
            };
        let vector = |name: &str, len: usize| -> Result<(), String> {
            let ok = shape(name)?.as_slice() == [len];
            expect(name, ok, format!("[{len}]"))
        };
        // The recurrent state `ssm_h` is `H*P*N` f32 per layer, allocated on
        // the host and the device from the config alone. The shape checks
        // below bound `H*P` and `G*N` by the checkpoint, but not their
        // product, so a few-MB checkpoint with a wide `d_inner` and a wide
        // `state_size` could still ask for gigabytes of state per layer.
        // Bound it by the layer's own in_proj, which every real Mamba2 meets
        // with a wide margin (`N <= 2 * hidden_size` suffices).
        let state = di.checked_mul(ssm.state_size);
        let in_proj = ssm.in_proj_dim().checked_mul(h);
        if !matches!((state, in_proj), (Some(state), Some(in_proj)) if state <= in_proj) {
            return Err(format!(
                "SSM state D*N = {di} * {} exceeds the in_proj weight \
                 [{}, {h}] it is derived from",
                ssm.state_size,
                ssm.in_proj_dim()
            ));
        }
        let names = crate::weight_names::scheme_names(self.weight_scheme);
        let jamba = self.weight_scheme == WeightScheme::Jamba;
        let named_blocks = self.falcon_h1.is_some() || jamba;
        proj(names.embed, self.vocab, h)?;
        vector(names.final_norm, h)?;
        if !self.tie_word_embeddings {
            proj("lm_head.weight", self.vocab, h)?;
        }
        for li in 0..self.n_layers {
            let p = format!("{}{li}.", names.layer_stem);
            vector(&format!("{p}{}", names.input_layernorm), h)?;
            let m = format!("{p}{}.", if named_blocks { "mamba" } else { "mixer" });
            let attention = self.attn_layer_index(li).is_some();
            if attention {
                let m = format!("{p}{}.", if named_blocks { "self_attn" } else { "mixer" });
                for (name, out, input) in [
                    ("q_proj", self.n_q * self.head_dim, h),
                    ("k_proj", self.n_kv * self.head_dim, h),
                    ("v_proj", self.n_kv * self.head_dim, h),
                    ("o_proj", h, self.n_q * self.head_dim),
                ] {
                    proj(&format!("{m}{name}.weight"), out, input)?;
                    if self.qkv_bias {
                        vector(&format!("{m}{name}.bias"), out)?;
                    }
                }
                if !named_blocks {
                    continue;
                }
            }
            // Falcon-H1 and Jamba blocks normalize again before their FFN; a
            // Jamba MoE FFN is checked by the expert branch below.
            if named_blocks && (attention || jamba) {
                vector(&format!("{p}pre_ff_layernorm.weight"), h)?;
            }
            if named_blocks && (attention || jamba) && !self.is_moe_layer(li) {
                for (name, out, input) in [
                    ("gate_proj", self.inter, h),
                    ("up_proj", self.inter, h),
                    ("down_proj", h, self.inter),
                ] {
                    proj(&format!("{p}feed_forward.{name}.weight"), out, input)?;
                    if self.mlp_bias {
                        vector(&format!("{p}feed_forward.{name}.bias"), out)?;
                    }
                }
            }
            if self.is_moe_layer(li) {
                let moe = self
                    .moe
                    .as_ref()
                    .ok_or("a Nemotron-H E block has no MoE configuration")?;
                let mp = format!("{p}{}.", moe.weight_prefix);
                let router = crate::weight_names::moe_router_stem(self.weight_scheme);
                proj(&format!("{mp}{router}.weight"), moe.n_experts, h)?;
                if matches!(
                    moe.scoring,
                    MoeScoring::Sigmoid {
                        correction_bias: true,
                        ..
                    }
                ) {
                    vector(&format!("{mp}gate.e_score_correction_bias"), moe.n_experts)?;
                }
                match moe.expert_act {
                    ExpertAct::SwiGlu => {
                        for projection in ["gate_proj", "up_proj"] {
                            stacked_proj(
                                &format!("{mp}switch_mlp.{projection}.weight"),
                                moe.n_experts,
                                moe.intermediate,
                                h,
                            )?;
                        }
                        stacked_proj(
                            &format!("{mp}switch_mlp.down_proj.weight"),
                            moe.n_experts,
                            h,
                            moe.intermediate,
                        )?;
                    }
                    ExpertAct::Relu2 => {
                        stacked_proj(
                            &format!("{mp}switch_mlp.fc1.weight"),
                            moe.n_experts,
                            moe.intermediate,
                            h,
                        )?;
                        stacked_proj(
                            &format!("{mp}switch_mlp.fc2.weight"),
                            moe.n_experts,
                            h,
                            moe.intermediate,
                        )?;
                    }
                }
                if let Some(shared) = moe.shared {
                    let stem = if moe.expert_act == ExpertAct::Relu2 {
                        "shared_experts"
                    } else {
                        "shared_expert"
                    };
                    if moe.expert_act == ExpertAct::SwiGlu {
                        proj(
                            &format!("{mp}{stem}.gate_proj.weight"),
                            shared.intermediate,
                            h,
                        )?;
                    }
                    proj(
                        &format!("{mp}{stem}.up_proj.weight"),
                        shared.intermediate,
                        h,
                    )?;
                    proj(
                        &format!("{mp}{stem}.down_proj.weight"),
                        h,
                        shared.intermediate,
                    )?;
                    if shared.gated {
                        proj(&format!("{mp}shared_expert_gate.weight"), 1, h)?;
                    }
                }
                if !jamba {
                    continue;
                }
            }
            // A Jamba attention block has no recurrent mixer.
            if jamba && attention {
                continue;
            }
            if self.layer_mixer(li) == super::config::MixerKind::None {
                proj(&format!("{m}up_proj.weight"), self.inter, h)?;
                proj(&format!("{m}down_proj.weight"), h, self.inter)?;
                if self.mlp_bias {
                    vector(&format!("{m}up_proj.bias"), self.inter)?;
                    vector(&format!("{m}down_proj.bias"), h)?;
                }
                continue;
            }
            proj(&format!("{m}in_proj.weight"), ssm.in_proj_dim(), h)?;
            proj(&format!("{m}out_proj.weight"), h, di)?;
            if ssm.use_bias {
                vector(&format!("{m}in_proj.bias"), ssm.in_proj_dim())?;
            }
            if self.falcon_h1.as_ref().map_or(ssm.use_bias, |f| f.out_bias) {
                vector(&format!("{m}out_proj.bias"), h)?;
            }
            // torch `[C, 1, K]`, or the MLX conversion's `[C, K, 1]` (same bytes).
            let conv = format!("{m}conv1d.weight");
            let conv_shape = shape(&conv)?.as_slice();
            let ok = conv_shape == [cd, 1, k] || conv_shape == [cd, k, 1];
            expect(&conv, ok, format!("[{cd}, 1, {k}]"))?;
            if ssm.use_conv_bias {
                vector(&format!("{m}conv1d.bias"), cd)?;
            }
            match ssm.kind {
                SsmKind::Mamba2 => {
                    for name in ["dt_bias", "A_log", "D"] {
                        vector(&format!("{m}{name}"), ssm.n_heads)?;
                    }
                    if self.falcon_h1.as_ref().is_none_or(|f| f.rms_norm) {
                        vector(&format!("{m}norm.weight"), di)?;
                    }
                }
                SsmKind::Mamba1 { dt_rank, bcdt_norm } => {
                    if bcdt_norm == super::config::BcdtNorm::Weighted {
                        vector(&format!("{m}dt_layernorm.weight"), dt_rank)?;
                        vector(&format!("{m}b_layernorm.weight"), ssm.state_size)?;
                        vector(&format!("{m}c_layernorm.weight"), ssm.state_size)?;
                    }
                    proj(
                        &format!("{m}x_proj.weight"),
                        dt_rank + 2 * ssm.state_size,
                        di,
                    )?;
                    proj(&format!("{m}dt_proj.weight"), di, dt_rank)?;
                    vector(&format!("{m}dt_proj.bias"), di)?;
                    let a_log = format!("{m}A_log");
                    expect(
                        &a_log,
                        shape(&a_log)?.as_slice() == [di, ssm.state_size],
                        format!("[{di}, {}]", ssm.state_size),
                    )?;
                    vector(&format!("{m}D"), di)?;
                }
                SsmKind::Plamo2 { .. } => {
                    unreachable!("PLaMo 2 checkpoints are checked by plamo2_config")
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A hostile packed width must not wrap into the expected one: with
    /// wrapping math `(2^59 + 2) * 32` equals `16 * 4` (hidden 16 at 4 bits).
    #[test]
    fn projection_packed_width_does_not_wrap() {
        let q = Some(QuantConfig {
            bits: 4,
            group_size: 16,
        });
        assert!(is_projection(&[8, 2], 8, 16, q), "16 * 4 / 32 = 2 words");
        assert!(!is_projection(&[8, (1usize << 59) + 2], 8, 16, q));
    }

    #[test]
    fn tensor_bytes_is_checked_and_knows_sub_byte_dtypes() {
        assert_eq!(tensor_bytes("F32", &[3, 5]), Some(60));
        assert_eq!(tensor_bytes("BF16", &[]), Some(2));
        assert_eq!(tensor_bytes("F4", &[4]), Some(2));
        assert_eq!(tensor_bytes("F4", &[3]), None, "a partial byte");
        assert_eq!(tensor_bytes("F32", &[usize::MAX, 2]), None, "overflow");
        assert_eq!(tensor_bytes("Q4_K", &[4]), None, "unknown dtype");
    }

    #[test]
    fn stacked_affine_projection_requires_exact_companion_shapes() {
        let quant = QuantConfig {
            bits: 4,
            group_size: 8,
        };
        let name = "backbone.layers.2.mixer.switch_mlp.fc1.weight";
        let stem = name.strip_suffix(".weight").unwrap();
        let mut shapes = HashMap::from([
            (name.to_string(), vec![8, 12, 2]),
            (format!("{stem}.scales"), vec![8, 12, 2]),
            (format!("{stem}.biases"), vec![8, 12, 2]),
        ]);
        assert!(is_packed_projection(
            shapes.get(name).unwrap(),
            &[8],
            12,
            16,
            Some(quant)
        ));
        validate_affine_metadata(&shapes, name, &[8], 12, 16, quant).unwrap();

        shapes.insert(format!("{stem}.scales"), vec![8, 12, 1]);
        let error = validate_affine_metadata(&shapes, name, &[8], 12, 16, quant).unwrap_err();
        assert!(error.contains("scales") && error.contains("[8, 12, 2]"));
    }
}
