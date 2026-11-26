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

//! The NVFP4 exports of Raon-1 mini on the tiny random model (#3437).
//!
//! The tiny model is quantized into each published layout (the vendor MLX
//! export and the ModelOpt mixed-precision export), written to disk with a
//! `config.json`, and loaded through `Raon1MiniModel::load`, so the shared
//! loader's detection, repack and global-scale resolution all run. Its logits
//! must equal a dense twin built from the dequantized weights, which pins
//! every scale: a per-tensor scale left out, applied twice, or a per-expert
//! scale read with the other export's meaning moves the logits far past the
//! tolerance.

use super::tests::{EXPERTS, HIDDEN, MOE_INTER, ids, noise, raw_weights, tiny_config, values};
use super::{Raon1MiniConfig, Raon1MiniModel};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};
use serde_json::{Value, json};

/// Linear leaves the exports quantize (routers, convs, norms, mHC, the
/// embedding and `kv_b_proj` stay unquantized, as in the release).
const QUANTIZED_LEAVES: [&str; 15] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "g_proj",
    "o_proj",
    "b_proj",
    "rda_w_proj",
    "f_a_proj",
    "f_b_proj",
    "q_a_proj",
    "q_b_proj",
    "kv_a_proj_with_mqa",
    "gate_proj",
    "up_proj",
    "down_proj",
];
const PROJS: [&str; 3] = ["gate_proj", "up_proj", "down_proj"];

#[derive(Clone, Copy, PartialEq)]
enum Export {
    Mlx,
    ModelOpt,
}

fn guard() -> crate::models::embedding_test_support::MlxTestGuard {
    crate::models::embedding_test_support::mlx_test_guard()
}

fn f32s(a: &MlxArray) -> Vec<f32> {
    values(a)
}

/// `(packed, scales, dequantized)` of `w` at an MLX block-float layout,
/// optionally with MLX's NVFP4 tensor scale.
fn quantize(
    w: &MlxArray,
    group: i32,
    bits: i32,
    mode: &str,
    tensor_scale: Option<f32>,
) -> (
    UniquePtr<MlxArray>,
    UniquePtr<MlxArray>,
    UniquePtr<MlxArray>,
) {
    let q = match tensor_scale {
        Some(g) => mlxcel_core::quantize_weights_with_global_scale(
            w,
            group,
            bits,
            mode,
            &mlxcel_core::from_slice_f32(&[g], &[]),
        ),
        None => mlxcel_core::quantize_weights_with_mode(w, group, bits, mode),
    };
    let (wq, sq) = (
        mlxcel_core::quantized_weights_w(&q),
        mlxcel_core::quantized_weights_scales(&q),
    );
    // SAFETY: both operands are live arrays; block-float modes take no biases.
    let deq = unsafe { mlxcel_core::dequantize(&wq, &sq, std::ptr::null(), group, bits, mode) };
    // MLX dequantizes block-float planes to a 16-bit dtype; keep the twin in
    // f32 so a scale applied to it is not rounded.
    (wq, sq, mlxcel_core::astype(&deq, dtype::FLOAT32))
}

fn scalar(v: f32, shape: &[i32]) -> UniquePtr<MlxArray> {
    mlxcel_core::from_slice_f32(&[v], shape)
}

fn max_abs(a: &MlxArray) -> f32 {
    f32s(a).iter().fold(0f32, |m, v| m.max(v.abs()))
}

fn as_bytes(a: &MlxArray) -> UniquePtr<MlxArray> {
    mlxcel_core::view(&mlxcel_core::contiguous(a, false), dtype::UINT8)
}

/// One export of the tiny model: the files' tensors (with the safetensors
/// dtype tag of each), the dense twin's raw weights, and the export's
/// `config.json` additions.
struct Built {
    files: Vec<(String, UniquePtr<MlxArray>, &'static str)>,
    twin: WeightMap,
    quantized_layers: serde_json::Map<String, Value>,
    global_scaled: Vec<String>,
}

impl Built {
    fn file(&mut self, key: String, a: UniquePtr<MlxArray>, tag: &'static str) {
        self.files.push((key, a, tag));
    }
}

/// A dense linear: NVFP4 codes of `w / g` with the per-tensor `g`.
fn dense_linear(b: &mut Built, export: Export, prefix: &str, w: &MlxArray) {
    let g = max_abs(w) / 3.0;
    let scaled = mlxcel_core::multiply_scalar(w, 1.0 / g);
    let (wq, sq, deq) = quantize(&scaled, 16, 4, "nvfp4", None);
    b.twin.insert(
        format!("{prefix}.weight"),
        mlxcel_core::multiply_scalar(&deq, g),
    );
    match export {
        Export::Mlx => {
            b.file(format!("{prefix}.weight"), wq, "U32");
            b.file(format!("{prefix}.scales"), sq, "U8");
            b.file(format!("{prefix}.global_scale"), scalar(g, &[]), "F32");
            b.global_scaled.push(prefix.to_string());
        }
        Export::ModelOpt => {
            b.file(format!("{prefix}.weight"), as_bytes(&wq), "U8");
            b.file(format!("{prefix}.weight_scale"), sq, "F8_E4M3");
            b.file(format!("{prefix}.weight_scale_2"), scalar(g, &[]), "F32");
            b.file(format!("{prefix}.input_scale"), scalar(1.0, &[]), "F32");
            b.quantized_layers.insert(
                prefix.to_string(),
                json!({"quant_algo": "W4A16_NVFP4", "group_size": 16}),
            );
        }
    }
}

/// One routed-expert projection, `[EXPERTS, out, in]`.
fn expert_plane(b: &mut Built, export: Export, moe: &str, proj: &str, seed: u32) {
    let (rows, cols) = if proj == "down_proj" {
        (HIDDEN, MOE_INTER)
    } else {
        (MOE_INTER, HIDDEN)
    };
    let (mut codes, mut scales, mut twin, mut tensor_scales) = (vec![], vec![], vec![], vec![]);
    for e in 0..EXPERTS {
        let magnitude = 0.4 * (1 + e) as f32;
        let data: Vec<f32> = noise(rows * cols, seed + e as u32)
            .iter()
            .map(|v| v * magnitude)
            .collect();
        let w = mlxcel_core::from_slice_f32(&data, &[rows as i32, cols as i32]);
        let g = max_abs(&w);
        match export {
            // mlx-lm: block scales relative to `max|w|`, dequantized with
            // `g / (448 * 6)` inside `gather_qmm`.
            Export::Mlx => {
                let (wq, sq, deq) = quantize(&w, 16, 4, "nvfp4", Some(g));
                twin.push(mlxcel_core::multiply_scalar(&deq, g / (448.0 * 6.0)));
                codes.push(wq);
                scales.push(sq);
                tensor_scales.push(g);
            }
            // ModelOpt: codes of `w / g` and `weight_scale_2 = g` per expert.
            Export::ModelOpt => {
                let s = g / 3.0;
                let (wq, sq, deq) = quantize(
                    &mlxcel_core::multiply_scalar(&w, 1.0 / s),
                    16,
                    4,
                    "nvfp4",
                    None,
                );
                twin.push(mlxcel_core::multiply_scalar(&deq, s));
                let key = format!("{moe}.experts.{e}.{proj}");
                b.file(format!("{key}.weight"), as_bytes(&wq), "U8");
                b.file(format!("{key}.weight_scale"), sq, "F8_E4M3");
                b.file(format!("{key}.weight_scale_2"), scalar(s, &[1]), "F32");
                b.file(format!("{key}.input_scale"), scalar(1.0, &[1]), "F32");
                b.quantized_layers
                    .insert(key, json!({"quant_algo": "NVFP4", "group_size": 16}));
            }
        }
    }
    let plane = format!("{moe}.switch_mlp.{proj}");
    if export == Export::Mlx {
        b.file(
            format!("{plane}.weight"),
            mlxcel_core::stack_owned(&codes, 0),
            "U32",
        );
        b.file(
            format!("{plane}.scales"),
            mlxcel_core::stack_owned(&scales, 0),
            "U8",
        );
        b.file(
            format!("{plane}.global_scale"),
            mlxcel_core::from_slice_f32(&tensor_scales, &[EXPERTS as i32]),
            "F32",
        );
    }
    b.twin.insert(
        format!("{plane}.weight"),
        mlxcel_core::stack_owned(&twin, 0),
    );
}

fn build(export: Export) -> Built {
    let config = tiny_config();
    let mut b = Built {
        files: Vec::new(),
        twin: WeightMap::new(),
        quantized_layers: serde_json::Map::new(),
        global_scaled: Vec::new(),
    };
    let mut raw = raw_weights();
    raw.retain(|k, _| !k.contains(".experts."));
    let mut keys: Vec<String> = raw.keys().cloned().collect();
    keys.sort();
    for key in keys {
        let w = raw.remove(&key).unwrap();
        let prefix = key.strip_suffix(".weight").unwrap_or("");
        let leaf = prefix.rsplit('.').next().unwrap_or("");
        let cols = mlxcel_core::array_shape(&w).last().copied().unwrap_or(0);
        if key == "lm_head.weight" {
            let (wq, sq, deq) = quantize(&w, 32, 8, "mxfp8", None);
            b.twin.insert(key.clone(), deq);
            match export {
                Export::Mlx => {
                    b.file(key, wq, "U32");
                    b.file("lm_head.scales".into(), sq, "U8");
                }
                Export::ModelOpt => {
                    b.file(key, as_bytes(&wq), "F8_E4M3");
                    b.file("lm_head.weight_scale".into(), sq, "U8");
                    b.quantized_layers.insert(
                        "lm_head".into(),
                        json!({"quant_algo": "MXFP8", "group_size": 32}),
                    );
                }
            }
        } else if QUANTIZED_LEAVES.contains(&leaf)
            && !prefix.contains(".moe.gate")
            && cols % 16 == 0
        {
            dense_linear(&mut b, export, prefix, &w);
        } else {
            b.twin.insert(key.clone(), mlxcel_core::copy(&w));
            b.file(key, w, "F32");
        }
    }
    for layer in (0..config.num_hidden_layers).filter(|&l| config.is_moe_layer(l)) {
        for (i, proj) in PROJS.iter().enumerate() {
            let moe = format!("model.layers.{layer}.moe");
            expert_plane(&mut b, export, &moe, proj, 900 + (layer * 10 + i) as u32);
        }
    }
    b
}

/// The tiny config (layer 2 steered) with the export's quantization keys.
fn export_config_json(export: Export, b: &Built) -> Value {
    let mut config = json!({
        "model_type": "raon1_mini", "vocab_size": 32, "hidden_size": HIDDEN,
        "intermediate_size": 48, "num_hidden_layers": 4, "num_attention_heads": 2,
        "head_dim": 32, "num_experts": EXPERTS, "num_experts_per_token": 2,
        "moe_intermediate_size": MOE_INTER, "num_shared_experts": 1, "q_lora_rank": 16,
        "kv_lora_rank": 16, "qk_nope_head_dim": 8, "qk_rope_head_dim": 4,
        "v_head_dim": 8, "rda_decay_rank": 8, "raon_mhc_streams": 4, "eos_token_id": [31],
        "rda_config": {"num_heads": 2, "head_dim": 32, "conv_kernel_size": 4,
            "full_rank_gate": true, "gate_lower_bound": -5.0, "rda_layers": [1, 2, 4], "mla_layers": [3]},
        "reasoning_effort_steering": {
            "alpha": {"none": 0, "low": 0.3, "medium": 0.1, "high": 0.05, "xhigh": 0.02, "max": 0},
            "default": "high", "layers": [2]}
    });
    match export {
        Export::Mlx => {
            let mut q = json!({"group_size": 16, "bits": 4, "mode": "nvfp4",
                "lm_head": {"group_size": 32, "bits": 8, "mode": "mxfp8"}});
            for (key, _, _) in &b.files {
                if let Some(plane) = key.strip_suffix(".global_scale")
                    && plane.contains(".switch_mlp.")
                {
                    q[plane] =
                        json!({"group_size": 16, "bits": 4, "mode": "nvfp4", "global_scale": true});
                }
            }
            config["quantization"] = q;
            config["raon_global_scaled_linears"] = json!(b.global_scaled);
            config["raon_merged_gate_up"] = json!(true);
            config["raon_fused_mhc_decode"] = json!(true);
            config["model_file"] = json!("raon_mlx_model.py");
        }
        Export::ModelOpt => {
            config["quantization_config"] = json!({
                "quant_method": "modelopt_fp4", "quant_algo": "MIXED_PRECISION",
                "group_size": 16, "kv_cache_scheme": null,
                "quantized_layers": b.quantized_layers,
            });
        }
    }
    config
}

/// `files` as one safetensors file; `tag` is each tensor's dtype on disk
/// (F8_E4M3 bytes are written as they are, as ModelOpt stores them).
fn write_safetensors(path: &std::path::Path, files: &[(String, UniquePtr<MlxArray>, &str)]) {
    use std::io::Write;
    let mut data: Vec<u8> = Vec::new();
    let mut header = serde_json::Map::new();
    for (name, tensor, tag) in files {
        let c = mlxcel_core::contiguous(tensor, false);
        mlxcel_core::eval(&c);
        let start = data.len();
        data.extend(mlxcel_core::array_to_raw_bytes(&c));
        header.insert(
            name.clone(),
            json!({"dtype": tag, "shape": mlxcel_core::array_shape(tensor),
                "data_offsets": [start, data.len()]}),
        );
    }
    let mut header = serde_json::to_string(&header).unwrap();
    while !header.len().is_multiple_of(8) {
        header.push(' ');
    }
    let mut file = std::fs::File::create(path).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(header.as_bytes()).unwrap();
    file.write_all(&data).unwrap();
}

/// Load the export from disk and its dense twin; `None` on a backend that
/// refuses native NVFP4 (the load error must then name it).
fn load_pair(export: Export) -> Option<(Raon1MiniModel, Raon1MiniModel)> {
    let b = build(export);
    let tag = if export == Export::Mlx {
        "mlx"
    } else {
        "modelopt"
    };
    let dir = std::env::temp_dir().join(format!("mlxcel_raon_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("config.json"),
        export_config_json(export, &b).to_string(),
    )
    .unwrap();
    write_safetensors(&dir.join("model.safetensors"), &b.files);
    let loaded = crate::models::switch_layers::expect_native_nvfp4_load(
        Raon1MiniModel::load(&dir).map(|(m, _)| m),
        "",
        "the export loads",
    );
    let _ = std::fs::remove_dir_all(&dir);
    let model = loaded?;
    let mut twin_config: Raon1MiniConfig = tiny_config();
    twin_config.reasoning_effort_steering = Some(
        crate::models::reasoning_effort_steering::ReasoningEffortSteering::from_config_value(
            &export_config_json(Export::ModelOpt, &b),
        )
        .unwrap()
        .unwrap(),
    );
    let twin = Raon1MiniModel::sanitize_weights(b.twin, &twin_config).unwrap();
    let twin = Raon1MiniModel::from_weights(&twin, &twin_config).unwrap();
    Some((model, twin))
}

fn check_export_matches_twin(export: Export) {
    let _guard = guard();
    let Some((model, twin)) = load_pair(export) else {
        return;
    };
    // The steering vector stays float32 through the export's load.
    let steering = model.reasoning_effort_steering();
    assert_eq!(steering.len(), 1);
    assert_eq!(mlxcel_core::array_dtype(steering[0].1), dtype::FLOAT32);

    let tokens: Vec<i32> = (0..10).map(|i| (i * 7 + 3) % 32).collect();
    for level in [None, Some("none")] {
        model.set_reasoning_effort(None, level).unwrap();
        twin.set_reasoning_effort(None, level).unwrap();
        let got = values(&model.forward_with_caches(
            &ids(&[&tokens]),
            &mut model.make_layer_caches(),
            None,
        ));
        let want = values(&twin.forward_with_caches(
            &ids(&[&tokens]),
            &mut twin.make_layer_caches(),
            None,
        ));
        let scale = want.iter().fold(1f32, |m, v| m.max(v.abs()));
        let worst = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            worst <= 2e-3 * scale,
            "effort {level:?}: max |delta| {worst} against logit scale {scale}"
        );
    }
}

#[test]
fn mlx_export_matches_its_dequantized_twin() {
    check_export_matches_twin(Export::Mlx);
}

#[test]
fn modelopt_export_matches_its_dequantized_twin() {
    check_export_matches_twin(Export::ModelOpt);
}

/// A per-tensor `global_scale` linear (the vendor's `GlobalScaledLinear`)
/// equals dequantize-then-matmul times the scale, and the MXFP8 head loads as
/// `mxfp8` although an NVFP4 or MXFP4 reading would also fit its widths.
#[test]
fn global_scaled_linear_equals_dequantized_matmul_times_the_scale() {
    let _guard = guard();
    let (rows, cols) = (8i32, 64i32);
    let w = mlxcel_core::from_slice_f32(&noise((rows * cols) as usize, 77), &[rows, cols]);
    let (wq, sq, deq) = quantize(&w, 16, 4, "nvfp4", None);
    let g = 0.37f32;
    let x = mlxcel_core::from_slice_f32(&noise(3 * cols as usize, 5), &[3, cols]);
    let want =
        mlxcel_core::multiply_scalar(&mlxcel_core::matmul(&x, &mlxcel_core::transpose(&deq)), g);
    let mut weights = WeightMap::new();
    weights.insert("p.weight".into(), wq);
    weights.insert("p.scales".into(), sq);
    weights.insert("p.global_scale".into(), scalar(g, &[]));
    let Some(linear) = crate::models::switch_layers::expect_native_nvfp4_load(
        super::super::raon1_mini_quant::linear(&weights, "p"),
        "p",
        "global-scaled linear",
    ) else {
        return;
    };
    let q = linear.quantized_weight().expect("quantized");
    assert_eq!((q.mode.as_str(), q.group_size, q.bits), ("nvfp4", 16, 4));
    let (got, want) = (f32s(&linear.forward(&x)), f32s(&want));
    for (a, b) in got.iter().zip(&want) {
        assert!((a - b).abs() <= 1e-4 * b.abs().max(1.0), "{a} vs {b}");
    }

    let (hq, hs, _) = quantize(&w, 32, 8, "mxfp8", None);
    let mut head = WeightMap::new();
    head.insert("lm_head.weight".into(), hq);
    head.insert("lm_head.scales".into(), hs);
    let head = super::super::raon1_mini_quant::linear(&head, "lm_head").unwrap();
    let q = head.quantized_weight().expect("quantized");
    assert_eq!((q.mode.as_str(), q.group_size, q.bits), ("mxfp8", 32, 8));
}
