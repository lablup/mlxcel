use std::io::Write;

use serde_json::json;

use super::config::InklingMtpConfig;
use super::model::{InklingMtpDraftModel, has_inkling_mtp_tensors};
use super::sanitize::sanitize_weights;
use crate::dtype;
use crate::inkling_layer::InklingLayerCache;
use crate::layers::{KVCache, RMSNorm, UnifiedEmbedding, UnifiedLinear};
use crate::weights::{WeightMap, load_weights_from_dir_index_filtered};

fn config_value() -> serde_json::Value {
    json!({
        "model_type": "inkling_mm_model",
        "mtp_config": {
            "num_nextn_predict_layers": 3,
            "local_layer_ids": [2, 0, 2]
        },
        "text_config": {
            "hidden_size": 8,
            "vocab_size": 17,
            "unpadded_vocab_size": 13,
            "rms_norm_eps": 0.000001,
            "logits_mup_width_multiplier": 2.0,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "swa_num_attention_heads": 4,
            "swa_num_key_value_heads": 2,
            "swa_head_dim": 2,
            "dense_intermediate_size": 16
        }
    })
}

fn write_single_tensor(path: &std::path::Path, name: &str) {
    let mut header =
        format!("{{\"{name}\":{{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}}}");
    while !header.len().is_multiple_of(8) {
        header.push(' ');
    }
    let mut file = std::fs::File::create(path).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(header.as_bytes()).unwrap();
    file.write_all(&1.0_f32.to_le_bytes()).unwrap();
}

fn assert_same(left: &crate::MlxArray, right: &crate::MlxArray) {
    let close = crate::allclose(left, right, 0.0, 0.0);
    crate::eval(&close);
    assert!(crate::item_bool(&close));
}

fn matrix(rows: i32, columns: i32, seed: usize) -> crate::UniquePtr<crate::MlxArray> {
    let values: Vec<f32> = (0..rows as usize * columns as usize)
        .map(|index| (((index + seed) % 13) as f32 - 6.0) / 40.0)
        .collect();
    crate::from_slice_f32(&values, &[rows, columns])
}

fn conv(channels: i32, seed: usize) -> crate::UniquePtr<crate::MlxArray> {
    let values: Vec<f32> = (0..channels as usize * 2)
        .map(|index| (((index + seed) % 7) as f32 - 3.0) / 30.0)
        .collect();
    crate::from_slice_f32(&values, &[channels, 2, 1])
}

struct TinyTarget {
    embed: UnifiedEmbedding,
    head: UnifiedLinear,
}

impl crate::generate::LanguageModel for TinyTarget {
    fn forward(
        &self,
        input: &crate::MlxArray,
        _caches: &mut [KVCache],
        _mask: Option<&crate::MlxArray>,
    ) -> crate::UniquePtr<crate::MlxArray> {
        self.head.forward(&self.embed.forward(input))
    }

    fn make_caches(&self) -> Vec<KVCache> {
        Vec::new()
    }

    fn num_layers(&self) -> usize {
        0
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        Vec::new()
    }

    fn embed_tokens(&self, input: &crate::MlxArray) -> Option<crate::UniquePtr<crate::MlxArray>> {
        Some(self.embed.forward(input))
    }

    fn embed_tokens_module(&self) -> Option<UnifiedEmbedding> {
        Some(self.embed.clone_shared())
    }

    fn lm_head_module(&self) -> Option<UnifiedLinear> {
        Some(self.head.clone_shared())
    }

    fn final_norm_module(&self) -> Option<RMSNorm> {
        Some(RMSNorm::new(crate::ones(&[4], dtype::FLOAT32), 1e-6))
    }
}

fn tiny_mtp_pair() -> (InklingMtpDraftModel, TinyTarget) {
    let mut value = config_value();
    value["mtp_config"]["num_nextn_predict_layers"] = json!(1);
    value["mtp_config"]["local_layer_ids"] = json!([0]);
    value["text_config"]["hidden_size"] = json!(4);
    value["text_config"]["vocab_size"] = json!(8);
    value["text_config"]["unpadded_vocab_size"] = json!(8);
    value["text_config"]["num_attention_heads"] = json!(2);
    value["text_config"]["num_key_value_heads"] = json!(1);
    value["text_config"]["head_dim"] = json!(2);
    value["text_config"]["swa_num_attention_heads"] = json!(2);
    value["text_config"]["swa_num_key_value_heads"] = json!(1);
    value["text_config"]["swa_head_dim"] = json!(2);
    value["text_config"]["sliding_window_size"] = json!(3);
    value["text_config"]["d_rel"] = json!(2);
    value["text_config"]["rel_extent"] = json!(4);
    value["text_config"]["sconv_kernel_size"] = json!(2);
    value["text_config"]["dense_intermediate_size"] = json!(6);
    let config = InklingMtpConfig::from_value(value).unwrap();
    let mut weights = WeightMap::new();
    for name in ["embed_norm.weight", "hidden_norm.weight"] {
        weights.insert(
            format!("blocks.0.{name}"),
            crate::ones(&[4], dtype::FLOAT32),
        );
    }
    weights.insert("blocks.0.input_proj.weight".into(), matrix(4, 8, 1));
    let layer = "blocks.0.transformer_block";
    for name in ["input_layernorm.weight", "post_attention_layernorm.weight"] {
        weights.insert(format!("{layer}.{name}"), crate::ones(&[4], dtype::FLOAT32));
    }
    weights.insert(format!("{layer}.attn_sconv.conv.weight"), conv(4, 2));
    weights.insert(format!("{layer}.mlp_sconv.conv.weight"), conv(4, 3));
    let attn = format!("{layer}.self_attn");
    for (name, rows, cols, seed) in [
        ("q_proj.weight", 4, 4, 4),
        ("k_proj.weight", 2, 4, 5),
        ("v_proj.weight", 2, 4, 6),
        ("r_proj.weight", 4, 4, 7),
        ("o_proj.weight", 4, 4, 8),
    ] {
        weights.insert(format!("{attn}.{name}"), matrix(rows, cols, seed));
    }
    for name in ["q_norm.weight", "k_norm.weight"] {
        weights.insert(format!("{attn}.{name}"), crate::ones(&[2], dtype::FLOAT32));
    }
    weights.insert(format!("{attn}.k_sconv.conv.weight"), conv(2, 9));
    weights.insert(format!("{attn}.v_sconv.conv.weight"), conv(2, 10));
    weights.insert(format!("{attn}.rel_proj"), matrix(2, 3, 11));
    let mlp = format!("{layer}.mlp");
    weights.insert(format!("{mlp}.gate_proj.weight"), matrix(6, 4, 12));
    weights.insert(format!("{mlp}.up_proj.weight"), matrix(6, 4, 13));
    weights.insert(format!("{mlp}.down_proj.weight"), matrix(4, 6, 14));
    weights.insert(
        format!("{mlp}.global_scale"),
        crate::ones(&[1], dtype::FLOAT32),
    );
    let mut target_weights = WeightMap::new();
    target_weights.insert("embed.weight".into(), matrix(8, 4, 15));
    target_weights.insert("head.weight".into(), matrix(8, 4, 16));
    let target = TinyTarget {
        embed: UnifiedEmbedding::from_weights(&target_weights, "embed", 64, 4).unwrap(),
        head: UnifiedLinear::from_weights(&target_weights, "head", 64, 4).unwrap(),
    };
    (
        InklingMtpDraftModel::from_weights(config, &weights).unwrap(),
        target,
    )
}

#[test]
fn config_uses_mtp_layer_count_local_ids_and_default_block_size() {
    let config = InklingMtpConfig::from_value(config_value()).unwrap();
    assert_eq!(config.num_mtp_layers(), 3);
    assert_eq!(config.local_layer_ids(), [0, 2]);
    assert_eq!(config.block_size(), 5);
    assert!(!config.layer_spec(1).unwrap().is_sliding);
    let local = config.layer_spec(2).unwrap();
    assert!(local.is_sliding);
    assert_eq!(local.num_attention_heads, 4);
    assert_eq!(local.num_key_value_heads, 2);
    assert_eq!(local.head_dim, 2);
}

#[test]
fn config_flag_without_mtp_tensors_does_not_enable_drafter() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("config.json"), config_value().to_string()).unwrap();
    write_single_tensor(
        &dir.path().join("model.safetensors"),
        "model.embed_tokens.weight",
    );
    assert!(!has_inkling_mtp_tensors(dir.path()).unwrap());
    assert_eq!(
        crate::drafter::resolve_drafter_kind(dir.path(), None).unwrap(),
        crate::drafter::DrafterKind::Dflash
    );
}

#[test]
fn actual_mtp_tensor_enables_detection() {
    let dir = tempfile::tempdir().unwrap();
    write_single_tensor(
        &dir.path().join("model.safetensors"),
        "model.mtp.layers.0.embed_norm.weight",
    );
    assert!(has_inkling_mtp_tensors(dir.path()).unwrap());
    std::fs::write(dir.path().join("config.json"), config_value().to_string()).unwrap();
    assert_eq!(
        crate::drafter::resolve_drafter_kind(dir.path(), None).unwrap(),
        crate::drafter::DrafterKind::Mtp
    );
}

#[test]
fn index_filter_opens_only_the_mtp_shard() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("target.safetensors"), b"not safetensors").unwrap();
    write_single_tensor(
        &dir.path().join("mtp.safetensors"),
        "model.mtp.layers.0.embed_norm.weight",
    );
    std::fs::write(
        dir.path().join("model.safetensors.index.json"),
        json!({
            "weight_map": {
                "model.embed_tokens.weight": "target.safetensors"
            }
        })
        .to_string(),
    )
    .unwrap();
    let weights = load_weights_from_dir_index_filtered(dir.path(), |name| {
        name.starts_with("model.mtp.layers.")
    })
    .unwrap();
    assert_eq!(weights.len(), 1);
    assert!(has_inkling_mtp_tensors(dir.path()).unwrap());
}

#[test]
fn sanitizer_maps_attention_and_splits_w13() {
    let mut weights = WeightMap::new();
    weights.insert(
        "model.mtp.layers.0.transformer_block.attn.wq_du.weight".into(),
        crate::zeros(&[8, 8], dtype::FLOAT32),
    );
    weights.insert(
        "model.mtp.layers.0.transformer_block.mlp.w13_dn.weight".into(),
        crate::zeros(&[32, 8], dtype::FLOAT32),
    );
    sanitize_weights(&mut weights).unwrap();
    assert!(weights.contains_key("blocks.0.transformer_block.self_attn.q_proj.weight"));
    assert_eq!(
        crate::array_shape(&weights["blocks.0.transformer_block.mlp.gate_proj.weight"]),
        [16, 8]
    );
    assert_eq!(
        crate::array_shape(&weights["blocks.0.transformer_block.mlp.up_proj.weight"]),
        [16, 8]
    );
}

#[test]
fn drafter_forward_shape_is_finite_after_five_token_prompt() {
    use crate::drafter::Drafter;

    let (mut drafter, target) = tiny_mtp_pair();
    drafter.bind(&target).unwrap();
    let tokens = [1, 2, 3, 4, 5];
    let hidden_values: Vec<f32> = (0..20).map(|index| index as f32 / 50.0).collect();
    let hidden = crate::from_slice_f32(&hidden_values, &[1, 5, 4]);
    let logits = drafter
        .forward_logits_for_test(&tokens, &hidden, 0)
        .unwrap();
    assert_eq!(crate::array_shape(&logits), [1, 1, 8]);
    let finite = crate::all_all(&crate::isfinite(&logits));
    crate::eval(&finite);
    assert!(crate::item_bool(&finite));
}

#[test]
fn flat_snapshot_restores_kv_and_all_four_convolution_states_exactly() {
    let mut original = InklingLayerCache::new();
    let keys = crate::from_slice_f32(&[1., 2., 3., 4., 5., 6.], &[1, 1, 3, 2]);
    let values = crate::from_slice_f32(&[6., 5., 4., 3., 2., 1.], &[1, 1, 3, 2]);
    let _ = original.kv.update_and_fetch(keys, values);
    for index in 0..4 {
        let ones = crate::ones(&[1, 2, 2], dtype::FLOAT32);
        original.conv[index] = Some(crate::multiply_scalar(&ones, index as f32 + 1.0));
    }
    let (expected_keys, expected_values) = original.kv.visible_state().unwrap();
    let expected_conv: Vec<_> = original
        .conv
        .iter()
        .map(|state| crate::copy(state.as_deref().unwrap()))
        .collect();
    let mut tensors = Vec::new();
    let mut scalars = Vec::new();
    original.capture_flat(&mut tensors, &mut scalars);

    let mut restored = InklingLayerCache::new();
    let mut tensors = tensors.into_iter();
    let mut scalar_index = 0;
    restored
        .restore_flat(&mut tensors, &scalars, &mut scalar_index)
        .unwrap();
    assert!(tensors.next().is_none());
    assert_eq!(scalar_index, 2);
    assert_eq!(restored.kv.offset, 3);
    let (actual_keys, actual_values) = restored.kv.visible_state().unwrap();
    assert_same(&actual_keys, &expected_keys);
    assert_same(&actual_values, &expected_values);
    for (actual, expected) in restored.conv.iter().zip(expected_conv) {
        assert_same(actual.as_deref().unwrap(), &expected);
    }
}
