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

//! Key mapping and metadata reading against upstream's `weights.py`.

use super::*;

#[test]
fn transformer_keys_follow_upstream_renames() {
    let cases = [
        (
            "transformer_blocks.0.img_mod.1.weight",
            "transformer_blocks.0.img_mod.linear.weight",
        ),
        (
            "transformer_blocks.3.txt_mod.1.bias",
            "transformer_blocks.3.txt_mod.linear.bias",
        ),
        (
            "transformer_blocks.1.img_mlp.net.0.proj.weight",
            "transformer_blocks.1.img_mlp.linear_in.weight",
        ),
        (
            "transformer_blocks.1.txt_mlp.net.2.bias",
            "transformer_blocks.1.txt_mlp.linear_out.bias",
        ),
        (
            "transformer_blocks.2.attn.to_out.0.weight",
            "transformer_blocks.2.attn.to_out.weight",
        ),
        (
            "transformer_blocks.2.attn.to_add_out.weight",
            "transformer_blocks.2.attn.to_add_out.weight",
        ),
        ("img_in.weight", "img_in.weight"),
    ];
    for (raw, want) in cases {
        assert_eq!(sanitize_transformer_key(raw), want);
    }
}

#[test]
fn vae_keys_follow_upstream_mapping() {
    let cases = [
        (
            "student.dconv_encoder.blocks.0.adaLN_modulation.1.weight",
            Some("dconv_encoder.blocks.0.adaLN_modulation.linear.weight"),
        ),
        (
            "student.dconv_encoder.head_blocks.1.ca.1.bias",
            Some("dconv_encoder.head_blocks.1.ca_conv.bias"),
        ),
        (
            "student.dconv_encoder.t_embedder.mlp.0.weight",
            Some("dconv_encoder.t_embedder.linear_1.weight"),
        ),
        (
            "pipeline.t_embedder.mlp.2.bias",
            Some("decoder_model.t_embedder.linear_2.bias"),
        ),
        (
            "pipeline.x_embedder.embedder.0.weight",
            Some("decoder_model.x_embedder.linear.weight"),
        ),
        (
            "pipeline.dec_net.res_blocks.2.mlp.0.weight",
            Some("decoder_model.dec_net.res_blocks.2.linear_1.weight"),
        ),
        (
            "pipeline.dec_net.res_blocks.12.mlp.2.bias",
            Some("decoder_model.dec_net.res_blocks.12.linear_2.bias"),
        ),
        (
            "pipeline.y_embedder.decoder.conv_in.weight",
            Some("decoder_model.y_embedder.decoder.conv_in.weight"),
        ),
        (
            "decoder_model.final_layer.linear.bias",
            Some("decoder_model.final_layer.linear.bias"),
        ),
        ("pipeline.y_embedder.encoder.conv_in.weight", None),
        ("teacher.blocks.0.weight", None),
        // `.mlp.0.` outside `dec_net.res_blocks.N` is left alone.
        (
            "pipeline.dec_net.mlp.0.weight",
            Some("decoder_model.dec_net.mlp.0.weight"),
        ),
    ];
    for (raw, want) in cases {
        assert_eq!(map_vae_key(raw).as_deref(), want, "{raw}");
    }
}

#[test]
fn vae_sanitize_drops_unread_tensors_and_permutes_torch_convs() {
    let mut raw = WeightMap::new();
    raw.insert(
        "pipeline.y_embedder.decoder.conv_in.weight".into(),
        mlxcel_core::zeros(&[384, 128, 3, 3], mlxcel_core::dtype::BFLOAT16),
    );
    raw.insert(
        "pipeline.y_embedder.decoder.norm_out.num_batches_tracked".into(),
        mlxcel_core::zeros(&[1], mlxcel_core::dtype::INT32),
    );
    raw.insert(
        "pipeline.y_embedder.encoder.conv_in.weight".into(),
        mlxcel_core::zeros(&[1, 1, 1, 1], mlxcel_core::dtype::BFLOAT16),
    );
    let layout = vae_layout(&raw, &Map::new()).unwrap();
    assert_eq!(layout, ConvLayout::Torch);
    let out = sanitize_vae_weights(raw, layout);
    assert_eq!(out.len(), 1);
    assert_eq!(
        mlxcel_core::array_shape(&out["decoder_model.y_embedder.decoder.conv_in.weight"]),
        [384, 3, 3, 128]
    );
    // Converted metadata (or an already-MLX conv) keeps the layout.
    let meta = serde_json::json!({"tensor_layout": "mlx_nhwc"});
    assert_eq!(
        vae_layout(&WeightMap::new(), meta.as_object().unwrap()).unwrap(),
        ConvLayout::Mlx
    );
    let mut mlx = WeightMap::new();
    mlx.insert(
        "decoder_model.y_embedder.decoder.conv_in.weight".into(),
        mlxcel_core::zeros(&[384, 3, 3, 128], mlxcel_core::dtype::BFLOAT16),
    );
    assert_eq!(vae_layout(&mlx, &Map::new()).unwrap(), ConvLayout::Mlx);
}

#[test]
fn component_metadata_merges_the_index_and_shard_headers() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("model.safetensors.index.json"),
        r#"{"metadata": {"mlx_vlm_format": "mage_flow", "bits": "8"}, "weight_map": {}}"#,
    )
    .unwrap();
    let mut weights = WeightMap::new();
    weights.insert(
        "a".into(),
        mlxcel_core::zeros(&[2], mlxcel_core::dtype::FLOAT32),
    );
    crate::imagegen::testing::write_bf16_safetensors(
        &dir.path().join("model.safetensors"),
        &weights,
    );
    let metadata = component_metadata(dir.path()).unwrap();
    assert_eq!(metadata["mlx_vlm_format"], "mage_flow");
    assert_eq!(metadata_int(&metadata, &["bits"]).unwrap(), Some(8));
    let bad = serde_json::json!({"bits": "eight"});
    assert!(metadata_int(bad.as_object().unwrap(), &["bits"]).is_err());
}
