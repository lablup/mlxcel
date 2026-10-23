// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
use super::*;

fn fixture() -> (WeightMap, Config) {
    let c = Config {
        embed_dim: 8,
        depth: 2,
        num_heads: 2,
        head_dim: 4,
        patch_size: 2,
        image_size: [4, 6],
        ffn_ratio: 2.,
        num_storage_tokens: 4,
        rope_periods: 1,
        ..Config::default()
    };
    let mut w = WeightMap::new();
    let mut put = |key: &str, dims: &[i32], value: f32| {
        w.insert(key.into(), mx::full_f32(dims, value, mx::dtype::BFLOAT16));
    };
    put("backbone.cls_token", &[1, 1, 8], 0.);
    put("backbone.storage_tokens", &[1, 4, 8], 0.);
    put("backbone.patch_embed.projection.weight", &[8, 2, 2, 3], 0.);
    put("backbone.rope_embed.periods", &[1], 1.);
    put("backbone.norm.weight", &[8], 1.);
    put("backbone.norm.bias", &[8], 0.);
    for i in 0..2 {
        let p = format!("backbone.blocks.{i}");
        for name in ["norm1", "norm2"] {
            put(&format!("{p}.{name}.weight"), &[8], 1.);
            put(&format!("{p}.{name}.bias"), &[8], 0.);
        }
        for name in ["ls1", "ls2"] {
            put(&format!("{p}.{name}.gamma"), &[8], 1.);
        }
        for name in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            put(&format!("{p}.attention.{name}.weight"), &[8, 8], 0.);
            if name != "k_proj" {
                put(&format!("{p}.attention.{name}.bias"), &[8], 0.);
            }
        }
        for (name, input, output) in [("w1", 8, 16), ("w2", 8, 16), ("w3", 16, 8)] {
            put(&format!("{p}.mlp.{name}.weight"), &[output, input], 0.);
            put(&format!("{p}.mlp.{name}.bias"), &[output], 0.);
        }
    }
    w.insert(
        "backbone.patch_embed.projection.bias".into(),
        mx::astype(
            &mx::from_slice_f32(&[1., -1., 1., -1., 1., -1., 1., -1.], &[8]),
            mx::dtype::BFLOAT16,
        ),
    );
    (w, c)
}

#[test]
fn sam3d_body_backbone_preserves_weights_and_f32_input_promotion() -> Result<()> {
    let (w, c) = fixture();
    let model = Backbone::from_weights(&w, &c)?;
    assert_eq!(mx::array_dtype(&model.projection), mx::dtype::BFLOAT16);
    let y = model.forward(&mx::zeros(&[1, 4, 6, 3], mx::dtype::FLOAT32))?;
    assert_eq!(mx::array_shape(&y), [1, 2, 3, 8]);
    assert_eq!(mx::array_dtype(&y), mx::dtype::FLOAT32);
    let normalized = 1. / (1. + 1e-5f32).sqrt();
    for (i, actual) in mx::utils::array_to_vec_f32(&y).iter().enumerate() {
        let expected = if i % 2 == 0 { normalized } else { -normalized };
        assert!((actual - expected).abs() < 1e-6);
    }
    Ok(())
}

#[test]
fn sam3d_body_backbone_rejects_wrong_channels_and_masked_key_bias() {
    let (mut w, c) = fixture();
    w.insert(
        "backbone.blocks.0.attention.k_proj.bias".into(),
        mx::zeros(&[8], mx::dtype::BFLOAT16),
    );
    assert!(Backbone::from_weights(&w, &c).is_err());
    w.remove("backbone.blocks.0.attention.k_proj.bias");
    w.insert(
        "backbone.blocks.1.norm2.weight".into(),
        mx::ones(&[7], mx::dtype::BFLOAT16),
    );
    assert!(Backbone::from_weights(&w, &c).is_err());
}

#[test]
fn sam3d_body_backbone_rejects_unconfigured_image_shapes() -> Result<()> {
    let (w, c) = fixture();
    let model = Backbone::from_weights(&w, &c)?;
    for dims in [&[1, 4, 6][..], &[1, 6, 4, 3], &[1, 4, 6, 1]] {
        assert!(model.forward(&mx::zeros(dims, mx::dtype::FLOAT32)).is_err());
    }
    assert!(
        model
            .forward(&mx::zeros(&[1, 4, 6, 3], mx::dtype::BFLOAT16))
            .is_err()
    );
    Ok(())
}
