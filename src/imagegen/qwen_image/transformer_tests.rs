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

//! The tiny transformer, its RoPE layout and its reference KV cache against
//! upstream `QwenImageTransformer` (`tests/fixtures/imagegen/qwen_image_tiny_refs.safetensors`,
//! written by `generate_qwen_image_fixtures.py tiny`), plus the KV-cache
//! on/off equivalence of one block.
//!
//! The fixture stores the tiny model's bf16 weights under their Diffusers
//! checkpoint names (`time_text_embed.timestep_embedder.*`, `modulation.1.*`),
//! so [`remap_transformer_key`] is exercised too. Inputs come from the
//! closed-form [`det`] generator both sides share.

use super::*;
use crate::imagegen::qwen_image::blocks::{rope_positions, rope_tables};
use crate::imagegen::qwen_image::kv_cache::KvCache;
use crate::imagegen::qwen_image::parity_tests::tiny_refs;
use crate::imagegen::testing::{det, rel_rms, values};

fn tiny_config() -> TransformerConfig {
    TransformerConfig {
        num_layers: 2,
        num_heads: 2,
        head_dim: 16,
        in_channels: 8,
        out_channels: 8,
        context_in_dim: 12,
        mlp_ratio: 3,
        axes_dims_rope: [4, 6, 6],
        eps: 1e-6,
        causal_condition: true,
    }
}

fn tiny_weights(refs: &WeightMap) -> WeightMap {
    refs.iter()
        .filter_map(|(k, v)| {
            k.strip_prefix("tf.w.")
                .map(|k| (remap_transformer_key(k), mlxcel_core::copy(v)))
        })
        .collect()
}

fn tiny_model(refs: &WeightMap) -> QwenImageTransformer {
    QwenImageTransformer::from_weights(tiny_config(), &tiny_weights(refs), None)
        .expect("tiny transformer builds")
}

fn bf16(a: UniquePtr<MlxArray>) -> UniquePtr<MlxArray> {
    mlxcel_core::astype(&a, dtype::BFLOAT16)
}

fn timestep(v: f32) -> UniquePtr<MlxArray> {
    bf16(mlxcel_core::from_slice_f32(&[v], &[1]))
}

/// Vision slots 3..7 of the 10-token edit prompt.
fn edit_mask() -> Vec<bool> {
    (0..10).map(|i| (3..7).contains(&i)).collect()
}

/// bf16 outputs of a 2-block model agree with upstream to bf16 rounding of
/// differently fused ops; a real layout or modulation bug is far above this.
const TINY_REL_RMS: f64 = 2e-3;

fn check(name: &str, ours: &MlxArray, refs: &WeightMap) {
    let rel = rel_rms(&values(ours), &values(&refs[name]));
    eprintln!("{name}: rel-RMS {rel:.3e}");
    assert!(rel < TINY_REL_RMS, "{name}: rel-RMS {rel:.3e}");
}

#[test]
fn checkpoint_keys_remap_onto_module_names() {
    assert_eq!(
        remap_transformer_key("time_text_embed.timestep_embedder.linear_1.weight"),
        "time_text_embed.linear_1.weight"
    );
    assert_eq!(
        remap_transformer_key("modulation.1.weight"),
        "modulation.0.weight"
    );
    // Idempotent on an mlx_format checkpoint.
    assert_eq!(
        remap_transformer_key("modulation.0.weight"),
        "modulation.0.weight"
    );
}

#[test]
fn rope_layouts_match_upstream() {
    let refs = tiny_refs();
    for (name, seq, blocks) in [
        ("rope.edit", 34, vec![(3, 4, 4), (22, 2, 6)]),
        ("rope.tail", 20, vec![(2, 3, 5)]),
    ] {
        let (cos, sin) = rope_tables(&rope_positions(seq, &blocks), [4, 6, 6]);
        for (ours, key) in [(cos, "cos"), (sin, "sin")] {
            let (a, b) = (values(&ours), values(&refs[&format!("{name}.{key}")]));
            let max = a
                .iter()
                .zip(&b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0, f32::max);
            assert!(max < 1e-6, "{name}.{key}: max abs {max}");
        }
    }
}

#[test]
fn generate_forward_matches_upstream() {
    let refs = tiny_refs();
    let model = tiny_model(&refs);
    let out = model
        .forward(
            &bf16(det("qit.gen_img", &[1, 24, 8], 1.0, 0.0)),
            &bf16(det("qit.gen_txt", &[1, 5, 12], 1.0, 0.0)),
            &timestep(0.875),
            (1, 4, 6),
            None,
            None,
        )
        .unwrap();
    check("tf.gen.out", &out, &refs);
}

struct EditCase {
    model: QwenImageTransformer,
    joint: UniquePtr<MlxArray>,
    target: UniquePtr<MlxArray>,
    text: UniquePtr<MlxArray>,
}

fn edit_case(refs: &WeightMap) -> EditCase {
    let reference = bf16(det("qit.ref_img", &[1, 16, 8], 1.0, 0.0));
    let target = bf16(det("qit.edit_img", &[1, 12, 8], 1.0, 0.0));
    EditCase {
        model: tiny_model(refs),
        joint: mlxcel_core::concatenate(&reference, &target, 1),
        target,
        text: bf16(det("qit.edit_txt", &[1, 10, 12], 1.0, 0.0)),
    }
}

#[test]
fn edit_forward_with_and_without_the_kv_cache_matches_upstream() {
    let refs = tiny_refs();
    let case = edit_case(&refs);
    let mask = edit_mask();
    let layout = EditLayout {
        reference_shapes: &[(1, 4, 4)],
        image_pad_mask: &mask,
    };
    let run = |hidden: &MlxArray, t: f32, cache: Option<(&mut KvCache, KvMode)>| {
        case.model
            .forward(
                hidden,
                &case.text,
                &timestep(t),
                (1, 2, 6),
                Some(&layout),
                cache,
            )
            .unwrap()
    };
    check("tf.edit.nokv0", &run(&case.joint, 0.875, None), &refs);
    check("tf.edit.nokv1", &run(&case.joint, 0.5, None), &refs);
    let mut cache = KvCache::new(2);
    let kv0 = run(&case.joint, 0.875, Some((&mut cache, KvMode::Extract)));
    check("tf.edit.kv0", &kv0, &refs);
    assert_eq!(cache.prefix_len, 22);
    let kv1 = run(&case.target, 0.5, Some((&mut cache, KvMode::Cached)));
    check("tf.edit.kv1", &kv1, &refs);
    // A cached call must keep the extracted target layout.
    let wrong = mlxcel_core::slice(&case.target, &[0, 0, 0], &[1, 6, 8]);
    assert!(
        case.model
            .forward(
                &wrong,
                &case.text,
                &timestep(0.5),
                (1, 1, 6),
                Some(&layout),
                Some((&mut cache, KvMode::Cached))
            )
            .is_err()
    );
}

#[test]
fn edit_layout_errors_like_upstream() {
    let refs = tiny_refs();
    let case = edit_case(&refs);
    let t = timestep(0.5);
    let forward = |mask: &[bool], shapes: &[ImageShape]| {
        case.model.forward(
            &case.joint,
            &case.text,
            &t,
            (1, 2, 6),
            Some(&EditLayout {
                reference_shapes: shapes,
                image_pad_mask: mask,
            }),
            None,
        )
    };
    // Three slots for a 4x4 reference (which needs four).
    let short: Vec<bool> = (0..10).map(|i| (3..6).contains(&i)).collect();
    assert!(forward(&short, &[(1, 4, 4)]).is_err());
    // No slots at all.
    assert!(forward(&[false; 10], &[(1, 4, 4)]).is_err());
    // A mask of the wrong length.
    assert!(forward(&edit_mask()[..9], &[(1, 4, 4)]).is_err());
}

/// The KV-cache on/off equivalence of one block: the target rows of a full
/// block-causal pass equal an `Extract` pass followed by a `Cached` pass of
/// the target alone.
#[test]
fn one_block_with_the_kv_cache_equals_the_full_sequence() {
    use crate::imagegen::qwen_image::blocks::Block;
    use crate::imagegen::qwen_image::kv_cache::KvLayer;

    let refs = tiny_refs();
    let weights = tiny_weights(&refs);
    let block = Block::load(&weights, "transformer_blocks.0", 2, 16, 1e-6, None).unwrap();
    let (seq, target) = (22, 8);
    let prefix = seq - target;
    let x = bf16(det("qit.block_x", &[1, seq, 32], 1.0, 0.0));
    let modulation = bf16(det("qit.block_mod", &[2, 4 * 32], 0.5, 0.0));
    let ids: Vec<i32> = (0..seq).map(|i| if i < prefix { -1 } else { 0 }).collect();
    let blocks = [(prefix, 2, 4)];
    let (cos, sin) = rope_tables(&rope_positions(seq, &blocks), [4, 6, 6]);
    let mask = block_causal_mask(&ids);
    let target_mask = bool_mask(&ids.iter().map(|&i| i == 0).collect::<Vec<_>>());
    let full = block
        .forward(
            &x,
            &modulation,
            &cos,
            &sin,
            Some(&mask),
            Some(&target_mask),
            None,
        )
        .unwrap();
    let rows = |a: &MlxArray, from: i32| {
        let s = mlxcel_core::array_shape(a);
        mlxcel_core::slice(a, &[from, 0], &[s[0], s[1]])
    };
    let mut layer = KvLayer::default();
    block
        .forward(
            &x,
            &modulation,
            &cos,
            &sin,
            Some(&mask),
            Some(&target_mask),
            Some(CacheUse {
                layer: &mut layer,
                mode: KvMode::Extract,
                prefix_len: prefix,
            }),
        )
        .unwrap();
    let cached = block
        .forward(
            &take_tokens(&x, prefix, seq),
            &modulation,
            &rows(&cos, prefix),
            &rows(&sin, prefix),
            None,
            Some(&mlxcel_core::ones(&[target], dtype::BOOL)),
            Some(CacheUse {
                layer: &mut layer,
                mode: KvMode::Cached,
                prefix_len: prefix,
            }),
        )
        .unwrap();
    let (a, b) = (values(&cached), values(&take_tokens(&full, prefix, seq)));
    let max = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max);
    let scale = b.iter().map(|v| v.abs()).fold(0.0, f32::max);
    eprintln!("one block, cached vs full: max abs {max} (scale {scale})");
    assert!(
        max <= scale * 1e-3,
        "cached vs full: max abs {max} (scale {scale})"
    );
    // Without the cached prefix the target sees only itself, which differs.
    let alone = block
        .forward(
            &take_tokens(&x, prefix, seq),
            &modulation,
            &rows(&cos, prefix),
            &rows(&sin, prefix),
            None,
            Some(&mlxcel_core::ones(&[target], dtype::BOOL)),
            None,
        )
        .unwrap();
    assert!(rel_rms(&values(&alone), &b) > 1e-2);
}
