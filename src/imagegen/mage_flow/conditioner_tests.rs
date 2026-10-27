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

//! The conditioner's prompt and image preprocessing against upstream's
//! `MageFlowTextEncoder` and its numpy Qwen3-VL processor, and (ignored, GPU
//! and the real checkpoints) its hidden states.
//!
//! The token ids, the edit M-RoPE positions and the 64-channel pools of
//! upstream's hidden states (bf16 as shipped, and widened to f32) are
//! committed in `tests/fixtures/imagegen/mage_flow_parity.safetensors`;
//! `check_cond` says why the hidden states are gated against the f32
//! reference rather than at a fixed distance from the bf16 one. The full-size
//! oracle tensors (`pixel_values`, `cond`) are read from
//! `MLXCEL_MAGE_FLOW_ORACLE_DIR` when it is set (the `--out` directory of
//! `generate_mage_flow_fixtures.py real`).

use std::path::PathBuf;

use super::*;
use crate::imagegen::mage_flow::transformer::tests::{rel_rms, to_vec};
use mlxcel_core::weights::{WeightMap, load_safetensors};

pub(crate) const PROMPT: &str = "Editorial photograph of a cobalt teapot on a red table";
pub(crate) const EDIT_PROMPT: &str = "Blend the object from image 2 naturally into image 1";
pub(crate) const EDIT_REFS: [&str; 2] = [
    "assets/multiref_000000_0.jpg",
    "assets/multiref_000000_1.png",
];

pub(crate) fn model_dir(name: &str) -> Option<PathBuf> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("models/mlx")
        .join(name);
    root.join("text_encoder/tokenizer.json")
        .is_file()
        .then_some(root)
}

pub(crate) fn parity() -> WeightMap {
    load_safetensors(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/imagegen/mage_flow_parity.safetensors"),
    )
    .expect("fixture loads")
}

/// `<MLXCEL_MAGE_FLOW_ORACLE_DIR>/<case>.safetensors`, when the variable is set.
pub(crate) fn oracle(case: &str) -> Option<WeightMap> {
    let dir = std::env::var_os("MLXCEL_MAGE_FLOW_ORACLE_DIR")?;
    let path = PathBuf::from(dir).join(format!("{case}.safetensors"));
    Some(load_safetensors(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display())))
}

pub(crate) fn edit_references(root: &std::path::Path) -> Vec<DynamicImage> {
    EDIT_REFS
        .iter()
        .map(|p| image::open(root.join(p)).expect("reference opens"))
        .collect()
}

fn ids(a: &MlxArray) -> Vec<i32> {
    to_vec(a).into_iter().map(|v| v as i32).collect()
}

#[test]
fn templates_match_upstream() {
    let text = generation_text("a cat");
    assert!(text.starts_with("<|im_start|>system\nDescribe the image by detailing"));
    assert!(text.ends_with("<|im_start|>user\na cat<|im_end|>\n<|im_start|>assistant\n"));
    let text = edit_text("swap them", 2);
    assert!(text.contains(
        "<|im_start|>user\nImage 1: <|vision_start|><|image_pad|><|vision_end|>Image 2: \
         <|vision_start|><|image_pad|><|vision_end|>swap them<|im_end|>"
    ));
}

#[test]
fn smart_resize_follows_the_hf_rule() {
    const MIN: u64 = 65536;
    const MAX: u64 = 16_777_216;
    assert_eq!(smart_resize(288, 384, 32, MIN, MAX).unwrap(), (288, 384));
    // Rounded to the factor: 300 / 32 = 9.375.
    assert_eq!(smart_resize(300, 384, 32, MIN, MAX).unwrap(), (288, 384));
    // Below the minimum area: scaled up and rounded up.
    assert_eq!(smart_resize(100, 384, 32, MIN, MAX).unwrap(), (160, 512));
    // Above the maximum area: scaled down and rounded down; 62.5 rounds to even.
    assert_eq!(
        smart_resize(2000, 1000, 32, 56 * 56, 14 * 14 * 4 * 1280).unwrap(),
        (1408, 704)
    );
    let err = smart_resize(10, 2500, 32, MIN, MAX)
        .unwrap_err()
        .to_string();
    assert!(err.contains("aspect ratio"), "{err}");
}

#[test]
fn resize_long_edge_only_shrinks() {
    let image = RgbImage::new(1000, 500);
    assert_eq!(
        resize_long_edge(&image, Some(384)).unwrap().dimensions(),
        (384, 192)
    );
    let small = RgbImage::new(300, 200);
    assert_eq!(
        resize_long_edge(&small, Some(384)).unwrap().dimensions(),
        (300, 200)
    );
    assert_eq!(
        resize_long_edge(&image, None).unwrap().dimensions(),
        (1000, 500)
    );
    // 333 * 384 / 1000 = 127.872.
    let odd = RgbImage::new(1000, 333);
    assert_eq!(
        resize_long_edge(&odd, Some(384)).unwrap().dimensions(),
        (384, 128)
    );
}

#[test]
fn generation_ids_match_upstream() {
    let Some(root) = model_dir("mage-flow-turbo") else {
        eprintln!("skipping: models/mlx/mage-flow-turbo is absent");
        return;
    };
    let tokenizer = crate::tokenizer::load_tokenizer(&root.join("text_encoder")).unwrap();
    let got = generation_ids(&tokenizer, PROMPT).unwrap();
    assert_eq!(got, ids(&parity()["turbo.input_ids"]));
    // The template prefix is the 34 positions upstream drops.
    let prefix = generation_ids(&tokenizer, "").unwrap();
    assert_eq!(prefix.len(), GENERATION_DROP + 5);
    // Truncated at 2048 + 34 tokens.
    let long = "teapot ".repeat(3000);
    assert_eq!(
        generation_ids(&tokenizer, &long).unwrap().len(),
        MAX_SEQUENCE_LENGTH + GENERATION_DROP
    );
}

#[test]
fn edit_inputs_match_upstream() {
    let Some(root) = model_dir("mage-flow-edit-turbo") else {
        eprintln!("skipping: models/mlx/mage-flow-edit-turbo is absent");
        return;
    };
    let dir = root.join("text_encoder");
    let tokenizer = crate::tokenizer::load_tokenizer(&dir).unwrap();
    let (min_pixels, max_pixels) = pixel_bounds(&dir);
    assert_eq!((min_pixels, max_pixels), (65536, 16_777_216));
    let mut processor = Qwen2VLProcessor::new_with_norm(16, 2, 2, [0.5; 3], [0.5; 3]);
    processor.min_pixels = min_pixels as usize;
    processor.max_pixels = max_pixels as usize;
    let pre = EditPreprocessor {
        processor: &processor,
        tokenizer: &tokenizer,
        image_token_id: 151_655,
        min_pixels,
        max_pixels,
    };
    let images = edit_references(&root);
    let inputs = pre.inputs(EDIT_PROMPT, &images, Some(384)).unwrap();
    let want = parity();
    assert_eq!(inputs.input_ids, ids(&want["edit.input_ids"]));
    assert_eq!(inputs.grid_thw, [(1, 16, 24), (1, 24, 24)]);
    // The M-RoPE positions the language model reads, exactly.
    let len = inputs.input_ids.len() as i32;
    let positions = crate::multimodal::qwen_vl::compute_qwen_vl_mrope_position_ids(
        &mlxcel_core::from_slice_i32(&inputs.input_ids, &[1, len]),
        &inputs.grid_thw,
        2,
        151_655,
        151_656,
    );
    assert_eq!(mlxcel_core::array_shape(&positions), [3, 1, len]);
    assert_eq!(ids(&positions), ids(&want["edit.position_ids"]));
    if let Some(oracle) = oracle("edit") {
        let grid: Vec<i32> = ids(&oracle["image_grid_thw"]);
        let got: Vec<i32> = inputs
            .grid_thw
            .iter()
            .flat_map(|&(t, h, w)| [t, h, w])
            .collect();
        assert_eq!(got, grid);
        let (err, max) = pixel_drift(&inputs.pixel_values, &oracle["pixel_values"]);
        eprintln!("edit pixel_values rel-RMS {err:.3e}, max abs {max:.4}");
        // Our bicubic (`image` crate) and PIL's differ by up to two 8-bit
        // levels at a few pixels (measured 1.95e-3, max 2/127.5).
        assert!(err < 5e-3, "pixel_values rel-RMS {err}");
        assert!(max <= 2.5 / 127.5, "pixel_values max abs {max}");
    }
}

/// Our processor rows are `[N * tp, C * P * P]` (one row per temporal copy);
/// HF's are `[N, C * tp * P * P]`. Compares the two layouts element for
/// element: `(rel-RMS, max abs)`.
fn pixel_drift(ours: &MlxArray, theirs: &MlxArray) -> (f32, f32) {
    const TP: usize = 2;
    const C: usize = 3;
    const AREA: usize = 256;
    let (ours, theirs) = (to_vec(ours), to_vec(theirs));
    assert_eq!(ours.len(), theirs.len());
    let (mut num, mut den, mut max) = (0f64, 0f64, 0f32);
    for row in 0..theirs.len() / (TP * C * AREA) {
        let base = row * TP * C * AREA;
        for tp in 0..TP {
            for c in 0..C {
                for k in 0..AREA {
                    let a = ours[base + (tp * C + c) * AREA + k];
                    let b = theirs[base + (c * TP + tp) * AREA + k];
                    num += f64::from(a - b).powi(2);
                    den += f64::from(b).powi(2);
                    max = max.max((a - b).abs());
                }
            }
        }
    }
    ((num / den).sqrt() as f32, max)
}

/// Upstream `encode` on the real Qwen3-VL. GPU and about 9 GB.
#[test]
#[ignore = "loads the real Mage-Flow text encoder (GPU, ~9 GB)"]
fn generation_hidden_states_match_upstream() {
    let root = model_dir("mage-flow-turbo").expect("models/mlx/mage-flow-turbo");
    let conditioner = Conditioner::load(&root).unwrap();
    let cond = conditioner.encode(PROMPT).unwrap();
    check_cond(&cond, "turbo");
}

/// Upstream `encode_edit` on the real Qwen3-VL with two references.
#[test]
#[ignore = "loads the real Mage-Flow text encoder (GPU, ~9 GB)"]
fn edit_hidden_states_match_upstream() {
    let root = model_dir("mage-flow-edit-turbo").expect("models/mlx/mage-flow-edit-turbo");
    let conditioner = Conditioner::load(&root).unwrap();
    let cond = conditioner
        .encode_edit(EDIT_PROMPT, &edit_references(&root), Some(384))
        .unwrap();
    check_cond(&cond, "edit");
}

/// Mean over each group of 64 channels: `[1, L, 2560]` to `[1, L, 40]`.
pub(crate) fn pool64(cond: &MlxArray) -> UniquePtr<MlxArray> {
    let s = mlxcel_core::array_shape(cond);
    let x = mlxcel_core::astype(cond, mlxcel_core::dtype::FLOAT32);
    let x = mlxcel_core::reshape(&x, &[s[0], s[1], s[2] / 64, 64]);
    mlxcel_core::mean_axis(&x, -1, false)
}

/// Headroom over upstream's own bf16 distance from the f32 reference, for
/// kernel accumulation order (measured ratios 0.94 turbo, 1.04 edit).
const F32_DISTANCE_RATIO: f32 = 1.2;

/// The conditioner against upstream. A bf16 Qwen3-VL cannot reproduce
/// upstream's bf16 hidden states to 1e-3: one same-input decoder layer already
/// differs by 5.7e-3 between our MLX and the oracle's mlx 0.32.4 kernels, and
/// upstream's own bf16 run sits 1.8e-2 (turbo) / 2.6e-1 (edit, through the
/// 24-layer vision tower and DeepStack) from the same model widened to f32.
/// So the gate is relative to that f32 reference (`cond_f32_pool64`): the port
/// may be no further from it than upstream's bf16 run is, times
/// [`F32_DISTANCE_RATIO`]. The distance from upstream's bf16 hidden states is
/// printed as drift. The structural inputs (token ids, grid, M-RoPE positions,
/// text-row embeddings) are compared exactly elsewhere.
fn check_cond(cond: &MlxArray, case: &str) {
    let want = parity();
    let ours = pool64(cond);
    let f32_ref = &want[&format!("{case}.cond_f32_pool64")];
    let bf16_ref = &want[&format!("{case}.cond_pool64")];
    let ours_to_f32 = rel_rms(&ours, f32_ref);
    let upstream_to_f32 = rel_rms(bf16_ref, f32_ref);
    let ratio = ours_to_f32 / upstream_to_f32;
    eprintln!(
        "{case} cond_pool64: ours to f32 {ours_to_f32:.3e}, upstream bf16 to f32 \
         {upstream_to_f32:.3e} (ratio {ratio:.3}); ours to upstream bf16 {:.3e} (drift)",
        rel_rms(&ours, bf16_ref)
    );
    if let Some(oracle) = oracle(case) {
        eprintln!(
            "{case} cond: ours to upstream bf16 {:.3e} (drift)",
            rel_rms(cond, &oracle["cond"])
        );
    }
    assert!(
        ratio <= F32_DISTANCE_RATIO,
        "{case} conditioner is {ratio}x as far from the f32 reference as upstream's bf16 run"
    );
}

/// A reference kept at its own size (`vl_cond_long_edge` null or `<= 0`) is
/// still capped: the checkpoint's 4096x4096 processor ceiling becomes
/// 2048x2048, at most 4096 visual tokens per reference.
#[test]
fn reference_pixels_are_capped_below_the_checkpoint_ceiling() {
    let cap = reference_max_pixels(16_777_216);
    assert_eq!(cap, 2048 * 2048);
    assert_eq!(reference_max_pixels(1 << 20), 1 << 20);
    for (h, w) in [(4096, 4096), (16_000, 9_000), (4096, 1024)] {
        let (th, tw) = smart_resize(h, w, 32, 65536, cap).unwrap();
        assert!(u64::from(th) * u64::from(tw) <= cap, "{h}x{w} -> {th}x{tw}");
        assert!((th / 32) * (tw / 32) <= 4096, "{h}x{w} -> {th}x{tw}");
    }
}

/// An over-long edit instruction and an unusable reference shape are request
/// errors (`400`), raised before any reference reaches the vision tower.
#[test]
fn edit_request_bounds_are_invalid_requests() {
    let Some(root) = model_dir("mage-flow-edit-turbo") else {
        eprintln!("skipping: models/mlx/mage-flow-edit-turbo is absent");
        return;
    };
    let dir = root.join("text_encoder");
    let tokenizer = crate::tokenizer::load_tokenizer(&dir).unwrap();
    check_edit_prompt_length(&tokenizer, EDIT_PROMPT).unwrap();
    let long = "teapot ".repeat(MAX_SEQUENCE_LENGTH + 1);
    let err = check_edit_prompt_length(&tokenizer, &long).unwrap_err();
    assert!(
        err.downcast_ref::<InvalidImageRequest>().is_some(),
        "{err:#}"
    );

    let mut processor = Qwen2VLProcessor::new_with_norm(16, 2, 2, [0.5; 3], [0.5; 3]);
    let max_pixels = reference_max_pixels(16_777_216);
    processor.min_pixels = 65536;
    processor.max_pixels = max_pixels as usize;
    let pre = EditPreprocessor {
        processor: &processor,
        tokenizer: &tokenizer,
        image_token_id: 151_655,
        min_pixels: 65536,
        max_pixels,
    };
    let tiny = vec![DynamicImage::new_rgb8(64, 64)];
    let err = pre
        .inputs(&long, &tiny, Some(384))
        .err()
        .expect("over-long prompt");
    assert!(
        err.downcast_ref::<InvalidImageRequest>().is_some(),
        "{err:#}"
    );
    // A client-supplied vision marker would add an image position without
    // features.
    let two = vec![
        DynamicImage::new_rgb8(64, 64),
        DynamicImage::new_rgb8(64, 64),
    ];
    for marker in VISION_MARKERS {
        let prompt = format!("Image 3: {marker} make it blue");
        let err = pre.inputs(&prompt, &two, Some(384)).err().expect("marker");
        assert!(
            err.downcast_ref::<InvalidImageRequest>().is_some(),
            "{err:#}"
        );
    }
    // 201:1 is past the Qwen3-VL processor's aspect limit.
    let sliver = vec![DynamicImage::new_rgb8(2010, 10)];
    let err = pre
        .inputs(EDIT_PROMPT, &sliver, None)
        .err()
        .expect("aspect ratio");
    assert!(
        err.downcast_ref::<InvalidImageRequest>().is_some(),
        "{err:#}"
    );
}
