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

//! Florence-2 processor parity: preprocessing, tokenization, and the whole
//! task path from a real image to structured coordinates.
//!
//! Where `tests/florence2_fusion_parity.rs` pins the fused model on a
//! synthetic pixel tensor, this file closes the loop on a real image:
//! `tests/fixtures/test_image.png` goes in as a task marker plus a picture,
//! and boxes and quadrilaterals in original-image pixels come out.
//!
//! The reference is the same checkpoint driven through Python: image
//! preprocessing by the checkpoint's own `CLIPImageProcessor` configuration,
//! tokenization by its `BartTokenizerFast`, and generation by the upstream
//! mlx-vlm florence2 `Model`
//! (https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/florence2/florence2.py)
//! with weights cast bf16 -> f16 to match mlxcel's Apple Silicon precision
//! policy. Skips when the checkpoint is absent (CI has no Metal and no
//! weights).
//!
//! Two things are needed to reproduce the reference run, both of which cost
//! real time to rediscover:
//!
//! - Upstream's `ModelConfig` reads `image_feature_source`, `image_pos_embed`
//!   and `visual_temporal_embedding` from the top level of `config.json`, but
//!   every real checkpoint stores them inside `vision_config`, and the
//!   dataclass default for `image_feature_source` is the reverse order. They
//!   have to be passed explicitly from `vision_config`.
//! - mlx's `nn.Module` starts in training mode and upstream's DaViT
//!   `DropPath` only short-circuits when `not self.training`, so without
//!   `model.eval()` the vision tower applies stochastic depth at inference
//!   and the output is not reproducible.

use std::path::{Path, PathBuf};

use mlxcel::models::{
    Florence2ImageSize, Florence2Model, Florence2Processor, Florence2Task, Florence2TaskResult,
};
use mlxcel::vision::processors::florence2::Florence2ImageProcessor;

const MODEL_DIR: &str = "models/Florence-2-base-ft-bf16";

/// The committed fixture: 224x224 8-bit RGB. Upscaled to 768x768 by the
/// processor, which is the interesting direction here because it is what makes
/// the location bins land on a 224-pixel extent.
const FIXTURE_SIZE: (u32, u32) = (224, 224);

/// Reference pixels from the checkpoint's `CLIPImageProcessor`
/// (`CLIPImageProcessorPil` backend, so PIL BICUBIC). All sixteen leading
/// values are the same saturated red-channel corner:
/// `(255/255 - 0.485) / 0.229`.
#[allow(clippy::excessive_precision)]
const REF_PIXELS_FIRST16: &[f32] = &[2.248908281326294; 16];

/// Mean and standard deviation over all 1,769,472 values.
#[allow(clippy::excessive_precision)]
const REF_PIXELS_STATS: (f32, f32) = (0.343636691570282, 1.3729557991027832);

/// Tokenized task prompts: `<s>` + the expanded English sentence + `</s>`.
const REF_PROMPT_IDS: &[(Florence2Task, &[i32])] = &[
    (
        Florence2Task::Caption,
        &[0, 2264, 473, 5, 2274, 6190, 116, 2],
    ),
    (
        Florence2Task::ObjectDetection,
        &[0, 574, 22486, 5, 8720, 19, 4120, 766, 11, 5, 2274, 4, 2],
    ),
    (
        Florence2Task::OcrWithRegion,
        &[0, 2264, 16, 5, 2788, 11, 5, 2274, 6, 19, 3806, 116, 2],
    ),
    (
        Florence2Task::Ocr,
        &[0, 2264, 16, 5, 2788, 11, 5, 2274, 116, 2],
    ),
    (
        Florence2Task::DenseRegionCaption,
        &[0, 574, 22486, 5, 8720, 11, 5, 2274, 6, 19, 49, 24173, 4, 2],
    ),
    (
        Florence2Task::RegionProposal,
        &[0, 574, 22486, 5, 976, 5327, 11, 5, 2274, 4, 2],
    ),
];

/// Answers the reference produced for the fixture, decoded with special
/// tokens kept. `<CAPTION>` says "unanswerable" and `<OCR>` reads a stray "0";
/// those are the honest outputs for this small synthetic-looking image and the
/// point is that both runtimes produce the same string, not that the string is
/// impressive.
const REF_ANSWERS: &[(Florence2Task, &str)] = &[
    (Florence2Task::Caption, "<s>unanswerable"),
    (
        Florence2Task::ObjectDetection,
        "<s>poster<loc_0><loc_0><loc_998><loc_998>",
    ),
    (
        Florence2Task::OcrWithRegion,
        "<s>0:00 PM<loc_0><loc_999><loc_76><loc_999><loc_76><loc_999><loc_0><loc_999>",
    ),
    (Florence2Task::Ocr, "<s>0"),
    (Florence2Task::DenseRegionCaption, "<s>orange square"),
    (
        Florence2Task::RegionProposal,
        "<s><loc_0><loc_0><loc_998><loc_998>",
    ),
];

/// Bins dequantized against the fixture's own 224x224 extent, from upstream's
/// `post_process_generation`. `<loc_998>` on a 224-pixel axis is
/// `(998 + 0.5) * 0.224`.
#[allow(clippy::excessive_precision)]
const REF_OD_BOX: [f32; 4] = [
    0.1120000034570694,
    0.1120000034570694,
    223.66400146484375,
    223.66400146484375,
];

#[allow(clippy::excessive_precision)]
const REF_OCR_QUAD: [f32; 8] = [
    0.1120000034570694,
    223.88800048828125,
    17.13599967956543,
    223.88800048828125,
    17.13599967956543,
    223.88800048828125,
    0.1120000034570694,
    223.88800048828125,
];

const MAX_NEW_TOKENS: usize = 96;

fn skip() -> bool {
    if Path::new(MODEL_DIR).exists() {
        return false;
    }
    eprintln!("skipping florence2_processor_parity: {MODEL_DIR} not present");
    true
}

fn fixture() -> image::DynamicImage {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/test_image.png");
    image::open(&path).expect("load tests/fixtures/test_image.png")
}

fn to_vec_f32(a: &mlxcel_core::MlxArray) -> Vec<f32> {
    let a = mlxcel_core::astype(a, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&a);
    mlxcel_core::array_to_raw_bytes(&a)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn assert_close(got: &[f32], want: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length mismatch");
    let mut worst = 0.0f32;
    let mut worst_at = 0usize;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let dev = (g - w).abs();
        if dev > worst {
            worst = dev;
            worst_at = i;
        }
    }
    eprintln!("{what}: max abs deviation {worst} at index {worst_at} (tol {tol})");
    assert!(
        worst <= tol,
        "{what}[{worst_at}]: got {}, reference {} (deviation {worst}, tol {tol})",
        got[worst_at],
        want[worst_at]
    );
}

/// Preprocessing parity on the real fixture.
///
/// The tolerance is generous relative to what was measured: comparing all
/// 1,769,472 values against the Python reference gave a maximum absolute
/// deviation of exactly 0.0, so the `image` crate's `CatmullRom` filter and
/// PIL's `BICUBIC` agree bit for bit on this 224 -> 768 upscale. The bound is
/// kept loose anyway because that agreement is a property of two independent
/// resampling implementations and is not guaranteed to survive a version bump
/// on either side.
#[test]
fn preprocessing_matches_the_checkpoint_processor() {
    if skip() {
        return;
    }
    let processor =
        Florence2ImageProcessor::from_pretrained(Path::new(MODEL_DIR)).expect("load processor");
    assert_eq!((processor.width, processor.height), (768, 768));

    let processed = processor.preprocess_with_sizes(std::slice::from_ref(&fixture()));
    assert_eq!(
        mlxcel_core::array_shape(&processed.pixel_values),
        vec![1, 3, 768, 768],
        "pixel tensor shape"
    );
    // The pre-resize extent, which is what the location bins are relative to.
    assert_eq!(processed.original_sizes, vec![FIXTURE_SIZE]);

    let values = to_vec_f32(&processed.pixel_values);
    assert_eq!(values.len(), 3 * 768 * 768, "one image, three planes");
    assert_close(
        &values[..16],
        REF_PIXELS_FIRST16,
        1e-4,
        "pixel_values first16",
    );

    let n = values.len() as f64;
    let mean = values.iter().map(|v| *v as f64).sum::<f64>() / n;
    let var = values
        .iter()
        .map(|v| {
            let d = *v as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / n;
    let (ref_mean, ref_std) = REF_PIXELS_STATS;
    eprintln!(
        "pixel_values: mean {mean:.6} (ref {ref_mean:.6}), std {:.6} (ref {ref_std:.6})",
        var.sqrt()
    );
    assert!((mean as f32 - ref_mean).abs() <= 1e-4, "pixel mean {mean}");
    assert!(
        (var.sqrt() as f32 - ref_std).abs() <= 1e-4,
        "pixel std {}",
        var.sqrt()
    );
}

/// Tokenizer parity. Unlike the pixel comparison this one is exact: the same
/// vocabulary and the same merges have to produce identical ids.
#[test]
fn task_prompts_tokenize_exactly_as_the_reference() {
    if skip() {
        return;
    }
    let processor = Florence2Processor::from_pretrained(Path::new(MODEL_DIR)).expect("load");
    for (task, want) in REF_PROMPT_IDS {
        let ids = processor.encode_prompt(*task, None).expect("encode prompt");
        assert_eq!(ids, want.to_vec(), "{task} prompt ids");
    }
}

/// The whole path the acceptance criteria describe: a task marker and an image
/// go in, a structured result comes out, and both halves match the reference.
///
/// Loads the model once and drives every task against it, because loading the
/// checkpoint dominates the runtime.
#[test]
fn task_runs_match_the_reference_end_to_end() {
    if skip() {
        return;
    }
    let model = Florence2Model::load(Path::new(MODEL_DIR)).expect("load model");
    let processor = Florence2Processor::from_pretrained(Path::new(MODEL_DIR)).expect("load");
    let image = fixture();

    for (task, want_text) in REF_ANSWERS {
        let output = processor
            .run(&model, *task, None, &image, MAX_NEW_TOKENS)
            .expect("run task");
        eprintln!("{task}: {:?}", output.raw_text);
        assert_eq!(&output.raw_text, want_text, "{task} answer text");
    }
}

/// Detection and OCR parse into the coordinates upstream computes, in
/// original-image pixels rather than in the 768x768 resized frame.
#[test]
fn detection_and_ocr_parse_into_reference_coordinates() {
    if skip() {
        return;
    }
    let model = Florence2Model::load(Path::new(MODEL_DIR)).expect("load model");
    let processor = Florence2Processor::from_pretrained(Path::new(MODEL_DIR)).expect("load");
    let image = fixture();

    let od = processor
        .run(
            &model,
            Florence2Task::ObjectDetection,
            None,
            &image,
            MAX_NEW_TOKENS,
        )
        .expect("run <OD>");
    let Florence2TaskResult::Boxes { boxes, labels } = od.result else {
        panic!("<OD> must produce boxes, got {:?}", od.result);
    };
    assert_eq!(labels, vec!["poster".to_string()]);
    assert_close(&boxes[0].to_array(), &REF_OD_BOX, 1e-4, "<OD> box");
    // A box in the resized 768 frame would run to roughly 767, not 224.
    assert!(
        boxes[0].xmax <= FIXTURE_SIZE.0 as f32,
        "box must be in original-image pixels, got {:?}",
        boxes[0]
    );

    let ocr = processor
        .run(
            &model,
            Florence2Task::OcrWithRegion,
            None,
            &image,
            MAX_NEW_TOKENS,
        )
        .expect("run <OCR_WITH_REGION>");
    let Florence2TaskResult::QuadBoxes { quad_boxes, labels } = ocr.result else {
        panic!("<OCR_WITH_REGION> must produce quads, got {:?}", ocr.result);
    };
    assert_eq!(labels, vec!["0:00 PM".to_string()]);
    assert_close(&quad_boxes[0].points, &REF_OCR_QUAD, 1e-4, "OCR quad");

    // The same answer parsed against a different declared size must move,
    // which is what proves the original extent is actually threaded through.
    let scaled = Florence2TaskResult::parse(
        &od.raw_text,
        Florence2Task::ObjectDetection,
        Florence2ImageSize::new(1000, 1000),
    );
    let Florence2TaskResult::Boxes { boxes: scaled, .. } = scaled else {
        panic!("expected boxes");
    };
    assert!(scaled[0].xmax > 900.0, "got {:?}", scaled[0]);
}
