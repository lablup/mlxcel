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

//! Florence-2 quantized-checkpoint parity against the mlx-vlm reference.
//!
//! Same shape as `tests/florence2_fusion_parity.rs` and driven by the same
//! deterministic synthetic pixel tensor and task prompt, with one difference
//! that matters: the checkpoint is `models/Florence-2-base-ft-4bit` and the
//! reference is upstream mlx-vlm running *that same 4-bit checkpoint*, not the
//! bf16 one.
//!
//! Picking the reference this way is the whole point. Quantization is lossy by
//! construction, so the bf16 activations are not a value a 4-bit run can
//! reproduce and comparing against them can only ever answer "how lossy",
//! never "is the graph right". Upstream on the same packed weights answers the
//! second question exactly: both runtimes dequantize identical bytes with
//! identical parameters, so anything beyond f16 rounding is a wiring defect in
//! this port. The tolerances below are therefore the same order as the dense
//! parity tests use, not quantization-sized.
//!
//! `florence2_quantization_cost_matches_mlx_vlm` closes the other half. It
//! measures how far the 4-bit checkpoint lands from the bf16 one and compares
//! that distance against the distance upstream measures between the same two
//! checkpoints. The first test says mlxcel computes what the reference
//! computes on packed weights; this one says the gap to dense is the
//! checkpoint's lossiness and not something this port introduced.
//!
//! Skips when the checkpoints are absent (CI has no Metal and no weights).
//!
//! To regenerate the pins: in a virtualenv holding `mlx` and `numpy`, with a
//! local checkout of <https://github.com/Blaizzy/mlx-vlm> on `sys.path` (the
//! `mlx_vlm` package entry and `transformers` can both be stubbed so the HF
//! processor shim is not required), build `ModelConfig` with
//! `image_feature_source`, `image_pos_embed`, and `visual_temporal_embedding`
//! lifted out of `vision_config` (upstream reads them from the top level,
//! every real checkpoint stores them nested), instantiate `florence2.Model`,
//! load `model.safetensors`, pass it through `VisionModel.sanitize` then
//! `Model.sanitize`, call `nn.quantize` with upstream's own class predicate
//! (`f"{p}.scales" in weights`, skipping modules whose weight size is not a
//! multiple of 64), and `load_weights`. Cast the synthetic pixels to
//! `mx.float16` before `_encode_image`: mlxcel casts pixel input to the
//! tower's own dtype, which comes off the dense conv weight, so an f32 input
//! on the reference side would compare two different activation precisions
//! rather than two implementations.

use std::path::Path;

use mlxcel::models::{Florence2Model, Florence2Quantization};

const QUANT_DIR: &str = "models/Florence-2-base-ft-4bit";
const DENSE_DIR: &str = "models/Florence-2-base-ft-bf16";

/// Extra bit widths smoke-tested for load when present, so the packing path
/// cannot become accidentally 4-bit-specific. `base-ft` and `large-ft` each
/// publish all four.
const OTHER_QUANTIZED_DIRS: &[&str] = &[
    "models/Florence-2-base-ft-8bit",
    "models/Florence-2-base-ft-6bit",
    "models/Florence-2-base-ft-3bit",
    "models/Florence-2-large-ft-8bit",
    // `large-ft` generates degenerate output (a run of BOS) on both the 4-bit
    // and the bf16 checkpoint, and upstream mlx-vlm reproduces that on the
    // same weights, so it is a property of that release rather than of this
    // load path. It stays in the smoke list because loading it still proves
    // the packing path is not `base-ft`-shaped: `d_model` 1024, twelve BART
    // layers, and `dim_embed` up to 2048.
    "models/Florence-2-large-ft-4bit",
];

const IMAGE_SIDE: i32 = 768;

/// `<s>What does the image describe?</s>` under the checkpoint tokenizer, the
/// task prompt Florence-2 uses for `<CAPTION>`-style captioning. Same prompt
/// the dense fusion parity test drives.
const PROMPT_IDS: &[i32] = &[0, 2264, 473, 5, 2274, 6190, 116, 2];

/// 1 spatially averaged token + 576 temporally averaged grid tokens.
const IMAGE_TOKENS: i32 = 577;
const D_MODEL: i32 = 768;

// Reference activations from upstream mlx-vlm on the same 4-bit checkpoint
// (f16 activations, f32 readout). The full digit strings are the exact
// f16-representable reference values.
#[allow(clippy::excessive_precision)]
const REF_IMAGE_FEATURES_FIRST16: &[f32] = &[
    -0.7451171875,
    0.7783203125,
    -0.0859375,
    -1.5341796875,
    1.19140625,
    0.28173828125,
    -0.02716064453125,
    -0.75146484375,
    0.2242431640625,
    -1.419921875,
    -0.4453125,
    -0.274169921875,
    -0.325439453125,
    0.384033203125,
    1.2900390625,
    -0.467041015625,
];
#[allow(clippy::excessive_precision)]
const REF_IMAGE_FEATURES_STATS: (f32, f32) = (-0.00416754, 0.81522191);

#[allow(clippy::excessive_precision)]
const REF_ENCODER_FIRST16: &[f32] = &[
    4.66796875,
    2.400390625,
    0.6220703125,
    -3.947265625,
    3.4609375,
    3.935546875,
    0.432373046875,
    0.64453125,
    -0.94287109375,
    -1.8203125,
    -1.4873046875,
    -1.1181640625,
    -6.40625,
    3.244140625,
    -0.6875,
    -0.77392578125,
];
#[allow(clippy::excessive_precision)]
const REF_ENCODER_STATS: (f32, f32) = (0.06744993, 3.56659889);

#[allow(clippy::excessive_precision)]
const REF_STEP0_LOGITS_FIRST16: &[f32] = &[
    20.9375,
    -5.8046875,
    4.71484375,
    -5.8046875,
    -1.705078125,
    -4.328125,
    -0.9140625,
    -0.01032257080078125,
    -0.8671875,
    -1.4609375,
    -4.234375,
    -1.6708984375,
    -1.6962890625,
    0.01312255859375,
    -3.21875,
    -0.80810546875,
];
#[allow(clippy::excessive_precision)]
const REF_STEP0_LOGITS_STATS: (f32, f32) = (-4.32664299, 1.71739292);

/// Greedy ids upstream produces on the 4-bit checkpoint for this prompt and
/// this synthetic input. Identical to the bf16 sequence
/// `tests/florence2_fusion_parity.rs` pins, so on this input the caption
/// survives 4-bit quantization token for token.
const REF_GENERATED: &[i32] = &[0, 879, 27740, 868];

/// How far upstream mlx-vlm's own 4-bit run lands from its own bf16 run on
/// this input, as `(relative RMS, cosine similarity)` per stage. Measured with
/// the same script that produced the pins above, on the same two checkpoints.
///
/// These are the quantization cost, and mlxcel has to pay exactly the same
/// one. Asserting mlxcel's measured divergence *against* these numbers is
/// stronger than asserting it against a hand-picked threshold: a threshold
/// only says "not absurd", while this says "the loss this port takes from
/// 4-bit weights is the loss the reference implementation takes".
const UPSTREAM_QUANT_VS_DENSE: &[(&str, f64, f64)] = &[
    ("image_features", 0.25540, 0.967458),
    ("encoder_hidden", 0.41583, 0.914181),
];

/// Tolerance on those two metrics.
///
/// Both implementations dequantize identical bytes and then run f16 arithmetic
/// in a different op order, so the metrics should agree to about one f16 ulp
/// relative. Observed on this box: relative RMS within 5.8e-4 and cosine
/// within 2.6e-4 at every stage. The bounds are roughly an order of magnitude
/// above that, which still leaves them far too tight for a wiring defect to
/// slip through: any of the failure modes this file guards against moves
/// cosine by 1e-1 or more.
const DIVERGENCE_RMS_TOL: f64 = 5e-3;
const DIVERGENCE_COS_TOL: f64 = 2e-3;

/// `x[0, c, h, w] = ((h * side + w) * 3 + c) % 251 / 251.0 - 0.5`, NCHW.
/// Identical to the dense parity tests, so the two runs differ only in the
/// checkpoint. Every operation is a single IEEE-754 f32 step over exactly
/// represented integers, so the Python reference reproduces it bit for bit.
fn synthetic_pixels(side: i32) -> Vec<f32> {
    let mut out = vec![0.0f32; (3 * side * side) as usize];
    for c in 0..3i64 {
        for h in 0..side as i64 {
            for w in 0..side as i64 {
                let raw = ((h * side as i64 + w) * 3 + c) % 251;
                let value = raw as f32 / 251.0f32 - 0.5f32;
                out[(c * side as i64 * side as i64 + h * side as i64 + w) as usize] = value;
            }
        }
    }
    out
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

fn assert_stats(values: &[f32], want: (f32, f32), tol: f32, what: &str) {
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
    let std = var.sqrt();
    let (ref_mean, ref_std) = want;
    eprintln!("{what}: mean {mean:.6} (ref {ref_mean:.6}), std {std:.6} (ref {ref_std:.6})");
    assert!(
        (mean as f32 - ref_mean).abs() <= tol,
        "{what} mean: got {mean}, reference {ref_mean} (tol {tol})"
    );
    assert!(
        (std as f32 - ref_std).abs() <= tol,
        "{what} std: got {std}, reference {ref_std} (tol {tol})"
    );
}

/// `||q - d|| / ||d||`, accumulated in f64 so a 450k-element tensor does not
/// lose the sum to f32 rounding.
fn relative_rms(quantized: &[f32], dense: &[f32]) -> f64 {
    assert_eq!(quantized.len(), dense.len(), "length mismatch");
    let mut diff_sq = 0.0f64;
    let mut ref_sq = 0.0f64;
    for (q, d) in quantized.iter().zip(dense) {
        let delta = *q as f64 - *d as f64;
        diff_sq += delta * delta;
        ref_sq += (*d as f64) * (*d as f64);
    }
    if ref_sq == 0.0 {
        return if diff_sq == 0.0 { 0.0 } else { f64::INFINITY };
    }
    (diff_sq / ref_sq).sqrt()
}

fn cosine_similarity(quantized: &[f32], dense: &[f32]) -> f64 {
    assert_eq!(quantized.len(), dense.len(), "length mismatch");
    let mut dot = 0.0f64;
    let mut qn = 0.0f64;
    let mut dn = 0.0f64;
    for (q, d) in quantized.iter().zip(dense) {
        let (q, d) = (*q as f64, *d as f64);
        dot += q * d;
        qn += q * q;
        dn += d * d;
    }
    if qn == 0.0 || dn == 0.0 {
        return 0.0;
    }
    dot / (qn.sqrt() * dn.sqrt())
}

#[test]
fn florence2_quantized_checkpoint_loads() {
    if !Path::new(QUANT_DIR).exists() {
        eprintln!("skipping florence2_quantized_parity load: {QUANT_DIR} not present");
        return;
    }

    let model = Florence2Model::load(Path::new(QUANT_DIR)).expect("load 4-bit florence2 model");
    let config = model.config();

    // Both halves have to carry the top-level quantization block: it sits
    // beside `text_config` / `vision_config` rather than inside either, so a
    // parser that only descended into the sub-object would silently leave both
    // at the dense default and mis-stride every dequantization.
    let expected = Florence2Quantization {
        group_size: 64,
        bits: 4,
    };
    assert_eq!(
        config.text.quantization, expected,
        "text-side quantization parameters"
    );
    assert_eq!(
        config.vision.quantization, expected,
        "vision-side quantization parameters"
    );

    // The activation dtype comes off the scales plane on a quantized
    // checkpoint. Reading it from `shared.weight` instead would return the
    // packed `uint32` type and make the decoder build an integer causal mask.
    assert_eq!(
        model.dtype(),
        mlxcel_core::dtype::FLOAT16,
        "activation dtype must be a float type, not the packed weight's uint32"
    );

    assert_eq!(config.text.d_model, D_MODEL);
    assert_eq!(config.vision.output_dim(), 1024);
}

#[test]
fn florence2_quantized_matches_mlx_vlm_reference() {
    if !Path::new(QUANT_DIR).exists() {
        eprintln!("skipping florence2_quantized_parity: {QUANT_DIR} not present");
        return;
    }

    let model = Florence2Model::load(Path::new(QUANT_DIR)).expect("load 4-bit florence2 model");
    let config = model.config();

    let pixels = synthetic_pixels(IMAGE_SIDE);
    let pixel_values = mlxcel_core::from_slice_f32(&pixels, &[1, 3, IMAGE_SIDE, IMAGE_SIDE]);

    // 1. DaViT tower + fusion projection. Every window and channel attention
    //    qkv / proj and every tower MLP is packed here; the conv stack, the
    //    layer norms, `image_projection`, and the cosine temporal buffer are
    //    not, so this stage alone exercises the mixed dense / quantized tower.
    let image_features = model.encode_image(&pixel_values).expect("encode image");
    assert_eq!(
        mlxcel_core::array_shape(&image_features),
        vec![1, IMAGE_TOKENS, D_MODEL],
        "image feature shape"
    );
    let image_values = to_vec_f32(&image_features);
    // Both sides run f16 with different op ordering. absmax here is 4.6, one
    // f16 ulp at that magnitude is 3.9e-3; the bound is a small multiple of
    // that and 80x below the tensor's own standard deviation.
    assert_close(
        &image_values[..16],
        REF_IMAGE_FEATURES_FIRST16,
        1e-2,
        "image_features first16",
    );
    assert_stats(
        &image_values,
        REF_IMAGE_FEATURES_STATS,
        2e-3,
        "image_features",
    );

    // 2. BART encoder over the fused image + prompt sequence: adds the
    //    quantized shared token table and the quantized encoder position
    //    table, gathered by `arange` rather than sliced.
    let encoder_hidden = model
        .encode(&pixel_values, PROMPT_IDS)
        .expect("fused encode");
    let fused_len = IMAGE_TOKENS + PROMPT_IDS.len() as i32;
    assert_eq!(
        mlxcel_core::array_shape(&encoder_hidden),
        vec![1, fused_len, D_MODEL],
        "encoder hidden state shape"
    );
    let encoder_values = to_vec_f32(&encoder_hidden);
    // absmax 22.3 here, so one f16 ulp is 1.6e-2.
    assert_close(
        &encoder_values[..16],
        REF_ENCODER_FIRST16,
        8e-2,
        "encoder_hidden_states first16",
    );
    assert_stats(&encoder_values, REF_ENCODER_STATS, 5e-3, "encoder_hidden");

    // 3. First decoder step: cross-attention against the cached encoder
    //    output, the quantized decoder position table, and the quantized LM
    //    head.
    let mut cache = model.make_cache();
    let start = mlxcel_core::from_slice_i32(&[config.text.decoder_start_token_id], &[1, 1]);
    let logits = model.decode(&start, &encoder_hidden, &mut cache);
    assert_eq!(
        mlxcel_core::array_shape(&logits),
        vec![1, 1, config.text.vocab_size],
        "decoder logits shape"
    );
    let logit_values = to_vec_f32(&logits);
    // absmax 21.0, one f16 ulp 1.6e-2.
    assert_close(
        &logit_values[..16],
        REF_STEP0_LOGITS_FIRST16,
        5e-2,
        "step0 logits first16",
    );
    assert_stats(&logit_values, REF_STEP0_LOGITS_STATS, 5e-3, "step0 logits");

    // 4. Whole greedy loop through the seq2seq pipeline.
    let generated = model
        .generate_greedy(&pixel_values, PROMPT_IDS, 8)
        .expect("greedy generation");
    assert_eq!(
        generated, REF_GENERATED,
        "greedy token ids must match the mlx-vlm reference on the same 4-bit weights"
    );
}

/// Check that the *cost* of quantization in mlxcel is the cost upstream pays.
///
/// The test above pins mlxcel's 4-bit activations against upstream's 4-bit
/// activations, which proves the two runtimes agree on the same weights. This
/// one measures how far those weights land from the bf16 ones and compares
/// that distance to the distance upstream measures between the same two
/// checkpoints. Together they close the loop: the first says mlxcel computes
/// what the reference computes, the second says the difference from dense is
/// the checkpoint's lossiness rather than anything this port introduced.
///
/// Cosine similarity is the discriminating metric. A wrong group size, a
/// wrong-rows position gather, or a mis-strided dequantization moves it by
/// 1e-1 or more; 4-bit reconstruction error and f16 op ordering move it by
/// 1e-4. The measured values are printed on every run, so a future MLX bump
/// that degrades the quantized kernels shows up as a number.
///
/// The image features are the loosest stage (cosine 0.967) because the DaViT
/// tower is pre-norm: twelve blocks of 4-bit projections accumulate along an
/// unnormalized residual stream before `image_proj_norm` renormalizes at the
/// end. The step-0 logits are the tightest (0.993), because the BART decoder
/// is post-norm and renormalizes after every sublayer.
#[test]
fn florence2_quantization_cost_matches_mlx_vlm() {
    if !Path::new(QUANT_DIR).exists() || !Path::new(DENSE_DIR).exists() {
        eprintln!(
            "skipping florence2_quantized_parity dense comparison: need both {QUANT_DIR} and {DENSE_DIR}"
        );
        return;
    }

    let quantized = Florence2Model::load(Path::new(QUANT_DIR)).expect("load 4-bit florence2 model");
    let dense = Florence2Model::load(Path::new(DENSE_DIR)).expect("load bf16 florence2 model");

    let pixels = synthetic_pixels(IMAGE_SIDE);
    let pixel_values = mlxcel_core::from_slice_f32(&pixels, &[1, 3, IMAGE_SIDE, IMAGE_SIDE]);

    let stages = [
        (
            "image_features",
            to_vec_f32(
                &quantized
                    .encode_image(&pixel_values)
                    .expect("quantized tower"),
            ),
            to_vec_f32(&dense.encode_image(&pixel_values).expect("dense tower")),
        ),
        (
            "encoder_hidden",
            to_vec_f32(
                &quantized
                    .encode(&pixel_values, PROMPT_IDS)
                    .expect("quantized encode"),
            ),
            to_vec_f32(
                &dense
                    .encode(&pixel_values, PROMPT_IDS)
                    .expect("dense encode"),
            ),
        ),
    ];

    for (what, quantized_values, dense_values) in &stages {
        let rms = relative_rms(quantized_values, dense_values);
        let cos = cosine_similarity(quantized_values, dense_values);
        let (_, ref_rms, ref_cos) = UPSTREAM_QUANT_VS_DENSE
            .iter()
            .find(|(name, _, _)| name == what)
            .copied()
            .unwrap_or_else(|| panic!("no upstream divergence pin for {what}"));
        eprintln!(
            "{what}: 4-bit vs bf16 relative RMS {rms:.5} (mlx-vlm {ref_rms:.5}), cosine {cos:.6} (mlx-vlm {ref_cos:.6})"
        );
        assert!(
            (rms - ref_rms).abs() <= DIVERGENCE_RMS_TOL,
            "{what}: mlxcel loses {rms} of the dense signal to 4-bit weights but mlx-vlm loses \
             {ref_rms} on the same two checkpoints (tol {DIVERGENCE_RMS_TOL})"
        );
        assert!(
            (cos - ref_cos).abs() <= DIVERGENCE_COS_TOL,
            "{what}: mlxcel's 4-bit direction agrees with dense at cosine {cos} but mlx-vlm's \
             agrees at {ref_cos} on the same two checkpoints (tol {DIVERGENCE_COS_TOL})"
        );
    }

    // The generated text is the only end-user-visible property, and on this
    // input it survives quantization exactly.
    let quantized_ids = quantized
        .generate_greedy(&pixel_values, PROMPT_IDS, 16)
        .expect("quantized greedy");
    let dense_ids = dense
        .generate_greedy(&pixel_values, PROMPT_IDS, 16)
        .expect("dense greedy");
    eprintln!("greedy ids: 4-bit {quantized_ids:?}, bf16 {dense_ids:?}");
    assert_eq!(
        quantized_ids, dense_ids,
        "the 4-bit and bf16 captions must agree token for token on this input"
    );
}

/// Load smoke test across the other published bit widths, so the packing path
/// cannot become accidentally 4-bit-specific. Every present directory is
/// exercised; absent ones are reported and skipped.
#[test]
fn florence2_other_quantized_bit_widths_load() {
    let present: Vec<&str> = OTHER_QUANTIZED_DIRS
        .iter()
        .copied()
        .filter(|dir| Path::new(dir).exists())
        .collect();
    if present.is_empty() {
        eprintln!(
            "skipping florence2_quantized_parity bit-width smoke: none of {OTHER_QUANTIZED_DIRS:?} present"
        );
        return;
    }

    for dir in present {
        let model = Florence2Model::load(Path::new(dir))
            .unwrap_or_else(|e| panic!("load quantized florence2 model {dir}: {e}"));
        let bits = model.config().text.quantization.bits;
        eprintln!("{dir}: loaded at {bits}-bit, dtype {}", model.dtype());
        assert_eq!(
            model.dtype(),
            mlxcel_core::dtype::FLOAT16,
            "{dir}: activation dtype must be a float type"
        );
        assert_eq!(
            model.config().vision.quantization,
            model.config().text.quantization,
            "{dir}: both halves must inherit the same top-level quantization block"
        );

        // One decoder step is enough to prove the packed planes actually reach
        // the kernels: a wrong bit width surfaces as a shape error inside the
        // first quantized matmul rather than as a load failure.
        let encoder = model
            .text_model()
            .encode_tokens(&mlxcel_core::from_slice_i32(
                PROMPT_IDS,
                &[1, PROMPT_IDS.len() as i32],
            ));
        let mut cache = model.make_cache();
        let start =
            mlxcel_core::from_slice_i32(&[model.config().text.decoder_start_token_id], &[1, 1]);
        let logits = model.decode(&start, &encoder, &mut cache);
        let logits = to_vec_f32(&logits);
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "{dir}: decoder logits must be finite"
        );
    }
}
