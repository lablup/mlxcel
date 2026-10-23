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

//! Opt-in authentic Falcon-OCR prefill/decode boundary capture.
//!
//! The hooks in `FalconOcrTextModel::forward_impl` are test-only. They record
//! the arrays and positional/cache metadata that the direct model and
//! `DirectEngine` actually consume, so a runtime regression can be localized
//! without changing the production graph.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use mlxcel_core::engine::{DirectEngine, DirectRequest};
use mlxcel_core::generate::{LanguageModel, SamplingConfig};
use mlxcel_core::layers::KVCache;
use mlxcel_core::sampling::TokenBiasMap;
use mlxcel_core::{MlxArray, UniquePtr};
use safetensors::Dtype;
use safetensors::tensor::TensorView;
use serde::Serialize;

use crate::LoadedModel;
use crate::vision::falcon_ocr_layout::{MIN_CROP_DIM, OcrCategory, crop_region};
use crate::vlm_runtime::{prepare_vlm_embeddings_with_budget, prepared_embedding_refs};

#[path = "falcon_ocr_diagnostic_reference.rs"]
mod reference;
use reference::{patchify_nhwc, reference_tensor_f32, reference_tensor_i64};

const OUTPUT_DIR_ENV: &str = "MLXCEL_TEST_FALCON_DIAGNOSTIC_DIR";
const REFERENCE_ENV: &str = "MLXCEL_TEST_FALCON_TITLE_REFERENCE";
const CHECKPOINT_ENV: &str = "MLXCEL_TEST_FALCON_OCR_DIR";

#[derive(Serialize)]
struct CallRecord {
    input_ids: Vec<i32>,
    input_shape: Vec<i32>,
    embeddings_shape: Option<Vec<i32>>,
    mask_shape: Option<Vec<i32>>,
    positions: Vec<i32>,
    sequence_id_present: bool,
    cache_offsets_before: Vec<i32>,
    cache_offsets_after: Vec<i32>,
    position_delta_from_cache: Option<i32>,
    last_logits_shape: Vec<i32>,
    top_token: i32,
    top_logit: f32,
}

struct CaptureState {
    tensor_path: PathBuf,
    receipt_path: PathBuf,
    tensors: Vec<(String, UniquePtr<MlxArray>)>,
    calls: Vec<CallRecord>,
}

thread_local! {
    static ACTIVE: RefCell<Option<CaptureState>> = const { RefCell::new(None) };
}

struct CaptureScope;

impl CaptureScope {
    fn begin(root: &Path, label: &str) -> Self {
        let tensor_path = root.join(format!("{label}.safetensors"));
        let receipt_path = root.join(format!("{label}.json"));
        assert!(
            !tensor_path.exists(),
            "refusing to overwrite {}",
            tensor_path.display()
        );
        assert!(
            !receipt_path.exists(),
            "refusing to overwrite {}",
            receipt_path.display()
        );
        ACTIVE.with(|active| {
            let mut active = active.borrow_mut();
            assert!(
                active.is_none(),
                "Falcon diagnostic capture is already active"
            );
            *active = Some(CaptureState {
                tensor_path,
                receipt_path,
                tensors: Vec::new(),
                calls: Vec::new(),
            });
        });
        Self
    }
}

impl Drop for CaptureScope {
    fn drop(&mut self) {
        let state = ACTIVE.with(|active| active.borrow_mut().take());
        if let Some(state) = state {
            write_capture(state);
        }
    }
}

pub(super) fn begin_forward(
    input_ids: &MlxArray,
    input_embeddings: Option<&MlxArray>,
    mask: Option<&MlxArray>,
    positions: &[i32],
    caches: &[KVCache],
    sequence_id_present: bool,
) {
    ACTIVE.with(|active| {
        let mut active = active.borrow_mut();
        let Some(state) = active.as_mut() else {
            return;
        };
        let call = state.calls.len();
        if let Some(value) = input_embeddings {
            push_tensor(state, format!("call{call}.input_embeddings"), value);
        }
        if let Some(value) = mask {
            push_tensor(state, format!("call{call}.attention_mask"), value);
        }
        let input_shape = mlxcel_core::array_shape(input_ids);
        let input_ids = mlxcel_core::utils::array_to_vec_f32(input_ids)
            .into_iter()
            .map(|value| value as i32)
            .collect();
        let cache_offsets_before: Vec<i32> = caches.iter().map(|cache| cache.offset).collect();
        let position_delta_from_cache = positions
            .first()
            .zip(cache_offsets_before.first())
            .map(|(position, offset)| position - offset);
        state.calls.push(CallRecord {
            input_ids,
            input_shape,
            embeddings_shape: input_embeddings.map(mlxcel_core::array_shape),
            mask_shape: mask.map(mlxcel_core::array_shape),
            positions: positions.to_vec(),
            sequence_id_present,
            cache_offsets_before,
            cache_offsets_after: Vec::new(),
            position_delta_from_cache,
            last_logits_shape: Vec::new(),
            top_token: -1,
            top_logit: f32::NEG_INFINITY,
        });
    });
}

pub(super) fn finish_forward(logits: &MlxArray, caches: &[KVCache]) {
    ACTIVE.with(|active| {
        let mut active = active.borrow_mut();
        let Some(state) = active.as_mut() else {
            return;
        };
        let call = state.calls.len() - 1;
        let last = mlxcel_core::slice_last_logits(logits);
        push_tensor(state, format!("call{call}.last_logits"), &last);
        evaluate_call_tensors(state, call);
        let values = mlxcel_core::utils::array_to_vec_f32(&last);
        let (top_token, top_logit) = argmax(&values);
        let record = state.calls.last_mut().expect("an active Falcon call");
        record.cache_offsets_after = caches.iter().map(|cache| cache.offset).collect();
        record.last_logits_shape = mlxcel_core::array_shape(&last);
        record.top_token = top_token;
        record.top_logit = top_logit;
    });
}

pub(super) fn capture_cache_before(layer: usize, cache: &KVCache) {
    if layer != 0 {
        return;
    }
    let Some((keys, values, live_len)) = cache.raw_kv_allocations() else {
        return;
    };
    let key_shape = mlxcel_core::array_shape(keys);
    let value_shape = mlxcel_core::array_shape(values);
    let keys = mlxcel_core::slice(
        keys,
        &[0, 0, 0, 0],
        &[key_shape[0], key_shape[1], live_len, key_shape[3]],
    );
    let values = mlxcel_core::slice(
        values,
        &[0, 0, 0, 0],
        &[value_shape[0], value_shape[1], live_len, value_shape[3]],
    );
    capture_layer_tensor(layer, "attention.cache_k.before", &keys);
    capture_layer_tensor(layer, "attention.cache_v.before", &values);
}

pub(super) fn capture_layer_tensor(layer: usize, name: &str, value: &MlxArray) {
    // Layer 0 localizes the first divergence; later blocks only need their
    // outputs to show where an already-observed error grows. Keeping every
    // later QKV/FFN temporary would retain several gigabytes for this prompt.
    if layer != 0 && name != "output" {
        return;
    }
    capture_model_tensor(&format!("layer{layer}.{name}"), value);
}

pub(super) fn capture_model_tensor(name: &str, value: &MlxArray) {
    ACTIVE.with(|active| {
        let mut active = active.borrow_mut();
        let Some(state) = active.as_mut() else {
            return;
        };
        let call = state
            .calls
            .len()
            .checked_sub(1)
            .expect("an active Falcon call");
        push_tensor(state, format!("call{call}.{name}"), value);
    });
}

fn evaluate_call_tensors(state: &CaptureState, call: usize) {
    let prefix = format!("call{call}.");
    let pointers: Vec<_> = state
        .tensors
        .iter()
        .filter(|(name, _)| name.starts_with(&prefix))
        .map(|(_, value)| value.as_ref().expect("captured tensor") as *const MlxArray)
        .collect();
    // Every pointer remains owned by `state.tensors` for the duration of the
    // call. Materializing here snapshots prefill cache values before the
    // subsequent one-token update can write into the same backing allocation.
    unsafe { mlxcel_core::eval_all(&pointers) };
}

fn push_tensor(state: &mut CaptureState, name: String, value: &MlxArray) {
    assert_eq!(
        mlxcel_core::array_dtype(value),
        mlxcel_core::dtype::FLOAT32,
        "captured Falcon tensor {name} must be F32"
    );
    state.tensors.push((name, mlxcel_core::copy(value)));
}

fn argmax(values: &[f32]) -> (i32, f32) {
    values
        .iter()
        .copied()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, value)| (index as i32, value))
        .expect("non-empty logits")
}

fn write_capture(state: CaptureState) {
    let values: Vec<_> = state
        .tensors
        .into_iter()
        .map(|(name, value)| {
            let shape: Vec<usize> = mlxcel_core::array_shape(&value)
                .into_iter()
                .map(|dimension| usize::try_from(dimension).expect("non-negative shape"))
                .collect();
            let bytes: Vec<u8> = mlxcel_core::utils::array_to_vec_f32(&value)
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect();
            (name, shape, bytes)
        })
        .collect();
    let views: BTreeMap<_, _> = values
        .iter()
        .map(|(name, shape, bytes)| {
            (
                name.as_str(),
                TensorView::new(Dtype::F32, shape.clone(), bytes).expect("valid diagnostic view"),
            )
        })
        .collect();
    safetensors::serialize_to_file(&views, None, &state.tensor_path)
        .expect("write Falcon diagnostic tensor bundle");
    let json = serde_json::to_vec_pretty(&state.calls).expect("serialize Falcon call records");
    std::fs::write(&state.receipt_path, json).expect("write Falcon call records");
}

fn prepare_title(
    checkpoint: &Path,
    crop: &image::DynamicImage,
) -> (
    LoadedModel,
    Vec<i32>,
    crate::vlm_runtime::PreparedVlmEmbeddings,
) {
    let (model, tokenizer) = crate::loading::load_model(checkpoint).expect("load Falcon-OCR");
    let prompt = format!("{}\n", OcrCategory::Title.instruction());
    let mut prompt_tokens: Vec<i32> = tokenizer
        .encode(&prompt, true)
        .expect("encode title prompt")
        .into_iter()
        .map(|token| token as i32)
        .collect();
    let prepared = prepare_vlm_embeddings_with_budget(
        &model,
        &mut prompt_tokens,
        &prompt,
        std::slice::from_ref(crop),
        None,
        |text, add_special| {
            tokenizer
                .encode(text, add_special)
                .expect("encode Falcon image prompt")
                .into_iter()
                .map(|token| token as i32)
                .collect()
        },
    )
    .expect("prepare Falcon title embeddings")
    .expect("Falcon is a VLM");
    (model, prompt_tokens, prepared)
}

#[derive(Serialize)]
struct Summary {
    processor_max_abs: f32,
    processor_rms: f64,
    processor_differing_values: usize,
    direct_tokens: [i32; 2],
    engine_tokens: Vec<i32>,
    python_tokens: [i32; 2],
    prompt_length: usize,
    grid: Vec<(i32, i32)>,
}

#[test]
#[ignore = "requires the pinned Falcon-OCR checkpoint and a ROCm device"]
fn authentic_title_direct_and_engine_capture() {
    let checkpoint = PathBuf::from(std::env::var(CHECKPOINT_ENV).expect(CHECKPOINT_ENV));
    let reference = PathBuf::from(std::env::var(REFERENCE_ENV).expect(REFERENCE_ENV));
    let output = PathBuf::from(std::env::var(OUTPUT_DIR_ENV).expect(OUTPUT_DIR_ENV));
    assert!(
        output.is_dir(),
        "{OUTPUT_DIR_ENV} must be an existing directory"
    );
    assert!(
        std::fs::read_dir(&output)
            .expect("read diagnostic output directory")
            .next()
            .is_none(),
        "{OUTPUT_DIR_ENV} must be empty"
    );

    let page = image::open(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/pp_doclayout_v3/two_column_page.png"),
    )
    .expect("open authentic BERT page");
    let crop = crop_region(
        &page,
        &[228.493_68, 137.169_86, 965.519_65, 203.076_87],
        MIN_CROP_DIM,
        1024,
    )
    .expect("title crop");
    assert_eq!((crop.width(), crop.height()), (738, 66));

    let (model, prompt_tokens, prepared) = prepare_title(&checkpoint, &crop);
    let LoadedModel::FalconOcrVL(falcon) = &model else {
        panic!("checkpoint did not load as Falcon-OCR");
    };
    let (rust_pixels, grid) = falcon
        .processor
        .preprocess_values_with_grid(std::slice::from_ref(&crop));
    assert_eq!(grid, vec![(4, 46)]);
    let (pixel_shape, python_pixels) = reference_tensor_f32(&reference, "processor.pixel_values");
    assert_eq!(pixel_shape, vec![1, 64, 736, 3]);
    let python_patches = patchify_nhwc(&python_pixels, 64, 736);
    assert_eq!(rust_pixels.len(), python_patches.len());
    assert!(rust_pixels.iter().all(|value| value.is_finite()));
    let mut max_abs = 0.0f32;
    let mut squared = 0.0f64;
    let mut differing = 0usize;
    for (rust, python) in rust_pixels.iter().zip(&python_patches) {
        let difference = (rust - python).abs();
        max_abs = max_abs.max(difference);
        squared += f64::from(difference).powi(2);
        differing += usize::from(difference != 0.0);
    }
    let rms = (squared / rust_pixels.len() as f64).sqrt();

    assert_eq!(
        prompt_tokens,
        reference_tensor_i64(&reference, "processor.input_ids")
    );
    assert_eq!(prompt_tokens.len(), 200);
    let (embeddings, mask) =
        prepared_embedding_refs(&prepared.embeddings).expect("prepared arrays");

    let mut bias = TokenBiasMap::new();
    bias.suppress_tokens(&model.output_suppressed_token_ids());
    let sampling = SamplingConfig::greedy();
    let engine_tokens = {
        let _capture = CaptureScope::begin(&output, "engine");
        let mut engine = DirectEngine::new(model, 0)
            .with_token_bias(bias)
            .with_force_sync(true);
        engine
            .generate(
                &DirectRequest {
                    prompt_tokens: &prompt_tokens,
                    embeddings: Some(embeddings),
                    mask,
                    max_tokens: 2,
                    sampling: &sampling,
                },
                |_| true,
            )
            .expect("run DirectEngine Falcon title diagnostic")
            .tokens
    };
    assert_eq!(engine_tokens.len(), 2, "engine must capture two tokens");
    assert_eq!(engine_tokens[0], 526, "engine prefill token changed");

    // Reload and prepare independently so the direct arm cannot inherit lazy
    // arrays, caches, or model state materialized by DirectEngine.
    let (direct_model, direct_prompt_tokens, direct_prepared) = prepare_title(&checkpoint, &crop);
    assert_eq!(direct_prompt_tokens, prompt_tokens);
    let direct_input_ids = mlxcel_core::from_slice_i32(&direct_prompt_tokens, &[1, 200]);
    let (direct_embeddings, direct_mask) =
        prepared_embedding_refs(&direct_prepared.embeddings).expect("fresh prepared arrays");
    let direct_tokens = {
        let _capture = CaptureScope::begin(&output, "direct");
        let mut caches = direct_model.make_caches();
        let logits = direct_model.forward_with_embeddings(
            &direct_input_ids,
            Some(direct_embeddings),
            &mut caches,
            direct_mask,
        );
        let first = argmax(&mlxcel_core::utils::array_to_vec_f32(
            &mlxcel_core::slice_last_logits(&logits),
        ))
        .0;
        assert_eq!(first, engine_tokens[0], "direct and engine decode inputs");
        assert_eq!(first, 526, "direct prefill token changed");
        let decode_ids = mlxcel_core::from_slice_i32(&[first], &[1, 1]);
        let logits = direct_model.forward(&decode_ids, &mut caches, None);
        let second = argmax(&mlxcel_core::utils::array_to_vec_f32(
            &mlxcel_core::slice_last_logits(&logits),
        ))
        .0;
        [first, second]
    };

    let summary = Summary {
        processor_max_abs: max_abs,
        processor_rms: rms,
        processor_differing_values: differing,
        direct_tokens,
        engine_tokens,
        python_tokens: [526, 969],
        prompt_length: prompt_tokens.len(),
        grid,
    };
    std::fs::write(
        output.join("summary.json"),
        serde_json::to_vec_pretty(&summary).expect("serialize diagnostic summary"),
    )
    .expect("write diagnostic summary");
}
