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

use super::*;

#[test]
fn pinned_gemma3_projector_loads_and_returns_finite_features() {
    let Ok(model) = std::env::var("MLXCEL_GEMMA3_FIXTURE") else {
        return;
    };
    let device = std::env::var("MLXCEL_XLA_DEVICE").unwrap_or_else(|_| "cuda".to_string());
    let mut projector = mlxcel_xla::IreeVisionProjector::load(Path::new(&model), &device)
        .expect("load pinned Gemma3 IREE vision projector");
    assert_eq!(projector.input_shape(), [1, 3, 896, 896]);
    assert_eq!(projector.output_shape(), [256, 2560]);
    let pixels = vec![0.0; projector.input_shape().into_iter().product()];
    let projection = projector
        .project(&pixels)
        .expect("execute pinned Gemma3 IREE vision projector");
    assert_eq!(projection.shape, [256, 2560]);
    assert!(projection.values.iter().all(|value| value.is_finite()));
}

#[cfg(feature = "xla-reference-diagnostics")]
pub mod reference_boundary {
    use std::fs;
    use std::io::{self, Write};
    use std::path::Path;
    #[cfg(test)]
    use std::path::PathBuf;
    use std::sync::mpsc::{self, RecvTimeoutError, Sender};
    use std::thread::{self, JoinHandle};
    use std::time::{Duration, Instant};

    use mlxcel_core::layers::UnifiedEmbedding;
    use mlxcel_core::session::{OwnedTensor, PreparedPrefill, PreparedTensorDType};
    use sha2::{Digest, Sha256};

    use super::*;
    use crate::multimodal::host_preprocessor::{
        Gemma3IreeHostPreprocessor, HostMultimodalPreprocessor,
    };
    use crate::multimodal::vlm_prompt::{ImageTokenBlockInfo, apply_image_token_blocks};
    use crate::vision::config::VLMConfig;
    use crate::vision::connectors::MultiModalConnector;
    use crate::vision::connectors::avg_pool::AvgPoolProjector;
    use crate::vision::encoders::VisionEncoder;
    use crate::vision::encoders::siglip::SigLipVisionModel;
    use crate::vision::processors::ImageProcessor;
    use crate::vision::processors::siglip::SigLipProcessor;

    const PINNED_REVISION: &str = "93724907d4ed1745d2fe50baadf3b0b01a65abf2";
    const PINNED_CONFIG_SHA256: &str =
        "5ccdde91da736e6e6f8f138268c620adcbf1219c973b884240b719a54122465b";
    const PINNED_PREPROCESSOR_SHA256: &str =
        "f688d6bb20c5017601c4011de7ca656da8485b540b05013efdaf986c0fcc918d";
    const PINNED_PROCESSOR_SHA256: &str =
        "3ffd5f11778dc73e2b69b3c00535e4121e1badf7018136263cd17b5b34fbaa53";
    const PINNED_IMAGE_SHA256: &str =
        "5e7d54e8a7d21802378c87d2d70cf551e29739fe27599ddf129ebccdad1e6261";
    const BLOCK0_STAGES: [&str; 12] = [
        "siglip.block0.layer_norm1",
        "siglip.block0.q_proj",
        "siglip.block0.k_proj",
        "siglip.block0.v_proj",
        "siglip.block0.attention_context",
        "siglip.block0.attention_output",
        "siglip.block0.attention_residual",
        "siglip.block0.layer_norm2",
        "siglip.block0.mlp_fc1",
        "siglip.block0.mlp_activation",
        "siglip.block0.mlp_fc2",
        "siglip.block0.output",
    ];

    #[derive(Debug, Clone, Copy)]
    struct Tolerance {
        atol: f64,
        rtol: f64,
    }

    const PIXEL_TOLERANCE: Tolerance = Tolerance {
        atol: 1e-6,
        rtol: 1e-6,
    };
    // The pinned MLX checkpoint executes the eager vision path in BF16 while
    // the qualified IREE graph widens immutable checkpoint weights to F32.
    // These are the same BF16 stage envelopes used by the #863 vision gate.
    const VISION_TOLERANCE: Tolerance = Tolerance {
        atol: 8e-2,
        rtol: 4e-2,
    };
    const IREE_REPLAY_TOLERANCE: Tolerance = Tolerance {
        atol: 1e-6,
        rtol: 1e-6,
    };
    const EXACT_TOLERANCE: Tolerance = Tolerance {
        atol: 0.0,
        rtol: 0.0,
    };

    #[derive(Debug, Clone, Copy, PartialEq)]
    struct ComparisonStats {
        max_absolute: f64,
        max_relative: f64,
        failures: usize,
        non_finite_count: usize,
        first_failure: Option<usize>,
    }

    fn comparison_stats(
        observed: &[f32],
        reference: &[f32],
        tolerance: Tolerance,
    ) -> ComparisonStats {
        assert_eq!(observed.len(), reference.len(), "comparison lengths differ");
        let mut stats = ComparisonStats {
            max_absolute: 0.0,
            max_relative: 0.0,
            failures: 0,
            non_finite_count: 0,
            first_failure: None,
        };
        for (index, (&observed, &reference)) in observed.iter().zip(reference).enumerate() {
            if !observed.is_finite() || !reference.is_finite() {
                stats.failures += 1;
                stats.non_finite_count += 1;
                stats.first_failure.get_or_insert(index);
                continue;
            }
            let absolute = f64::from((observed - reference).abs());
            let relative = absolute / f64::from(reference.abs()).max(f64::MIN_POSITIVE);
            stats.max_absolute = stats.max_absolute.max(absolute);
            stats.max_relative = stats.max_relative.max(relative);
            if absolute > tolerance.atol + tolerance.rtol * f64::from(reference.abs()) {
                stats.failures += 1;
                stats.first_failure.get_or_insert(index);
            }
        }
        stats
    }

    fn progress(stage: &str) {
        eprintln!("[gemma3-vlm-boundary] {stage}");
        io::stderr().flush().expect("flush diagnostic progress");
    }

    fn create_after_diagnostic_iree_configuration<T>(
        configure: impl FnOnce() -> Result<(), String>,
        create: impl FnOnce() -> Result<T, String>,
    ) -> Result<T, String> {
        configure()?;
        create()
    }

    struct ProgressHeartbeat {
        stop: Option<Sender<()>>,
        worker: Option<JoinHandle<()>>,
    }

    impl ProgressHeartbeat {
        fn start(stage: &str) -> Self {
            progress(stage);
            let (stop, receiver) = mpsc::channel();
            let stage = stage.to_string();
            let started = Instant::now();
            let worker = thread::spawn(move || {
                while let Err(RecvTimeoutError::Timeout) =
                    receiver.recv_timeout(Duration::from_secs(60))
                {
                    progress(&format!(
                        "heartbeat stage={stage} elapsed={}s",
                        started.elapsed().as_secs()
                    ));
                }
            });
            Self {
                stop: Some(stop),
                worker: Some(worker),
            }
        }
    }

    impl Drop for ProgressHeartbeat {
        fn drop(&mut self) {
            if let Some(stop) = self.stop.take() {
                let _ = stop.send(());
            }
            if let Some(worker) = self.worker.take() {
                let _ = worker.join();
            }
        }
    }

    fn compare_stage(
        stage: &str,
        observed: &[f32],
        reference: &[f32],
        tolerance: Tolerance,
        first_divergence: &mut Option<String>,
    ) {
        if observed.len() != reference.len() {
            let detail = format!("{stage}: length {} != {}", observed.len(), reference.len());
            first_divergence.get_or_insert(detail.clone());
            eprintln!("[gemma3-vlm-boundary] stage={stage} status=FAIL {detail}");
            return;
        }
        let stats = comparison_stats(observed, reference, tolerance);
        let status = if stats.failures == 0 { "PASS" } else { "FAIL" };
        eprintln!(
            "[gemma3-vlm-boundary] stage={stage} status={status} elements={} \
             atol={:.3e} rtol={:.3e} max_abs={:.6e} max_rel={:.6e} \
             failures={} non_finite={} first_failure={:?}",
            observed.len(),
            tolerance.atol,
            tolerance.rtol,
            stats.max_absolute,
            stats.max_relative,
            stats.failures,
            stats.non_finite_count,
            stats.first_failure,
        );
        io::stderr().flush().expect("flush diagnostic stage");
        if stats.failures != 0 {
            first_divergence.get_or_insert_with(|| {
                format!(
                    "{stage} at flat index {}",
                    stats
                        .first_failure
                        .expect("a failed comparison has an index")
                )
            });
        }
    }

    fn sha256(path: &Path) -> String {
        let bytes =
            fs::read(path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
        format!("{:x}", Sha256::digest(bytes))
    }

    fn assert_sha256(path: &Path, expected: &str, label: &str) {
        assert_eq!(
            sha256(path),
            expected,
            "{label} differs from the pinned #869 fixture: {}",
            path.display()
        );
    }

    fn pinned_revision(model: &Path) -> String {
        if let Ok(revision) = std::env::var("MLXCEL_GEMMA3_REVISION") {
            return revision;
        }
        let metadata = model.join(".cache/huggingface/download/config.json.metadata");
        fs::read_to_string(&metadata)
            .unwrap_or_else(|error| {
                panic!(
                    "read pinned revision from {} ({error}); set MLXCEL_GEMMA3_REVISION",
                    metadata.display()
                )
            })
            .lines()
            .next()
            .expect("Hugging Face metadata contains a revision")
            .to_string()
    }

    fn tensor_f32(tensor: &OwnedTensor, label: &str) -> Vec<f32> {
        assert_eq!(
            tensor.dtype,
            PreparedTensorDType::Float32,
            "{label} must be float32"
        );
        tensor
            .bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte f32 chunk")))
            .collect()
    }

    fn mlx_f32(array: &mlxcel_core::MlxArray, label: &str) -> Vec<f32> {
        let widened = mlxcel_core::astype(array, mlxcel_core::dtype::FLOAT32);
        mlxcel_core::try_array_to_raw_bytes(&widened)
            .unwrap_or_else(|error| panic!("export {label}: {error}"))
            .chunks_exact(4)
            .map(|bytes| f32::from_ne_bytes(bytes.try_into().expect("four-byte f32 chunk")))
            .collect()
    }

    fn reference_weight(name: &str) -> bool {
        let canonical = name.strip_prefix("language_model.").unwrap_or(name);
        canonical.starts_with("vision_tower.")
            || canonical.starts_with("multi_modal_projector.")
            || canonical.starts_with("model.embed_tokens.")
    }

    fn image_block_info(config: &VLMConfig, tokens_per_image: usize) -> ImageTokenBlockInfo {
        ImageTokenBlockInfo {
            use_boi_eoi: true,
            image_token_id: config.image_token_index,
            mm_tokens_per_image: tokens_per_image,
            boi_token_id: config.boi_token_index,
            eoi_token_id: config.eoi_token_index,
            has_bos: true,
            separator_token_id: None,
            suffix_tokens: Vec::new(),
            block_prefix_tokens: vec![108],
            block_suffix_tokens: vec![108],
        }
    }

    fn image_rows(prepared: &PreparedPrefill, image_token_id: i32, hidden: usize) -> Vec<f32> {
        let embeddings = tensor_f32(&prepared.embeddings, "prepared embeddings");
        prepared
            .token_ids
            .iter()
            .enumerate()
            .filter(|(_, token)| **token == image_token_id)
            .flat_map(|(position, _)| {
                embeddings[position * hidden..(position + 1) * hidden]
                    .iter()
                    .copied()
            })
            .collect()
    }

    fn expected_mask(token_ids: &[i32], pad_token_id: i32) -> Vec<f32> {
        let mut expected = vec![f32::MIN; token_ids.len() * token_ids.len()];
        for (query, query_token) in token_ids.iter().enumerate() {
            for (key, key_token) in token_ids.iter().enumerate() {
                if *query_token != pad_token_id && *key_token != pad_token_id {
                    expected[query * token_ids.len() + key] = 0.0;
                }
            }
        }
        expected
    }

    fn expected_post_scale(
        token_ids: &[i32],
        raw_text: &[f32],
        projected: &[f32],
        hidden: usize,
        pad_token_id: i32,
        image_token_id: i32,
    ) -> Vec<f32> {
        let normalizer = (hidden as f64).sqrt() as f32;
        let mut expected = vec![0.0; token_ids.len() * hidden];
        let mut image_row = 0;
        for (position, token) in token_ids.iter().enumerate() {
            let destination = position * hidden;
            if *token == pad_token_id {
                continue;
            }
            if *token == image_token_id {
                let source = image_row * hidden;
                expected[destination..destination + hidden]
                    .copy_from_slice(&projected[source..source + hidden]);
                image_row += 1;
            } else {
                for offset in 0..hidden {
                    expected[destination + offset] = raw_text[destination + offset] * normalizer;
                }
            }
        }
        expected
    }

    #[cfg(test)]
    #[test]
    fn comparison_reports_the_first_failed_element() {
        let stats = comparison_stats(
            &[1.0, 2.2, f32::NAN],
            &[1.0, 2.0, 3.0],
            Tolerance {
                atol: 0.01,
                rtol: 0.01,
            },
        );
        assert_eq!(stats.failures, 2);
        assert_eq!(stats.non_finite_count, 1);
        assert_eq!(stats.first_failure, Some(1));
    }

    /// Run the pinned mixed-runtime boundary gate for #869.
    ///
    /// This entry point loads only the eager MLX SigLIP/projector/text embedding
    /// weights and the resident IREE vision projector. The caller selects the
    /// MLX runtime at build time; the dedicated example requires CUDA while the
    /// IREE side remains pinned to `local-task`.
    ///
    /// # Panics
    ///
    /// Panics if the pinned fixture identity or any ordered comparison differs.
    pub fn run_gemma3_eager_mlx_iree_prepared_boundary(
        model: &Path,
        image_path: &Path,
        device: &str,
    ) {
        assert_eq!(
            device, "local-task",
            "the pinned #869 mixed-runtime gate qualifies IREE local-task only"
        );

        let _heartbeat = ProgressHeartbeat::start("validate pinned checkpoint and image");
        assert_eq!(
            pinned_revision(model),
            PINNED_REVISION,
            "checkpoint revision differs from pinned mlx-community/gemma-3-4b-it-4bit"
        );
        assert_sha256(
            &model.join("config.json"),
            PINNED_CONFIG_SHA256,
            "checkpoint config",
        );
        assert_sha256(
            &model.join("preprocessor_config.json"),
            PINNED_PREPROCESSOR_SHA256,
            "checkpoint image preprocessor",
        );
        assert_sha256(
            &model.join("processor_config.json"),
            PINNED_PROCESSOR_SHA256,
            "checkpoint processor",
        );
        assert_sha256(image_path, PINNED_IMAGE_SHA256, "image");
        drop(_heartbeat);

        let _heartbeat =
            ProgressHeartbeat::start("load filtered MLX SigLIP/projector/text embedding weights");
        let (_config_str, full_config) =
            read_sanitized_vlm_config(model).expect("read pinned Gemma3 config");
        let config: VLMConfig =
            serde_json::from_value(full_config.clone()).expect("parse pinned Gemma3 VLM config");
        let text_config: models::gemma3::ModelArgs =
            serde_json::from_value(config.text_config.clone())
                .expect("parse pinned Gemma3 text config");
        let weights = load_vlm_weights_common_filtered_canonical(model, reference_weight)
            .map(strip_language_model_prefix)
            .expect("load only Gemma3 vision/projector/embedding weights");
        let quant_group_size = full_config
            .pointer("/quantization/group_size")
            .and_then(Value::as_i64)
            .unwrap_or(64) as i32;
        let quant_bits = full_config
            .pointer("/quantization/bits")
            .and_then(Value::as_i64)
            .unwrap_or(4) as i32;
        let text_embeddings = UnifiedEmbedding::from_weights(
            &weights,
            "model.embed_tokens",
            quant_group_size,
            quant_bits,
        )
        .expect("load filtered Gemma3 text embedding table");
        let encoder = SigLipVisionModel::from_weights(
            &weights,
            &config.vision_config,
            "vision_tower.vision_model",
        )
        .expect("load filtered Gemma3 SigLIP tower");
        let tokens_per_image = config.get_mm_tokens_per_image();
        let connector = AvgPoolProjector::from_weights(
            &weights,
            "multi_modal_projector",
            config.vision_config.hidden_size,
            config.vision_config.image_size,
            config.vision_config.patch_size,
            tokens_per_image,
            config.vision_config.layer_norm_eps,
        )
        .expect("load filtered Gemma3 average-pool projector");
        let image = image::open(image_path)
            .unwrap_or_else(|error| panic!("decode {}: {error}", image_path.display()));
        let images = vec![image];
        drop(_heartbeat);

        progress("compare independently constructed processor pixels");
        let mlx_processor = SigLipProcessor::new(config.vision_config.image_size);
        let iree_processor = SigLipProcessor::new(config.vision_config.image_size);
        let mlx_pixels = mlx_processor.preprocess(&images);
        let iree_pixels = iree_processor.preprocess(&images);
        let mlx_pixel_values = mlx_f32(&mlx_pixels, "MLX processor pixels");
        let iree_pixel_values = mlx_f32(&iree_pixels, "IREE host processor pixels");
        let mut first_divergence = None;
        compare_stage(
            "processor.pixel_values",
            &iree_pixel_values,
            &mlx_pixel_values,
            PIXEL_TOLERANCE,
            &mut first_divergence,
        );

        progress("expand pinned padded image-token fixture and embed text");
        let mut logical_tokens = vec![
            config.pad_token_id,
            2,
            config.boi_token_index,
            1,
            config.pad_token_id,
        ];
        apply_image_token_blocks(
            &mut logical_tokens,
            image_block_info(&config, tokens_per_image),
            images.len(),
        )
        .expect("expand Gemma3 image-token block");
        let input_ids = mlxcel_core::from_slice_i32(
            &logical_tokens,
            &[
                1,
                i32::try_from(logical_tokens.len()).expect("fixture sequence length fits i32"),
            ],
        );
        let raw_text_array = text_embeddings.forward(&input_ids);
        let embed_dtype = mlxcel_core::array_dtype(&raw_text_array);
        let raw_text = mlx_f32(&raw_text_array, "raw text embeddings");

        let _heartbeat =
            ProgressHeartbeat::start("capture eager MLX SigLIP hidden and projector stages");
        let mlx_vision_input = mlxcel_core::astype(
            &mlxcel_core::transpose_axes(&mlx_pixels, &[0, 2, 3, 1]),
            embed_dtype,
        );
        let (mlx_selected, mlx_hidden, mlx_block0) =
            encoder.forward_with_hidden_state_diagnostics(&mlx_vision_input);
        let mlx_selected_values =
            mlx_f32(&mlx_selected.hidden_states, "MLX selected vision features");
        let mlx_projected = connector.forward(&mlx_selected.hidden_states);
        let mlx_projected_values = mlx_f32(&mlx_projected, "MLX projected image features");
        let mlx_hidden0 = mlx_f32(&mlx_hidden[0], "MLX SigLIP embedding output");
        let mlx_last_hidden = mlx_f32(
            mlx_hidden
                .last()
                .expect("MLX captured a final hidden state"),
            "MLX SigLIP last hidden",
        );
        let mlx_block0_values = mlx_block0
            .iter()
            .map(|stage| mlx_f32(stage, "MLX SigLIP block 0 stage"))
            .collect::<Vec<_>>();
        assert_eq!(mlx_block0_values.len(), BLOCK0_STAGES.len());
        drop(_heartbeat);

        let _heartbeat =
            ProgressHeartbeat::start("run IREE diagnostic SigLIP and average-pool projector");
        let mut diagnostic = create_after_diagnostic_iree_configuration(
            mlxcel_xla::configure_diagnostic_local_task_threads,
            || mlxcel_xla::IreeVisionDiagnosticProjector::load(model, device),
        )
        .expect("configure and load Gemma3 IREE diagnostic projector");
        let iree = diagnostic
            .project(&iree_pixel_values)
            .expect("execute Gemma3 IREE diagnostic projector");
        compare_stage(
            "siglip.hidden.embedding",
            &iree.hidden_states[0],
            &mlx_hidden0,
            VISION_TOLERANCE,
            &mut first_divergence,
        );
        assert_eq!(iree.block0_states.len(), BLOCK0_STAGES.len());
        for ((stage, observed), reference) in BLOCK0_STAGES
            .iter()
            .zip(&iree.block0_states)
            .zip(&mlx_block0_values)
        {
            compare_stage(
                stage,
                observed,
                reference,
                VISION_TOLERANCE,
                &mut first_divergence,
            );
        }
        compare_stage(
            "siglip.hidden.last_pre_layernorm",
            iree.hidden_states
                .last()
                .expect("IREE captured a final hidden state"),
            &mlx_last_hidden,
            VISION_TOLERANCE,
            &mut first_divergence,
        );
        compare_stage(
            "siglip.selected.post_layernorm",
            &iree.selected_vision_features,
            &mlx_selected_values,
            VISION_TOLERANCE,
            &mut first_divergence,
        );
        compare_stage(
            "projector.avg_pool_projection",
            &iree.projected_image_features,
            &mlx_projected_values,
            VISION_TOLERANCE,
            &mut first_divergence,
        );
        drop(_heartbeat);

        let _heartbeat = ProgressHeartbeat::start(
            "construct MLX-reference and production IREE prepared prefills",
        );
        let attention_mask = logical_tokens
            .iter()
            .map(|token| i32::from(*token != config.pad_token_id))
            .collect::<Vec<_>>();
        let mlx_prepared = mlxcel_xla::prepare_gemma3_vlm_prefill(
            logical_tokens.clone(),
            &raw_text,
            &mlx_projected_values,
            &attention_mask,
            text_config.hidden_size,
            text_config.max_position_embeddings,
            config.pad_token_id,
            config.image_token_index,
            images.len(),
        )
        .expect("construct eager MLX reference prepared prefill");
        let production = Gemma3IreeHostPreprocessor::load(model, device)
            .expect("load production Gemma3 IREE host preprocessor")
            .prepare(
                &[
                    config.pad_token_id,
                    2,
                    config.boi_token_index,
                    1,
                    config.pad_token_id,
                ],
                &images,
            )
            .expect("construct production Gemma3 IREE prepared prefill");
        assert_eq!(production.token_ids, logical_tokens);
        assert_eq!(production.positions, mlx_prepared.positions);
        assert_eq!(production.modalities, mlx_prepared.modalities);
        drop(_heartbeat);

        progress("compare final projected image rows and one-time scaling");
        let mlx_image_rows = image_rows(
            &mlx_prepared,
            config.image_token_index,
            text_config.hidden_size,
        );
        let iree_image_rows = image_rows(
            &production,
            config.image_token_index,
            text_config.hidden_size,
        );
        compare_stage(
            "prepared.mlx_projected_image_rows",
            &mlx_image_rows,
            &mlx_projected_values,
            EXACT_TOLERANCE,
            &mut first_divergence,
        );
        compare_stage(
            "prepared.iree_projected_image_rows",
            &iree_image_rows,
            &iree.projected_image_features,
            IREE_REPLAY_TOLERANCE,
            &mut first_divergence,
        );
        compare_stage(
            "prepared.mlx_vs_iree_image_rows",
            &iree_image_rows,
            &mlx_image_rows,
            VISION_TOLERANCE,
            &mut first_divergence,
        );
        let expected_mlx = expected_post_scale(
            &logical_tokens,
            &raw_text,
            &mlx_projected_values,
            text_config.hidden_size,
            config.pad_token_id,
            config.image_token_index,
        );
        let mlx_embeddings = tensor_f32(&mlx_prepared.embeddings, "MLX prepared embeddings");
        compare_stage(
            "prepared.text_sqrt_hidden_and_image_identity",
            &mlx_embeddings,
            &expected_mlx,
            EXACT_TOLERANCE,
            &mut first_divergence,
        );
        let expected_iree = expected_post_scale(
            &logical_tokens,
            &raw_text,
            &iree.projected_image_features,
            text_config.hidden_size,
            config.pad_token_id,
            config.image_token_index,
        );
        let production_embeddings = tensor_f32(
            &production.embeddings,
            "IREE production prepared embeddings",
        );
        compare_stage(
            "prepared.iree_text_sqrt_hidden_and_image_identity",
            &production_embeddings,
            &expected_iree,
            IREE_REPLAY_TOLERANCE,
            &mut first_divergence,
        );

        progress("compare exact additive bidirectional padding mask");
        assert!(!production.attention_bias.causal);
        assert!(!mlx_prepared.attention_bias.causal);
        assert_eq!(
            production.attention_bias.tensor.shape,
            vec![1, 1, logical_tokens.len(), logical_tokens.len()]
        );
        assert_eq!(
            production.attention_bias.tensor.dtype,
            PreparedTensorDType::Float32
        );
        let expected_attention = expected_mask(&logical_tokens, config.pad_token_id);
        let mlx_attention = tensor_f32(&mlx_prepared.attention_bias.tensor, "MLX attention bias");
        let iree_attention = tensor_f32(&production.attention_bias.tensor, "IREE attention bias");
        compare_stage(
            "prepared.mask.mlx_cell_exact",
            &mlx_attention,
            &expected_attention,
            EXACT_TOLERANCE,
            &mut first_divergence,
        );
        compare_stage(
            "prepared.mask.iree_cell_exact",
            &iree_attention,
            &expected_attention,
            EXACT_TOLERANCE,
            &mut first_divergence,
        );

        if let Some(first_divergence) = first_divergence {
            panic!("Gemma3 eager MLX/IREE first divergence: {first_divergence}");
        }
        progress("PASS all pinned Gemma3 eager MLX/IREE boundary stages");
    }

    /// Retained libtest wrapper for developers who already use the historical
    /// ignored gate. The dedicated example avoids linking the libtest harness.
    #[cfg(test)]
    #[test]
    #[ignore = "requires pinned Gemma3 checkpoint, image, MLX CUDA, and IREE local-task"]
    fn pinned_gemma3_eager_mlx_matches_iree_prepared_boundary() {
        let model = PathBuf::from(
            std::env::var("MLXCEL_GEMMA3_FIXTURE")
                .expect("MLXCEL_GEMMA3_FIXTURE must name the pinned checkpoint"),
        );
        let image_path = std::env::var("MLXCEL_GEMMA3_IMAGE")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("tests/fixtures/test_image.png"));
        let device =
            std::env::var("MLXCEL_XLA_DEVICE").unwrap_or_else(|_| "local-task".to_string());
        run_gemma3_eager_mlx_iree_prepared_boundary(&model, &image_path, &device);
    }

    #[cfg(test)]
    #[test]
    fn diagnostic_runner_configures_threads_before_creating_iree() {
        use std::cell::Cell;

        let configured = Cell::new(false);
        let result = create_after_diagnostic_iree_configuration(
            || {
                configured.set(true);
                Ok(())
            },
            || {
                assert!(configured.get());
                Ok("created")
            },
        );
        assert_eq!(result.as_deref(), Ok("created"));
    }

    #[cfg(test)]
    #[test]
    fn diagnostic_runner_propagates_configuration_failure_before_creation() {
        let created = std::cell::Cell::new(false);
        let result = create_after_diagnostic_iree_configuration(
            || Err("configuration failed".to_string()),
            || {
                created.set(true);
                Ok(())
            },
        );
        assert_eq!(result, Err("configuration failed".to_string()));
        assert!(!created.get());
    }

    #[cfg(test)]
    #[test]
    fn diagnostics_thread_overrides_stay_out_of_production_iree_paths() {
        let production_rust = include_str!("../lib/mlxcel-xla/src/aux.rs");
        let production_aux_c = include_str!("../lib/mlxcel-xla/csrc/xla_aux.c");
        let production_iree_c = include_str!("../lib/mlxcel-xla/csrc/xla_iree.c");
        for source in [production_rust, production_aux_c, production_iree_c] {
            assert!(!source.contains("task_topology_group_count"));
            assert!(!source.contains("task_worker_stack_size"));
            assert!(!source.contains("configure_diagnostic_local_task_threads"));
        }
    }

    #[cfg(test)]
    #[test]
    fn diagnostic_iree_thread_flags_pin_group_and_use_host_stack_default() {
        let diagnostic_c = include_str!("../lib/mlxcel-xla/csrc/xla_diagnostic_flags.c");
        assert!(diagnostic_c.contains("--task_topology_group_count=1"));
        assert!(diagnostic_c.contains("--task_worker_stack_size=0"));
        assert!(diagnostic_c.contains("if (argc != 1)"));
    }

    #[cfg(test)]
    #[test]
    fn native_diagnostic_thread_configuration_is_reusable() {
        mlxcel_xla::configure_diagnostic_local_task_threads()
            .expect("configure diagnostics-only IREE task threads");
        assert!(mlxcel_xla::diagnostic_local_task_threads_are_configured());
        mlxcel_xla::configure_diagnostic_local_task_threads()
            .expect("reuse diagnostics-only IREE task thread configuration");
    }
}
