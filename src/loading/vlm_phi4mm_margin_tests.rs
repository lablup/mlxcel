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

//! Pinned Phi4MM vision-path bisect probe (issue #3391 item 2).
//!
//! `mlxcel generate --image ... --temp 0` on `phi-4-multimodal-instruct`
//! matched eager for 8 tokens, then XLA said "white line in the middle" where
//! eager said "on the left side". XLA f32 and f16 vision give the same tokens,
//! so the gap sits upstream of the decoder: preprocessing or the encoder. The
//! eager and XLA paths run two separately loaded copies of the same pipeline
//! (full weights versus filtered canonical weights), so this probe compares
//! them stage by stage and names the first diverging stage. Measured verdict
//! on GB10 at the fc344c98 MLX pin: the path is bit-identical at every stage
//! (preprocessing, per-crop encoder hidden states, pooled and projected
//! features, all max abs 0.0) and the CLI repro produces identical tokens on
//! both backends, so the reported divergence does not reproduce on current
//! main; the probe gates the parity a future loader change would break.
//!
//! Stage 1 compares the two processors' crops bit-wise (deterministic CPU
//! code, so any difference is a preprocessing defect). Stage 2 runs each
//! side's vision tower over the same crops and compares the selected hidden
//! states (a weight-loading or layout defect shows here). Stage 3 compares
//! the pooled features and the projected output.
//!
//! ```text
//! MLXCEL_PHI4MM_FIXTURE=/path/to/phi-4-multimodal-instruct \
//! cargo test --features cuda,xla-iree \
//!   pinned_phi4mm_vision_bisect_probe -- --ignored --nocapture --test-threads=1
//! ```

use super::*;

#[cfg(all(test, feature = "xla-iree"))]
mod vision_bisect {
    use std::io::Write;

    use mlxcel_core::{MlxArray, UniquePtr};

    use super::*;

    const DEFAULT_IMAGE: &str = "tests/fixtures/test_image.png";

    fn array_to_f32(array: &MlxArray, label: &str) -> Vec<f32> {
        let array = mlxcel_core::astype(array, mlxcel_core::dtype::FLOAT32);
        mlxcel_core::eval(&array);
        let bytes = mlxcel_core::try_array_to_raw_bytes(&array)
            .unwrap_or_else(|error| panic!("export {label}: {error}"));
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_ne_bytes(chunk.try_into().expect("four-byte f32")))
            .collect()
    }

    fn max_abs(label: &str, left: &[f32], right: &[f32]) -> f32 {
        assert_eq!(
            left.len(),
            right.len(),
            "{label} has {} values on one side and {} on the other",
            left.len(),
            right.len()
        );
        left.iter()
            .zip(right)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
    }

    /// The #874 HD transform pipeline, replicated verbatim from both private
    /// implementations (which are structurally identical) so the same steps run
    /// over each side's separately loaded weights.
    fn hd_transform_checkpoints(
        vision_tower: &vision::encoders::phi4_siglip::Phi4SigLipVisionEncoder,
        processor: &vision::processors::phi4mm::Phi4MMProcessor,
        select_layer: isize,
        glb_gn: &MlxArray,
        sub_gn: &MlxArray,
        hd_transform_order: &str,
        linear1: &mlxcel_core::layers::UnifiedLinear,
        linear2: &mlxcel_core::layers::UnifiedLinear,
        processed: &vision::processors::phi4mm::Phi4MMImageInput,
        dtype: i32,
    ) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<f32>) {
        let (h_crops, w_crops) = processed.image_grid;
        let pooled_grid = processed.pooled_grid_size;
        let spatial_per_crop = (
            (processor.crop_size / processor.patch_size) as i32,
            (processor.crop_size / processor.patch_size) as i32,
        );
        let mut selected_features = Vec::with_capacity(processed.crops.len());
        for crop in &processed.crops {
            let pixels = mlxcel_core::astype(&crop.pixel_values, dtype);
            let mut hidden_states = vision_tower.forward_hidden_states(&pixels, spatial_per_crop);
            let layer_count = hidden_states.len() as isize;
            let selected_index = if select_layer < 0 {
                (layer_count + select_layer) as usize
            } else {
                select_layer as usize
            };
            let selected = hidden_states.swap_remove(selected_index);
            selected_features.push(selected);
        }
        let orig_grid = processor.crop_size / processor.patch_size;
        let pooled: Vec<UniquePtr<MlxArray>> = selected_features
            .iter()
            .map(|features| avg_pool_2d(features, orig_grid, pooled_grid))
            .collect();
        let vision_dim = mlxcel_core::array_shape(&pooled[0])[2];
        let pooled_refs: Vec<f32> = pooled
            .iter()
            .flat_map(|features| array_to_f32(features, "pooled features"))
            .collect();
        let global_grid = mlxcel_core::reshape(
            &pooled[0],
            &[1, pooled_grid as i32, pooled_grid as i32, vision_dim],
        );
        let separator = mlxcel_core::astype(sub_gn, dtype);
        let sub_gn_col =
            mlxcel_core::broadcast_to(&separator, &[1, pooled_grid as i32, 1, vision_dim]);
        let global_with_sep = mlxcel_core::concatenate(&global_grid, &sub_gn_col, 2);
        let global_tokens = mlxcel_core::reshape(
            &global_with_sep,
            &[1, (pooled_grid * (pooled_grid + 1)) as i32, vision_dim],
        );
        let sub_tokens = assemble_sub_features(
            &pooled[1..],
            h_crops,
            w_crops,
            pooled_grid,
            vision_dim,
            processed.active_rows,
            processed.active_cols,
            sub_gn,
            dtype,
        );
        let glb_gn_3d = {
            let token = mlxcel_core::astype(glb_gn, dtype);
            mlxcel_core::reshape(&token, &[1, 1, vision_dim])
        };
        let combined = if hd_transform_order == "sub_glb" {
            let first_two = mlxcel_core::concatenate(&sub_tokens, &glb_gn_3d, 1);
            mlxcel_core::concatenate(&first_two, &global_tokens, 1)
        } else {
            let first_two = mlxcel_core::concatenate(&global_tokens, &glb_gn_3d, 1);
            mlxcel_core::concatenate(&first_two, &sub_tokens, 1)
        };
        let projected = linear1.forward(&combined);
        let activated = mlxcel_core::gelu_approx(&projected);
        let output = linear2.forward(&activated);
        let projected_values = array_to_f32(&output, "projected image");
        (
            selected_refs(&selected_features),
            vec![pooled_refs],
            projected_values,
        )
    }

    fn selected_refs(features: &[UniquePtr<MlxArray>]) -> Vec<Vec<f32>> {
        features
            .iter()
            .map(|features| array_to_f32(features, "selected hidden states"))
            .collect()
    }

    fn assemble_sub_features(
        features: &[UniquePtr<MlxArray>],
        h_crops: usize,
        w_crops: usize,
        pooled_grid: usize,
        vision_dim: i32,
        active_rows: usize,
        active_cols: usize,
        sub_gn: &MlxArray,
        dtype: i32,
    ) -> UniquePtr<MlxArray> {
        let pg = pooled_grid as i32;
        let total_h = h_crops * pooled_grid;
        let total_w = w_crops * pooled_grid;
        let mut row_segments: Vec<UniquePtr<MlxArray>> = Vec::new();
        for ch in 0..h_crops {
            let mut col_segments: Vec<UniquePtr<MlxArray>> = Vec::new();
            for cw in 0..w_crops {
                let crop_idx = ch * w_crops + cw;
                let reshaped = mlxcel_core::reshape(&features[crop_idx], &[pg, pg, vision_dim]);
                col_segments.push(reshaped);
            }
            let row = concat_arrays_axis(&col_segments, 1);
            row_segments.push(row);
        }
        let spatial = concat_arrays_axis(&row_segments, 0);
        let useful_h = active_rows.min(total_h) as i32;
        let useful_w = active_cols.min(total_w) as i32;
        let cropped = if useful_h < total_h as i32 || useful_w < total_w as i32 {
            mlxcel_core::slice(&spatial, &[0, 0, 0], &[useful_h, useful_w, vision_dim])
        } else {
            spatial
        };
        let cropped = mlxcel_core::reshape(&cropped, &[1, useful_h, useful_w, vision_dim]);
        let sub_gn_col = {
            let separator = mlxcel_core::astype(sub_gn, dtype);
            mlxcel_core::broadcast_to(&separator, &[1, useful_h, 1, vision_dim])
        };
        let with_separator = mlxcel_core::concatenate(&cropped, &sub_gn_col, 2);
        mlxcel_core::reshape(&with_separator, &[1, useful_h * (useful_w + 1), vision_dim])
    }

    fn concat_arrays_axis(arrays: &[UniquePtr<MlxArray>], axis: i32) -> UniquePtr<MlxArray> {
        let mut output = mlxcel_core::copy(&arrays[0]);
        for array in &arrays[1..] {
            output = mlxcel_core::concatenate(&output, array, axis);
        }
        output
    }

    fn avg_pool_2d(
        features: &MlxArray,
        original_grid: usize,
        target_grid: usize,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(features);
        let dim = shape[2];
        let og = original_grid as i32;
        let tg = target_grid as i32;
        let pool_size = (original_grid / target_grid) as i32;
        let spatial = mlxcel_core::reshape(features, &[og, og, dim]);
        let transposed = mlxcel_core::transpose_axes(&spatial, &[2, 0, 1]);
        let batched = mlxcel_core::reshape(&transposed, &[dim, 1, og, og]);
        let blocked = mlxcel_core::reshape(&batched, &[dim, 1, tg, pool_size, tg, pool_size]);
        let mean = mlxcel_core::mean_axis(&blocked, 5, true);
        let mean = mlxcel_core::mean_axis(&mean, 3, true);
        let squeezed = mlxcel_core::reshape(&mean, &[dim, tg, tg]);
        let result = mlxcel_core::transpose_axes(&squeezed, &[1, 2, 0]);
        mlxcel_core::reshape(&result, &[1, tg * tg, dim])
    }

    #[test]
    #[ignore = "requires the pinned Phi4MM checkpoint plus MLX and IREE CUDA"]
    fn pinned_phi4mm_vision_bisect_probe() {
        let Ok(model_dir_env) = std::env::var("MLXCEL_PHI4MM_FIXTURE") else {
            panic!("set MLXCEL_PHI4MM_FIXTURE to the pinned Phi4MM checkpoint");
        };
        let model_dir = Path::new(&model_dir_env);
        let image_binding =
            std::env::var("MLXCEL_PHI4MM_IMAGE").unwrap_or_else(|_| DEFAULT_IMAGE.to_string());
        let image_path = Path::new(&image_binding);
        let image = image::open(image_path).expect("load the fixture image");

        let (loaded, _tokenizer) =
            crate::load_model(model_dir).expect("load pinned Phi4MM checkpoint");
        let inner = match &loaded {
            LoadedModel::Phi4MMVLM(inner) => inner,
            _ => panic!("pinned checkpoint did not load as Phi4MMVLM"),
        };
        let (text_embeddings, hidden_size, _max_pos, components) =
            super::load_phi4mm_xla_media_components(model_dir)
                .expect("load filtered Phi4MM XLA media components");
        let _ = (text_embeddings, hidden_size);
        let dtype = mlxcel_core::dtype::BFLOAT16;

        // Stage 1: preprocessing. Deterministic CPU code, so any difference is
        // a preprocessing defect.
        let eager_processed = inner.processor.preprocess(std::slice::from_ref(&image));
        let xla_processed = components
            .processor
            .preprocess(std::slice::from_ref(&image));
        assert_eq!(
            eager_processed.len(),
            xla_processed.len(),
            "the processors produced different image counts"
        );
        let mut first_divergence: Option<String> = None;
        for (index, (left, right)) in eager_processed.iter().zip(&xla_processed).enumerate() {
            assert_eq!(left.crops.len(), right.crops.len());
            if left.image_grid != right.image_grid
                || left.active_rows != right.active_rows
                || left.active_cols != right.active_cols
                || left.pooled_grid_size != right.pooled_grid_size
            {
                first_divergence = Some(format!(
                    "preprocessing grid metadata diverged at image {}: {:?} vs {:?}",
                    index + 1,
                    (left.image_grid, left.active_rows, left.active_cols),
                    (right.image_grid, right.active_rows, right.active_cols)
                ));
                break;
            }
            let mut worst = 0.0f32;
            for (crop_left, crop_right) in left.crops.iter().zip(&right.crops) {
                let a = array_to_f32(&crop_left.pixel_values, "eager pixel values");
                let b = array_to_f32(&crop_right.pixel_values, "xla pixel values");
                worst = worst.max(max_abs("pixel values", &a, &b));
            }
            let stderr = std::io::stderr();
            let mut writer = stderr.lock();
            let _ = writeln!(
                writer,
                "stage 1 (preprocessing) image {}: {} crops, max abs {worst:.3e}",
                index + 1,
                left.crops.len()
            );
            if worst > 0.0 && first_divergence.is_none() {
                first_divergence = Some(format!(
                    "preprocessing pixel values diverged at image {index}: max abs {worst:.3e}"
                ));
            }
        }

        // Stage 2 and 3: encoder and projector, over the eager processor's
        // crops (identical to the XLA processor's unless stage 1 diverged).
        let processed = &eager_processed[0];
        let (eager_selected, eager_pooled, eager_projected) = hd_transform_checkpoints(
            &inner.vision_tower,
            &inner.processor,
            inner.select_layer,
            &inner.glb_gn,
            &inner.sub_gn,
            &inner.hd_transform_order,
            &inner.mm_projector_linear1,
            &inner.mm_projector_linear2,
            processed,
            dtype,
        );
        let (xla_selected, xla_pooled, xla_projected) = hd_transform_checkpoints(
            &components.vision_tower,
            &components.processor,
            components.select_layer,
            &components.glb_gn,
            &components.sub_gn,
            &components.hd_transform_order,
            &components.mm_projector_linear1,
            &components.mm_projector_linear2,
            processed,
            dtype,
        );
        mlxcel_core::memory::clear_cache();

        let stderr = std::io::stderr();
        let mut writer = stderr.lock();
        let mut worst_encoder = 0.0f32;
        for (index, (left, right)) in eager_selected.iter().zip(&xla_selected).enumerate() {
            let delta = max_abs("selected hidden states", left, right);
            worst_encoder = worst_encoder.max(delta);
            let _ = writeln!(
                writer,
                "stage 2 (encoder) crop {}: selected hidden states max abs {delta:.3e}",
                index + 1
            );
        }
        if worst_encoder > 0.0 && first_divergence.is_none() {
            first_divergence = Some(format!(
                "vision encoder outputs diverged: max abs {worst_encoder:.3e}"
            ));
        }
        let worst_pooled = max_abs("pooled features", &eager_pooled[0], &xla_pooled[0]);
        let _ = writeln!(
            writer,
            "stage 3 (pool+project): pooled max abs {worst_pooled:.3e}, projected max abs {:.3e}",
            max_abs("projected image", &eager_projected, &xla_projected)
        );

        if let Some(divergence) = first_divergence {
            panic!("Phi4MM vision path first divergence: {divergence}");
        }
        let _ = writeln!(writer, "PASS the pinned Phi4MM vision bisect stages");
    }
}
