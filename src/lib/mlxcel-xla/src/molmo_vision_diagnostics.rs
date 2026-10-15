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
use crate::emitter::emit_molmo_vision_diagnostics;

/// Diagnostics-only outputs from the resident Molmo v1 vision graph.
#[derive(Debug, Clone, PartialEq)]
pub struct MolmoVisionDiagnostics {
    pub patch_embeddings: Vec<f32>,
    pub patch_shape: [usize; 3],
    pub selected_hidden_states: Vec<Vec<f32>>,
    pub selected_shape: [usize; 3],
    pub selected_layers: Vec<usize>,
    pub projected_features: Vec<f32>,
    pub projected_shape: [usize; 3],
    pub elapsed_seconds: f64,
    pub upload_bytes: usize,
    pub transfer_bytes: usize,
}

/// Diagnostics-only Molmo graph. Production continues to load the one-output
/// [`IreeMolmoVisionProjector`] artifact.
pub struct IreeMolmoVisionDiagnosticProjector {
    module: IreeAuxiliaryModule,
    config: MolmoVisionConfig,
}

impl IreeMolmoVisionDiagnosticProjector {
    pub fn load(model_dir: &Path, device: &str) -> Result<Self, String> {
        let config = MolmoVisionConfig::from_model_dir(model_dir)?;
        let mlir = emit_molmo_vision_diagnostics(&config);
        let module = compile_and_load(
            model_dir,
            device,
            &config,
            &mlir,
            "molmo-v1-vision-diagnostics",
        )?;
        Ok(Self { module, config })
    }

    #[must_use]
    pub fn artifact_fingerprint(&self) -> u64 {
        self.module.fingerprint()
    }

    pub fn project(
        &mut self,
        pixels: &[f32],
        masks: &[f32],
        crop_count: usize,
    ) -> Result<MolmoVisionDiagnostics, String> {
        let input = validate_and_pad_input(&self.config, pixels, masks, crop_count)?;
        let patch_shape = [
            self.config.max_crops,
            self.config.patches_per_crop,
            self.config.hidden,
        ];
        let selected_shape = [
            self.config.max_crops * self.config.positions,
            self.config.hidden,
        ];
        let projected_shape = [
            self.config.max_crops,
            self.config.projected_rows_per_crop(),
            self.config.text_hidden,
        ];
        let mut shapes = Vec::with_capacity(self.config.selected_layers.len() + 2);
        shapes.push(patch_shape.to_vec());
        shapes.extend((0..self.config.selected_layers.len()).map(|_| selected_shape.to_vec()));
        shapes.push(projected_shape.to_vec());
        let mut buffers = shapes
            .iter()
            .map(|shape| {
                checked_product(shape, "diagnostic output").and_then(|elements| {
                    elements
                        .checked_mul(std::mem::size_of::<f32>())
                        .map(|bytes| vec![0u8; bytes])
                        .ok_or_else(|| {
                            "Molmo diagnostic output byte count overflows usize".to_string()
                        })
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let transfer_bytes = buffers.iter().map(Vec::len).sum();
        let mut outputs = buffers
            .iter_mut()
            .zip(&shapes)
            .map(|(bytes, shape)| AuxiliaryOutput {
                bytes,
                dtype: AuxiliaryTensorDType::Float32,
                shape,
            })
            .collect::<Vec<_>>();
        let started = Instant::now();
        self.module.invoke(
            &[
                AuxiliaryInput {
                    bytes: f32_as_bytes(&input.padded_pixels),
                    dtype: AuxiliaryTensorDType::Float32,
                    shape: &input.pixel_shape,
                },
                AuxiliaryInput {
                    bytes: f32_as_bytes(&input.padded_masks),
                    dtype: AuxiliaryTensorDType::Float32,
                    shape: &input.mask_shape,
                },
            ],
            &mut outputs,
        )?;
        let elapsed_seconds = started.elapsed().as_secs_f64();
        drop(outputs);
        let values = buffers
            .into_iter()
            .enumerate()
            .map(|(index, bytes)| {
                checked_f32_output(&format!("IREE Molmo diagnostic output {index}"), bytes)
            })
            .collect::<Result<Vec<_>, String>>()?;
        let mut values = values.into_iter();
        let mut patch_embeddings = values
            .next()
            .expect("Molmo diagnostics include patch embeddings");
        let active_patch_count = checked_product(
            &[
                input.crop_count,
                self.config.patches_per_crop,
                self.config.hidden,
            ],
            "active diagnostic patch output",
        )?;
        patch_embeddings.truncate(active_patch_count);
        let active_selected_count = checked_product(
            &[input.crop_count, self.config.positions, self.config.hidden],
            "active diagnostic selected output",
        )?;
        let selected_hidden_states = values
            .by_ref()
            .take(self.config.selected_layers.len())
            .map(|mut state| {
                state.truncate(active_selected_count);
                state
            })
            .collect::<Vec<_>>();
        let mut projected_features = values
            .next()
            .expect("Molmo diagnostics include projected features");
        let active_projected_count = checked_product(
            &[
                input.crop_count,
                self.config.projected_rows_per_crop(),
                self.config.text_hidden,
            ],
            "active diagnostic projected output",
        )?;
        projected_features.truncate(active_projected_count);
        debug_assert!(values.next().is_none());
        Ok(MolmoVisionDiagnostics {
            patch_embeddings,
            patch_shape: [
                input.crop_count,
                self.config.patches_per_crop,
                self.config.hidden,
            ],
            selected_hidden_states,
            selected_shape: [input.crop_count, self.config.positions, self.config.hidden],
            selected_layers: self.config.selected_layers.clone(),
            projected_features,
            projected_shape: [
                input.crop_count,
                self.config.projected_rows_per_crop(),
                self.config.text_hidden,
            ],
            elapsed_seconds,
            upload_bytes: input.upload_bytes,
            transfer_bytes,
        })
    }
}
