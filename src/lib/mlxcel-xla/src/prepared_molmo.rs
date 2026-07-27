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

//! Owned Molmo v1 image input and sparse additive embedding merge contract.
//!
//! Molmo v1 is deliberately distinct from token-scanned VLM merges. Every
//! `image_input_idx` entry corresponds to the projected feature row at the same
//! flat index. Negative entries are padding sentinels: removing one must not
//! renumber later feature rows. Valid target positions receive an addition to
//! the post-scale OLMo text embedding, never a replacement.

use std::collections::HashSet;
use std::fmt;

/// Artifact-identity spelling for the only qualified Molmo v1 merge.
pub const MOLMO_V1_MERGE_MODE: &str = "processor-indexed-sparse-add-v1";

/// One active processor-supplied sparse merge pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MolmoSparseAddPair {
    /// Row in the projected feature tensor before sentinel removal.
    pub feature_row: usize,
    /// Position in the expanded OLMo text sequence.
    pub target_position: usize,
}

/// Owned processor output retained across the host/compiler boundary.
#[derive(Debug, Clone, PartialEq)]
pub struct MolmoPreparedImage {
    pub pixel_values: Vec<f32>,
    pub crop_count: usize,
    pub patches_per_crop: usize,
    pub patch_width: usize,
    pub image_masks: Vec<f32>,
    /// Rows emitted by attention pooling for each crop.
    pub projected_rows_per_crop: usize,
    /// Authoritative mapping, including negative sentinels.
    pub image_input_idx: Vec<i32>,
}

/// A validated sparse-add plan whose row mapping cannot drift after filtering.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MolmoSparseAddPlan {
    pairs: Vec<MolmoSparseAddPair>,
    feature_rows: usize,
    text_len: usize,
    hidden_size: usize,
}

/// Floating-point storage dtype of the post-scale OLMo text embeddings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MolmoEmbeddingDType {
    Float16,
    BFloat16,
    Float32,
}

impl MolmoEmbeddingDType {
    fn round(self, value: f32) -> f32 {
        match self {
            Self::Float16 => crate::float::half_to_f32(crate::float::f32_to_f16_bits(value)),
            Self::BFloat16 => crate::float::round_bf16_f32(value),
            Self::Float32 => value,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MolmoSparseAddError {
    ZeroDimension(&'static str),
    Capacity {
        dimension: &'static str,
        actual: usize,
        maximum: usize,
    },
    ShapeOverflow,
    PixelCount {
        expected: usize,
        actual: usize,
    },
    MaskCount {
        expected: usize,
        actual: usize,
    },
    MaskValue {
        flat_index: usize,
        value_bits: u32,
    },
    IndexCount {
        expected: usize,
        actual: usize,
    },
    TargetOutOfRange {
        feature_row: usize,
        target_position: i32,
        text_len: usize,
    },
    DuplicateTarget {
        first_feature_row: usize,
        feature_row: usize,
        target_position: usize,
    },
    TensorCount {
        tensor: &'static str,
        expected: usize,
        actual: usize,
    },
    NonFinite {
        tensor: &'static str,
        flat_index: usize,
    },
}

impl fmt::Display for MolmoSparseAddError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension(name) => write!(formatter, "Molmo {name} must be non-zero"),
            Self::Capacity {
                dimension,
                actual,
                maximum,
            } => write!(
                formatter,
                "Molmo {dimension} {actual} exceeds static maximum {maximum}"
            ),
            Self::ShapeOverflow => formatter.write_str("Molmo tensor shape overflowed"),
            Self::PixelCount { expected, actual } => write!(
                formatter,
                "Molmo pixel_values has {actual} elements, expected {expected}"
            ),
            Self::MaskCount { expected, actual } => write!(
                formatter,
                "Molmo image_masks has {actual} elements, expected {expected}"
            ),
            Self::MaskValue {
                flat_index,
                value_bits,
            } => write!(
                formatter,
                "Molmo image_masks[{flat_index}]={} is outside [-1,1]",
                f32::from_bits(*value_bits)
            ),
            Self::IndexCount { expected, actual } => write!(
                formatter,
                "Molmo image_input_idx has {actual} rows, expected {expected}"
            ),
            Self::TargetOutOfRange {
                feature_row,
                target_position,
                text_len,
            } => write!(
                formatter,
                "Molmo image_input_idx feature row {feature_row} targets {target_position}, outside [0,{text_len})"
            ),
            Self::DuplicateTarget {
                first_feature_row,
                feature_row,
                target_position,
            } => write!(
                formatter,
                "Molmo image_input_idx rows {first_feature_row} and {feature_row} both target {target_position}"
            ),
            Self::TensorCount {
                tensor,
                expected,
                actual,
            } => write!(
                formatter,
                "Molmo {tensor} has {actual} elements, expected {expected}"
            ),
            Self::NonFinite { tensor, flat_index } => {
                write!(formatter, "Molmo {tensor}[{flat_index}] is not finite")
            }
        }
    }
}

impl std::error::Error for MolmoSparseAddError {}

fn checked_product(dimensions: &[usize]) -> Result<usize, MolmoSparseAddError> {
    dimensions.iter().try_fold(1usize, |size, dimension| {
        size.checked_mul(*dimension)
            .ok_or(MolmoSparseAddError::ShapeOverflow)
    })
}

fn validate_capacity(
    dimension: &'static str,
    actual: usize,
    maximum: usize,
) -> Result<(), MolmoSparseAddError> {
    if actual > maximum {
        return Err(MolmoSparseAddError::Capacity {
            dimension,
            actual,
            maximum,
        });
    }
    Ok(())
}

impl MolmoPreparedImage {
    /// Validate all processor shapes and static buckets before native execution.
    pub fn validate(
        &self,
        max_crops: usize,
        max_patches_per_crop: usize,
        max_feature_rows: usize,
    ) -> Result<(), MolmoSparseAddError> {
        for (name, value) in [
            ("crop count", self.crop_count),
            ("patches per crop", self.patches_per_crop),
            ("patch width", self.patch_width),
            ("projected rows per crop", self.projected_rows_per_crop),
        ] {
            if value == 0 {
                return Err(MolmoSparseAddError::ZeroDimension(name));
            }
        }
        validate_capacity("crop count", self.crop_count, max_crops)?;
        validate_capacity(
            "patches per crop",
            self.patches_per_crop,
            max_patches_per_crop,
        )?;
        let patch_rows = checked_product(&[self.crop_count, self.patches_per_crop])?;
        let feature_rows = checked_product(&[self.crop_count, self.projected_rows_per_crop])?;
        validate_capacity("feature rows", feature_rows, max_feature_rows)?;
        let expected_pixels =
            checked_product(&[self.crop_count, self.patches_per_crop, self.patch_width])?;
        if self.pixel_values.len() != expected_pixels {
            return Err(MolmoSparseAddError::PixelCount {
                expected: expected_pixels,
                actual: self.pixel_values.len(),
            });
        }
        if self.image_masks.len() != patch_rows {
            return Err(MolmoSparseAddError::MaskCount {
                expected: patch_rows,
                actual: self.image_masks.len(),
            });
        }
        if self.image_input_idx.len() != feature_rows {
            return Err(MolmoSparseAddError::IndexCount {
                expected: feature_rows,
                actual: self.image_input_idx.len(),
            });
        }
        for (flat_index, value) in self.pixel_values.iter().enumerate() {
            if !value.is_finite() {
                return Err(MolmoSparseAddError::NonFinite {
                    tensor: "pixel_values",
                    flat_index,
                });
            }
        }
        for (flat_index, value) in self.image_masks.iter().enumerate() {
            if !value.is_finite() {
                return Err(MolmoSparseAddError::NonFinite {
                    tensor: "image_masks",
                    flat_index,
                });
            }
            if !(-1.0..=1.0).contains(value) {
                return Err(MolmoSparseAddError::MaskValue {
                    flat_index,
                    value_bits: value.to_bits(),
                });
            }
        }
        Ok(())
    }

    /// Validate the processor payload and build its authoritative sparse plan.
    #[allow(clippy::too_many_arguments)]
    pub fn sparse_add_plan(
        &self,
        text_len: usize,
        hidden_size: usize,
        max_crops: usize,
        max_patches_per_crop: usize,
        max_feature_rows: usize,
        max_text_len: usize,
    ) -> Result<MolmoSparseAddPlan, MolmoSparseAddError> {
        self.validate(max_crops, max_patches_per_crop, max_feature_rows)?;
        MolmoSparseAddPlan::from_image_input_idx(
            &self.image_input_idx,
            text_len,
            hidden_size,
            max_feature_rows,
            max_text_len,
        )
    }
}

impl MolmoSparseAddPlan {
    /// Build the authoritative mapping from the processor output.
    ///
    /// Sentinel rows are skipped, but their indices remain consumed so a later
    /// active entry keeps its original projected feature row.
    pub fn from_image_input_idx(
        image_input_idx: &[i32],
        text_len: usize,
        hidden_size: usize,
        max_feature_rows: usize,
        max_text_len: usize,
    ) -> Result<Self, MolmoSparseAddError> {
        if text_len == 0 {
            return Err(MolmoSparseAddError::ZeroDimension("text length"));
        }
        if hidden_size == 0 {
            return Err(MolmoSparseAddError::ZeroDimension("hidden size"));
        }
        validate_capacity("feature rows", image_input_idx.len(), max_feature_rows)?;
        validate_capacity("expanded text length", text_len, max_text_len)?;
        checked_product(&[text_len, hidden_size])?;
        checked_product(&[image_input_idx.len(), hidden_size])?;

        let mut pairs = Vec::new();
        let mut targets = std::collections::HashMap::new();
        for (feature_row, &target) in image_input_idx.iter().enumerate() {
            if target < 0 {
                continue;
            }
            let target_position =
                usize::try_from(target).map_err(|_| MolmoSparseAddError::TargetOutOfRange {
                    feature_row,
                    target_position: target,
                    text_len,
                })?;
            if target_position >= text_len {
                return Err(MolmoSparseAddError::TargetOutOfRange {
                    feature_row,
                    target_position: target,
                    text_len,
                });
            }
            if let Some(first_feature_row) = targets.insert(target_position, feature_row) {
                return Err(MolmoSparseAddError::DuplicateTarget {
                    first_feature_row,
                    feature_row,
                    target_position,
                });
            }
            pairs.push(MolmoSparseAddPair {
                feature_row,
                target_position,
            });
        }
        debug_assert_eq!(
            pairs
                .iter()
                .map(|pair| pair.target_position)
                .collect::<HashSet<_>>()
                .len(),
            pairs.len()
        );
        Ok(Self {
            pairs,
            feature_rows: image_input_idx.len(),
            text_len,
            hidden_size,
        })
    }

    #[must_use]
    pub fn pairs(&self) -> &[MolmoSparseAddPair] {
        &self.pairs
    }

    /// Add F32 projected rows to already post-scaled F32 OLMo embeddings.
    pub fn apply(
        &self,
        text_embeddings: &mut [f32],
        projected_features: &[f32],
    ) -> Result<(), MolmoSparseAddError> {
        self.apply_in_dtype(
            text_embeddings,
            projected_features,
            MolmoEmbeddingDType::Float32,
        )
    }

    /// Cast projected rows to the text-embedding dtype, add in that dtype, and
    /// widen the rounded result to F32 for the prepared-prefill boundary.
    pub fn apply_in_dtype(
        &self,
        text_embeddings: &mut [f32],
        projected_features: &[f32],
        dtype: MolmoEmbeddingDType,
    ) -> Result<(), MolmoSparseAddError> {
        let expected_text = checked_product(&[self.text_len, self.hidden_size])?;
        let expected_features = checked_product(&[self.feature_rows, self.hidden_size])?;
        for (tensor, expected, actual) in [
            ("text embeddings", expected_text, text_embeddings.len()),
            (
                "projected features",
                expected_features,
                projected_features.len(),
            ),
        ] {
            if expected != actual {
                return Err(MolmoSparseAddError::TensorCount {
                    tensor,
                    expected,
                    actual,
                });
            }
        }
        if let Some((flat_index, _)) = text_embeddings
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(MolmoSparseAddError::NonFinite {
                tensor: "text embeddings",
                flat_index,
            });
        }
        if let Some((flat_index, _)) = projected_features
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(MolmoSparseAddError::NonFinite {
                tensor: "projected features",
                flat_index,
            });
        }
        for pair in &self.pairs {
            let text_start = pair.target_position * self.hidden_size;
            let feature_start = pair.feature_row * self.hidden_size;
            for offset in 0..self.hidden_size {
                let text = dtype.round(text_embeddings[text_start + offset]);
                let feature = dtype.round(projected_features[feature_start + offset]);
                if !dtype.round(text + feature).is_finite() {
                    return Err(MolmoSparseAddError::NonFinite {
                        tensor: "merged embeddings",
                        flat_index: text_start + offset,
                    });
                }
            }
        }
        for pair in &self.pairs {
            let text_start = pair.target_position * self.hidden_size;
            let feature_start = pair.feature_row * self.hidden_size;
            for offset in 0..self.hidden_size {
                let text = dtype.round(text_embeddings[text_start + offset]);
                let feature = dtype.round(projected_features[feature_start + offset]);
                text_embeddings[text_start + offset] = dtype.round(text + feature);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "prepared_molmo_tests.rs"]
mod tests;
