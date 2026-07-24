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

//! Molmo2-specific indexed-pooling and additive-merge contracts.
//!
//! Molmo2 processor output contains negative pooling sentinels. Native graph
//! inputs retain those signed indices verbatim; this module separately derives
//! safe gather indices and a valid mask so clamped patch zero can never
//! contribute to a pooled token. Molmo v1's sparse `image_input_idx` mapping is
//! deliberately not represented here.

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Molmo2InputError {
    ZeroHiddenSize,
    EmptyPoolingGroup,
    PoolingShape {
        values: usize,
        groups: usize,
        group_size: usize,
    },
    PoolingIndex {
        group: usize,
        offset: usize,
        value: i32,
        patch_count: usize,
    },
    EmbeddingShape {
        values: usize,
        tokens: usize,
        hidden_size: usize,
    },
    ProjectedShape {
        values: usize,
        tokens: usize,
        hidden_size: usize,
    },
    ProjectedTokenCount {
        positions: usize,
        projected_tokens: usize,
    },
    ActiveTokenCount {
        active_groups: usize,
        grid_groups: usize,
        prompt_positions: usize,
        all_invalid_groups: Vec<usize>,
    },
    NonFinite {
        tensor: &'static str,
        index: usize,
    },
    ShapeOverflow,
}

impl fmt::Display for Molmo2InputError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroHiddenSize => f.write_str("Molmo2 hidden size must be positive"),
            Self::EmptyPoolingGroup => {
                f.write_str("Molmo2 pooling groups must contain at least one patch index")
            }
            Self::PoolingShape {
                values,
                groups,
                group_size,
            } => write!(
                f,
                "Molmo2 pooling has {values} indices, expected {groups} * {group_size}"
            ),
            Self::PoolingIndex {
                group,
                offset,
                value,
                patch_count,
            } => write!(
                f,
                "Molmo2 pooling group {group} offset {offset} nonnegative index {value} is outside [0,{patch_count})"
            ),
            Self::EmbeddingShape {
                values,
                tokens,
                hidden_size,
            } => write!(
                f,
                "Molmo2 text embeddings have {values} values, expected {tokens} * {hidden_size}"
            ),
            Self::ProjectedShape {
                values,
                tokens,
                hidden_size,
            } => write!(
                f,
                "Molmo2 projected features have {values} values, expected {tokens} * {hidden_size}"
            ),
            Self::ProjectedTokenCount {
                positions,
                projected_tokens,
            } => write!(
                f,
                "Molmo2 prompt has {positions} image_patch_id positions but projector returned {projected_tokens} tokens"
            ),
            Self::ActiveTokenCount {
                active_groups,
                grid_groups,
                prompt_positions,
                all_invalid_groups,
            } => write!(
                f,
                "Molmo2 projected active rows ({active_groups}) and grid rows ({grid_groups}) must match prompt image_patch_id positions ({prompt_positions}) before native invocation; all-invalid pooling groups: {all_invalid_groups:?}"
            ),
            Self::NonFinite { tensor, index } => {
                write!(
                    f,
                    "Molmo2 {tensor} contains a non-finite value at flat index {index}"
                )
            }
            Self::ShapeOverflow => f.write_str("Molmo2 tensor shape overflowed"),
        }
    }
}

impl std::error::Error for Molmo2InputError {}

/// Signed processor indices plus the separate values consumed by a safe gather.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Molmo2SafePooling {
    /// Original processor output, including every negative invalid sentinel.
    pub signed_indices: Vec<i32>,
    /// Gather-only indices. Invalid entries are clamped to patch zero.
    pub gather_indices: Vec<i32>,
    /// One byte per index (`1` valid, `0` invalid), suitable for an IREE bool input.
    pub valid_mask: Vec<u8>,
    /// Number of valid patches in each group. Zero is retained for all-invalid groups.
    pub valid_counts: Vec<i32>,
    pub groups: usize,
    pub group_size: usize,
}

impl Molmo2SafePooling {
    /// Validate processor indices without erasing their negative sentinel values.
    ///
    /// The graph must multiply gathered K/V values by `valid_mask` before every
    /// reduction. Query-mean denominators use `max(valid_counts, 1)`, so an
    /// all-invalid group is deterministic and cannot divide by zero.
    pub fn prepare(
        signed_indices: &[i32],
        groups: usize,
        group_size: usize,
        patch_count: usize,
    ) -> Result<Self, Molmo2InputError> {
        if group_size == 0 {
            return Err(Molmo2InputError::EmptyPoolingGroup);
        }
        let expected = groups
            .checked_mul(group_size)
            .ok_or(Molmo2InputError::ShapeOverflow)?;
        if signed_indices.len() != expected {
            return Err(Molmo2InputError::PoolingShape {
                values: signed_indices.len(),
                groups,
                group_size,
            });
        }

        let mut gather_indices = Vec::with_capacity(expected);
        let mut valid_mask = Vec::with_capacity(expected);
        let mut valid_counts = vec![0i32; groups];
        for (flat_index, &value) in signed_indices.iter().enumerate() {
            let group = flat_index / group_size;
            let offset = flat_index % group_size;
            if value < 0 {
                gather_indices.push(0);
                valid_mask.push(0);
                continue;
            }
            let index = usize::try_from(value).map_err(|_| Molmo2InputError::PoolingIndex {
                group,
                offset,
                value,
                patch_count,
            })?;
            if index >= patch_count {
                return Err(Molmo2InputError::PoolingIndex {
                    group,
                    offset,
                    value,
                    patch_count,
                });
            }
            gather_indices.push(value);
            valid_mask.push(1);
            valid_counts[group] += 1;
        }

        Ok(Self {
            signed_indices: signed_indices.to_vec(),
            gather_indices,
            valid_mask,
            valid_counts,
            groups,
            group_size,
        })
    }

    /// Safe denominators for query means under the configured attention-mask policy.
    ///
    /// Masked pooling averages only valid patches and clamps all-invalid groups
    /// to one. Unmasked pooling matches the MLX reference by averaging the
    /// zero-masked values over the full fixed-size pooling window.
    #[must_use]
    pub fn mean_denominators(&self, pooling_attention_mask: bool) -> Vec<i32> {
        if pooling_attention_mask {
            self.valid_counts
                .iter()
                .map(|&count| count.max(1))
                .collect()
        } else {
            vec![i32::try_from(self.group_size).unwrap_or(i32::MAX); self.groups]
        }
    }

    pub(crate) fn active_groups_for_prompt(
        &self,
        grid_groups: usize,
        prompt_positions: usize,
    ) -> Result<Vec<usize>, Molmo2InputError> {
        let active = self
            .valid_counts
            .iter()
            .enumerate()
            .filter_map(|(index, &count)| (count > 0).then_some(index))
            .collect::<Vec<_>>();
        if grid_groups != prompt_positions || active.len() != prompt_positions {
            let all_invalid_groups = self
                .valid_counts
                .iter()
                .enumerate()
                .filter_map(|(index, &count)| (count == 0).then_some(index))
                .collect::<Vec<_>>();
            return Err(Molmo2InputError::ActiveTokenCount {
                active_groups: active.len(),
                grid_groups,
                prompt_positions,
                all_invalid_groups,
            });
        }
        Ok(active)
    }
}

/// Find logical image positions and add projected Molmo2 features in order.
///
/// This intentionally requires the caller's logical expanded token sequence;
/// it never accepts Molmo v1 `image_input_idx` coordinates.
pub fn add_projected_features(
    token_ids: &[i32],
    image_patch_id: i32,
    text_embeddings: &mut [f32],
    hidden_size: usize,
    projected_features: &[f32],
) -> Result<Vec<usize>, Molmo2InputError> {
    if hidden_size == 0 {
        return Err(Molmo2InputError::ZeroHiddenSize);
    }
    let expected_embeddings = token_ids
        .len()
        .checked_mul(hidden_size)
        .ok_or(Molmo2InputError::ShapeOverflow)?;
    if text_embeddings.len() != expected_embeddings {
        return Err(Molmo2InputError::EmbeddingShape {
            values: text_embeddings.len(),
            tokens: token_ids.len(),
            hidden_size,
        });
    }
    if let Some((index, _)) = text_embeddings
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(Molmo2InputError::NonFinite {
            tensor: "text embeddings",
            index,
        });
    }
    if !projected_features.len().is_multiple_of(hidden_size) {
        return Err(Molmo2InputError::ProjectedShape {
            values: projected_features.len(),
            tokens: projected_features.len() / hidden_size,
            hidden_size,
        });
    }
    if let Some((index, _)) = projected_features
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(Molmo2InputError::NonFinite {
            tensor: "projected features",
            index,
        });
    }

    let positions = token_ids
        .iter()
        .enumerate()
        .filter_map(|(index, &token)| (token == image_patch_id).then_some(index))
        .collect::<Vec<_>>();
    let projected_tokens = projected_features.len() / hidden_size;
    if positions.len() != projected_tokens {
        return Err(Molmo2InputError::ProjectedTokenCount {
            positions: positions.len(),
            projected_tokens,
        });
    }

    // Preflight every sum so an overflow cannot leave a partially-mutated
    // prepared payload behind.
    for (feature_index, &position) in positions.iter().enumerate() {
        let destination = position * hidden_size;
        let source = feature_index * hidden_size;
        for offset in 0..hidden_size {
            if !(text_embeddings[destination + offset] + projected_features[source + offset])
                .is_finite()
            {
                return Err(Molmo2InputError::NonFinite {
                    tensor: "merged embeddings",
                    index: destination + offset,
                });
            }
        }
    }
    for (feature_index, &position) in positions.iter().enumerate() {
        let destination = position * hidden_size;
        let source = feature_index * hidden_size;
        for offset in 0..hidden_size {
            text_embeddings[destination + offset] += projected_features[source + offset];
        }
    }
    Ok(positions)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn negative_pooling_sentinels_are_preserved_but_cannot_leak_patch_zero() {
        let pooling = Molmo2SafePooling::prepare(&[4, -1, 2, -7, -1, -1, -1, -1], 2, 4, 5)
            .expect("valid signed pooling");
        assert_eq!(pooling.signed_indices, vec![4, -1, 2, -7, -1, -1, -1, -1]);
        assert_eq!(pooling.gather_indices, vec![4, 0, 2, 0, 0, 0, 0, 0]);
        assert_eq!(pooling.valid_mask, vec![1, 0, 1, 0, 0, 0, 0, 0]);
        assert_eq!(pooling.valid_counts, vec![2, 0]);
        assert_eq!(pooling.mean_denominators(true), vec![2, 1]);
        assert_eq!(pooling.mean_denominators(false), vec![4, 4]);

        let patches = [100.0, 1.0, 2.0, 3.0, 4.0];
        let masked_sums = (0..pooling.groups)
            .map(|group| {
                (0..pooling.group_size)
                    .map(|offset| {
                        let flat = group * pooling.group_size + offset;
                        patches[pooling.gather_indices[flat] as usize]
                            * f32::from(pooling.valid_mask[flat])
                    })
                    .sum::<f32>()
            })
            .collect::<Vec<_>>();
        assert_eq!(masked_sums, vec![6.0, 0.0]);
    }

    #[test]
    fn pooling_rejects_shape_and_positive_out_of_range_indices() {
        assert!(matches!(
            Molmo2SafePooling::prepare(&[0, 1, 2], 1, 4, 3),
            Err(Molmo2InputError::PoolingShape { .. })
        ));
        assert!(matches!(
            Molmo2SafePooling::prepare(&[0, 1, 3, -1], 1, 4, 3),
            Err(Molmo2InputError::PoolingIndex {
                group: 0,
                offset: 2,
                value: 3,
                ..
            })
        ));
    }

    #[test]
    fn active_rows_are_rejected_before_native_invoke_when_cardinality_drifts() {
        let partial = Molmo2SafePooling::prepare(&[0, -1, -1, -1, 1, -1, -1, -1], 2, 4, 2)
            .expect("partially filled groups are valid");
        assert_eq!(partial.active_groups_for_prompt(2, 2).unwrap(), vec![0, 1]);

        let all_invalid = Molmo2SafePooling::prepare(&[0, -1, -1, -1, -1, -1, -1, -1], 2, 4, 2)
            .expect("negative sentinels remain valid inputs");
        assert!(matches!(
            all_invalid.active_groups_for_prompt(2, 2),
            Err(Molmo2InputError::ActiveTokenCount {
                active_groups: 1,
                grid_groups: 2,
                prompt_positions: 2,
                all_invalid_groups,
            }) if all_invalid_groups == vec![1]
        ));

        assert!(matches!(
            partial.active_groups_for_prompt(2, 1),
            Err(Molmo2InputError::ActiveTokenCount {
                active_groups: 2,
                grid_groups: 2,
                prompt_positions: 1,
                all_invalid_groups,
            }) if all_invalid_groups.is_empty()
        ));
    }

    #[test]
    fn image_patch_features_are_added_in_scanned_position_order() {
        let tokens = [9, 151_938, 8, 151_938];
        let mut embeddings = vec![
            1.0, 1.0, // text
            10.0, 20.0, // image slot zero
            2.0, 2.0, // text
            30.0, 40.0, // image slot one
        ];
        let positions =
            add_projected_features(&tokens, 151_938, &mut embeddings, 2, &[0.5, 1.5, 2.5, 3.5])
                .expect("matching image positions");
        assert_eq!(positions, vec![1, 3]);
        assert_eq!(embeddings, vec![1.0, 1.0, 10.5, 21.5, 2.0, 2.0, 32.5, 43.5]);
        assert_ne!(embeddings[2..4], [0.5, 1.5]);
    }

    #[test]
    fn additive_merge_rejects_count_mismatch_and_non_finite_values() {
        let tokens = [151_938, 7];
        let mut embeddings = vec![0.0; 4];
        assert!(matches!(
            add_projected_features(&tokens, 151_938, &mut embeddings, 0, &[]),
            Err(Molmo2InputError::ZeroHiddenSize)
        ));
        assert!(matches!(
            add_projected_features(&tokens, 151_938, &mut embeddings, 2, &[]),
            Err(Molmo2InputError::ProjectedTokenCount {
                positions: 1,
                projected_tokens: 0
            })
        ));
        assert!(matches!(
            add_projected_features(&tokens, 151_938, &mut embeddings, 2, &[f32::NAN, 0.0]),
            Err(Molmo2InputError::NonFinite {
                tensor: "projected features",
                index: 0
            })
        ));

        let mut overflowing = vec![f32::MAX, 7.0, 0.0, 0.0];
        let snapshot = overflowing.clone();
        assert!(matches!(
            add_projected_features(&tokens, 151_938, &mut overflowing, 2, &[f32::MAX, 1.0]),
            Err(Molmo2InputError::NonFinite {
                tensor: "merged embeddings",
                index: 0
            })
        ));
        assert_eq!(overflowing, snapshot, "overflow rejection must be atomic");
    }
}
