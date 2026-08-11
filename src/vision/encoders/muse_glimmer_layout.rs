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

//! Pure Muse Glimmer vision layout helpers.
//!
//! These helpers own the host-side index arithmetic shared by the future tower:
//! per-frame full-attention boundaries and the window-attention reorder plus
//! inverse restore. They do not load or execute any checkpoint weights.

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MuseWindowIndexPlan {
    pub indices: Vec<i32>,
    pub inverse_indices: Vec<i32>,
    pub cu_window_seqlens: Vec<i32>,
}

pub fn full_cu_seqlens(grid_thw: &[(i32, i32, i32)]) -> Result<Vec<i32>, String> {
    let mut out = vec![0];
    let mut total = 0;
    for &(t, h, w) in grid_thw {
        validate_grid((t, h, w))?;
        let frame_tokens = h * w;
        for _ in 0..t {
            total += frame_tokens;
            out.push(total);
        }
    }
    Ok(out)
}

pub fn window_index_plan(
    grid_thw: &[(i32, i32, i32)],
    window_patch_size: i32,
) -> Result<MuseWindowIndexPlan, String> {
    if window_patch_size <= 0 {
        return Err("Muse Glimmer window_patch_size must be positive".to_string());
    }

    let total_tokens: usize = grid_thw
        .iter()
        .map(|&(t, h, w)| {
            validate_grid((t, h, w))?;
            Ok((t * h * w) as usize)
        })
        .collect::<Result<Vec<_>, String>>()?
        .into_iter()
        .sum();

    let mut indices = Vec::with_capacity(total_tokens);
    let mut cu_window_seqlens = vec![0];
    let mut base = 0;
    let mut cumulative = 0;
    for &(t, h, w) in grid_thw {
        for frame in 0..t {
            let frame_base = base + frame * h * w;
            let mut row = 0;
            while row < h {
                let row_end = (row + window_patch_size).min(h);
                let mut col = 0;
                while col < w {
                    let col_end = (col + window_patch_size).min(w);
                    let before = indices.len() as i32;
                    for y in row..row_end {
                        for x in col..col_end {
                            indices.push(frame_base + y * w + x);
                        }
                    }
                    cumulative += indices.len() as i32 - before;
                    cu_window_seqlens.push(cumulative);
                    col += window_patch_size;
                }
                row += window_patch_size;
            }
        }
        base += t * h * w;
    }

    let mut inverse_indices = vec![0; indices.len()];
    for (new_pos, &old_pos) in indices.iter().enumerate() {
        let old_pos = old_pos as usize;
        if old_pos >= inverse_indices.len() {
            return Err("Muse Glimmer window index exceeded token count".to_string());
        }
        inverse_indices[old_pos] = new_pos as i32;
    }

    Ok(MuseWindowIndexPlan {
        indices,
        inverse_indices,
        cu_window_seqlens,
    })
}

fn validate_grid(grid: (i32, i32, i32)) -> Result<(), String> {
    let (t, h, w) = grid;
    if t <= 0 || h <= 0 || w <= 0 {
        Err(format!("Muse Glimmer grid must be positive, got {grid:?}"))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_cu_seqlens_splits_each_static_frame() {
        assert_eq!(
            full_cu_seqlens(&[(1, 4, 4), (2, 2, 3)]).unwrap(),
            vec![0, 16, 22, 28]
        );
    }

    #[test]
    fn window_plan_is_identity_when_grid_fits_single_window() {
        let plan = window_index_plan(&[(1, 4, 4)], 32).unwrap();
        assert_eq!(plan.indices, (0..16).collect::<Vec<_>>());
        assert_eq!(plan.inverse_indices, (0..16).collect::<Vec<_>>());
        assert_eq!(plan.cu_window_seqlens, vec![0, 16]);
    }

    #[test]
    fn window_plan_reorders_and_inverse_restores() {
        let plan = window_index_plan(&[(1, 3, 5)], 2).unwrap();
        assert_eq!(
            plan.indices,
            vec![0, 1, 5, 6, 2, 3, 7, 8, 4, 9, 10, 11, 12, 13, 14]
        );
        assert_eq!(plan.cu_window_seqlens, vec![0, 4, 8, 10, 12, 14, 15]);

        for (new_pos, old_pos) in plan.indices.iter().enumerate() {
            assert_eq!(plan.inverse_indices[*old_pos as usize], new_pos as i32);
        }
        let reordered = plan.indices.clone();
        let restored = plan
            .inverse_indices
            .iter()
            .map(|&idx| reordered[idx as usize])
            .collect::<Vec<_>>();
        assert_eq!(restored, (0..15).collect::<Vec<_>>());
    }
}
