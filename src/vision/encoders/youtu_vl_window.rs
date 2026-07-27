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

//! Pure-CPU windowed-attention bookkeeping for the Youtu-VL vision encoder.
//!
//! `get_window_index` is the only reason the encoder file would otherwise
//! exceed 500 lines. Lifting it into its own module keeps the encoder body
//! focused on tensor work and makes the windowed-attention permutation logic
//! independently testable.
//!
//! Used by: `YoutuVLVisionEncoder::forward_with_spatial`.

/// Compute the window-index permutation and per-window cumulative sequence
/// lengths for Youtu-VL's windowed attention.
///
/// Given `spatial_shapes` of `(h_patches, w_patches)` per image and a
/// `merger_window_size` measured in merged-patch units, returns:
///
/// - `window_index` — a permutation of `0..sum(h*w/merge^2)` that reorders
///   merged patches into window-major layout (so a single SDPA over a
///   contiguous slice of length `merge^2 * window_tokens` attends within
///   one window).
/// - `cu_window_seqlens` — strictly-increasing cumulative seqlens (in
///   pre-merge tokens) used as the segment boundaries for windowed SDPA.
///   Duplicate entries (which would cause empty SDPA slices) are removed.
pub(super) fn get_window_index(
    spatial_shapes: &[(i32, i32)],
    spatial_merge_size: i32,
    window_size: i32,
    patch_size: i32,
    spatial_merge_unit: i32,
) -> (Vec<i32>, Vec<i32>) {
    let merger_window_size = window_size / spatial_merge_size / patch_size;

    let mut window_index: Vec<i32> = Vec::new();
    let mut cu_window_seqlens: Vec<i32> = vec![0];
    let mut window_index_id: i32 = 0;

    for &(h, w) in spatial_shapes {
        let llm_h = h / spatial_merge_size;
        let llm_w = w / spatial_merge_size;
        let total = llm_h * llm_w;

        // Build a [llm_h, llm_w] index array padded with -100 to a multiple
        // of merger_window_size.
        let pad_h = if llm_h % merger_window_size == 0 {
            0
        } else {
            merger_window_size - llm_h % merger_window_size
        };
        let pad_w = if llm_w % merger_window_size == 0 {
            0
        } else {
            merger_window_size - llm_w % merger_window_size
        };
        let num_windows_h = (llm_h + pad_h) / merger_window_size;
        let num_windows_w = (llm_w + pad_w) / merger_window_size;
        let padded_h = llm_h + pad_h;
        let padded_w = llm_w + pad_w;

        let mut padded = vec![-100i32; (padded_h * padded_w) as usize];
        for hi in 0..llm_h {
            for wi in 0..llm_w {
                let src = (hi * llm_w + wi) as usize;
                let dst = (hi * padded_w + wi) as usize;
                padded[dst] = src as i32;
            }
        }

        // Reorder to window-major: [num_windows_h, num_windows_w, ws, ws].
        let ws = merger_window_size;
        let num_windows = (num_windows_h * num_windows_w) as usize;
        let ws2 = (ws * ws) as usize;
        let mut reordered = vec![-100i32; num_windows * ws2];
        for wh in 0..num_windows_h {
            for ww in 0..num_windows_w {
                for sh in 0..ws {
                    for sw in 0..ws {
                        let src_h = wh * ws + sh;
                        let src_w = ww * ws + sw;
                        let src = (src_h * padded_w + src_w) as usize;
                        let win = wh * num_windows_w + ww;
                        let dst = (win * ws * ws + sh * ws + sw) as usize;
                        reordered[dst] = padded[src];
                    }
                }
            }
        }

        // Per-window valid count.
        let mut seqlens: Vec<i32> = Vec::with_capacity(num_windows);
        for win in 0..num_windows {
            let mut count = 0i32;
            for j in 0..ws2 {
                if reordered[win * ws2 + j] != -100 {
                    count += 1;
                }
            }
            seqlens.push(count);
        }

        // Append valid (non-padding) indices in window-major order, biased by
        // the running window_index_id offset.
        for &val in &reordered {
            if val != -100 {
                window_index.push(val + window_index_id);
            }
        }

        let last_cum = *cu_window_seqlens.last().unwrap();
        let mut cum = last_cum;
        for &sl in &seqlens {
            cum += sl * spatial_merge_unit;
            cu_window_seqlens.push(cum);
        }

        window_index_id += total;
    }

    // Deduplicate the cumulative seqlens (consecutive duplicates would create
    // empty SDPA windows).
    let mut deduped: Vec<i32> = Vec::with_capacity(cu_window_seqlens.len());
    let mut seen = std::collections::HashSet::new();
    for &val in &cu_window_seqlens {
        if seen.insert(val) {
            deduped.push(val);
        }
    }

    (window_index, deduped)
}

/// Compute the inverse permutation of `window_index` so the encoder can
/// restore the original spatial token order after the merger projects
/// window-grouped tokens.
pub(super) fn reverse_window_indices(window_index: &[i32]) -> Vec<i32> {
    let mut indexed: Vec<(i32, usize)> = window_index
        .iter()
        .enumerate()
        .map(|(i, &v)| (v, i))
        .collect();
    indexed.sort_by_key(|&(v, _)| v);
    let mut reverse = vec![0i32; window_index.len()];
    for (original_position, &(_, window_position)) in indexed.iter().enumerate() {
        reverse[original_position] = window_position as i32;
    }
    reverse
}

#[cfg(test)]
mod tests {
    use super::reverse_window_indices;

    #[test]
    fn reverse_indices_restore_original_group_order() {
        // A two-row, three-window layout with 2x2 windows. This permutation
        // contains a four-cycle, so it cannot accidentally pass as its own inverse.
        let window_index = vec![0, 1, 6, 7, 2, 3, 8, 9, 4, 5, 10, 11];
        let window_order = vec![
            "g0", "g1", "g6", "g7", "g2", "g3", "g8", "g9", "g4", "g5", "g10", "g11",
        ];
        let reverse = reverse_window_indices(&window_index);
        let restored = reverse
            .iter()
            .map(|&position| window_order[position as usize])
            .collect::<Vec<_>>();
        assert_ne!(reverse, window_index);
        assert_eq!(reverse, vec![0, 1, 4, 5, 8, 9, 2, 3, 6, 7, 10, 11]);
        assert_eq!(
            restored,
            vec![
                "g0", "g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9", "g10", "g11",
            ]
        );
    }
}
