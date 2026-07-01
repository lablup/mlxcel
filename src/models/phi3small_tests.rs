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

//! Unit tests for the Phi-3-small blocksparse attention path.
//!
//! Config-parse tests are checkpoint-free. The mask/selection and parity tests
//! build small `MlxArray`s and run on the Metal device (`--features
//! metal,accelerate`); they do not need a real checkpoint.

use super::{ModelArgs, build_blocksparse_mask};
use mlxcel_core::MlxArray;

// ── Config surface ──────────────────────────────────────────────────────────

/// Trimmed config matching `microsoft/Phi-3-small-8k-instruct`.
const PHI3_SMALL_CONFIG: &str = r#"{
    "model_type": "phi3small",
    "hidden_size": 4096,
    "dense_attention_every_n_layers": 4,
    "ff_intermediate_size": 14336,
    "gegelu_limit": 20.0,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "layer_norm_epsilon": 1e-5,
    "vocab_size": 100352,
    "num_key_value_heads": 8,
    "blocksparse_block_size": 64,
    "blocksparse_num_local_blocks": 16,
    "blocksparse_vert_stride": 8
}"#;

#[test]
fn config_parses_blocksparse_fields() {
    let args: ModelArgs = serde_json::from_str(PHI3_SMALL_CONFIG).unwrap();
    assert_eq!(args.blocksparse_block_size, 64);
    assert_eq!(args.blocksparse_num_local_blocks, 16);
    assert_eq!(args.blocksparse_vert_stride, 8);
    assert_eq!(args.dense_attention_every_n_layers, 4);
    assert_eq!(args.head_dim(), 128);
}

#[test]
fn config_defaults_apply_when_blocksparse_fields_absent() {
    // Same config without the blocksparse_* keys: defaults must kick in.
    let args: ModelArgs = serde_json::from_str(
        r#"{
        "model_type": "phi3small",
        "hidden_size": 4096,
        "dense_attention_every_n_layers": 4,
        "ff_intermediate_size": 14336,
        "gegelu_limit": 20.0,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "layer_norm_epsilon": 1e-5,
        "vocab_size": 100352,
        "num_key_value_heads": 8
    }"#,
    )
    .unwrap();
    assert_eq!(args.blocksparse_block_size, 64);
    assert_eq!(args.blocksparse_num_local_blocks, 16);
    assert_eq!(args.blocksparse_vert_stride, 8);
}

/// The per-layer `block_sparse` flag follows the Microsoft reference: a layer is
/// dense when `(layer_idx + 1) % dense_attention_every_n_layers == 0`, else it
/// is blocksparse.
#[test]
fn block_sparse_flag_marks_every_nth_layer_dense() {
    let n = 4usize;
    let is_sparse = |layer_idx: usize| !(layer_idx + 1).is_multiple_of(n);
    // Layers 0,1,2 blocksparse; layer 3 dense; then it repeats.
    assert!(is_sparse(0));
    assert!(is_sparse(1));
    assert!(is_sparse(2));
    assert!(!is_sparse(3));
    assert!(is_sparse(4));
    assert!(!is_sparse(7));
    assert!(!is_sparse(31));
}

// ── Mask / selection ────────────────────────────────────────────────────────

/// Host reference for a single mask cell, mirroring
/// `mlx_lm/models/phi3small.py::Attention._block_sparse_mask` (block pattern)
/// plus token-level causality.
#[allow(clippy::too_many_arguments)]
fn ref_attend(
    head: i32,
    q_row: i32,
    k_col: i32,
    q_len: i32,
    kv_len: i32,
    offset: i32,
    block_size: i32,
    local_blocks: i32,
    vert_stride: i32,
) -> bool {
    let kv_blocks = (kv_len + block_size - 1) / block_size;
    let q_blocks = (q_len + block_size - 1) / block_size;
    let front_pad = q_blocks * block_size - q_len;
    let offset_blocks = kv_blocks - q_blocks;

    let qb = offset_blocks + (front_pad + q_row) / block_size;
    let kb = k_col / block_size;

    let block_ok =
        qb >= kb && ((qb - kb < local_blocks) || ((kb + head + 1).rem_euclid(vert_stride) == 0));
    let token_ok = (offset + q_row) >= k_col;
    block_ok && token_ok
}

/// Read a single additive-mask cell `[0, head, q, k]` back to the host.
fn mask_cell(mask: &MlxArray, head: i32, q: i32, k: i32) -> f32 {
    let s = mlxcel_core::slice(mask, &[0, head, q, k], &[1, head + 1, q + 1, k + 1]);
    mlxcel_core::eval(&s);
    mlxcel_core::item_f32(&s)
}

fn is_attend(v: f32) -> bool {
    // Attended cells are exactly 0.0; masked cells are -inf.
    v > -1.0e30
}

/// Max absolute difference between two same-shaped arrays, as an f32 scalar.
fn max_abs_diff(a: &MlxArray, b: &MlxArray) -> f32 {
    let d = mlxcel_core::abs(&mlxcel_core::subtract(a, b));
    let m = mlxcel_core::max_all(&d);
    mlxcel_core::eval(&m);
    mlxcel_core::item_f32(&m)
}

/// RMS difference between two same-shaped arrays.
fn rms_diff(a: &MlxArray, b: &MlxArray) -> f32 {
    let d = mlxcel_core::subtract(a, b);
    let sq = mlxcel_core::square(&d);
    let s = mlxcel_core::sum_all(&sq);
    mlxcel_core::eval(&s);
    let total = mlxcel_core::item_f32(&s);
    let n: i32 = mlxcel_core::array_shape(a).iter().product();
    (total / n as f32).sqrt()
}

/// Convert an additive 0/-inf mask into a 0/1 attend indicator (f32).
fn attend_indicator(mask: &MlxArray) -> mlxcel_core::UniquePtr<MlxArray> {
    let zero = mlxcel_core::zeros(&[1], mlxcel_core::dtype::FLOAT32);
    let is_attend = mlxcel_core::greater_equal(mask, &zero); // mask (0) >= 0 -> true; -inf -> false
    let ones = mlxcel_core::ones(&[1], mlxcel_core::dtype::FLOAT32);
    mlxcel_core::where_cond(&is_attend, &ones, &zero)
}

/// The device-built mask must match the host reference formula cell-for-cell,
/// including the local-block window and per-head vertical stride, on a
/// block-aligned constructed case.
#[test]
fn blocksparse_mask_matches_reference_pattern() {
    // Small, block-aligned case so the block structure is unambiguous.
    let n_heads = 4;
    let block_size = 2;
    let local_blocks = 1; // only the query's own block is "local"
    let vert_stride = 2;
    let q_len = 8; // 4 query blocks
    let kv_len = 8; // prefill, offset 0
    let offset = 0;

    let mask = build_blocksparse_mask(
        n_heads,
        q_len,
        kv_len,
        offset,
        block_size,
        local_blocks,
        vert_stride,
    );
    mlxcel_core::eval(&mask);
    assert_eq!(
        mlxcel_core::array_shape(&mask),
        vec![1, n_heads, q_len, kv_len]
    );

    // Build the reference indicator on the host and compare in bulk.
    let mut ref_ind = Vec::with_capacity((n_heads * q_len * kv_len) as usize);
    for h in 0..n_heads {
        for q in 0..q_len {
            for k in 0..kv_len {
                let a = ref_attend(
                    h,
                    q,
                    k,
                    q_len,
                    kv_len,
                    offset,
                    block_size,
                    local_blocks,
                    vert_stride,
                );
                ref_ind.push(if a { 1.0f32 } else { 0.0f32 });
            }
        }
    }
    let ref_arr = mlxcel_core::from_slice_f32(&ref_ind, &[1, n_heads, q_len, kv_len]);
    let dev_ind = attend_indicator(&mask);
    assert_eq!(
        max_abs_diff(&dev_ind, &ref_arr),
        0.0,
        "device mask disagrees with the reference blocksparse pattern"
    );
}

/// Explicitly assert the selected key-BLOCK set for one query, so the local +
/// vertical-stride semantics are pinned independent of the bulk comparison.
#[test]
fn blocksparse_selected_key_blocks_are_local_plus_vertical() {
    let n_heads = 4;
    let block_size = 2;
    let local_blocks = 2; // current + previous block are local
    let vert_stride = 3;
    let q_len = 12; // 6 query blocks (indices 0..6)
    let kv_len = 12;
    let offset = 0;

    let mask = build_blocksparse_mask(
        n_heads,
        q_len,
        kv_len,
        offset,
        block_size,
        local_blocks,
        vert_stride,
    );
    mlxcel_core::eval(&mask);

    // Look at the last query row (absolute pos 11 -> query block 5) for head 0.
    let head = 0;
    let q_row = q_len - 1; // 11, block 5
    let q_block = 5;

    // Expected attended key blocks: causal (kb <= 5) AND
    //   local: q_block - kb < 2  => kb in {4, 5}
    //   vertical: (kb + head + 1) % 3 == 0 => (kb + 1) % 3 == 0 => kb in {2, 5}
    // Union over kb in 0..=5 => {2, 4, 5}.
    let mut expected_blocks: Vec<i32> = Vec::new();
    for kb in 0..=q_block {
        let local = q_block - kb < local_blocks;
        let vert = (kb + head + 1).rem_euclid(vert_stride) == 0;
        if local || vert {
            expected_blocks.push(kb);
        }
    }
    assert_eq!(expected_blocks, vec![2, 4, 5]);

    // Every key column inside an expected block is attended; every column of a
    // non-selected causal block is masked.
    for kb in 0..=q_block {
        let want = expected_blocks.contains(&kb);
        for within in 0..block_size {
            let k = kb * block_size + within;
            let cell = mask_cell(&mask, head, q_row, k);
            assert_eq!(
                is_attend(cell),
                want,
                "head {head} q_row {q_row} key {k} (block {kb}): attend={} expected {want}",
                is_attend(cell)
            );
        }
    }
}

// ── Short-context parity with the dense fallback ────────────────────────────

/// Deterministic pseudo-random f32 buffer in [-1, 1) via a small LCG.
fn pseudo_random(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let bits = (state >> 33) as u32;
            (bits as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

fn make_qkv(
    n_heads: i32,
    n_kv_heads: i32,
    seq: i32,
    head_dim: i32,
) -> (
    mlxcel_core::UniquePtr<MlxArray>,
    mlxcel_core::UniquePtr<MlxArray>,
    mlxcel_core::UniquePtr<MlxArray>,
) {
    let q_n = (n_heads * seq * head_dim) as usize;
    let kv_n = (n_kv_heads * seq * head_dim) as usize;
    let q = mlxcel_core::from_slice_f32(&pseudo_random(q_n, 1), &[1, n_heads, seq, head_dim]);
    let k = mlxcel_core::from_slice_f32(&pseudo_random(kv_n, 2), &[1, n_kv_heads, seq, head_dim]);
    let v = mlxcel_core::from_slice_f32(&pseudo_random(kv_n, 3), &[1, n_kv_heads, seq, head_dim]);
    (q, k, v)
}

/// When the whole sequence fits inside the local window, the blocksparse mask
/// is exactly the plain causal mask (no key masked that causality allows).
#[test]
fn short_context_mask_equals_causal() {
    let n_heads = 8;
    let block_size = 4;
    let local_blocks = 8; // window = 32 tokens
    let vert_stride = 3;
    let q_len = 16; // 16 <= 32 -> in-window
    let kv_len = 16;
    let offset = 0;

    let bs = build_blocksparse_mask(
        n_heads,
        q_len,
        kv_len,
        offset,
        block_size,
        local_blocks,
        vert_stride,
    );
    // Plain causal additive mask, broadcast to the head dim.
    let causal = mlxcel_core::utils::create_causal_mask(q_len, offset);
    let causal = mlxcel_core::reshape(&causal, &[1, 1, q_len, kv_len]);
    let causal = mlxcel_core::broadcast_to(&causal, &[1, n_heads, q_len, kv_len]);

    let bs_ind = attend_indicator(&bs);
    let causal_ind = attend_indicator(&causal);
    assert_eq!(
        max_abs_diff(&bs_ind, &causal_ind),
        0.0,
        "in-window blocksparse mask must equal the causal mask"
    );
}

/// Short-context RMS parity: running fused SDPA with the blocksparse mask on a
/// within-window sequence produces output RMS-equivalent to the dense fallback
/// (`causal_attention`).
#[test]
fn short_context_output_matches_dense_fallback() {
    let n_heads = 8;
    let n_kv_heads = 2;
    let head_dim = 16;
    let block_size = 4;
    let local_blocks = 8; // window = 32 tokens
    let vert_stride = 3;
    let seq = 16; // in-window
    let offset = 0;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let (q, k, v) = make_qkv(n_heads, n_kv_heads, seq, head_dim);

    // Dense fallback path (what non-sparse layers and in-window sparse layers use).
    let dense_out = mlxcel_core::causal_attention(&q, &k, &v, scale, 0.0, 0);

    // Blocksparse mask path forced over the same q/k/v.
    let bs_mask = build_blocksparse_mask(
        n_heads,
        seq,
        seq,
        offset,
        block_size,
        local_blocks,
        vert_stride,
    );
    let bs_out = unsafe {
        mlxcel_core::layers::attention_from_ptr(&q, &k, &v, scale, &*bs_mask as *const _, 0.0, 0)
    };

    let rms = rms_diff(&dense_out, &bs_out);
    assert!(
        rms < 1.0e-3,
        "short-context blocksparse output diverged from dense fallback: rms={rms}"
    );
}

// ── Long-context sparsity (not dense) ───────────────────────────────────────

/// Beyond the local window, the blocksparse mask masks at least one key that
/// plain causality would allow, i.e. it is genuinely sparser than dense.
#[test]
fn long_context_mask_is_sparser_than_causal() {
    let n_heads = 8;
    let block_size = 4;
    let local_blocks = 2; // window = 8 tokens
    let vert_stride = 4;
    let q_len = 32; // 8 blocks, well beyond the 2-block window
    let kv_len = 32;
    let offset = 0;

    let mask = build_blocksparse_mask(
        n_heads,
        q_len,
        kv_len,
        offset,
        block_size,
        local_blocks,
        vert_stride,
    );
    mlxcel_core::eval(&mask);

    // Count how many (head, q, k) cells are causal-allowed but blocksparse-masked.
    let mut masked_causal = 0usize;
    let mut example: Option<(i32, i32, i32)> = None;
    for h in 0..n_heads {
        for q in 0..q_len {
            for k in 0..kv_len {
                let causal_ok = q >= k; // offset 0
                let bs_ok = ref_attend(
                    h,
                    q,
                    k,
                    q_len,
                    kv_len,
                    offset,
                    block_size,
                    local_blocks,
                    vert_stride,
                );
                if causal_ok && !bs_ok {
                    masked_causal += 1;
                    if example.is_none() {
                        example = Some((h, q, k));
                    }
                }
            }
        }
    }
    assert!(
        masked_causal > 0,
        "long-context blocksparse mask should mask some causal keys (found none)"
    );

    // Spot-check the device mask agrees at the first such example.
    let (h, q, k) = example.unwrap();
    let cell = mask_cell(&mask, h, q, k);
    assert!(
        !is_attend(cell),
        "device mask should mask the far, non-local, non-vertical key at ({h},{q},{k})"
    );
    // And a local-diagonal key stays attended.
    assert!(is_attend(mask_cell(&mask, h, q, q)));
}

/// Long-context runtime output must differ from the dense fallback (the
/// blocksparse layer drops far, non-selected keys that dense would include).
#[test]
fn long_context_output_diverges_from_dense() {
    let n_heads = 8;
    let n_kv_heads = 2;
    let head_dim = 16;
    let block_size = 4;
    let local_blocks = 2; // window = 8 tokens
    let vert_stride = 4;
    let seq = 32; // beyond window
    let offset = 0;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let (q, k, v) = make_qkv(n_heads, n_kv_heads, seq, head_dim);

    let dense_out = mlxcel_core::causal_attention(&q, &k, &v, scale, 0.0, 0);
    let bs_mask = build_blocksparse_mask(
        n_heads,
        seq,
        seq,
        offset,
        block_size,
        local_blocks,
        vert_stride,
    );
    let bs_out = unsafe {
        mlxcel_core::layers::attention_from_ptr(&q, &k, &v, scale, &*bs_mask as *const _, 0.0, 0)
    };

    let rms = rms_diff(&dense_out, &bs_out);
    assert!(
        rms > 1.0e-4,
        "long-context blocksparse output should differ from dense: rms={rms}"
    );
}

/// Decode-shaped mask (`q_len == 1`) beyond the window is a single row of shape
/// `[1, n_heads, 1, kv_len]` and follows the per-head vertical-stride pattern.
#[test]
fn decode_single_row_mask_follows_pattern() {
    let n_heads = 8;
    let block_size = 4;
    let local_blocks = 2; // window = 8 tokens
    let vert_stride = 4;
    let q_len = 1;
    let kv_len = 40; // decode step at absolute position 39
    let offset = kv_len - q_len; // 39

    let mask = build_blocksparse_mask(
        n_heads,
        q_len,
        kv_len,
        offset,
        block_size,
        local_blocks,
        vert_stride,
    );
    mlxcel_core::eval(&mask);
    assert_eq!(mlxcel_core::array_shape(&mask), vec![1, n_heads, 1, kv_len]);

    // Full agreement with the host reference on the single query row.
    for h in 0..n_heads {
        for k in 0..kv_len {
            let want = ref_attend(
                h,
                0,
                k,
                q_len,
                kv_len,
                offset,
                block_size,
                local_blocks,
                vert_stride,
            );
            let got = is_attend(mask_cell(&mask, h, 0, k));
            assert_eq!(got, want, "decode mask mismatch head {h} key {k}");
        }
    }
}
