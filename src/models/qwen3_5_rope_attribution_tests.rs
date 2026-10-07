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

//! Attribution diagnostics for issue #2191: does the verify block's RoPE
//! call differ from classic decode's on CUDA, and is that the Qwen 3.5
//! DFlash width 2/4 residual (#1935)?
//!
//! ```text
//! MLX_CUDA_GRAPH_CACHE_SIZE=2000 \
//! cargo test --release --features cuda -p mlxcel --lib -- --ignored \
//!   --test-threads=1 --nocapture models::qwen3_5::qwen3_5_rope_attribution_tests
//! ```
//!
//! The cache size is what `mlxcel-server` applies by default; the test binary
//! does not, and the probe tests cross MLX's lifetime "Cache thrashing" limit
//! at MLX's own default of 400.

use super::qwen3_5_dflash_probe_tests::{
    apply_row_rope_override, differing, env_ids, ids_of, load_text_model, row_bytes, text_model,
    text_model_mut,
};
use crate::models::qwen3_next::verify_rope::{capture, fast_rope_rows_like_decode};
use crate::models::speculative_exactness::BlockChainExactness;
use mlxcel_core::{MlxArray, UniquePtr};

/// Deterministic pseudo-random values in [-2, 2).
fn draw(len: usize, seed: u64) -> Vec<f32> {
    let mut s = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..len)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0
        })
        .collect()
}

fn slice_axis(x: &MlxArray, axis: usize, start: i32, stop: i32) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    let mut lo = vec![0; shape.len()];
    let mut hi = shape.clone();
    lo[axis] = start;
    hi[axis] = stop;
    mlxcel_core::slice(x, &lo, &hi)
}

fn bytes(x: &MlxArray) -> Vec<u8> {
    mlxcel_core::array_to_raw_bytes(&mlxcel_core::copy(x))
}

/// Step 1.2: `fast_rope` on a `[1, H, L, D]` verify-shaped block against `L`
/// decode-shaped `L = 1` calls at `offset + row`, in bytes, with the model's
/// geometry (`dims 64`, `base 1e7`, not traditional) and layout (the block is
/// a transpose of `[1, L, H, D]`, as in `forward_hidden_with_position_ids_verify`).
#[test]
#[ignore = "needs a GPU; run on the CUDA host"]
fn op_level_fast_rope_block_versus_decode_rows() {
    let (dims, base, d) = (64, 1.0e7_f32, 256);
    let mut block_total = 0usize;
    let mut rows_total = 0usize;
    for heads in [16, 4] {
        for l in [2, 4] {
            for offset in [0, 1, 37, 158, 194, 195, 326, 1000, 4093] {
                for seed in 0..4u64 {
                    let n = (l * heads * d) as usize;
                    let x = mlxcel_core::from_slice_f32(
                        &draw(n, seed * 7919 + offset as u64),
                        &[1, l, heads, d],
                    );
                    let x = mlxcel_core::astype(&x, mlxcel_core::dtype::BFLOAT16);
                    let xt = mlxcel_core::transpose_axes(&x, &[0, 2, 1, 3]);
                    let block = mlxcel_core::fast_rope(&xt, dims, false, base, 1.0, offset);
                    let rows = fast_rope_rows_like_decode(&xt, dims, base, offset);
                    let (mut db, mut dr) = (0usize, 0usize);
                    for r in 0..l {
                        let dec_in = mlxcel_core::transpose_axes(
                            &slice_axis(&x, 1, r, r + 1),
                            &[0, 2, 1, 3],
                        );
                        let dec = bytes(&mlxcel_core::fast_rope(
                            &dec_in,
                            dims,
                            false,
                            base,
                            1.0,
                            offset + r,
                        ));
                        db += differing(&dec, &bytes(&slice_axis(&block, 2, r, r + 1)));
                        dr += differing(&dec, &bytes(&slice_axis(&rows, 2, r, r + 1)));
                    }
                    if db > 0 || dr > 0 {
                        eprintln!(
                            "[2191] op H={heads} L={l} offset={offset} seed={seed}: \
                             block differs in {db} bytes, per-row in {dr}"
                        );
                    }
                    block_total += db;
                    rows_total += dr;
                }
            }
        }
    }
    eprintln!(
        "[2191] op-level total: block-vs-decode {block_total} differing bytes, \
         per-row-vs-decode {rows_total}"
    );
    assert_eq!(rows_total, 0, "per-row RoPE must reproduce decode's bytes");
}

/// Which axis carries the token rows for a captured tag.
fn row_axis(tag: &str) -> usize {
    match tag {
        "q_rope" | "k_rope" | "attn" => 2,
        _ => 1,
    }
}

/// Step 1.3: replay the served round structure, capture every full-attention
/// layer's post-RoPE Q and K, attention output, `o_proj` and MLP output in both
/// arms, and name the first sub-op whose bytes differ on each kept row.
///
/// Env as in `block_versus_chain_byte_bisect_on_the_real_transcript`, plus
/// `MLXCEL_Q35_PROBE_ROWS` (default 60) to bound how many kept rows are
/// compared sub-op by sub-op.
#[test]
#[ignore = "needs the real Qwen 3.5 4B checkpoint, a GPU and a recorded transcript"]
fn post_rope_capture_bisect_on_the_real_transcript() {
    let Some((mut model, _dir)) = load_text_model() else {
        return;
    };
    apply_row_rope_override(&mut model);
    let text = text_model(&model);
    let Some(prompt) = env_ids("MLXCEL_Q35_PROBE_PROMPT") else {
        eprintln!("[2191] skipping: set MLXCEL_Q35_PROBE_PROMPT and MLXCEL_Q35_PROBE_REFERENCE");
        return;
    };
    let reference = env_ids("MLXCEL_Q35_PROBE_REFERENCE").expect("MLXCEL_Q35_PROBE_REFERENCE");
    let env_usize = |k: &str, d: usize| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    let block_size = env_usize("MLXCEL_Q35_PROBE_BLOCK", 4);
    let max_rows = env_usize("MLXCEL_Q35_PROBE_ROWS", 60);
    let accepts: Vec<usize> = std::env::var("MLXCEL_Q35_PROBE_ACCEPTS")
        .ok()
        .map(|v| {
            v.split(',')
                .filter(|s| !s.trim().is_empty())
                .map(|s| s.trim().parse::<usize>().expect("accept count"))
                .collect()
        })
        .unwrap_or_else(|| vec![1, 2, 3]);

    let mut chain_caches = text.make_speculative_caches();
    let _ = text.forward_prefill_with_capture_layers(&ids_of(&prompt), &mut chain_caches, &[]);
    let mut burst_caches = text.make_speculative_caches();
    let _ = text.forward_prefill_with_capture_layers(&ids_of(&prompt), &mut burst_caches, &[]);

    let (mut i, mut round, mut compared) = (0usize, 0usize, 0usize);
    let mut logit_rows = 0usize;
    let mut first_by_tag: std::collections::BTreeMap<String, usize> = Default::default();
    while i < reference.len() {
        let end = (i + block_size).min(reference.len());
        let rows = end - i;
        let keep = accepts[round % accepts.len()].clamp(1, rows);
        let fed: Vec<i32> = reference[i..end].to_vec();
        capture::begin();
        let burst = text.forward_speculative(&ids_of(&fed), &mut burst_caches, &[]);
        let burst_caps = capture::take();
        for (r, tok) in fed.iter().enumerate().take(keep) {
            capture::begin();
            let chain = text.forward_speculative(&ids_of(&[*tok]), &mut chain_caches, &[]);
            let chain_caps = capture::take();
            let dl = differing(
                &row_bytes(&chain.logits, 0),
                &row_bytes(&burst.logits, r as i32),
            );
            if dl > 0 {
                logit_rows += 1;
            }
            if compared >= max_rows {
                continue;
            }
            compared += 1;
            assert_eq!(
                chain_caps.len(),
                burst_caps.len(),
                "capture lists must pair"
            );
            let mut first: Option<String> = None;
            let mut diffs: Vec<String> = Vec::new();
            for (k, ((ct, c), (bt, b))) in chain_caps.iter().zip(&burst_caps).enumerate() {
                assert_eq!(ct, bt);
                let b_row = slice_axis(b, row_axis(bt), r as i32, r as i32 + 1);
                let d = differing(&bytes(c), &bytes(&b_row));
                if d > 0 {
                    let label = format!("fa{}:{ct}", k / 5);
                    diffs.push(format!("{label}={d}"));
                    if first.is_none() {
                        first = Some(label);
                    }
                }
            }
            let pos = prompt.len() + i + r;
            if let Some(f) = &first {
                *first_by_tag
                    .entry(f.split(':').nth(1).unwrap_or("").to_string())
                    .or_default() += 1;
                eprintln!(
                    "[2191] pos {pos} round {round} row {r}: first differing sub-op {f}; \
                     logits {dl} bytes; {}",
                    diffs.iter().take(8).cloned().collect::<Vec<_>>().join(" ")
                );
            } else if dl > 0 {
                eprintln!("[2191] pos {pos} round {round} row {r}: sub-ops equal, logits {dl}");
            }
        }
        if keep < rows {
            text.rollback_speculative_cache(
                &mut burst_caches,
                &burst.gdn_states,
                &[keep as i32 - 1],
                rows as i32,
            );
        }
        i += keep;
        round += 1;
    }
    eprintln!(
        "[2191] SUMMARY block {block_size}: {logit_rows} of {} kept rows differ in logit bytes; \
         first differing sub-op over the first {compared} rows: {first_by_tag:?}",
        reference.len()
    );
}

/// Step 1.4's MLP cell: the dense MLP's `compiled_silu` (and the `silu * up`
/// product after it) on an `[1, L, I]` block against `L` one-row calls, in
/// bytes. #2185's compiled-versus-eager cause has no visible switch here, so
/// this is a check rather than a hypothesis.
#[test]
#[ignore = "needs a GPU; run on the CUDA host"]
fn op_level_compiled_silu_block_versus_rows() {
    let inter = 9216;
    let mut total = 0usize;
    for l in [2, 4] {
        for seed in 0..4u64 {
            let n = (l * inter) as usize;
            let g = mlxcel_core::astype(
                &mlxcel_core::from_slice_f32(&draw(n, seed + 11), &[1, l, inter]),
                mlxcel_core::dtype::BFLOAT16,
            );
            let u = mlxcel_core::astype(
                &mlxcel_core::from_slice_f32(&draw(n, seed + 97), &[1, l, inter]),
                mlxcel_core::dtype::BFLOAT16,
            );
            let block = mlxcel_core::multiply(&mlxcel_core::utils::silu(&g), &u);
            for r in 0..l {
                let gr = slice_axis(&g, 1, r, r + 1);
                let ur = slice_axis(&u, 1, r, r + 1);
                let row = mlxcel_core::multiply(&mlxcel_core::utils::silu(&gr), &ur);
                total += differing(&bytes(&row), &bytes(&slice_axis(&block, 1, r, r + 1)));
            }
        }
    }
    eprintln!("[2191] compiled_silu block-vs-rows: {total} differing bytes");
}

/// Acceptance test for the probe's long draw (#2191): with the CUDA per-row
/// verify RoPE reverted, the probe must return a divergent verdict at width 2
/// or 4, and with it on, `Equal` at both. Without the first half a passing
/// probe proves nothing, because the probe without the long draw also passed
/// while the served path diverged.
#[test]
#[ignore = "needs the real Qwen 3.5 4B checkpoint and a CUDA GPU"]
fn long_probe_draw_sees_the_verify_rope_hazard_and_passes_with_the_fix() {
    if !mlxcel_core::cuda_is_available() {
        eprintln!("[2191] skipping: the long draw and the per-row RoPE are CUDA-only");
        return;
    }
    let Some((mut model, _dir)) = load_text_model() else {
        return;
    };
    text_model_mut(&mut model).set_verify_rope_rows_for_test(false);
    let reverted: Vec<(usize, BlockChainExactness)> = [2, 4]
        .into_iter()
        .map(|w| (w, text_model(&model).probe_block_chain_exactness(w)))
        .collect();
    text_model_mut(&mut model).set_verify_rope_rows_for_test(true);
    let fixed: Vec<(usize, BlockChainExactness)> = [2, 4]
        .into_iter()
        .map(|w| (w, text_model(&model).probe_block_chain_exactness(w)))
        .collect();
    for (w, v) in &reverted {
        eprintln!("[2191] fix reverted, width {w}: {}", v.reason());
    }
    for (w, v) in &fixed {
        eprintln!("[2191] fix on, width {w}: {}", v.reason());
    }
    assert!(
        reverted
            .iter()
            .any(|(_, v)| matches!(v, BlockChainExactness::Diverges { .. })),
        "with the per-row RoPE reverted the long draw must see the divergence"
    );
    for (w, v) in &fixed {
        assert!(
            v.is_equal(),
            "width {w} must be Equal with the fix: {}",
            v.reason()
        );
    }
}
