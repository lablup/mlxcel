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

//! The affine int8 MLA latent on the DSA attention (issue #3163).
//!
//! `deepseek_v32`, `deepseek_v3.2` and `glm_moe_dsa` cache `[kv_latent |
//! indexer key]` in one slot. Under `int8` the whole slot is stored as the
//! #3155 affine group-64 triple with fp16 rope keys, and a decode step reads
//! it without dequantizing the latent:
//!
//! 1. the indexer scores the cached keys from the indexer columns alone
//!    (`LatentWindow::dense_key_columns`, the last `index_head_dim / 64`
//!    groups);
//! 2. the latent columns are kept packed (`LatentWindow::narrow_keys`);
//! 3. when the indexer selected a top-k, those rows are gathered from the
//!    packed triple and the rope stream (`LatentWindow::gather_rows`);
//! 4. `mla::quantized_absorbed_attention` attends over the result: the fused
//!    kernel's quantized variant where it applies, the composed
//!    `quantized_matmul` decode otherwise.
//!
//! A multi-token step dequantizes the window once and runs the fp16 code in
//! `deepseek_v32.rs`, so prefill reads the values decode reads.
//!
//! The mode resolver does not hand these families an `int8` cache: on
//! DeepSeek-V3.2-4bit the trace gate passed and the top-k overlap gate did not
//! (`docs/benchmark_results/mla-quantized-latent-dsa-m3ultra-2026-11-19.md`),
//! so they are not in `mla::QUANTIZED_LATENT_FAMILIES` and this path is reached
//! only by a harness that sets the cache mode itself
//! (`MLXCEL_TRACE_KV_CACHE_MODE=int8`).
//!
//! This file also holds the selection trace (`MLXCEL_TRACE_DSA_TOPK`) that the
//! top-k overlap gate of #3163 is measured with.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::{Mutex, OnceLock};

use mlxcel_core::mla::{LatentWindow, MlaGeometry};
use mlxcel_core::{MlxArray, UniquePtr};

use super::MLAAttention;

/// Environment variable naming a file the decode-step top-k selections are
/// appended to, one line per layer per step: `layer<TAB>kv_len<TAB>` and the
/// selected positions in ascending order, comma separated. Compare two arms
/// with `scripts/compare_dsa_topk.py`. Unset (the default) costs one
/// `OnceLock` read per selection.
pub(super) const TOPK_TRACE_ENV: &str = "MLXCEL_TRACE_DSA_TOPK";

fn topk_trace() -> Option<&'static Mutex<BufWriter<File>>> {
    static SINK: OnceLock<Option<Mutex<BufWriter<File>>>> = OnceLock::new();
    SINK.get_or_init(|| {
        let path = std::env::var_os(TOPK_TRACE_ENV)?;
        match File::create(&path) {
            Ok(file) => Some(Mutex::new(BufWriter::new(file))),
            Err(err) => {
                tracing::warn!(?path, %err, "{TOPK_TRACE_ENV}: cannot create the trace file");
                None
            }
        }
    })
    .as_ref()
}

/// Record one decode step's selection for `layer` when the trace is on.
///
/// `indices` is `[B, 1, 1, K]`; row zero is written. `kv_len` is called only
/// when the trace is on, so an untraced decode step pays for no shape query.
/// Evaluates the indices, which a traced run pays for and an untraced run
/// never reaches.
pub(super) fn trace_topk(layer: i32, kv_len: impl FnOnce() -> i32, indices: &MlxArray) {
    let Some(sink) = topk_trace() else {
        return;
    };
    let kv_len = kv_len();
    let shape = mlxcel_core::array_shape(indices);
    let k = shape.last().copied().unwrap_or(0);
    let row = mlxcel_core::slice(indices, &[0, 0, 0, 0], &[1, 1, 1, k]);
    // Positions are far below 2^24, so f32 carries them exactly.
    let row = mlxcel_core::astype(&row, mlxcel_core::dtype::FLOAT32);
    let mut positions: Vec<i32> = mlxcel_core::utils::array_to_vec_f32(&row)
        .into_iter()
        .map(|position| position.round() as i32)
        .collect();
    positions.sort_unstable();
    let line = positions
        .iter()
        .map(i32::to_string)
        .collect::<Vec<_>>()
        .join(",");
    if let Ok(mut out) = sink.lock() {
        // A failed write loses a trace line, not a decode step.
        let _ = writeln!(out, "{layer}\t{kv_len}\t{line}");
        let _ = out.flush();
    }
}

impl MLAAttention {
    /// The layer's MLA shape, for the latent cache view.
    pub(super) fn geometry(&self) -> MlaGeometry {
        crate::models::mla_batched::mla_geometry(
            self.num_heads,
            self.kv_lora_rank,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
        )
    }

    /// The window one decode query attends over, from a quantized `[kv_latent
    /// | indexer key]` window: the latent columns, at the indexer's top-k rows
    /// when it selects any. Nothing but the indexer columns is dequantized.
    ///
    /// `index_query` is the step's `(indexer query, indexer weights)`, absent
    /// when the layer has no indexer. Returns the selected window and whether
    /// a top-k was applied.
    pub(super) fn select_quantized_window(
        &self,
        window: LatentWindow,
        index_query: Option<(&MlxArray, &MlxArray)>,
        mask: Option<&MlxArray>,
    ) -> (LatentWindow, bool) {
        let rank = self.kv_lora_rank;
        let indices = match (&self.indexer, index_query) {
            (Some(indexer), Some((query, weights))) => {
                let keys = window.dense_key_columns(rank, window.key_width());
                indexer.top_indices(query, &keys, weights, mask)
            }
            _ => None,
        };
        let latent = window.narrow_keys(0, rank);
        match indices {
            Some(indices) => {
                trace_topk(self.layer, || latent.seq_len(), &indices);
                (latent.gather_rows(&indices), true)
            }
            None => (latent, false),
        }
    }

    /// One absorbed query over a quantized window, in latent value space
    /// (`[B, H, 1, kv_lora_rank]`).
    ///
    /// A step whose top-k was applied attends maskless, as the fp16 sparse
    /// decode does: every gathered position is causally valid and the caller's
    /// mask is indexed by the ungathered positions.
    pub(super) fn quantized_decode(
        &self,
        window: LatentWindow,
        q_projected: &MlxArray,
        q_pe: &MlxArray,
        index_query: Option<(&MlxArray, &MlxArray)>,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let (selected, sparse) = self.select_quantized_window(window, index_query, mask);
        let LatentWindow::Quantized {
            ckv_q,
            scales,
            biases,
            kpe,
            bits,
            group_size,
        } = selected
        else {
            unreachable!("selection keeps a quantized window quantized")
        };
        mlxcel_core::mla::quantized_absorbed_attention(
            q_projected,
            q_pe,
            &ckv_q,
            &scales,
            &biases,
            &kpe,
            bits,
            group_size,
            self.scale,
            if sparse { None } else { mask },
        )
    }
}
