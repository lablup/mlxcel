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

//! Autotuner consumers (issue #906).
//!
//! | Op | Backend | Knob | Status |
//! |----|---------|------|--------|
//! | [`paged_decode_splits`] | Metal + CUDA | v1 paged-decode `NumSplits` launch shape | Wired end to end; the Apple-Silicon-tunable op. |
//! | [`paged_decode_v2_chunk`] | Metal + CUDA | v2 paged-decode `pages_per_chunk` | Wired end to end (issue #898); reachable only when `MLXCEL_PAGED_ATTENTION_V2=1` selects the v2 path. |
//! | [`cuda_kernel_knobs::QmmTileOp`] | CUDA | Blackwell qmm CTA `tile_m` | Wired, **unvalidated** (no CUDA host was available). |
//! | [`cuda_kernel_knobs::QmvMultirowOp`] | CUDA | multirow-qmv row-window ceiling | Wired, **unvalidated** (no CUDA host was available). |
//!
//! [`paged_decode_v2_chunk`] is the consumer the reserved
//! [`crate::autotune::OP_PAGED_DECODE_V2_KV_CHUNK`] name was held for. Nothing
//! in the autotuner itself changed to accept it, as that module's extension
//! note predicted: it registers under the reserved name, enumerates its own
//! feasible chunk sizes, and returns the plan's binary-search heuristic as its
//! default tactic.

pub mod cuda_kernel_knobs;
pub mod paged_decode_splits;
pub mod paged_decode_v2_chunk;

pub use cuda_kernel_knobs::{
    QmmShape, QmmTileOp, QmvMultirowOp, QmvShape, apply_tuned_cuda_kernel_env,
};
pub use paged_decode_splits::{DecodeShape, PagedDecodeSplitsOp, resolve_num_splits};
pub use paged_decode_v2_chunk::{
    PagedDecodeV2ChunkOp, V2ChunkShape, chunk_candidates, resolve_pages_per_chunk,
};
