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
//! | [`cuda_kernel_knobs::QmmTileOp`] | CUDA | Blackwell qmm CTA `tile_m` | Wired, **unvalidated** (no CUDA host was available). |
//! | [`cuda_kernel_knobs::QmvMultirowOp`] | CUDA | multirow-qmv row-window ceiling | Wired, **unvalidated** (no CUDA host was available). |
//!
//! A fourth consumer, issue #898's paged-decode v2 `kv_chunk_size`, is
//! deliberately absent: #898 is not implemented yet and its feasible chunk
//! sizes depend on the v2 plan's own memory accounting. The seam it should
//! plug into is documented in [`crate::autotune`] under
//! [`crate::autotune::OP_PAGED_DECODE_V2_KV_CHUNK`]; no code here needs to
//! change to accept it.

pub mod cuda_kernel_knobs;
pub mod paged_decode_splits;

pub use cuda_kernel_knobs::{
    QmmShape, QmmTileOp, QmvMultirowOp, QmvShape, apply_tuned_cuda_kernel_env,
};
pub use paged_decode_splits::{DecodeShape, PagedDecodeSplitsOp, resolve_num_splits};
