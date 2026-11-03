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

//! Weight remapping for Xiaomi MiMo-V2 / V2.5 / V2.6 (`mimo_v2`).
//!
//! The decoder stack is the `mimo_v2_flash` one (issue #842): upstream's
//! `MiMoV2` `LanguageModel` subclasses the Flash model and only overrides
//! `sanitize`. The V2.6 checkpoint stores the same parameters in two layouts
//! that the Flash loader cannot read directly:
//!
//! # Fused, tensor-parallel QKV
//!
//! `attention_projection_layout: "fused_qkv"` replaces `q_proj`, `k_proj` and
//! `v_proj` with one `self_attn.qkv_proj.weight` plus a block-FP8
//! `weight_scale_inv` grid. The fused rows are the concatenation of the
//! tensor-parallel shards, and each shard holds its own q, then k, then v rows.
//! The scale grid is likewise per shard and padded up to whole 128-row blocks,
//! which is why a degree-4 full-attention layer ships 108 grid rows for a
//! 106-block weight (4 x 27) and a degree-4 sliding layer 116 (4 x 29). Reading
//! the grid as one tensor neither divides nor lines up.
//!
//! [`unfuse_qkv`] ports `_candidate_degrees`, `_tensor_parallel_degree` and
//! `_unfuse_qkv` from
//! [`mlx_vlm/models/mimo_v2/language.py`](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/mimo_v2/language.py)
//! exactly, including `max(common)` when several degrees explain the shapes.
//!
//! # MXFP4 routed experts
//!
//! The native release ships the routed experts as E2M1 codes (half a byte per
//! value) with one E8M0 scale per 32 values, under `<plane>.weight_scale`.
//! MLX's `gather_qmm` wants the codes packed as `uint32` under `.weight` and the
//! scales under `.scales`; [`convert_mxfp4_experts`] performs that rename and
//! reinterpretation, and the Flash loader infers `mode: "mxfp4"`,
//! `group_size: 32` from the resulting bias-free uint8 scales.
//!
//! The mlx-vlm-converted release (`nativ-community/MiMo-V2.6-Flash-RL-MLX-4bit`)
//! is already unfused and repacked, so both passes are no-ops on it.

// The weight remap lives in a sibling file so this one stays a short, scan-friendly
// entry point; every item is re-exported at the family path the loader and tests use.
pub(crate) use super::mimo_v2_weights::*;
