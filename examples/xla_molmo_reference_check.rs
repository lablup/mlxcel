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

//! Dedicated Molmo v1 actual-checkpoint reference gate.
//!
//! This executable keeps eager MLX on CUDA while the IREE vision and decoder
//! references remain on `local-task` or `local-sync`. It deliberately avoids
//! the root crate's full libtest harness and emits flushed progress plus
//! periodic heartbeats around every potentially long eager materialization.

fn main() {
    mlxcel::multimodal::host_preprocessor::run_pinned_molmo_eager_mlx_iree_boundaries();
}
