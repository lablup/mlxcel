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

//! The SSM transition `A = -exp(A_log)` shared by the state-space families:
//! Mamba and Falcon-Mamba, Jamba's Mamba layers, and the Mamba2 SSD scans
//! (`mamba2`, `ssd_scan`). Each computes `A` from a float32 copy of `A_log`,
//! so the load-time dtype policy cannot overflow it (#3046, #3057).

use mlxcel_core::{MlxArray, UniquePtr};

/// The SSM transition `A = -exp(A_log)`, computed once in float32.
///
/// Issue #3046: on Apple Silicon a bf16 checkpoint is cast to f16 at load, and
/// `exp` of a trained `A_log` overflows f16, whose largest finite value
/// (65504) is `exp(11.09)`. `state-spaces/mamba-130m` carries `A_log` up to
/// 19.5, so 1530 of its 589824 `A` entries (in 23 of 24 layers) became
/// `-inf`, and a channel whose `softplus` step underflowed to zero in f16 then
/// computed `0 * -inf = NaN` for its decay. The NaN reached every logit on the
/// first token, greedy sampling took token 0, which is this model's EOS, and
/// generation stopped with no output. Without the f16 conversion (an f32
/// checkpoint, or `MLXCEL_KEEP_BF16`) it never appeared, because f32 and bf16
/// both hold `exp(19.5)`. Computing `A` from a float32 copy keeps the value
/// the checkpoint describes whatever the load-time dtype policy did to
/// `A_log`; `MambaBlock::scan_a` (`mamba.rs`) decides the dtype each scan reads it in.
/// Models that keep `A` call this once at load; it evaluates the result.
pub(crate) fn ssm_a_from_log(a_log: &MlxArray) -> UniquePtr<MlxArray> {
    let a = ssm_a_f32(a_log);
    mlxcel_core::eval(&a);
    a
}

/// The unevaluated graph of [`ssm_a_from_log`], for scans that take `A_log`
/// itself (the Mamba2 SSD scans, whose decode kernels read `A_log`): `exp`
/// runs on a float32 copy, so an f16 `A_log` past 11.09 stays finite (#3057).
/// It adds no evaluation, so calling it per forward pass costs no sync.
pub(crate) fn ssm_a_f32(a_log: &MlxArray) -> UniquePtr<MlxArray> {
    let a_log = mlxcel_core::astype(a_log, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::negative(&mlxcel_core::exp(&a_log))
}
