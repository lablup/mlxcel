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

//! The kernel ABI of the cuda-oxide fused residual-add + RMSNorm entries
//! (issue #2833): element types, what each of the 13 parameters carries, and
//! the check that a manifest's argument list is one the packing in
//! `mlxcelverse/kernels/fused_norm_oxide.cpp` gets right.
//!
//! Split from [`crate::cuda_oxide_artifact`], which validates everything else
//! about an artifact and calls [`argument_roles`] per entry.

use serde::Deserialize;

/// Number of kernel parameters under ABI version 1.
pub const ARG_COUNT: usize = 13;

/// Element type of an activation or weight buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Lane {
    F32,
    F16,
    Bf16,
}

impl Lane {
    pub(crate) fn parse(name: &str) -> Option<Self> {
        match name {
            "f32" => Some(Self::F32),
            "f16" => Some(Self::F16),
            "bf16" => Some(Self::Bf16),
            _ => None,
        }
    }

    /// Elements per 16-byte vector of the compute dtype of an activation in
    /// `self` under a weight in `weight`: 8 when that dtype is 16 bits wide,
    /// 4 when it is f32.
    ///
    /// The compute dtype is MLX's `promote_types` with this tree's table
    /// (`overlays/*/mlx/dtype.cpp`), which `fast::rms_norm` applies: equal
    /// dtypes stay, bf16 with f32 is bf16, and f16 with either other dtype is
    /// f32. The launch width `fused_norm_cuda_block(dim, n_reads)` depends on
    /// it, so the manifest's `n_reads` is checked against this.
    #[must_use]
    pub fn n_reads_under(self, weight: Self) -> u32 {
        let sixteen_bit = match (self, weight) {
            (a, w) if a == w => a != Self::F32,
            (Self::Bf16, Self::F32) | (Self::F32, Self::Bf16) => true,
            _ => false,
        };
        if sixteen_bit { 8 } else { 4 }
    }

    /// How the kernel sees one element: 16-bit lanes travel as bit patterns.
    fn pointee(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 | Self::Bf16 => "u16",
        }
    }

    /// The MLX dtype of a buffer with this element type.
    #[must_use]
    pub fn mlx_dtype(self) -> i32 {
        match self {
            Self::F32 => crate::dtype::FLOAT32,
            Self::F16 => crate::dtype::FLOAT16,
            Self::Bf16 => crate::dtype::BFLOAT16,
        }
    }

    /// Wire code of `mlxcel_oxide_fused_norm_install` (`fused_norm_oxide.cpp`).
    pub(crate) fn code(self) -> i32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Bf16 => 2,
        }
    }
}

/// What one kernel parameter carries. The discriminants are the wire format of
/// `mlxcel_oxide_fused_norm_install` and mirror `Role` in
/// `fused_norm_oxide.cpp`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ArgRole {
    XPtr = 0,
    XLen = 1,
    ResidualPtr = 2,
    ResidualLen = 3,
    WeightPtr = 4,
    WeightLen = 5,
    Eps = 6,
    WeightBias = 7,
    Dim = 8,
    NormedPtr = 9,
    NormedLen = 10,
    NewResidualPtr = 11,
    NewResidualLen = 12,
}

/// Which buffer a pointer or length parameter belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Buffer {
    Activation,
    Weight,
}

/// What the ABI requires of a parameter with a given role.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Expect {
    Pointer {
        buffer: Buffer,
        write: bool,
        length: ArgRole,
    },
    Length,
    Scalar(&'static str),
}

impl ArgRole {
    fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "x_ptr" => Self::XPtr,
            "x_len" => Self::XLen,
            "residual_ptr" => Self::ResidualPtr,
            "residual_len" => Self::ResidualLen,
            "weight_ptr" => Self::WeightPtr,
            "weight_len" => Self::WeightLen,
            "eps" => Self::Eps,
            "weight_bias" => Self::WeightBias,
            "dim" => Self::Dim,
            "normed_ptr" => Self::NormedPtr,
            "normed_len" => Self::NormedLen,
            "new_residual_ptr" => Self::NewResidualPtr,
            "new_residual_len" => Self::NewResidualLen,
            _ => return None,
        })
    }

    fn expect(self) -> Expect {
        let pointer = |buffer, write, length| Expect::Pointer {
            buffer,
            write,
            length,
        };
        match self {
            Self::XPtr => pointer(Buffer::Activation, false, Self::XLen),
            Self::ResidualPtr => pointer(Buffer::Activation, false, Self::ResidualLen),
            Self::WeightPtr => pointer(Buffer::Weight, false, Self::WeightLen),
            Self::NormedPtr => pointer(Buffer::Activation, true, Self::NormedLen),
            Self::NewResidualPtr => pointer(Buffer::Activation, true, Self::NewResidualLen),
            Self::XLen
            | Self::ResidualLen
            | Self::WeightLen
            | Self::NormedLen
            | Self::NewResidualLen => Expect::Length,
            Self::Eps | Self::WeightBias => Expect::Scalar("f32"),
            Self::Dim => Expect::Scalar("u32"),
        }
    }
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
pub(crate) struct Argument {
    ordinal: usize,
    name: String,
    cuda_param_type: String,
    width_bits: u32,
    kind: String,
    #[serde(default)]
    pointee: Option<String>,
    #[serde(default)]
    access: Option<String>,
    #[serde(default)]
    unit: Option<String>,
    #[serde(default)]
    length_argument: Option<usize>,
}

/// Map the manifest's argument list to roles, refusing anything the packing in
/// `fused_norm_oxide.cpp` would get wrong: a missing or repeated parameter, a
/// width other than the one the role is packed with, a buffer of another
/// element type, or a pointer whose length parameter is not its own.
pub(crate) fn argument_roles(
    arguments: &[Argument],
    activation: Lane,
    weight: Lane,
) -> Result<[ArgRole; ARG_COUNT], String> {
    if arguments.len() != ARG_COUNT {
        return Err(format!(
            "{} kernel parameters, the adapter packs {ARG_COUNT}",
            arguments.len()
        ));
    }
    let mut roles = Vec::with_capacity(ARG_COUNT);
    for (index, arg) in arguments.iter().enumerate() {
        if arg.ordinal != index {
            return Err(format!(
                "parameter {index} has ordinal {} (not in order)",
                arg.ordinal
            ));
        }
        let role = ArgRole::from_name(&arg.name)
            .ok_or_else(|| format!("unknown parameter {:?}", arg.name))?;
        if roles.contains(&role) {
            return Err(format!("parameter {:?} appears twice", arg.name));
        }
        roles.push(role);
    }
    for (arg, role) in arguments.iter().zip(&roles) {
        let wrong = |what: &str| format!("parameter {:?} {what}", arg.name);
        let is = |ty: &str, bits: u32, kind: &str| {
            arg.cuda_param_type == ty && arg.width_bits == bits && arg.kind == kind
        };
        match role.expect() {
            Expect::Pointer {
                buffer,
                write,
                length,
            } => {
                let lane = match buffer {
                    Buffer::Activation => activation,
                    Buffer::Weight => weight,
                };
                if !is("u64", 64, "device_pointer") {
                    return Err(wrong("is not a 64-bit device pointer"));
                }
                if arg.pointee.as_deref() != Some(lane.pointee()) {
                    return Err(wrong("points at another element type"));
                }
                if arg.access.as_deref() != Some(if write { "write" } else { "read" }) {
                    return Err(wrong("has the wrong access"));
                }
                if arg.length_argument.and_then(|i| roles.get(i)) != Some(&length) {
                    return Err(wrong("is not paired with its own length"));
                }
            }
            Expect::Length => {
                if !is("u64", 64, "length") || arg.unit.as_deref() != Some("elements") {
                    return Err(wrong("is not a 64-bit element count"));
                }
            }
            Expect::Scalar(ty) => {
                if !is(ty, 32, "value") {
                    return Err(wrong("has the wrong scalar type"));
                }
            }
        }
    }
    roles
        .try_into()
        .map_err(|_| "parameter count changed during validation".to_string())
}
