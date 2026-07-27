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

//! Scalar floating-point conversions shared by feature-neutral host contracts.

/// Round one f32 value to BF16 using round-to-nearest, ties-to-even, then widen
/// it back to an f32 carrier.
#[inline]
pub(crate) fn round_bf16_f32(value: f32) -> f32 {
    let bits = value.to_bits();
    if bits & 0x7f80_0000 == 0x7f80_0000 {
        return value;
    }
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
    f32::from_bits(rounded & 0xffff_0000)
}

/// Widen one IEEE 754 half bit pattern to f32.
pub(crate) fn half_to_f32(half: u16) -> f32 {
    let sign = if half >> 15 == 1 { -1.0 } else { 1.0 };
    let exponent = (half >> 10) & 0x1f;
    let mantissa = (half & 0x3ff) as f32;
    match exponent {
        0 => sign * mantissa * 2f32.powi(-24),
        0x1f if mantissa == 0.0 => sign * f32::INFINITY,
        0x1f => f32::NAN,
        _ => sign * (1.0 + mantissa / 1024.0) * 2f32.powi(exponent as i32 - 15),
    }
}

/// Convert one f32 to an IEEE 754 half bit pattern using round-to-nearest,
/// ties-to-even.
pub(crate) fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let absolute = bits & 0x7fff_ffff;

    if absolute >= 0x7f80_0000 {
        return sign
            | if absolute > 0x7f80_0000 {
                0x7e00
            } else {
                0x7c00
            };
    }

    let exponent = (absolute >> 23) as i32 - 127 + 15;
    if exponent >= 0x1f {
        return sign | 0x7c00;
    }

    if exponent <= 0 {
        if exponent < -10 {
            return sign;
        }
        let mantissa = (absolute & 0x007f_ffff) | 0x0080_0000;
        let shift = (14 - exponent) as u32;
        let quotient = mantissa >> shift;
        let remainder = mantissa & ((1 << shift) - 1);
        let halfway = 1u32 << (shift - 1);
        let round = u32::from(remainder > halfway || (remainder == halfway && quotient & 1 == 1));
        return sign | (quotient + round) as u16;
    }

    let mantissa = absolute & 0x007f_ffff;
    let base = ((exponent as u32) << 10) | (mantissa >> 13);
    let remainder = mantissa & 0x1fff;
    let halfway = 0x1000u32;
    let round = u32::from(remainder > halfway || (remainder == halfway && base & 1 == 1));
    sign | (base + round) as u16
}
