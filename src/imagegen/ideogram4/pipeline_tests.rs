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

// The literals are upstream's f32 values printed at full f64 precision.
#![allow(clippy::excessive_precision)]

use super::*;
use crate::imagegen::testing::values;

fn ints(a: &MlxArray) -> Vec<i32> {
    values(a).into_iter().map(|v| v as i32).collect()
}

/// Distance in f32 representations between two finite values of one sign
/// (`u32::MAX` when the signs differ, which no ulp bound should accept).
fn ulps(a: f32, b: f32) -> u32 {
    if a == b {
        return 0;
    }
    if a.is_sign_negative() != b.is_sign_negative() {
        return u32::MAX;
    }
    a.to_bits().abs_diff(b.to_bits())
}

/// The backends' `erfinv` and `log` differ in the last bit or two (CUDA and
/// the CPU stream by one ulp at element 510, ROCm by one at element 5), so
/// the draw is gated to this many ulps instead of bit equality. A wrong key
/// derivation or sampling order moves every element by far more.
const MAX_NOISE_ULPS: u32 = 2;

fn assert_noise(got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            ulps(*g, *w) <= MAX_NOISE_ULPS,
            "element {i}: {g} vs {w} ({} ulps)",
            ulps(*g, *w)
        );
    }
}

/// `mx.random.seed(s); mx.random.normal((1, 4, 128), float32)` from
/// mlx 0.32.4: the global-seed draw, which is NOT `normal(key=key(s))`.
#[test]
fn noise_reproduces_upstream_global_seeding() {
    let noise = values(&initial_noise(0, 4));
    assert_eq!(noise.len(), 4 * 128);
    assert_noise(
        &noise[..6],
        &[
            -0.0766015499830246_f32,
            -0.29556146264076233,
            -1.0921014547348022,
            1.4498533010482788,
            -1.098831295967102,
            -1.3240625858306885,
        ],
    );
    // Element 510 is 0.4459344148635864 on the CUDA backend and
    // 0.4459344744682312 on the CPU one; the whole-tensor CUDA checksum is
    // gated in `parity_tests`.
    assert_noise(
        &noise[510..],
        &[0.4459344148635864_f32, 0.09173143655061722],
    );
    let noise = values(&initial_noise(7, 4));
    assert_noise(
        &noise[..3],
        &[
            0.47220805287361145_f32,
            0.09765862673521042,
            -0.013432363979518414,
        ],
    );
}

#[test]
fn the_sequence_layout_matches_upstream_build_inputs() {
    // 3 text tokens and a 256x288 image: a 16x18 grid.
    let layout = SequenceLayout::new(3, 288, 256);
    assert_eq!((layout.grid_h, layout.grid_w), (16, 18));
    assert_eq!(layout.num_image_tokens, 288);
    let pos = ints(&layout.position_ids);
    assert_eq!(mlxcel_core::array_shape(&layout.position_ids), [1, 291, 3]);
    assert_eq!(pos[..9], [0, 0, 0, 1, 1, 1, 2, 2, 2]);
    let o = IMAGE_POSITION_OFFSET;
    // First image token (0, 0, 0), then (0, 0, 1); token 18 starts row 1.
    assert_eq!(pos[9..15], [o, o, o, o, o, o + 1]);
    assert_eq!(pos[9 + 18 * 3..9 + 18 * 3 + 3], [o, o + 1, o]);
    let ind = ints(&layout.indicator);
    assert_eq!(ind[..4], [3, 3, 3, 2]);
    assert!(ind[3..].iter().all(|&v| v == 2));
    assert_eq!(ints(&layout.image_position_ids), pos[9..]);
    assert_eq!(ints(&layout.image_indicator), ind[3..]);
}

#[test]
fn oversized_prompts_are_rejected_by_bytes_before_tokenizing() {
    assert!(check_prompt_bytes(&"a".repeat(MAX_PROMPT_BYTES)).is_ok());
    let err = check_prompt_bytes(&"a".repeat(MAX_PROMPT_BYTES + 1))
        .unwrap_err()
        .to_string();
    assert!(err.contains("max_text_tokens=2048"), "{err}");
    assert!(err.contains(&(MAX_PROMPT_BYTES + 1).to_string()), "{err}");
    assert!(err.contains(&MAX_PROMPT_BYTES.to_string()), "{err}");
    // The cap sits above the proven lower bound for a prompt over the token limit.
    const { assert!(MAX_PROMPT_BYTES > MAX_TEXT_TOKENS * 128 * 3) };
}
