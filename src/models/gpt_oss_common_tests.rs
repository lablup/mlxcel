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

//! The shared YaRN helper against the two upstream references it serves:
//! mlx-lm's `YarnRoPE` (GPT-OSS, `truncate=True`) and mlx-vlm's
//! `PrivacyFilterRoPE` (`truncate=False`). Both reference arrays were read
//! off the oracle venv (mlx-lm and mlx-vlm 0.7.8 git main, mlx CPU) for the
//! geometry the two families share: `head_dim` 64, `rope_theta` 150000,
//! `factor` 32, `beta_fast` 32, `beta_slow` 1, `original_max_position_embeddings`
//! 4096.

use super::*;

fn shared_geometry(truncate: bool) -> YarnRopeParams {
    YarnRopeParams {
        dims: 64,
        base: 150_000.0,
        factor: 32.0,
        original_max_position_embeddings: 4096.0,
        beta_fast: 32.0,
        beta_slow: 1.0,
        mscale: 1.0,
        mscale_all_dim: 0.0,
        truncate,
    }
}

/// `mlx_lm.models.rope_utils.YarnRoPE(...)._freqs` for the geometry above.
const MLX_LM_TRUNCATED_FREQS: [f32; 32] = [
    1.0, 1.4512855, 2.1062295, 3.0567403, 4.436203, 6.438197, 9.343662, 13.560322, 19.679897,
    31.6248, 51.41132, 84.80188, 142.53716, 245.72711, 439.12195, 829.09735, 1721.326, 4386.9688,
    26103.654, 37883.855, 54980.29, 79792.09, 115801.1, 168060.47, 243903.73, 353973.94, 513717.22,
    745550.4, 1082006.4, 1570300.2, 2278953.8, 3307412.5,
];

/// `PrivacyFilterRoPE(config)._freqs` from mlx-vlm for the checkpoint's
/// `rope_parameters` (`truncate: false`).
const MLX_VLM_PRIVACY_FILTER_FREQS: [f32; 32] = [
    1.0, 1.4512855, 2.1062295, 3.0567403, 4.436203, 6.438197, 9.343662, 13.560321, 19.679897,
    31.540073, 51.719673, 86.26602, 147.16792, 259.04327, 477.6022, 950.02655, 2190.6577, 7732.834,
    26103.654, 37883.855, 54980.29, 79792.09, 115801.11, 168060.47, 243903.72, 353973.94,
    513717.22, 745550.4, 1082006.4, 1570300.1, 2278953.8, 3307412.8,
];

fn assert_close(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let tolerance = 2e-5 * w.abs().max(1.0);
        assert!(
            (g - w).abs() <= tolerance,
            "{what}: frequency {i} is {g}, upstream has {w}"
        );
    }
}

#[test]
fn truncated_yarn_matches_mlx_lm() {
    let (freqs, factor) = yarn_rope_freqs(&shared_geometry(true));
    assert_close(&freqs, &MLX_LM_TRUNCATED_FREQS, "truncate=true");
    assert!((factor - 1.346_573_6).abs() < 1e-6, "{factor}");
}

#[test]
fn untruncated_yarn_matches_mlx_vlm_privacy_filter() {
    let (freqs, factor) = yarn_rope_freqs(&shared_geometry(false));
    assert_close(&freqs, &MLX_VLM_PRIVACY_FILTER_FREQS, "truncate=false");
    // `1 + 0.1 ln(32)`, the factor upstream multiplies into q and k.
    assert!((factor - (1.0 + 0.1 * 32f32.ln())).abs() < 1e-6, "{factor}");
}

#[test]
fn truncate_only_moves_the_ramp_band() {
    let (truncated, _) = yarn_rope_freqs(&shared_geometry(true));
    let (fractional, _) = yarn_rope_freqs(&shared_geometry(false));
    // Dimensions fully inside either regime are identical; the fractional
    // correction bounds change only the dimensions on the ramp.
    assert_eq!(truncated[..7], fractional[..7]);
    assert_eq!(truncated[18..21], fractional[18..21]);
    assert_ne!(truncated[9..18], fractional[9..18]);
}

#[test]
fn factor_one_disables_the_attention_scale() {
    let (_, factor) = yarn_rope_freqs(&YarnRopeParams {
        factor: 1.0,
        ..shared_geometry(false)
    });
    assert_eq!(factor, 1.0);
}
