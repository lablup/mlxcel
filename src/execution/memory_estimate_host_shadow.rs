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

//! Host-shadow term of the memory estimate (issue #2643).
//!
//! On Windows the ROCm allocator gives every GPU buffer the CPU touches a
//! pinned host copy, its shadow (`overlays/rocm/LOCAL_FIXES.md` items 55 and
//! 67). Shadows are `hipHostMalloc` memory: they draw from the RAM Windows
//! sees (about 28 GiB on the validated gfx1151 host), not from the device
//! carve-out the rest of the estimate is compared against (92+ GiB there), so
//! they get their own term and their own comparison. Weights never get a
//! shadow: `Load::eval_gpu` uploads them from a staging buffer through the
//! device pointer. What the host touches per sequence is the prompt it builds
//! (int32 token ids over the context) and, at most, one f32 logits row (a
//! sampler or logprob path that pulls the distribution down); the sampled
//! token itself is a few bytes. Linux ROCm and every other backend add 0, so
//! their totals and verdicts are unchanged.

/// Bytes per element of the f32 logits row a host-side sampler can pull down.
pub const HOST_SHADOW_LOGIT_BYTES: u64 = 4;
/// Bytes per prompt token id the host builds (int32).
pub const HOST_SHADOW_TOKEN_BYTES: u64 = 4;

/// Pinned host bytes the Windows ROCm shadows can hold for one load: per
/// sequence, one f32 logits row (`vocab × 4`) plus the int32 token ids of the
/// whole context (`ctx_len × 4`). `batch` and `ctx_len` of 0 count as 1. Pure;
/// the platform gate is [`backend_host_shadow_reserve_bytes`].
#[must_use]
pub fn rocm_host_shadow_reserve_bytes(vocab: u64, ctx_len: u64, batch: u64) -> u64 {
    let batch = batch.max(1);
    let logits = vocab
        .saturating_mul(batch)
        .saturating_mul(HOST_SHADOW_LOGIT_BYTES);
    let tokens = ctx_len
        .max(1)
        .saturating_mul(batch)
        .saturating_mul(HOST_SHADOW_TOKEN_BYTES);
    logits.saturating_add(tokens)
}

/// The shadow reserve this build adds: [`rocm_host_shadow_reserve_bytes`] on
/// a Windows ROCm build, 0 everywhere else. Gated at compile time like
/// `backend_inflight_reserve_bytes`, so the estimator never brings up a runtime
/// to ask which backend or OS it has.
#[cfg(all(feature = "rocm", windows))]
#[must_use]
pub(super) fn backend_host_shadow_reserve_bytes(vocab: u64, ctx_len: u64, batch: u64) -> u64 {
    rocm_host_shadow_reserve_bytes(vocab, ctx_len, batch)
}

#[cfg(not(all(feature = "rocm", windows)))]
#[must_use]
pub(super) fn backend_host_shadow_reserve_bytes(_vocab: u64, _ctx_len: u64, _batch: u64) -> u64 {
    0
}

/// The RAM the shadows draw from, when the estimate has a shadow term: the
/// available physical memory `GlobalMemoryStatusEx` reports on Windows. `None`
/// when there is nothing to compare (no shadow term) or the query fails.
#[must_use]
pub(super) fn host_ram_available_bytes(shadow_bytes: u64) -> Option<u64> {
    if shadow_bytes == 0 {
        return None;
    }
    host_ram_available_bytes_probe()
}

#[cfg(windows)]
fn host_ram_available_bytes_probe() -> Option<u64> {
    super::read_windows_available_memory_bytes()
}

#[cfg(not(windows))]
fn host_ram_available_bytes_probe() -> Option<u64> {
    None
}

/// Whether the shadows fit the host RAM. No shadows always fit; a shadow term
/// with no RAM figure does not, the same safe direction `fits` takes when
/// nothing about available memory is detectable.
#[must_use]
pub(super) fn host_shadow_fits(shadow_bytes: u64, host_ram: Option<u64>) -> bool {
    shadow_bytes == 0 || host_ram.is_some_and(|ram| shadow_bytes <= ram)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1 << 20;

    #[test]
    fn reserve_is_one_logits_row_plus_the_context_ids_per_sequence() {
        // Qwen3: 151936 vocab. One sequence at 4096 tokens.
        let got = rocm_host_shadow_reserve_bytes(151_936, 4096, 1);
        assert_eq!(got, 151_936 * 4 + 4096 * 4);
        // The term scales with the batch, not with the weights.
        assert_eq!(
            rocm_host_shadow_reserve_bytes(151_936, 4096, 8),
            8 * (151_936 * 4 + 4096 * 4)
        );
    }

    #[test]
    fn reserve_treats_zero_batch_and_context_as_one() {
        assert_eq!(
            rocm_host_shadow_reserve_bytes(1000, 0, 0),
            rocm_host_shadow_reserve_bytes(1000, 1, 1)
        );
        // A config without vocab_size still reserves the token ids.
        assert_eq!(rocm_host_shadow_reserve_bytes(0, 2048, 1), 2048 * 4);
    }

    #[test]
    fn reserve_saturates_instead_of_overflowing() {
        assert_eq!(
            rocm_host_shadow_reserve_bytes(u64::MAX, u64::MAX, u64::MAX),
            u64::MAX
        );
    }

    #[test]
    fn gate_adds_the_term_only_on_windows_rocm() {
        let got = backend_host_shadow_reserve_bytes(151_936, 4096, 1);
        if cfg!(all(feature = "rocm", windows)) {
            assert_eq!(got, rocm_host_shadow_reserve_bytes(151_936, 4096, 1));
        } else {
            assert_eq!(got, 0, "Linux and non-ROCm builds must add no shadow term");
        }
    }

    #[test]
    fn host_ram_is_not_queried_without_a_shadow_term() {
        assert_eq!(host_ram_available_bytes(0), None);
    }

    #[test]
    fn host_ram_is_the_windows_figure_when_there_is_a_term() {
        let ram = host_ram_available_bytes(MIB);
        if cfg!(windows) {
            assert!(
                ram.is_some_and(|b| b > 0),
                "GlobalMemoryStatusEx must report available physical memory"
            );
        } else {
            assert_eq!(ram, None);
        }
    }

    #[test]
    fn shadows_fit_when_absent_or_within_host_ram() {
        assert!(host_shadow_fits(0, None));
        assert!(host_shadow_fits(0, Some(0)));
        assert!(host_shadow_fits(MIB, Some(MIB)));
        assert!(host_shadow_fits(MIB, Some(2 * MIB)));
        assert!(!host_shadow_fits(2 * MIB, Some(MIB)));
        assert!(
            !host_shadow_fits(MIB, None),
            "a term with no RAM figure is the unsafe case"
        );
    }
}
