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

//! Streaming state for the codec decoder.
//!
//! Port of `CausalConv1dCache` from
//! `mlx_audio/codec/models/nemotron_voicechat/codec.py`. One cache holds the
//! left context of every causal depthwise convolution in the decoder plus the
//! overlapping spectrogram frames the iSTFT needs, so
//! [`super::NemotronCodec::decode_step`] can decode one frame at a time
//! without replaying history. Use one cache per output stream.
//!
//! Used by: [`super::NemotronCodec::decode_with_cache`] and
//! [`super::NemotronCodec::decode_step`].

use std::collections::HashMap;

use mlxcel_core::{MlxArray, UniquePtr};

/// Identifies one cached tensor (the reference keys by `int | str`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum CacheKey {
    /// Depthwise conv left context of decoder ConvNeXt block `n`.
    Block(usize),
    /// Real part of the trailing spectrogram frames.
    IstftReal,
    /// Imaginary part of the trailing spectrogram frames.
    IstftImag,
}

/// Per-stream causal convolution and spectrogram overlap state.
#[derive(Default)]
pub struct CausalConv1dCache {
    cache: HashMap<CacheKey, UniquePtr<MlxArray>>,
}

impl std::fmt::Debug for CausalConv1dCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CausalConv1dCache")
            .field("entries", &self.cache.len())
            .finish()
    }
}

impl CausalConv1dCache {
    /// Create an empty cache (the first step pads with zeros).
    pub fn new() -> Self {
        Self::default()
    }

    /// Drop all cached state so the next step starts a fresh stream.
    pub fn clear(&mut self) {
        self.cache.clear();
    }

    /// True when no state has been recorded (fresh or flushed stream).
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Prepend the cached `padding` frames to `states` (`[B, T, C]`) and keep
    /// the last `padding` frames of the result for the next call.
    ///
    /// Mirrors the reference exactly: a missing entry is zero-filled in the
    /// input dtype, a shape mismatch is an error, and `flush` drops the entry
    /// after producing the padded output.
    pub(crate) fn update(
        &mut self,
        states: &MlxArray,
        key: CacheKey,
        padding: usize,
        flush: bool,
    ) -> Result<UniquePtr<MlxArray>, String> {
        if padding == 0 {
            return Ok(mlxcel_core::copy(states));
        }
        let shape = mlxcel_core::array_shape(states);
        if shape.len() != 3 {
            return Err(format!(
                "codec cache: states must be [B, T, C], got {shape:?}"
            ));
        }
        let pad = i32::try_from(padding)
            .map_err(|_| format!("codec cache: padding {padding} exceeds i32 range"))?;
        let expected = [shape[0], pad, shape[2]];
        let previous = match self.cache.get(&key) {
            Some(prev) => {
                let prev_shape = mlxcel_core::array_shape(prev);
                if prev_shape.as_slice() != expected {
                    return Err(format!(
                        "codec cache entry {key:?} has shape {prev_shape:?}, expected {expected:?}"
                    ));
                }
                mlxcel_core::copy(prev)
            }
            None => mlxcel_core::zeros(&expected, mlxcel_core::array_dtype(states)),
        };
        let padded = mlxcel_core::concatenate(&previous, states, 1);
        let total = mlxcel_core::array_shape(&padded)[1];
        let tail = mlxcel_core::slice(
            &padded,
            &[0, total - pad, 0],
            &[expected[0], total, expected[2]],
        );
        if flush {
            self.cache.remove(&key);
        } else {
            self.cache.insert(key, tail);
        }
        Ok(padded)
    }
}
