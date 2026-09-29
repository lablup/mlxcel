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

//! Cache-aware streaming for the FastConformer encoder.
//!
//! Port of `ConformerStreamingState` / `_stream_block` in
//! `mlx_audio/stt/models/nemotron_asr/streaming.py`. Each Conformer layer keeps
//! an attention cache (the last `left_context` attention-input frames) and a
//! causal-conv cache (the last `conv_kernel - 1` GLU-output frames);
//! subsampling is incremental over a 16-frame mel cache. With the key window
//! sized to the allowed left context no attention mask is needed.
//!
//! The layer stack is exactly the offline `chunked_limited` encoder with
//! right context 0. The incremental subsampling follows the reference window
//! rule (`base = (consumed - cache_len) / 8`), which matches the offline
//! subsampling only while the mel cache window starts on an 8-frame boundary;
//! later frames are numerically close to, not equal to, the offline encoder,
//! exactly like the reference.
//!
//! Used by: NemotronLabs VoiceChat streaming perception.

use mlxcel_core::{MlxArray, UniquePtr};

use super::FastConformerEncoder;
use super::attention::rel_pos_embedding;

/// Mel frames kept for the causal receptive field of the 8x subsampling stack.
pub const PRE_ENCODE_MEL_CACHE: usize = 16;

/// Incremental encoder state for one input stream.
pub struct ConformerStreamingState<'a> {
    encoder: &'a FastConformerEncoder,
    left_cache: usize,
    chunk_mel: usize,
    subsampling_factor: usize,
    conv_left: usize,
    attn_cache: Vec<Option<UniquePtr<MlxArray>>>,
    conv_cache: Vec<Option<UniquePtr<MlxArray>>>,
    mel_cache: Option<UniquePtr<MlxArray>>,
    emitted: i64,
    consumed: i64,
    pending: Option<UniquePtr<MlxArray>>,
    closed: bool,
    /// Last `rel_pos_embedding(len)` in the activation dtype, keyed by
    /// `(len, dtype)`; every layer of one push shares the window length, and
    /// the length saturates at `left_context + chunk`.
    pos_emb: Option<((usize, i32), UniquePtr<MlxArray>)>,
}

impl<'a> ConformerStreamingState<'a> {
    /// `chunk_frames` encoder frames per native chunk (`right + 1` in the
    /// reference default; the VoiceChat session uses 1) and the `[left,
    /// right]` attention context (`[70, 0]` for VoiceChat).
    pub fn new(
        encoder: &'a FastConformerEncoder,
        chunk_frames: usize,
        att_context: [i64; 2],
    ) -> Result<Self, String> {
        let [left, right] = att_context;
        if left < 0 || right < 0 {
            return Err(format!(
                "streaming needs a bounded attention context, got [{left}, {right}]"
            ));
        }
        if chunk_frames == 0 {
            return Err("chunk_frames must be positive".to_string());
        }
        let args = encoder.args();
        let n = encoder.layers.len();
        Ok(Self {
            encoder,
            left_cache: left as usize,
            chunk_mel: chunk_frames * args.subsampling_factor,
            subsampling_factor: args.subsampling_factor,
            conv_left: args.conv_kernel_size.saturating_sub(1),
            attn_cache: (0..n).map(|_| None).collect(),
            conv_cache: (0..n).map(|_| None).collect(),
            mel_cache: None,
            emitted: 0,
            consumed: 0,
            pending: None,
            closed: false,
            pos_emb: None,
        })
    }

    /// Encoder frames emitted so far.
    pub fn emitted_frames(&self) -> usize {
        self.emitted.max(0) as usize
    }

    pub fn is_closed(&self) -> bool {
        self.closed
    }

    /// Push mel frames (`[1, n, feat_in]` or `[n, feat_in]`) and return the
    /// newly encoded `[1, t, d_model]` chunks.
    ///
    /// `emit_partial` is for callers whose input boundary is known to align
    /// with a valid causal encoder frame: it emits the subsampler's current
    /// right-boundary output without closing the state. `final_` does the
    /// same and closes the state.
    pub fn push(
        &mut self,
        mel: &MlxArray,
        final_: bool,
        emit_partial: bool,
    ) -> Result<Vec<UniquePtr<MlxArray>>, String> {
        if self.closed {
            return Err("conformer streaming state is closed".to_string());
        }
        self.append_pending(mel)?;
        let boundary = final_ || emit_partial;
        let mut outputs = Vec::new();
        while let Some(pending) = self.pending.as_ref() {
            let shape = mlxcel_core::array_shape(pending);
            let available = shape[1] as usize;
            if available == 0 || (available < self.chunk_mel && !boundary) {
                break;
            }
            let take = if boundary && available <= self.chunk_mel {
                available
            } else {
                self.chunk_mel.min(available)
            };
            let m = mlxcel_core::slice(pending, &[0, 0, 0], &[1, take as i32, shape[2]]);
            let rest = mlxcel_core::slice(pending, &[0, take as i32, 0], &shape);
            let rest_len = available - take;
            self.pending = Some(rest);
            let include_boundary = boundary && rest_len == 0;
            if let Some(encoded) = self.encode_mel_chunk(&m, include_boundary)? {
                outputs.push(encoded);
            }
        }
        if final_ {
            self.closed = true;
        }
        Ok(outputs)
    }

    /// Evaluate `extra` (typically this step's outputs) together with every
    /// cache slice, so lazy concatenate/slice graphs do not grow across a long
    /// stream.
    pub fn materialize(&self, extra: &[&MlxArray]) {
        let mut ptrs: Vec<*const MlxArray> = extra.iter().map(|a| *a as *const MlxArray).collect();
        let caches = self
            .mel_cache
            .iter()
            .chain(self.attn_cache.iter().flatten())
            .chain(self.conv_cache.iter().flatten())
            .chain(self.pending.iter());
        ptrs.extend(caches.map(|a| &**a as *const MlxArray));
        // SAFETY: every pointer refers to an array borrowed from `extra` or
        // owned by `self`, all of which outlive this call.
        unsafe { mlxcel_core::eval_all(&ptrs) };
    }

    fn append_pending(&mut self, mel: &MlxArray) -> Result<(), String> {
        let feat = self.encoder.args().feat_in as i32;
        let shape = mlxcel_core::array_shape(mel);
        let chunk = match shape.as_slice() {
            [n, f] if *f == feat => mlxcel_core::reshape(mel, &[1, *n, feat]),
            [1, _, f] if *f == feat => mlxcel_core::copy(mel),
            _ => {
                return Err(format!(
                    "streaming encoder expects mel [1, n, {feat}] or [n, {feat}], got {shape:?}"
                ));
            }
        };
        if mlxcel_core::array_shape(&chunk)[1] == 0 {
            return Ok(());
        }
        self.pending = Some(match self.pending.take() {
            Some(prev) if mlxcel_core::array_shape(&prev)[1] > 0 => {
                mlxcel_core::concatenate(&prev, &chunk, 1)
            }
            _ => chunk,
        });
        Ok(())
    }

    fn encode_mel_chunk(
        &mut self,
        m: &MlxArray,
        include_boundary: bool,
    ) -> Result<Option<UniquePtr<MlxArray>>, String> {
        let factor = self.subsampling_factor as i64;
        let m_len = mlxcel_core::array_shape(m)[1] as i64;
        let (win, cache_len) = match &self.mel_cache {
            Some(cache) => (
                mlxcel_core::concatenate(cache, m, 1),
                mlxcel_core::array_shape(cache)[1] as i64,
            ),
            None => (mlxcel_core::copy(m), 0),
        };
        let mut sub = self.encoder.pre_encode.forward(&win)?;
        if self.encoder.args().xscaling {
            let scale = (self.encoder.args().d_model as f32).sqrt();
            sub = mlxcel_core::multiply_scalar(&sub, scale);
        }
        let sub_shape = mlxcel_core::array_shape(&sub);
        let sub_len = sub_shape[1] as i64;

        let end = self.consumed + m_len;
        let base = (self.consumed - cache_len).div_euclid(factor);
        let lo = self.emitted - base;
        let hi = if include_boundary {
            sub_len
        } else {
            end.div_euclid(factor) - base
        };
        self.consumed = end;
        let win_shape = mlxcel_core::array_shape(&win);
        let keep_from = (win_shape[1] - PRE_ENCODE_MEL_CACHE as i32).max(0);
        self.mel_cache = Some(mlxcel_core::slice(&win, &[0, keep_from, 0], &win_shape));

        if hi <= lo {
            self.emitted = base + lo.max(hi);
            return Ok(None);
        }
        self.emitted = base + hi;
        let (lo_c, hi_c) = (lo.clamp(0, sub_len) as i32, hi.clamp(0, sub_len) as i32);
        if hi_c <= lo_c {
            return Err(format!(
                "streaming subsampler produced {sub_len} frames, cannot emit [{lo}, {hi})"
            ));
        }
        let h = mlxcel_core::slice(&sub, &[0, lo_c, 0], &[sub_shape[0], hi_c, sub_shape[2]]);
        self.stream_layers(&h).map(Some)
    }

    /// Run already-subsampled frames `h: [1, c, d_model]` through every layer,
    /// advancing the attention and conv caches.
    pub(crate) fn stream_layers(&mut self, h: &MlxArray) -> Result<UniquePtr<MlxArray>, String> {
        let dtype = mlxcel_core::array_dtype(h);
        let c = mlxcel_core::array_shape(h)[1] as usize;
        let mut x = mlxcel_core::copy(h);
        for li in 0..self.encoder.layers.len() {
            let cached = self.attn_cache[li]
                .as_ref()
                .map_or(0, |a| mlxcel_core::array_shape(a)[1] as usize);
            self.ensure_pos_emb(cached + c, dtype);
            let Some((_, pos_emb)) = self.pos_emb.as_ref() else {
                return Err("streaming positional embedding missing".to_string());
            };
            let out = self.encoder.layers[li].stream(
                &x,
                pos_emb,
                self.attn_cache[li].as_deref(),
                self.conv_cache[li].as_deref(),
                self.left_cache,
                self.conv_left,
            )?;
            self.attn_cache[li] = out.attn_cache;
            self.conv_cache[li] = Some(out.conv_cache);
            x = out.output;
        }
        Ok(x)
    }

    /// Make `self.pos_emb` hold `RelPositionalEncoding.pos_emb_for(len)` in
    /// `dtype`.
    fn ensure_pos_emb(&mut self, len: usize, dtype: i32) {
        let key = (len, dtype);
        if self.pos_emb.as_ref().is_some_and(|(k, _)| *k == key) {
            return;
        }
        let pe = rel_pos_embedding(len, self.encoder.args().d_model);
        let pe = if dtype == mlxcel_core::dtype::FLOAT32 {
            pe
        } else {
            mlxcel_core::astype(&pe, dtype)
        };
        self.pos_emb = Some((key, pe));
    }
}
