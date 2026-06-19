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

//! Kokoro PLBert: a custom ALBERT text encoder over phoneme ids.
//!
//! Embedding dim 128, hidden 768, 12 attention heads, 12 layers that all reuse
//! one shared layer group (ALBERT cross-layer weight sharing). The phoneme
//! sequence has no padding in single-utterance inference, so the attention mask
//! is all-ones and omitted from the scores. The activation is the tanh-approx
//! GELU (`gelu_new`) the weights were trained with, not the exact erf form.
//!
//! Layout note: this runs on a single unbatched `(L, *)` sequence. Output is the
//! `last_hidden_state` `(L, 768)`, consumed by the duration predictor.

use mlxcel_core::{MlxArray, UniquePtr};

use super::ops::{self};
use super::weights::Weights;

const HIDDEN: i32 = 768;
const N_HEADS: i32 = 12;
const HEAD_DIM: i32 = HIDDEN / N_HEADS; // 64
const LN_EPS: f32 = 1e-12;

/// One reused ALBERT layer's weights.
struct AlbertLayer {
    q_w: UniquePtr<MlxArray>,
    q_b: UniquePtr<MlxArray>,
    k_w: UniquePtr<MlxArray>,
    k_b: UniquePtr<MlxArray>,
    v_w: UniquePtr<MlxArray>,
    v_b: UniquePtr<MlxArray>,
    dense_w: UniquePtr<MlxArray>,
    dense_b: UniquePtr<MlxArray>,
    attn_ln_w: UniquePtr<MlxArray>,
    attn_ln_b: UniquePtr<MlxArray>,
    ffn_w: UniquePtr<MlxArray>,
    ffn_b: UniquePtr<MlxArray>,
    ffn_out_w: UniquePtr<MlxArray>,
    ffn_out_b: UniquePtr<MlxArray>,
    full_ln_w: UniquePtr<MlxArray>,
    full_ln_b: UniquePtr<MlxArray>,
}

/// The Kokoro PLBert model.
pub(crate) struct PlBert {
    word_emb: UniquePtr<MlxArray>,
    pos_emb: UniquePtr<MlxArray>,
    tok_type_emb: UniquePtr<MlxArray>,
    emb_ln_w: UniquePtr<MlxArray>,
    emb_ln_b: UniquePtr<MlxArray>,
    map_in_w: UniquePtr<MlxArray>,
    map_in_b: UniquePtr<MlxArray>,
    layer: AlbertLayer,
    n_layers: usize,
}

impl PlBert {
    pub(crate) fn load(w: &Weights, n_layers: usize) -> Result<Self, String> {
        let p = "bert.encoder.albert_layer_groups.0.albert_layers.0";
        let lin = |name: &str| -> Result<(UniquePtr<MlxArray>, UniquePtr<MlxArray>), String> {
            Ok((
                w.get(&format!("{p}.{name}.weight"))?,
                w.get(&format!("{p}.{name}.bias"))?,
            ))
        };
        let (q_w, q_b) = lin("attention.query")?;
        let (k_w, k_b) = lin("attention.key")?;
        let (v_w, v_b) = lin("attention.value")?;
        let (dense_w, dense_b) = lin("attention.dense")?;
        let (attn_ln_w, attn_ln_b) = lin("attention.LayerNorm")?;
        let (ffn_w, ffn_b) = lin("ffn")?;
        let (ffn_out_w, ffn_out_b) = lin("ffn_output")?;
        let (full_ln_w, full_ln_b) = lin("full_layer_layer_norm")?;

        Ok(Self {
            word_emb: w.get("bert.embeddings.word_embeddings.weight")?,
            pos_emb: w.get("bert.embeddings.position_embeddings.weight")?,
            tok_type_emb: w.get("bert.embeddings.token_type_embeddings.weight")?,
            emb_ln_w: w.get("bert.embeddings.LayerNorm.weight")?,
            emb_ln_b: w.get("bert.embeddings.LayerNorm.bias")?,
            map_in_w: w.get("bert.encoder.embedding_hidden_mapping_in.weight")?,
            map_in_b: w.get("bert.encoder.embedding_hidden_mapping_in.bias")?,
            layer: AlbertLayer {
                q_w,
                q_b,
                k_w,
                k_b,
                v_w,
                v_b,
                dense_w,
                dense_b,
                attn_ln_w,
                attn_ln_b,
                ffn_w,
                ffn_b,
                ffn_out_w,
                ffn_out_b,
                full_ln_w,
                full_ln_b,
            },
            n_layers,
        })
    }

    /// Run the encoder over a phoneme id sequence (length `l`), returning the
    /// `(L, 768)` last hidden state.
    pub(crate) fn forward(&self, ids: &[i32]) -> UniquePtr<MlxArray> {
        let l = ids.len() as i32;

        // Embeddings: word + position + token-type(0), then LayerNorm.
        let word = ops::embed(&self.word_emb, ids); // (L,128)
        let pos_ids: Vec<i32> = (0..l).collect();
        let pos = ops::embed(&self.pos_emb, &pos_ids); // (L,128)
        let tok = ops::embed(&self.tok_type_emb, &vec![0_i32; ids.len()]); // (L,128)
        let emb = ops::add(&ops::add(&word, &pos), &tok);
        let emb = ops::layer_norm(&emb, &self.emb_ln_w, &self.emb_ln_b, LN_EPS);

        // Project embedding dim -> hidden.
        let mut h = ops::linear(&emb, &self.map_in_w, Some(&self.map_in_b)); // (L,768)

        for _ in 0..self.n_layers {
            h = self.layer_forward(&h, l);
        }
        h
    }

    fn layer_forward(&self, h: &UniquePtr<MlxArray>, l: i32) -> UniquePtr<MlxArray> {
        let lyr = &self.layer;
        // Self-attention.
        let q = ops::linear(h, &lyr.q_w, Some(&lyr.q_b));
        let k = ops::linear(h, &lyr.k_w, Some(&lyr.k_b));
        let v = ops::linear(h, &lyr.v_w, Some(&lyr.v_b));
        // (L,768) -> (n_heads, L, head_dim)
        let split = |x: &UniquePtr<MlxArray>| {
            let r = ops::reshape(x, &[l, N_HEADS, HEAD_DIM]);
            ops::transpose(&r, &[1, 0, 2])
        };
        let qh = split(&q);
        let kh = split(&k);
        let vh = split(&v);
        // scores = q @ k^T / sqrt(d)
        let kt = ops::transpose(&kh, &[0, 2, 1]);
        let scores = ops::mul_scalar(&ops::matmul(&qh, &kt), 1.0 / (HEAD_DIM as f32).sqrt());
        let probs = mlxcel_core::softmax_precise(scores.as_ref().expect("scores"), -1);
        let ctx = ops::matmul(&probs, &vh); // (n_heads, L, head_dim)
        let ctx = ops::reshape(&ops::transpose(&ctx, &[1, 0, 2]), &[l, HIDDEN]);
        let ctx = ops::linear(&ctx, &lyr.dense_w, Some(&lyr.dense_b));
        // residual + LayerNorm
        let attn_out = ops::layer_norm(&ops::add(&ctx, h), &lyr.attn_ln_w, &lyr.attn_ln_b, LN_EPS);

        // Feed-forward with gelu_new.
        let ffn = ops::linear(&attn_out, &lyr.ffn_w, Some(&lyr.ffn_b));
        let ffn = ops::gelu_new(&ffn);
        let ffn = ops::linear(&ffn, &lyr.ffn_out_w, Some(&lyr.ffn_out_b));
        ops::layer_norm(
            &ops::add(&ffn, &attn_out),
            &lyr.full_ln_w,
            &lyr.full_ln_b,
            LN_EPS,
        )
    }
}
