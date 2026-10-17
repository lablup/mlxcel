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

//! Runtime (unfused) LoRA terms shared by the linear layers (issue #1439).
//!
//! llama-server b10621 keeps LoRA adapters as runtime-swappable layers:
//! `POST /lora-adapters` changes adapter scales while the server runs, the
//! native `lora` request field selects per-request scales, and
//! `--lora-init-without-apply` loads adapters at scale 0.0 to be applied
//! later. mlxcel's historical path fuses adapters into the base weights at
//! load, which makes all three impossible. This module is the unfused
//! alternative: each adapter contributes a low-rank term
//! `y += scale * (x @ Aᵀ) @ Bᵀ` at forward time, where `scale` is the user
//! scale behind a shared atomic handle (times the adapter's own
//! `alpha / rank`), so the serving layer can change it without touching the
//! layers.
//!
//! Attachment happens at model construction: the loader stages pending terms
//! keyed by layer prefix (thread-local, because loading and construction run
//! on one worker thread and two models must never see each other's
//! adapters), and the layer constructors in [`crate::layers`] claim the
//! terms for the prefixes they build. Whatever is left after construction is
//! reported back to the loader, which logs it.
//!
//! Note that an unclaimed term is not the same hole as an unmatched adapter
//! tensor. Issue #1328 made an adapter tensor that maps onto no base weight a
//! hard load failure on the fused and the unfused path alike. An unclaimed
//! term is one whose tensor did map onto a real base weight, but whose prefix
//! no constructor here reads runtime terms for, so those layers still serve
//! base weights after a load that succeeded. That is still only logged; issue
//! #1577 owns making it strict.
//!
//! Input embeddings take their own term type (issue #2421), in mlx-lm's
//! `LoRAEmbedding` layout: [`RuntimeLoraEmbeddingTerm`] adds
//! `scale * a[ids] @ b` to the lookup and, when the output head reads the same
//! matrix, `scale * (x @ bᵀ) @ aᵀ` to the logits. They are staged in a map of
//! their own, claimed by `UnifiedEmbedding::from_weights`, and shared with a
//! tied head through an `Arc` and one scale handle.
//!
//! A term whose user scale reads `0.0` contributes nothing and, crucially,
//! changes nothing about the base computation path, so a model loaded with
//! adapters at scale zero is byte-identical to the base model. That is the
//! control arm the numerical validation of the unfused path gates on.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use cxx::UniquePtr;

use crate::ffi::{self, MlxArray};

/// A user scale shared between the serving layer (which sets it) and every
/// linear layer holding a term of the adapter (which read it per forward).
/// Stored as `f32` bits in an `AtomicU32`.
#[derive(Clone)]
pub struct SharedLoraScale(Arc<AtomicU32>);

impl SharedLoraScale {
    #[must_use]
    pub fn new(value: f32) -> Self {
        Self(Arc::new(AtomicU32::new(value.to_bits())))
    }

    #[must_use]
    pub fn get(&self) -> f32 {
        f32::from_bits(self.0.load(Ordering::Relaxed))
    }

    pub fn set(&self, value: f32) {
        self.0.store(value.to_bits(), Ordering::Relaxed);
    }
}

impl std::fmt::Debug for SharedLoraScale {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SharedLoraScale({})", self.get())
    }
}

/// One adapter's contribution to one linear layer, ready to apply at forward
/// time. `a_t` is `[in_features, rank]` and `b_t` is `[rank, out_features]`
/// (the transposed orientation, so the forward is two plain matmuls), held
/// in `float32`: the fused path computes its delta in f32 and its fusion
/// promotes the adapted weight to f32, so the unfused term runs the same
/// arithmetic precision and only the final addition casts back to the base
/// output's dtype.
pub struct RuntimeLoraTerm {
    /// The adapter's user scale (what `POST /lora-adapters` sets).
    pub handle: SharedLoraScale,
    /// The adapter's own `alpha / rank`, fixed at load.
    pub base_scale: f32,
    pub a_t: UniquePtr<MlxArray>,
    pub b_t: UniquePtr<MlxArray>,
}

impl RuntimeLoraTerm {
    /// The low-rank contribution for input `x`, or `None` when the user
    /// scale is `0.0` (the term then adds no operation at all, which is what
    /// keeps the scale-zero configuration byte-identical to the base model).
    #[must_use]
    pub fn contribution(&self, x: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        let user_scale = self.handle.get();
        if user_scale == 0.0 {
            return None;
        }
        // f32 term arithmetic (see the struct docs): matches the fused
        // path's f32 delta and promoted weight, so the two paths agree to
        // the rounding class instead of diverging by a bf16-truncated
        // low-rank product.
        let x_f32 = ffi::astype(x, crate::dtype::FLOAT32);
        let low = ffi::matmul(&x_f32, &self.a_t);
        let out = ffi::matmul(&low, &self.b_t);
        Some(crate::multiply_scalar(&out, user_scale * self.base_scale))
    }

    /// Whether this term currently contributes (user scale non-zero).
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.handle.get() != 0.0
    }
}

/// Apply every active term in `terms` to `y` for input `x`.
#[must_use]
pub fn apply_terms(
    terms: &[RuntimeLoraTerm],
    x: &MlxArray,
    mut y: UniquePtr<MlxArray>,
) -> UniquePtr<MlxArray> {
    let y_dtype = ffi::array_dtype(&y);
    let mut any = false;
    for term in terms {
        if let Some(contribution) = term.contribution(x) {
            y = ffi::add(&y, &contribution);
            any = true;
        }
    }
    if !any {
        return y;
    }
    // The f32 contributions promoted the sum; return to the base output's
    // dtype so downstream layers see the dtype they were built for.
    if ffi::array_dtype(&y) == y_dtype {
        y
    } else {
        ffi::astype(&y, y_dtype)
    }
}

/// True when any term in `terms` is active. The fused-kernel launchers on
/// [`crate::layers::FusedQKVLinear`] consult this to fall back to the plain
/// projection graph (the only place the terms can be added before RoPE);
/// with every scale at zero they keep the base kernel path, which is what
/// makes scale zero byte-identical.
#[must_use]
pub fn any_active(terms: &[RuntimeLoraTerm]) -> bool {
    terms.iter().any(RuntimeLoraTerm::is_active)
}

/// One adapter's contribution to one input embedding, in mlx-lm's
/// `LoRAEmbedding` layout (issue #2421): `a` is `[vocab, rank]` and `b` is
/// `[rank, hidden]`, so the adapted table is `W + scale * a @ b`, the matrix
/// `mlx_lm.fuse` writes. Held in `float32` like [`RuntimeLoraTerm`].
///
/// The term is shared through an `Arc` between the embedding that claimed it
/// (the lookup end, [`Self::lookup_contribution`]) and, on a tied checkpoint,
/// the output head that reads the same matrix: either the embedding's own
/// `as_linear` ([`apply_embedding_head_terms`]) or a linear layer built from
/// the embedding's prefix ([`Self::head_term`], attached by [`claim`]). Both
/// ends read the same [`SharedLoraScale`], so `POST /lora-adapters` moves
/// them together.
pub struct RuntimeLoraEmbeddingTerm {
    /// The adapter's user scale (what `POST /lora-adapters` sets).
    pub handle: SharedLoraScale,
    /// The adapter's own scale, fixed at load.
    pub base_scale: f32,
    /// `[vocab, rank]`, `float32`.
    pub a: UniquePtr<MlxArray>,
    /// `[rank, hidden]`, `float32`.
    pub b: UniquePtr<MlxArray>,
}

impl RuntimeLoraEmbeddingTerm {
    /// The lookup end, `scale * a[indices] @ b` in `float32`, or `None` when
    /// the user scale is `0.0` (no operation is added at all).
    #[must_use]
    pub fn lookup_contribution(&self, indices: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        let user_scale = self.handle.get();
        if user_scale == 0.0 {
            return None;
        }
        let rows = ffi::embedding(&self.a, indices);
        let out = ffi::matmul(&rows, &self.b);
        Some(crate::multiply_scalar(&out, user_scale * self.base_scale))
    }

    /// The tied-head end as a linear term over `[hidden] -> [vocab]`,
    /// `scale * (x @ bᵀ) @ aᵀ` (mlx-lm's `LoRAEmbedding.as_linear`). The
    /// factors are transposed views of this term's own buffers, so attaching
    /// it copies no tensor data, and the handle is this term's.
    #[must_use]
    pub fn head_term(&self) -> RuntimeLoraTerm {
        RuntimeLoraTerm {
            handle: self.handle.clone(),
            base_scale: self.base_scale,
            a_t: ffi::transpose(&self.b),
            b_t: ffi::transpose(&self.a),
        }
    }

    /// Whether this term currently contributes (user scale non-zero).
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.handle.get() != 0.0
    }
}

/// Add every active embedding term's lookup contribution to the lookup output
/// `y` for `indices`.
///
/// Each contribution is cast to `y`'s dtype before the add, as mlx-lm's
/// `LoRAEmbedding.__call__` does, which also makes an adapted row equal to the
/// same row of a table fused as `W + (scale * a @ b).astype(W.dtype)`.
#[must_use]
pub fn apply_embedding_lookup_terms(
    terms: &[Arc<RuntimeLoraEmbeddingTerm>],
    indices: &MlxArray,
    mut y: UniquePtr<MlxArray>,
) -> UniquePtr<MlxArray> {
    if terms.is_empty() {
        return y;
    }
    let y_dtype = ffi::array_dtype(&y);
    for term in terms {
        if let Some(contribution) = term.lookup_contribution(indices) {
            y = ffi::add(&y, &ffi::astype(&contribution, y_dtype));
        }
    }
    y
}

/// Add every active embedding term's tied-head contribution to the head
/// output `y` for input `x` (the embedding's `as_linear`). It goes through
/// [`apply_terms`], the arithmetic a linear head built from the embedding's
/// prefix applies, so the two tied-head constructions agree.
#[must_use]
pub fn apply_embedding_head_terms(
    terms: &[Arc<RuntimeLoraEmbeddingTerm>],
    x: &MlxArray,
    y: UniquePtr<MlxArray>,
) -> UniquePtr<MlxArray> {
    if !terms.iter().any(|term| term.is_active()) {
        return y;
    }
    let head_terms: Vec<RuntimeLoraTerm> = terms.iter().map(|term| term.head_term()).collect();
    apply_terms(&head_terms, x, y)
}

/// A staged embedding term in the [`RuntimeLoraEmbeddingTerm`] layout
/// (`a = [vocab, rank]`, `b = [rank, hidden]`). The loader canonicalizes every
/// on-disk orientation it accepts for an embedding pair into this one before
/// staging.
pub struct PendingLoraEmbeddingTerm {
    pub handle: SharedLoraScale,
    pub base_scale: f32,
    pub a: UniquePtr<MlxArray>,
    pub b: UniquePtr<MlxArray>,
}

/// The staged embedding terms for one prefix and whether an embedding lookup
/// has claimed them. They stay staged after the lookup claims them, because a
/// tied output head built later from the same prefix attaches the same terms.
struct StagedEmbedding {
    terms: Vec<Arc<RuntimeLoraEmbeddingTerm>>,
    lookup_claimed: bool,
}

/// A staged, not-yet-claimed adapter term for one layer prefix. The loader
/// normalizes both on-disk orientations (mlx-lm `[in, rank]` / `[rank, out]`
/// and PEFT `[rank, in]` / `[out, rank]`) into the forward orientation
/// before staging: `a_t` is `[in_features, rank]` and `b_t` is
/// `[rank, out_features]`, so the claim only casts to the layer's dtype.
pub struct PendingLoraTerm {
    pub handle: SharedLoraScale,
    pub base_scale: f32,
    pub a_t: UniquePtr<MlxArray>,
    pub b_t: UniquePtr<MlxArray>,
}

thread_local! {
    /// Terms staged by the loader for the model construction that runs next
    /// on this thread, keyed by layer prefix (the weight-map key minus its
    /// `.weight` suffix).
    static PENDING: RefCell<HashMap<String, Vec<PendingLoraTerm>>> =
        RefCell::new(HashMap::new());

    /// Embedding terms staged next to [`PENDING`], keyed by embedding prefix
    /// (issue #2421).
    static PENDING_EMBEDDINGS: RefCell<HashMap<String, StagedEmbedding>> =
        RefCell::new(HashMap::new());
}

/// Stage adapter terms for the model construction about to run on this
/// thread. Replaces any previous staging, embedding terms included (a failed
/// load must not leak terms into the next one).
pub fn stage(terms: HashMap<String, Vec<PendingLoraTerm>>) {
    stage_with_embeddings(terms, HashMap::new());
}

/// [`stage`] with embedding terms as well (issue #2421). The embedding terms
/// are converted to `float32` here, once, because the lookup and the tied head
/// share them.
pub fn stage_with_embeddings(
    terms: HashMap<String, Vec<PendingLoraTerm>>,
    embeddings: HashMap<String, Vec<PendingLoraEmbeddingTerm>>,
) {
    let embeddings = embeddings
        .into_iter()
        .map(|(prefix, terms)| {
            let terms = terms
                .into_iter()
                .map(|term| {
                    Arc::new(RuntimeLoraEmbeddingTerm {
                        handle: term.handle,
                        base_scale: term.base_scale,
                        a: ffi::astype(&term.a, crate::dtype::FLOAT32),
                        b: ffi::astype(&term.b, crate::dtype::FLOAT32),
                    })
                })
                .collect();
            (
                prefix,
                StagedEmbedding {
                    terms,
                    lookup_claimed: false,
                },
            )
        })
        .collect();
    PENDING.with(|pending| *pending.borrow_mut() = terms);
    PENDING_EMBEDDINGS.with(|pending| *pending.borrow_mut() = embeddings);
}

/// Whether anything is staged on this thread. The layer constructors check
/// this first so the no-adapter path costs one thread-local read.
#[must_use]
pub fn has_pending() -> bool {
    PENDING.with(|pending| !pending.borrow().is_empty())
        || PENDING_EMBEDDINGS.with(|pending| !pending.borrow().is_empty())
}

/// Claim the staged terms for `prefix`. The terms are held in f32 (see
/// [`RuntimeLoraTerm`]) whatever the claiming layer's activation dtype is,
/// and [`apply_terms`] casts the summed output back to the base dtype, so
/// the claim needs no dtype from the caller. Returns an empty vector when
/// nothing is staged for the prefix.
///
/// A linear layer built from an embedding's prefix is that embedding read as
/// a tied output head (`UnifiedLinear::from_weights(.., "model.embed_tokens",
/// ..)` on the Llama family), so it also receives the tied-head end of every
/// embedding term staged for the prefix ([`RuntimeLoraEmbeddingTerm::head_term`]).
/// That attaches the term without claiming it: the lookup end still has to be
/// claimed by the embedding, or the load fails as unclaimed.
#[must_use]
pub fn claim(prefix: &str) -> Vec<RuntimeLoraTerm> {
    if !has_pending() {
        return Vec::new();
    }
    let staged = PENDING.with(|pending| pending.borrow_mut().remove(prefix));
    let mut claimed: Vec<RuntimeLoraTerm> = staged
        .unwrap_or_default()
        .into_iter()
        .map(|term| RuntimeLoraTerm {
            handle: term.handle,
            base_scale: term.base_scale,
            a_t: ffi::astype(&term.a_t, crate::dtype::FLOAT32),
            b_t: ffi::astype(&term.b_t, crate::dtype::FLOAT32),
        })
        .collect();
    PENDING_EMBEDDINGS.with(|pending| {
        if let Some(staged) = pending.borrow().get(prefix) {
            claimed.extend(staged.terms.iter().map(|term| term.head_term()));
        }
    });
    claimed
}

/// Claim the staged embedding terms for `prefix`, for the embedding lookup
/// built there. The terms stay staged for a tied head built later from the
/// same prefix (see [`claim`]); this marks them claimed so the load does not
/// report them. Returns an empty vector when nothing is staged for the prefix.
#[must_use]
pub fn claim_embedding(prefix: &str) -> Vec<Arc<RuntimeLoraEmbeddingTerm>> {
    if !has_pending() {
        return Vec::new();
    }
    PENDING_EMBEDDINGS.with(|pending| {
        pending
            .borrow_mut()
            .get_mut(prefix)
            .map(|staged| {
                staged.lookup_claimed = true;
                staged.terms.clone()
            })
            .unwrap_or_default()
    })
}

/// Drain whatever construction did not claim, returning the layer prefixes in
/// no particular order.
///
/// This is the single source of the unclaimed list. The loader sorts it and
/// fails the load when it is non-empty (issue #1577): an unclaimed term means
/// the model built that layer outside the claiming constructors, so the
/// adapter would never reach it. An embedding term counts as claimed once an
/// embedding lookup claimed it; a tied head attaching it is not enough, since
/// the lookup would still serve the base table.
pub fn drain_unclaimed() -> Vec<String> {
    let mut unclaimed: Vec<String> =
        PENDING.with(|pending| pending.borrow_mut().drain().map(|(k, _)| k).collect());
    PENDING_EMBEDDINGS.with(|pending| {
        unclaimed.extend(
            pending
                .borrow_mut()
                .drain()
                .filter(|(_, staged)| !staged.lookup_claimed)
                .map(|(prefix, _)| prefix),
        );
    });
    unclaimed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::{FusedQKVLinear, Linear, UnifiedEmbedding, UnifiedLinear};
    use crate::weights::WeightMap;

    fn weights_from(entries: Vec<(&str, Vec<f32>, Vec<i32>)>) -> WeightMap {
        let mut map = WeightMap::new();
        for (name, data, shape) in entries {
            map.insert(name.to_string(), crate::from_slice_f32(&data, &shape));
        }
        map
    }

    fn to_vec_f32(arr: &crate::ffi::MlxArray) -> Vec<f32> {
        let a = crate::astype(arr, crate::dtype::FLOAT32);
        crate::eval(&a);
        crate::array_to_raw_bytes(&a)
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn stage_one(prefix: &str, handle: &SharedLoraScale, base_scale: f32) {
        let mut pending = std::collections::HashMap::new();
        pending.insert(
            prefix.to_string(),
            vec![PendingLoraTerm {
                handle: handle.clone(),
                base_scale,
                // a_t: [in=2, rank=1], b_t: [rank=1, out=2]
                a_t: crate::from_slice_f32(&[1.0f32, 1.0], &[2, 1]),
                b_t: crate::from_slice_f32(&[1.0f32, 2.0], &[1, 2]),
            }],
        );
        stage(pending);
    }

    /// The staged term attaches at `Linear::from_weights` and contributes
    /// `user * base * (x @ Aᵀ) @ Bᵀ`; at user scale zero the output is
    /// byte-identical to a base layer that never saw an adapter.
    #[test]
    fn linear_claims_terms_and_scale_zero_is_byte_identical() {
        let weights = weights_from(vec![("proj.weight", vec![1.0, 0.0, 0.0, 1.0], vec![2, 2])]);
        let handle = SharedLoraScale::new(1.0);
        stage_one("proj", &handle, 2.0);
        let adapted = Linear::from_weights(&weights, "proj").expect("adapted");
        assert!(drain_unclaimed().is_empty(), "term claimed");
        let base = Linear::from_weights(&weights, "proj").expect("base");

        let x = crate::from_slice_f32(&[1.0f32, 2.0], &[1, 2]);
        // base: x @ I = [1, 2]; term: (x @ aT) = [3], @ bT = [3, 6], scaled
        // by user(1) * base(2) = [6, 12]; total [7, 14].
        assert_eq!(to_vec_f32(&adapted.forward(&x)), vec![7.0, 14.0]);

        handle.set(0.5);
        assert_eq!(to_vec_f32(&adapted.forward(&x)), vec![4.0, 8.0]);

        handle.set(0.0);
        let zero = adapted.forward(&x);
        let base_out = base.forward(&x);
        crate::eval(&zero);
        crate::eval(&base_out);
        assert_eq!(
            crate::array_to_raw_bytes(&zero),
            crate::array_to_raw_bytes(&base_out),
            "scale zero must be byte-identical to the base layer"
        );
    }

    /// `UnifiedLinear::from_weights` routes a dense layer through
    /// `Linear::from_weights`, so the claim happens there too.
    #[test]
    fn unified_linear_dense_claims_terms() {
        let weights = weights_from(vec![("up.weight", vec![1.0, 0.0, 0.0, 1.0], vec![2, 2])]);
        let handle = SharedLoraScale::new(1.0);
        stage_one("up", &handle, 1.0);
        let layer = UnifiedLinear::from_weights(&weights, "up", 64, 4).expect("layer");
        assert!(drain_unclaimed().is_empty());
        assert!(layer.has_active_runtime_lora());
        handle.set(0.0);
        assert!(!layer.has_active_runtime_lora());
    }

    /// The fused q/k/v representation claims per-projection terms and applies
    /// them to the right slice of the concatenated projection.
    #[test]
    fn fused_qkv_applies_the_q_term_to_the_q_slice_only() {
        // hidden = 2, one head, head_dim = 2 (q), one kv head.
        let weights = weights_from(vec![
            ("attn.q_proj.weight", vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]),
            ("attn.k_proj.weight", vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]),
            ("attn.v_proj.weight", vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]),
        ]);
        let handle = SharedLoraScale::new(1.0);
        stage_one("attn.q_proj", &handle, 1.0);
        let fused =
            FusedQKVLinear::from_weights_separate(&weights, "attn", 64, 4, 1, 1, 2).expect("fused");
        assert!(drain_unclaimed().is_empty(), "q term claimed");

        let x = crate::from_slice_f32(&[1.0f32, 2.0], &[1, 2]);
        let (q, k, v) = fused.forward(&x);
        // q = x + (x @ aT) @ bT = [1,2] + [3,6] = [4, 8]; k = v = x.
        assert_eq!(to_vec_f32(&q), vec![4.0, 8.0]);
        assert_eq!(to_vec_f32(&k), vec![1.0, 2.0]);
        assert_eq!(to_vec_f32(&v), vec![1.0, 2.0]);

        handle.set(0.0);
        let (q, _, _) = fused.forward(&x);
        assert_eq!(
            to_vec_f32(&q),
            vec![1.0, 2.0],
            "scale zero leaves q at base"
        );
    }

    /// Staging is replaced wholesale and unclaimed terms drain with their
    /// layer names, the source of the loader's unclaimed-layer error.
    #[test]
    fn unclaimed_terms_drain_with_their_prefixes() {
        let handle = SharedLoraScale::new(1.0);
        stage_one("nonexistent.layer", &handle, 1.0);
        assert!(has_pending());
        let unclaimed = drain_unclaimed();
        assert_eq!(unclaimed, vec!["nonexistent.layer".to_string()]);
        assert!(!has_pending());
    }

    /// Stage one embedding term for `emb`: a = [[1], [2], [3]] (vocab 3,
    /// rank 1), b = [[1, 10]] (hidden 2), so the delta is a @ b.
    fn stage_embedding(handle: &SharedLoraScale, base_scale: f32) {
        let mut embeddings = std::collections::HashMap::new();
        embeddings.insert(
            "emb".to_string(),
            vec![PendingLoraEmbeddingTerm {
                handle: handle.clone(),
                base_scale,
                a: crate::from_slice_f32(&[1.0f32, 2.0, 3.0], &[3, 1]),
                b: crate::from_slice_f32(&[1.0f32, 10.0], &[1, 2]),
            }],
        );
        stage_with_embeddings(std::collections::HashMap::new(), embeddings);
    }

    fn table() -> WeightMap {
        weights_from(vec![(
            "emb.weight",
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            vec![3, 2],
        )])
    }

    /// The embedding claims its term: the lookup adds `scale * a[ids] @ b`,
    /// `as_linear` adds `scale * (x @ bᵀ) @ aᵀ`, and a linear head built from
    /// the same prefix gets the same tied-head term (issue #2421).
    #[test]
    fn embedding_claims_the_lookup_term_and_shares_it_with_the_tied_head() {
        let weights = table();
        let handle = SharedLoraScale::new(1.0);
        stage_embedding(&handle, 2.0);
        let embedding = UnifiedEmbedding::from_weights(&weights, "emb", 64, 4).expect("emb");
        let head = UnifiedLinear::from_weights(&weights, "emb", 64, 4).expect("head");
        assert!(drain_unclaimed().is_empty(), "the lookup claimed the term");
        assert!(embedding.has_active_runtime_lora());
        assert!(head.has_active_runtime_lora());

        // Rows 0 and 2: base [0, 1] and [4, 5]; delta 2 * a[i] @ b = [2, 20]
        // and [6, 60].
        let ids = crate::from_slice_i32(&[0, 2], &[2]);
        assert_eq!(
            to_vec_f32(&embedding.forward(&ids)),
            vec![2.0, 21.0, 10.0, 65.0]
        );

        // x = [1, 1]: base x @ Wᵀ = [1, 5, 9]; term 2 * (x @ bᵀ) @ aᵀ =
        // 2 * 11 * [1, 2, 3] = [22, 44, 66].
        let x = crate::from_slice_f32(&[1.0f32, 1.0], &[1, 2]);
        let expected = vec![23.0, 49.0, 75.0];
        assert_eq!(to_vec_f32(&embedding.as_linear(&x)), expected);
        assert_eq!(to_vec_f32(&head.forward(&x)), expected);

        // One handle drives both ends; at zero both are byte-identical to the
        // base table.
        handle.set(0.0);
        let base = UnifiedEmbedding::from_weights(&weights, "emb", 64, 4).expect("base");
        for (adapted, plain) in [
            (embedding.forward(&ids), base.forward(&ids)),
            (embedding.as_linear(&x), base.as_linear(&x)),
            (head.forward(&x), base.as_linear(&x)),
        ] {
            crate::eval(&adapted);
            crate::eval(&plain);
            assert_eq!(
                crate::array_to_raw_bytes(&adapted),
                crate::array_to_raw_bytes(&plain)
            );
        }
    }

    /// A tied head attaching an embedding term does not claim it: the lookup
    /// would still serve the base table, so the load must report it.
    #[test]
    fn an_embedding_term_only_a_head_attached_is_unclaimed() {
        let weights = table();
        let handle = SharedLoraScale::new(1.0);
        stage_embedding(&handle, 1.0);
        let _head = UnifiedLinear::from_weights(&weights, "emb", 64, 4).expect("head");
        assert_eq!(drain_unclaimed(), vec!["emb".to_string()]);
        assert!(!has_pending());
    }

    /// `stage` replaces embedding staging too, so a failed load cannot leak
    /// an embedding term into the next one.
    #[test]
    fn staging_linear_terms_clears_embedding_terms() {
        let handle = SharedLoraScale::new(1.0);
        stage_embedding(&handle, 1.0);
        stage(std::collections::HashMap::new());
        assert!(!has_pending());
        assert!(claim_embedding("emb").is_empty());
    }
}
