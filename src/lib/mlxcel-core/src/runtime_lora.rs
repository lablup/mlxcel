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
/// `as_linear` ([`EmbeddingLoraTerms::apply_head`]) or a linear layer built from
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

/// The runtime LoRA embedding terms one embedding table holds (issue #2421):
/// the shared terms for the lookup, and their tied-head form, built once at
/// claim time, for the table's own `as_linear`.
#[derive(Default)]
pub struct EmbeddingLoraTerms {
    terms: Vec<Arc<RuntimeLoraEmbeddingTerm>>,
    heads: Vec<RuntimeLoraTerm>,
}

impl EmbeddingLoraTerms {
    /// Hold `terms`, checked against the table they adapt: every term must be
    /// `a = [rows, rank]`, `b = [rank, width]` for the table's `rows` and its
    /// logical (dequantized) `width`.
    ///
    /// The loader's check on a packed table only knows the scales' group count,
    /// so a pair of the wrong hidden width can reach this point, and an MLX
    /// shape mismatch in the forward crosses the cxx bridge as
    /// `std::terminate`. The claim is where the real width is known, so it is
    /// pinned here and a mismatch fails the load instead.
    ///
    /// # Errors
    ///
    /// Returns an error naming `prefix` when a term does not fit the table.
    pub fn new(
        prefix: &str,
        terms: Vec<Arc<RuntimeLoraEmbeddingTerm>>,
        rows: i32,
        width: i32,
    ) -> Result<Self, String> {
        for term in &terms {
            let a = ffi::array_shape(&term.a);
            let b = ffi::array_shape(&term.b);
            let fits =
                a.len() == 2 && b.len() == 2 && a[0] == rows && a[1] == b[0] && b[1] == width;
            if !fits {
                return Err(format!(
                    "LoRA adapter layer {prefix}: embedding pair A {a:?}, B {b:?} does not fit the \
                     [{rows}, {width}] embedding table"
                ));
            }
        }
        let heads = terms.iter().map(|term| term.head_term()).collect();
        Ok(Self { terms, heads })
    }

    /// Whether no term is held (the table serves no adapter).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    /// Whether any held term currently contributes.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.terms.iter().any(|term| term.is_active())
    }

    /// Add every active term's lookup contribution to the lookup output `y`
    /// for `indices`.
    ///
    /// Each contribution is cast to `y`'s dtype before the add, as mlx-lm's
    /// `LoRAEmbedding.__call__` does, which also makes an adapted row equal to
    /// the same row of a table fused as `W + (scale * a @ b).astype(W.dtype)`.
    #[must_use]
    pub fn apply_lookup(
        &self,
        indices: &MlxArray,
        mut y: UniquePtr<MlxArray>,
    ) -> UniquePtr<MlxArray> {
        if self.terms.is_empty() {
            return y;
        }
        let y_dtype = ffi::array_dtype(&y);
        for term in &self.terms {
            if let Some(contribution) = term.lookup_contribution(indices) {
                y = ffi::add(&y, &ffi::astype(&contribution, y_dtype));
            }
        }
        y
    }

    /// Add every active term's tied-head contribution to the head output `y`
    /// for input `x` (the table's `as_linear`), through [`apply_terms`], the
    /// arithmetic a linear head built from the embedding's prefix applies, so
    /// the two tied-head constructions agree.
    #[must_use]
    pub fn apply_head(&self, x: &MlxArray, y: UniquePtr<MlxArray>) -> UniquePtr<MlxArray> {
        if self.heads.is_empty() {
            return y;
        }
        apply_terms(&self.heads, x, y)
    }
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
///
/// The `Arc` holds MLX arrays, which are not `Send`, and the model that owns it
/// stays on its worker thread, so `Rc` would do today. `Arc` is kept so that a
/// model moved under one of the crate's `unsafe impl Send` wrappers still has
/// an atomic count.
#[allow(clippy::arc_with_non_send_sync)]
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

/// Check one linear term's shapes, `a_t` and `b_t`, against the layer it is
/// claimed for: they must be `[in_features, rank]` and `[rank, out_features]`
/// for the layer's logical (dequantized) input width and its output rows.
fn check_linear_term_shape(
    prefix: &str,
    a: &[i32],
    b: &[i32],
    in_features: i32,
    out_features: i32,
) -> Result<(), String> {
    let fits =
        a.len() == 2 && b.len() == 2 && a[0] == in_features && a[1] == b[0] && b[1] == out_features;
    if fits {
        return Ok(());
    }
    Err(format!(
        "LoRA adapter layer {prefix}: the pair (A {a:?}, B {b:?} in [in, rank] / [rank, out] \
         orientation) does not fit the linear layer, which takes {in_features} input features \
         and produces {out_features}"
    ))
}

/// [`claim`] the staged terms for a linear layer built at `prefix`, after
/// checking every one of them against the layer's real widths. Every linear
/// constructor that claims runtime terms goes through this.
///
/// The loader's check on a quantized base only knows the scales' group count
/// (`validate_pair_shapes` in the consuming crate), so a pair of the wrong
/// input width can get here, and an MLX shape mismatch in the forward crosses
/// the cxx bridge as `std::terminate`. The constructor is where the real width
/// is known, so it is pinned here and a mismatch fails the load instead
/// (issue #2437), as [`EmbeddingLoraTerms::new`] does for embedding tables.
/// The tied-head form of an embedding term staged for `prefix` is checked too.
///
/// The check runs before anything leaves the staging slot. A caller that turns
/// a constructor error into an absent layer (`UnifiedLinear::from_weights(..)
/// .ok()` for an optional projection) then still leaves the term staged, and
/// the load fails with the layer reported as unclaimed instead of serving a
/// model without it.
///
/// # Errors
///
/// Returns an error naming `prefix` when a staged term does not fit the layer.
pub fn claim_linear(
    prefix: &str,
    in_features: i32,
    out_features: i32,
) -> Result<Vec<RuntimeLoraTerm>, String> {
    if !has_pending() {
        return Ok(Vec::new());
    }
    PENDING.with(|pending| {
        pending
            .borrow()
            .get(prefix)
            .into_iter()
            .flatten()
            .try_for_each(|term| {
                check_linear_term_shape(
                    prefix,
                    &ffi::array_shape(&term.a_t),
                    &ffi::array_shape(&term.b_t),
                    in_features,
                    out_features,
                )
            })
    })?;
    PENDING_EMBEDDINGS.with(|pending| {
        pending
            .borrow()
            .get(prefix)
            .into_iter()
            .flat_map(|staged| staged.terms.iter())
            .try_for_each(|term| {
                // The head form is `a_t = bᵀ`, `b_t = aᵀ` (see
                // [`RuntimeLoraEmbeddingTerm::head_term`]).
                let mut a_t = ffi::array_shape(&term.b);
                let mut b_t = ffi::array_shape(&term.a);
                a_t.reverse();
                b_t.reverse();
                check_linear_term_shape(prefix, &a_t, &b_t, in_features, out_features)
            })
    })?;
    Ok(claim(prefix))
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

    /// A term whose hidden width does not match a packed table fails the
    /// claim with an error instead of reaching the forward, where the shape
    /// mismatch would abort the process. The loader's packed-table check only
    /// knows the group count, so a `lora_b` of half the width gets this far.
    #[test]
    fn a_term_of_the_wrong_width_on_a_quantized_table_fails_the_claim() {
        // vocab 4, hidden 128, 4-bit, group 64: two groups per row.
        let dense = crate::from_slice_f32(&vec![0.25f32; 4 * 128], &[4, 128]);
        let q = crate::quantize_weights(&dense, 64, 4);
        let mut weights = WeightMap::new();
        weights.insert("emb.weight".to_string(), crate::quantized_weights_w(&q));
        weights.insert(
            "emb.scales".to_string(),
            crate::quantized_weights_scales(&q),
        );
        weights.insert(
            "emb.biases".to_string(),
            crate::quantized_weights_biases(&q),
        );

        let handle = SharedLoraScale::new(1.0);
        for (width, fits) in [(64usize, false), (128, true)] {
            let mut embeddings = std::collections::HashMap::new();
            embeddings.insert(
                "emb".to_string(),
                vec![PendingLoraEmbeddingTerm {
                    handle: handle.clone(),
                    base_scale: 1.0,
                    a: crate::from_slice_f32(&[1.0f32; 4], &[4, 1]),
                    b: crate::from_slice_f32(&vec![1.0f32; width], &[1, width as i32]),
                }],
            );
            stage_with_embeddings(std::collections::HashMap::new(), embeddings);
            let built = UnifiedEmbedding::from_weights(&weights, "emb", 64, 4);
            let _ = drain_unclaimed();
            match built {
                Ok(embedding) => {
                    assert!(fits, "a [1, {width}] lora_b must not fit a 128-wide table");
                    let ids = crate::from_slice_i32(&[1], &[1]);
                    // 0.25 dequantizes exactly; the term adds 1.0 everywhere.
                    assert_eq!(to_vec_f32(&embedding.forward(&ids)), vec![1.25f32; 128]);
                }
                Err(err) => {
                    assert!(!fits, "{err}");
                    assert!(
                        err.contains("does not fit the [4, 128] embedding table"),
                        "{err}"
                    );
                }
            }
        }
    }

    /// A packed `[out, in]` linear weight of all `0.25` under `prefix`, in
    /// `mode` at `group_size` / `bits`, as a checkpoint stores it.
    fn quantized_linear(
        prefix: &str,
        out_features: i32,
        in_features: i32,
        group_size: i32,
        bits: i32,
        mode: &str,
    ) -> WeightMap {
        let dense = crate::from_slice_f32(
            &vec![0.25f32; (out_features * in_features) as usize],
            &[out_features, in_features],
        );
        let q = crate::quantize_weights_with_mode(&dense, group_size, bits, mode);
        let mut weights = WeightMap::new();
        weights.insert(format!("{prefix}.weight"), crate::quantized_weights_w(&q));
        weights.insert(
            format!("{prefix}.scales"),
            crate::quantized_weights_scales(&q),
        );
        if crate::quantized_weights_has_biases(&q) {
            weights.insert(
                format!("{prefix}.biases"),
                crate::quantized_weights_biases(&q),
            );
        }
        weights
    }

    /// Stage one all-ones linear term of `[in_width] -> [out_width]`, rank 1,
    /// for `prefix`.
    fn stage_linear(prefix: &str, handle: &SharedLoraScale, in_width: i32, out_width: i32) {
        let mut pending = std::collections::HashMap::new();
        pending.insert(
            prefix.to_string(),
            vec![PendingLoraTerm {
                handle: handle.clone(),
                base_scale: 1.0,
                a_t: crate::from_slice_f32(&vec![1.0f32; in_width as usize], &[in_width, 1]),
                b_t: crate::from_slice_f32(&vec![1.0f32; out_width as usize], &[1, out_width]),
            }],
        );
        stage(pending);
    }

    /// The input widths the loader's packed-base check accepts for a base of
    /// `groups` quantization groups (`in / groups` in {16, 32, 64, 128}), other
    /// than the real `in_features`. Every one of them passes the loader and
    /// must fail the claim.
    fn loader_plausible_wrong_widths(groups: i32, in_features: i32) -> Vec<i32> {
        [16, 32, 64, 128]
            .into_iter()
            .map(|group| group * groups)
            .filter(|width| *width != in_features)
            .collect()
    }

    /// A term whose input width does not match a packed linear weight fails
    /// the claim with an error instead of reaching the forward, where the
    /// shape mismatch would abort the process (issue #2437). The loader's
    /// packed-base check only knows the group count, so on a 4-bit group-64
    /// `[8, 256]` weight (four groups) widths 64, 128 and 512 get this far.
    /// The right width still loads and adds its term in the forward.
    #[test]
    fn a_term_of_the_wrong_width_on_a_quantized_linear_fails_the_claim() {
        let weights = quantized_linear("proj", 8, 256, 64, 4, "affine");
        let handle = SharedLoraScale::new(1.0);
        let wrong = loader_plausible_wrong_widths(4, 256);
        assert_eq!(wrong, vec![64, 128, 512]);
        for width in wrong {
            stage_linear("proj", &handle, width, 8);
            let built = UnifiedLinear::from_weights(&weights, "proj", 64, 4);
            // The refused term is still staged, so a caller that maps the
            // error to an absent layer still fails the load as unclaimed.
            assert_eq!(drain_unclaimed(), vec!["proj".to_string()]);
            match built {
                Ok(_) => panic!("a [{width}, 1] lora_a must not fit a 256-wide quantized weight"),
                Err(err) => {
                    assert!(err.contains("LoRA adapter layer proj"), "{err}");
                    assert!(
                        err.contains("takes 256 input features and produces 8"),
                        "{err}"
                    );
                }
            }
        }
        // A wrong output width is pinned as well.
        stage_linear("proj", &handle, 256, 16);
        let built = UnifiedLinear::from_weights(&weights, "proj", 64, 4);
        let _ = drain_unclaimed();
        assert!(
            built.is_err(),
            "a [1, 16] lora_b must not fit 8 output rows"
        );

        stage_linear("proj", &handle, 256, 8);
        let layer = UnifiedLinear::from_weights(&weights, "proj", 64, 4).expect("right width");
        assert!(drain_unclaimed().is_empty(), "the term was claimed");
        assert!(layer.is_quantized());
        let x = crate::from_slice_f32(&vec![1.0f32; 256], &[1, 256]);
        // 0.25 dequantizes exactly: base 256 * 0.25 = 64 per output; the term
        // adds (x @ a_t) @ b_t = 256 per output.
        assert_eq!(to_vec_f32(&layer.forward(&x)), vec![320.0f32; 8]);
    }

    /// Every packing the reconciler accepts pins the input width at claim:
    /// affine at every bit width and group size, and the block-float modes.
    /// Two layouts are loaded under a declared group size the reconciler
    /// overrides (mxfp4 stored at 32 under a declared 64, as minicpm-v ships,
    /// and the 4-bit group-16 NVFP4 fallback stored at 64), so the pin must use
    /// the reconciled group size, not the declared one.
    #[test]
    fn every_quantized_linear_layout_pins_the_term_width() {
        use crate::hardware::{QuantMode, QuantModeSupport, quant_mode_support};
        let native = |mode: QuantMode| matches!(quant_mode_support(mode), QuantModeSupport::Native);
        let (out_features, in_features) = (8, 256);
        // (mode, stored group size, declared group size, bits)
        let mut layouts: Vec<(&str, i32, i32, i32)> = Vec::new();
        for group_size in [32, 64, 128] {
            for bits in [2, 3, 4, 5, 6, 8] {
                layouts.push(("affine", group_size, group_size, bits));
            }
        }
        layouts.push(("affine", 64, 16, 4));
        if native(QuantMode::Mxfp4) {
            layouts.push(("mxfp4", 32, 32, 4));
            layouts.push(("mxfp4", 32, 64, 4));
        }
        if native(QuantMode::Mxfp8) {
            layouts.push(("mxfp8", 32, 32, 8));
        }
        if native(QuantMode::Nvfp4) {
            layouts.push(("nvfp4", 16, 16, 4));
        }
        let handle = SharedLoraScale::new(1.0);
        for (mode, stored, declared, bits) in layouts {
            let weights = quantized_linear("proj", out_features, in_features, stored, bits, mode);
            let what = format!("{mode} {bits}-bit gs{stored} (declared gs{declared})");
            let groups = in_features / stored;
            let mut wrong = loader_plausible_wrong_widths(groups, in_features);
            if groups * declared != in_features && !wrong.contains(&(groups * declared)) {
                wrong.push(groups * declared);
            }
            for width in wrong {
                stage_linear("proj", &handle, width, out_features);
                let built =
                    UnifiedLinear::from_weights_with_mode(&weights, "proj", declared, bits, mode);
                let _ = drain_unclaimed();
                match built {
                    Ok(_) => panic!("{what}: a {width}-wide term must fail the claim"),
                    Err(err) => assert!(err.contains("LoRA adapter layer proj"), "{what}: {err}"),
                }
            }
            stage_linear("proj", &handle, in_features, out_features);
            let built =
                UnifiedLinear::from_weights_with_mode(&weights, "proj", declared, bits, mode);
            assert!(drain_unclaimed().is_empty());
            if let Err(err) = built {
                panic!("{what}: the right width must load: {err}");
            }
        }
    }

    /// The fused q/k/v representation pins each projection's terms to its own
    /// packed plane: a k term as wide as the q plane's output, or of a
    /// loader-plausible wrong input width, fails the claim (issue #2437).
    #[test]
    fn fused_qkv_pins_each_projection_term_to_its_own_plane() {
        // hidden 128, two q heads and one kv head of head_dim 32: q [64, 128],
        // k and v [32, 128], 4-bit group 64.
        let mut weights = WeightMap::new();
        for (name, rows) in [("q_proj", 64), ("k_proj", 32), ("v_proj", 32)] {
            let plane = quantized_linear(&format!("attn.{name}"), rows, 128, 64, 4, "affine");
            weights.extend(plane);
        }
        let handle = SharedLoraScale::new(1.0);
        for (in_width, out_width) in [(256, 32), (64, 32), (128, 64)] {
            stage_linear("attn.k_proj", &handle, in_width, out_width);
            let built = FusedQKVLinear::from_weights_separate(&weights, "attn", 64, 4, 2, 1, 32);
            let _ = drain_unclaimed();
            match built {
                Ok(_) => panic!("a [{in_width}] -> [{out_width}] k term must not fit"),
                Err(err) => {
                    assert!(err.contains("LoRA adapter layer attn.k_proj"), "{err}");
                    assert!(
                        err.contains("takes 128 input features and produces 32"),
                        "{err}"
                    );
                }
            }
        }
        stage_linear("attn.k_proj", &handle, 128, 32);
        let fused = FusedQKVLinear::from_weights_separate(&weights, "attn", 64, 4, 2, 1, 32)
            .expect("right width");
        assert!(drain_unclaimed().is_empty(), "the k term was claimed");
        let x = crate::from_slice_f32(&vec![1.0f32; 128], &[1, 128]);
        let (q, k, v) = fused.forward(&x);
        // base 128 * 0.25 = 32 per output; the k term adds 128.
        assert_eq!(to_vec_f32(&q), vec![32.0f32; 64]);
        assert_eq!(to_vec_f32(&k), vec![160.0f32; 32]);
        assert_eq!(to_vec_f32(&v), vec![32.0f32; 32]);
    }
}
