//! Emits the `decode_step` / `prefill` StableHLO modules for the supported
//! Llama-family architectures (Llama, Qwen2) from Rust.
//!
//! Signature mirrors spike/openxla/model_jax.py `decode_step`:
//!   main(params..., token, pos, cache_len, kcache, vcache)
//!       -> (logits[V], kcache, vcache)
//! Weights are individual tensor inputs in the same order JAX emitted
//! (alphabetical within each layer), each carrying its pytree-path loc so the
//! arg-to-weight mapping is self-documenting and reuses the JAX weight glue. For
//! a `qkv_bias` architecture (Qwen2) the per-layer q/k/v projection biases follow
//! the layer's weights (see [`take_layer_weights`]). For an untied checkpoint
//! (`tie_word_embeddings = false`) a separate `params['lm_head']` weight follows
//! `final_norm` and feeds the final logits projection in place of the shared
//! `embed` matrix (see [`take_lm_head`]); a tied checkpoint emits no such arg and
//! is byte-identical to before.

use super::builder::{Builder, Precision, Ty, Val, precision_from_env};
use super::config::Config;
use super::rope;

const MAX_SEQ: usize = 256;

/// Bucketed padded prompt length for prefill. Set to MAX_SEQ so the one bucket
/// covers any prompt the cache holds (e.g. the ~103-token Llama-3.2 chat-template
/// prompt), not just the 46-token spike prompt. KV for real positions is
/// identical to a smaller bucket (padding positions are causally masked), so the
/// generated tokens are unchanged.
const PREFILL_LP: usize = MAX_SEQ;

/// Per-layer weight handles (JAX alphabetical order: down, gate, in_ln,
/// post_ln, up, wk, wo, wq, wv). `bk`/`bq`/`bv` are the q/k/v projection biases,
/// present only for an architecture with `qkv_bias` (Qwen2); `None` for Llama,
/// where the bias add emits no op so the graph is byte-identical to before.
struct LayerW {
    down: Val,
    gate: Val,
    in_ln: Val,
    post_ln: Val,
    up: Val,
    wk: Val,
    wo: Val,
    wq: Val,
    wv: Val,
    bk: Option<Val>,
    bq: Option<Val>,
    bv: Option<Val>,
    /// Gemma2 pre/post feed-forward norms (`None` for Llama / Qwen2). Gemma2 wraps
    /// each sublayer in a pre- and a post-norm: `post_ln` becomes the POST-attn
    /// norm, `pre_ff_ln` the pre-MLP norm, `post_ff_ln` the post-MLP norm.
    pre_ff_ln: Option<Val>,
    post_ff_ln: Option<Val>,
}

struct Args {
    embed: Val,
    final_norm: Val,
    /// Untied LM head (`None` when tied; the tail then reuses `embed`).
    lm_head: Option<Val>,
    layers: Vec<LayerW>,
    token: Val,
    pos: Val,
    cache_len: Val,
    kcache: Val,
    vcache: Val,
}

/// One (arg index, type, pytree-path loc) entry used to render the signature.
struct ArgDecl {
    ty: Ty,
    loc: String,
}

/// Append one (type, pytree-path loc) arg, returning a handle to it. `idx` is the
/// running arg counter; sharing it across every graph kind keeps arg numbering
/// identical to the hand-written builders this replaced.
fn take_arg(decls: &mut Vec<ArgDecl>, idx: &mut usize, ty: Ty, loc: String) -> Val {
    let val = Builder::arg(*idx, ty.clone());
    decls.push(ArgDecl { ty, loc });
    *idx += 1;
    val
}

/// Take the untied LM head weight `params['lm_head']` (`[V, H]`), or `None` for a
/// tied checkpoint (which reuses `embed` for the final projection). Called right
/// after `final_norm` and before the layers, so the weight arg order is embed,
/// final_norm, [lm_head when untied], layers..., matching `weight_names` in
/// `iree.rs`. For a tied model nothing is emitted, so the graph stays byte-
/// identical (the guard that keeps every tied checkpoint unchanged).
fn take_lm_head(decls: &mut Vec<ArgDecl>, idx: &mut usize, c: &Config) -> Option<Val> {
    if c.tie_word_embeddings {
        None
    } else {
        Some(take_arg(
            decls,
            idx,
            Ty::f32(vec![c.vocab, c.hidden]),
            "params['lm_head']".into(),
        ))
    }
}

/// The weight the final logits projection multiplies by: the dedicated `lm_head`
/// for an untied checkpoint, else the tied token-embedding matrix. Both are
/// `[V, H]` (`linear` computes `x @ W^T`), so the tail is identical apart from
/// which buffer it reads.
fn head_weight<'a>(embed: &'a Val, lm_head: &'a Option<Val>) -> &'a Val {
    lm_head.as_ref().unwrap_or(embed)
}

/// Append layer `li`'s weights (and, for `qkv_bias`, its q/k/v biases) in the one
/// canonical order every graph kind shares, so the emitted arg order matches
/// `weight_names` in `iree.rs` exactly. JAX-alphabetical weights (down, gate,
/// in_ln, post_ln, up, wk, wo, wq, wv), then — when `c.qkv_bias` — the k/q/v
/// projection biases (alphabetical, matching the wk<wq<wv weight order). The
/// biases are rank-1: `bk`/`bv` are `[n_kv*head_dim]`, `bq` is `[n_q*head_dim]`.
fn take_layer_weights(decls: &mut Vec<ArgDecl>, idx: &mut usize, c: &Config, li: usize) -> LayerW {
    let h = c.hidden;
    let inter = c.inter;
    let kv = c.n_kv * c.head_dim;
    let qd = c.n_q * c.head_dim;
    let p = |k: &str| format!("params['layers'][{}]['{}']", li, k);
    let down = take_arg(decls, idx, Ty::f32(vec![h, inter]), p("down"));
    let gate = take_arg(decls, idx, Ty::f32(vec![inter, h]), p("gate"));
    let in_ln = take_arg(decls, idx, Ty::f32(vec![h]), p("in_ln"));
    let post_ln = take_arg(decls, idx, Ty::f32(vec![h]), p("post_ln"));
    let up = take_arg(decls, idx, Ty::f32(vec![inter, h]), p("up"));
    let wk = take_arg(decls, idx, Ty::f32(vec![kv, h]), p("wk"));
    // o_proj maps `[n_q*head_dim]` -> `[hidden]`, so its weight is `[h, qd]` (HF's
    // `[out, in]`). For Llama / Qwen2 `qd == h`, so this renders the same square
    // type as before (byte-identical); Gemma2 is genuinely non-square.
    let wo = take_arg(decls, idx, Ty::f32(vec![h, qd]), p("wo"));
    let wq = take_arg(decls, idx, Ty::f32(vec![qd, h]), p("wq"));
    let wv = take_arg(decls, idx, Ty::f32(vec![kv, h]), p("wv"));
    let (bk, bq, bv) = if c.qkv_bias {
        let bk = take_arg(decls, idx, Ty::f32(vec![kv]), p("bk"));
        let bq = take_arg(decls, idx, Ty::f32(vec![qd]), p("bq"));
        let bv = take_arg(decls, idx, Ty::f32(vec![kv]), p("bv"));
        (Some(bk), Some(bq), Some(bv))
    } else {
        (None, None, None)
    };
    // Gemma2's two extra per-layer norms, appended after the q/k/v biases slot in
    // the same order `weight_names` lists them (pre then post feed-forward).
    let (pre_ff_ln, post_ff_ln) = if c.gemma2 {
        let pre = take_arg(decls, idx, Ty::f32(vec![h]), p("pre_ff_ln"));
        let post = take_arg(decls, idx, Ty::f32(vec![h]), p("post_ff_ln"));
        (Some(pre), Some(post))
    } else {
        (None, None)
    };
    LayerW {
        down,
        gate,
        in_ln,
        post_ln,
        up,
        wk,
        wo,
        wq,
        wv,
        bk,
        bq,
        bv,
        pre_ff_ln,
        post_ff_ln,
    }
}

/// Add an optional q/k/v projection bias to a single-token `[K]` projection (the
/// single-sequence decode path). When the bias is absent (Llama) this emits no op
/// and returns the projection unchanged.
fn add_proj_bias(b: &mut Builder, x: Val, bias: &Option<Val>) -> Val {
    match bias {
        Some(bias) => b.add(&x, bias),
        None => x,
    }
}

/// Add an optional q/k/v projection bias to `[N, K]` projections (the prefill /
/// batched / ragged paths): the `[K]` bias broadcasts over the leading row axis.
/// No-op (and no emitted op) when the bias is absent.
fn add_proj_bias_seq(b: &mut Builder, x: Val, bias: &Option<Val>, n: usize, k: usize) -> Val {
    match bias {
        Some(bias) => {
            let bb = b.broadcast(bias, &[1], vec![n, k]);
            b.add(&x, &bb)
        }
        None => x,
    }
}

fn build_arg_schema(c: &Config) -> (Vec<ArgDecl>, Args) {
    let h = c.hidden;
    let v = c.vocab;

    let mut decls: Vec<ArgDecl> = Vec::new();
    let mut idx = 0usize;

    let embed = take_arg(
        &mut decls,
        &mut idx,
        Ty::f32(vec![v, h]),
        "params['embed']".into(),
    );
    let final_norm = take_arg(
        &mut decls,
        &mut idx,
        Ty::f32(vec![h]),
        "params['final_norm']".into(),
    );
    let lm_head = take_lm_head(&mut decls, &mut idx, c);

    let mut layers = Vec::with_capacity(c.n_layers);
    for li in 0..c.n_layers {
        layers.push(take_layer_weights(&mut decls, &mut idx, c, li));
    }

    let token = take_arg(&mut decls, &mut idx, Ty::scalar("i32"), "token".into());
    let pos = take_arg(&mut decls, &mut idx, Ty::scalar("i32"), "pos".into());
    let cache_len = take_arg(&mut decls, &mut idx, Ty::scalar("i32"), "cache_len".into());
    let kcache = take_arg(
        &mut decls,
        &mut idx,
        Ty::f32(vec![c.n_layers, MAX_SEQ, c.n_kv, c.head_dim]),
        "kcache".into(),
    );
    let vcache = take_arg(
        &mut decls,
        &mut idx,
        Ty::f32(vec![c.n_layers, MAX_SEQ, c.n_kv, c.head_dim]),
        "vcache".into(),
    );

    (
        decls,
        Args {
            embed,
            final_norm,
            lm_head,
            layers,
            token,
            pos,
            cache_len,
            kcache,
            vcache,
        },
    )
}

fn render_signature(decls: &[ArgDecl]) -> String {
    let parts: Vec<String> = decls
        .iter()
        .enumerate()
        .map(|(i, d)| format!("%arg{}: {} loc(\"{}\")", i, d.ty.render(), d.loc))
        .collect();
    parts.join(", ")
}

/// Shared scalar/table constants, emitted once at the top of the body.
struct Consts {
    cos_table: Val,
    sin_table: Val,
    zero: Val,
    one: Val,
    neg_inf: Val,
    neg_big: Val,
    eps: Val,
    hidden_f: Val,
    scale: Val,
    c0: Val,
    layer_idx: Vec<Val>,
}

fn emit_consts(b: &mut Builder, c: &Config) -> Consts {
    let (cos, sin) = rope::rope_tables(c, MAX_SEQ);
    let cos_table = b.const_tensor_f32(&cos, vec![MAX_SEQ, c.head_dim]);
    let sin_table = b.const_tensor_f32(&sin, vec![MAX_SEQ, c.head_dim]);
    let zero = b.const_f32(0.0);
    let one = b.const_f32(1.0);
    let neg_inf = b.const_f32(f32::NEG_INFINITY);
    let neg_big = b.const_f32(-1e30);
    let eps = b.const_f32(c.eps);
    let hidden_f = b.const_f32(c.hidden as f32);
    let scale = b.const_f32(c.scale());
    let c0 = b.const_i32(0);
    let layer_idx: Vec<Val> = (0..c.n_layers).map(|i| b.const_i32(i as i32)).collect();
    Consts {
        cos_table,
        sin_table,
        zero,
        one,
        neg_inf,
        neg_big,
        eps,
        hidden_f,
        scale,
        c0,
        layer_idx,
    }
}

/// RMSNorm: x * rsqrt(mean(x*x) + eps) * w, all over the single feature axis.
fn rms_norm(b: &mut Builder, x: &Val, w: &Val, k: &Consts, hidden: usize) -> Val {
    let sq = b.multiply(x, x);
    let ssum = b.reduce_add(&sq, 0, &k.zero); // scalar
    let mean = b.divide(&ssum, &k.hidden_f); // scalar
    let meps = b.add(&mean, &k.eps);
    let r = b.rsqrt(&meps);
    let rb = b.broadcast(&r, &[], vec![hidden]);
    let xr = b.multiply(x, &rb);
    b.multiply(&xr, w)
}

/// Gemma2 `(1 + weight)` norm scale (`weight + 1` over the `[hidden]` feature
/// axis). Gemma stores the RMSNorm weight offset by one, so the gemma2 paths pass
/// `gemma_norm_w(...)` where Llama / Qwen2 pass the raw weight.
fn gemma_norm_w(b: &mut Builder, w: &Val, k: &Consts, hidden: usize) -> Val {
    let one = b.broadcast(&k.one, &[], vec![hidden]);
    b.add(w, &one)
}

/// The RMSNorm weight to feed `rms_norm`: `1 + w` for Gemma2, the raw `w`
/// otherwise. A `Val` clone is just a handle copy (no emitted op), so the
/// Llama / Qwen2 graphs are unchanged.
fn norm_w(b: &mut Builder, w: &Val, c: &Config, k: &Consts, hidden: usize) -> Val {
    if c.gemma2 {
        gemma_norm_w(b, w, k, hidden)
    } else {
        w.clone()
    }
}

/// Gemma2 `gelu_pytorch_tanh` activation, elementwise over `x` (any shape):
/// `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
fn gelu_tanh(b: &mut Builder, x: &Val) -> Val {
    let shape = x.ty.shape.clone();
    let bc = |b: &mut Builder, v: f32, shape: &[usize]| {
        let c = b.const_f32(v);
        b.broadcast(&c, &[], shape.to_vec())
    };
    let c0 = bc(b, (2.0f64 / std::f64::consts::PI).sqrt() as f32, &shape);
    let c1 = bc(b, 0.044715, &shape);
    let half = bc(b, 0.5, &shape);
    let one = bc(b, 1.0, &shape);
    let x2 = b.multiply(x, x);
    let x3 = b.multiply(&x2, x);
    let c1x3 = b.multiply(&c1, &x3);
    let inner1 = b.add(x, &c1x3);
    let inner = b.multiply(&c0, &inner1);
    let t = b.tanh(&inner);
    let onept = b.add(&one, &t);
    let hx = b.multiply(&half, x);
    b.multiply(&hx, &onept)
}

/// Gemma2 logit soft-cap, elementwise over `x`: `cap * tanh(x / cap)`.
fn softcap(b: &mut Builder, x: &Val, cap: f32) -> Val {
    let shape = x.ty.shape.clone();
    let capc = b.const_f32(cap);
    let capb = b.broadcast(&capc, &[], shape);
    let xd = b.divide(x, &capb);
    let t = b.tanh(&xd);
    b.multiply(&t, &capb)
}

/// The seq-shaped (`[n, H]`) per-layer MLP plus its surrounding norms, shared by
/// every multi-row graph (prefill, ragged decode). Llama / Qwen2: a pre-MLP
/// `post_attention_layernorm` then SwiGLU. Gemma2: a pre-MLP
/// `pre_feedforward_layernorm`, GeGLU, and a post-MLP `post_feedforward_layernorm`.
/// Returns the residual already added (`x + down`). For a non-Gemma2 config it
/// emits exactly the op sequence the graphs carried inline, so their text is
/// byte-identical. Writing it once is the lever that makes a new architecture's
/// MLP delta (here, GeGLU + the two FF norms) reach every serve graph at once.
fn seq_mlp(b: &mut Builder, c: &Config, lw: &LayerW, k: &Consts, x: &Val, n: usize) -> Val {
    let h = c.hidden;
    let pre_mlp = if c.gemma2 {
        lw.pre_ff_ln.as_ref().expect("gemma2 pre_ff_ln")
    } else {
        &lw.post_ln
    };
    let pre_mlp_w = norm_w(b, pre_mlp, c, k, h);
    let hn2 = rms_norm_seq(b, x, &pre_mlp_w, k, n, h);
    let gate = b.linear_seq(&hn2, &lw.gate);
    let up = b.linear_seq(&hn2, &lw.up);
    let act = if c.gemma2 {
        gelu_tanh(b, &gate)
    } else {
        let neg = b.negate(&gate);
        let ex = b.exponential(&neg);
        let one_b = b.broadcast(&k.one, &[], vec![n, c.inter]);
        let denom = b.add(&one_b, &ex);
        let sig = b.divide(&one_b, &denom);
        b.multiply(&gate, &sig)
    };
    let act = b.multiply(&act, &up);
    let down = b.linear_seq(&act, &lw.down);
    let down = if c.gemma2 {
        let w = norm_w(
            b,
            lw.post_ff_ln.as_ref().expect("gemma2 post_ff_ln"),
            c,
            k,
            h,
        );
        rms_norm_seq(b, &down, &w, k, n, h)
    } else {
        down
    };
    b.add(x, &down)
}

/// HF half-split RoPE on x:[heads, d]; cos/sin are [d] for the position.
fn apply_rope(b: &mut Builder, x: &Val, cos: &Val, sin: &Val, heads: usize, d: usize) -> Val {
    let half = d / 2;
    let cos_b = b.broadcast(cos, &[1], vec![heads, d]);
    let sin_b = b.broadcast(sin, &[1], vec![heads, d]);
    let xc = b.multiply(x, &cos_b);
    let x1 = b.slice(x, &[(0, heads), (0, half)]);
    let x2 = b.slice(x, &[(0, heads), (half, d)]);
    let nx2 = b.negate(&x2);
    let rh = b.concatenate(&nx2, &x1, 1);
    let rs = b.multiply(&rh, &sin_b);
    b.add(&xc, &rs)
}

// ===========================================================================
// shared per-layer attention core (issue #494)
// ===========================================================================
//
// One driver, [`emit_attention`], emits the complete per-layer attention block
// (input norm, q/k/v projection + bias, RoPE, KV cache write/read, GQA scores,
// scale, soft-cap, mask, softmax, context, o_proj, post-attn norm, residual) for
// the single-sequence decode, ragged decode, and prefill graph kinds. The
// architecture-level surface (the norm offset, the projection bias, the
// attention scale, the Gemma2 soft-cap and post-attn norm, and the reserved
// per-head q/k-norm hook) lives in the driver and its shared free helpers, so a
// new dense family customizes attention once and reaches all three paths
// together. The graph-kind-specific layout (activation rank, RoPE tables and
// their broadcast, KV cache indexing, GQA `dot_general` dims, softmax axis) is
// supplied per kind by [`AttnLayout`]. Each method emits the exact op sequence
// the path previously inlined, so every existing graph stays byte-for-byte
// identical. The uniform-B batched decode is a superseded Stage-1 graph off the
// serve path and keeps its own inline attention (out of this refactor's scope).

/// The graph-kind-specific attention layout: everything that differs between the
/// single-sequence, ragged, and prefill paths. Each variant owns the per-graph
/// constants its methods read (the RoPE cos/sin tensors, the additive key mask,
/// and any per-row index vectors), all built once in the graph's head before the
/// layer loop. `mask` is the global causal mask every layer uses; `mask_local` is
/// the Gemma2 sliding-window mask (issue #495), `Some` only for a windowed config
/// and selected on the local (even) layers by [`AttnLayout::add_mask`].
enum AttnLayout {
    /// Single-token decode: rank-reduced activations (`[heads, d]`), a shared
    /// `[d]` RoPE vector, an `[S]` key mask, and a shared-offset KV write at
    /// `cache_len`.
    Single {
        cos: Val,
        sin: Val,
        mask: Val,
        mask_local: Option<Val>,
        cache_len: Val,
    },
    /// Ragged (continuous-batching) decode: `[B, ...]` activations, a per-row
    /// `[B, d]` RoPE gather, a per-row `[B, S]` mask, and an unrolled per-row KV
    /// write at each row's own `pos[b]`.
    Ragged {
        bsz: usize,
        cos: Val,
        sin: Val,
        mask: Val,
        mask_local: Option<Val>,
        pos: Val,
        row_idx: Vec<Val>,
    },
    /// Prefill: `[Lp, ...]` activations, a per-position `[Lp, d]` RoPE gather, an
    /// `[Lp, Lp]` causal mask, and a whole-block KV write into the zero cache.
    /// Scores read the freshly projected K/V directly (no cache read-back).
    Prefill {
        lp: usize,
        cos: Val,
        sin: Val,
        mask: Val,
        mask_local: Option<Val>,
    },
}

impl AttnLayout {
    /// RMSNorm at this kind's activation rank: rank-reduced for single decode,
    /// per-row over the sequence axis otherwise.
    fn norm(&self, b: &mut Builder, c: &Config, k: &Consts, x: &Val, w: &Val) -> Val {
        match self {
            AttnLayout::Single { .. } => rms_norm(b, x, w, k, c.hidden),
            AttnLayout::Ragged { bsz, .. } => rms_norm_seq(b, x, w, k, *bsz, c.hidden),
            AttnLayout::Prefill { lp, .. } => rms_norm_seq(b, x, w, k, *lp, c.hidden),
        }
    }

    /// Project q/k/v, add the optional q/k/v bias, and reshape to head layout
    /// (`[heads, d]` single; `[N, heads, d]` seq). RoPE is applied separately so
    /// a future per-head q/k norm can slot in between (see [`emit_attention`]).
    fn project_qkv(&self, b: &mut Builder, c: &Config, hn: &Val, lw: &LayerW) -> (Val, Val, Val) {
        let d = c.head_dim;
        let (nq, nkv) = (c.n_q, c.n_kv);
        match self {
            AttnLayout::Single { .. } => {
                let q = b.linear(hn, &lw.wq);
                let q = add_proj_bias(b, q, &lw.bq);
                let q = b.reshape(&q, vec![nq, d]);
                let kk = b.linear(hn, &lw.wk);
                let kk = add_proj_bias(b, kk, &lw.bk);
                let kk = b.reshape(&kk, vec![nkv, d]);
                let vv = b.linear(hn, &lw.wv);
                let vv = add_proj_bias(b, vv, &lw.bv);
                let vv = b.reshape(&vv, vec![nkv, d]);
                (q, kk, vv)
            }
            AttnLayout::Ragged { bsz, .. } => Self::project_qkv_seq(b, c, hn, lw, *bsz),
            AttnLayout::Prefill { lp, .. } => Self::project_qkv_seq(b, c, hn, lw, *lp),
        }
    }

    /// The `[N, ...]` (seq) q/k/v projection shared by ragged decode and prefill.
    fn project_qkv_seq(
        b: &mut Builder,
        c: &Config,
        hn: &Val,
        lw: &LayerW,
        n: usize,
    ) -> (Val, Val, Val) {
        let d = c.head_dim;
        let (nq, nkv) = (c.n_q, c.n_kv);
        let q = b.linear_seq(hn, &lw.wq);
        let q = add_proj_bias_seq(b, q, &lw.bq, n, nq * d);
        let q = b.reshape(&q, vec![n, nq, d]);
        let kk = b.linear_seq(hn, &lw.wk);
        let kk = add_proj_bias_seq(b, kk, &lw.bk, n, nkv * d);
        let kk = b.reshape(&kk, vec![n, nkv, d]);
        let vv = b.linear_seq(hn, &lw.wv);
        let vv = add_proj_bias_seq(b, vv, &lw.bv, n, nkv * d);
        let vv = b.reshape(&vv, vec![n, nkv, d]);
        (q, kk, vv)
    }

    /// Apply this kind's RoPE to q and k (v is never rotated).
    fn rope_qk(&self, b: &mut Builder, c: &Config, q: &Val, kk: &Val) -> (Val, Val) {
        let d = c.head_dim;
        let (nq, nkv) = (c.n_q, c.n_kv);
        match self {
            AttnLayout::Single { cos, sin, .. } => {
                let q = apply_rope(b, q, cos, sin, nq, d);
                let kk = apply_rope(b, kk, cos, sin, nkv, d);
                (q, kk)
            }
            AttnLayout::Ragged { bsz, cos, sin, .. } => {
                let q = apply_rope_ragged(b, q, cos, sin, *bsz, nq, d);
                let kk = apply_rope_ragged(b, kk, cos, sin, *bsz, nkv, d);
                (q, kk)
            }
            AttnLayout::Prefill { lp, cos, sin, .. } => {
                let q = apply_rope_seq(b, q, cos, sin, *lp, nq, d);
                let kk = apply_rope_seq(b, kk, cos, sin, *lp, nkv, d);
                (q, kk)
            }
        }
    }

    /// Write the new K/V into the cache and return the (K, V) tensors the scores
    /// read: the freshly projected block for prefill (no read-back), the layer's
    /// cache slab otherwise. Mutates `kcache` / `vcache` in place.
    #[allow(clippy::too_many_arguments)]
    fn write_read_kv(
        &self,
        b: &mut Builder,
        k: &Consts,
        c: &Config,
        li: usize,
        kk: &Val,
        vv: &Val,
        kcache: &mut Val,
        vcache: &mut Val,
    ) -> (Val, Val) {
        let d = c.head_dim;
        let nkv = c.n_kv;
        match self {
            AttnLayout::Single { cache_len, .. } => {
                let k_upd = b.reshape(kk, vec![1, 1, nkv, d]);
                *kcache = b.dynamic_update_slice(
                    &*kcache,
                    &k_upd,
                    &[&k.layer_idx[li], cache_len, &k.c0, &k.c0],
                );
                let v_upd = b.reshape(vv, vec![1, 1, nkv, d]);
                *vcache = b.dynamic_update_slice(
                    &*vcache,
                    &v_upd,
                    &[&k.layer_idx[li], cache_len, &k.c0, &k.c0],
                );
                let kl = b.slice(&*kcache, &[(li, li + 1), (0, MAX_SEQ), (0, nkv), (0, d)]);
                let kl = b.reshape(&kl, vec![MAX_SEQ, nkv, d]);
                let vl = b.slice(&*vcache, &[(li, li + 1), (0, MAX_SEQ), (0, nkv), (0, d)]);
                let vl = b.reshape(&vl, vec![MAX_SEQ, nkv, d]);
                (kl, vl)
            }
            AttnLayout::Ragged {
                bsz, pos, row_idx, ..
            } => {
                // Row r writes its `[1,1,1,nkv,d]` K/V at `[r, li, pos[r]]`. `r`
                // indexes both the row consts and the (r, r+1) slice ranges, so a
                // plain iterator does not fit; keep the range loop.
                #[allow(clippy::needless_range_loop)]
                for r in 0..*bsz {
                    let pos_r = b.slice(pos, &[(r, r + 1)]);
                    let pos_r = b.reshape(&pos_r, vec![]);
                    let kk_r = b.slice(kk, &[(r, r + 1), (0, nkv), (0, d)]);
                    let kk_upd = b.reshape(&kk_r, vec![1, 1, 1, nkv, d]);
                    *kcache = b.dynamic_update_slice(
                        &*kcache,
                        &kk_upd,
                        &[&row_idx[r], &k.layer_idx[li], &pos_r, &k.c0, &k.c0],
                    );
                    let vv_r = b.slice(vv, &[(r, r + 1), (0, nkv), (0, d)]);
                    let vv_upd = b.reshape(&vv_r, vec![1, 1, 1, nkv, d]);
                    *vcache = b.dynamic_update_slice(
                        &*vcache,
                        &vv_upd,
                        &[&row_idx[r], &k.layer_idx[li], &pos_r, &k.c0, &k.c0],
                    );
                }
                let kl = b.slice(
                    &*kcache,
                    &[(0, *bsz), (li, li + 1), (0, MAX_SEQ), (0, nkv), (0, d)],
                );
                let kl = b.reshape(&kl, vec![*bsz, MAX_SEQ, nkv, d]);
                let vl = b.slice(
                    &*vcache,
                    &[(0, *bsz), (li, li + 1), (0, MAX_SEQ), (0, nkv), (0, d)],
                );
                let vl = b.reshape(&vl, vec![*bsz, MAX_SEQ, nkv, d]);
                (kl, vl)
            }
            AttnLayout::Prefill { lp, .. } => {
                let k_upd = b.reshape(kk, vec![1, *lp, nkv, d]);
                *kcache = b.dynamic_update_slice(
                    &*kcache,
                    &k_upd,
                    &[&k.layer_idx[li], &k.c0, &k.c0, &k.c0],
                );
                let v_upd = b.reshape(vv, vec![1, *lp, nkv, d]);
                *vcache = b.dynamic_update_slice(
                    &*vcache,
                    &v_upd,
                    &[&k.layer_idx[li], &k.c0, &k.c0, &k.c0],
                );
                (kk.clone(), vv.clone())
            }
        }
    }

    /// The GQA scores `dot_general` in this kind's score shape (single / ragged
    /// reshape to `[.., nq, S]`; prefill keeps `[nkv, Lp, g, Lp]`), pre-scale.
    fn raw_scores(&self, b: &mut Builder, c: &Config, q: &Val, kslab: &Val) -> Val {
        let d = c.head_dim;
        let (nq, nkv, g) = (c.n_q, c.n_kv, c.group());
        match self {
            AttnLayout::Single { .. } => {
                let q_r = b.reshape(q, vec![nkv, g, d]);
                let scores =
                    b.dot_general(&q_r, kslab, &[0], &[1], &[2], &[2], vec![nkv, g, MAX_SEQ]);
                b.reshape(&scores, vec![nq, MAX_SEQ])
            }
            AttnLayout::Ragged { bsz, .. } => {
                let q_r = b.reshape(q, vec![*bsz, nkv, g, d]);
                let scores = b.dot_general(
                    &q_r,
                    kslab,
                    &[0, 1],
                    &[0, 2],
                    &[3],
                    &[3],
                    vec![*bsz, nkv, g, MAX_SEQ],
                );
                b.reshape(&scores, vec![*bsz, nq, MAX_SEQ])
            }
            AttnLayout::Prefill { lp, .. } => {
                let q4 = b.reshape(q, vec![*lp, nkv, g, d]);
                b.dot_general(&q4, kslab, &[1], &[1], &[3], &[2], vec![nkv, *lp, g, *lp])
            }
        }
    }

    /// Broadcast the additive key mask for layer `li` to the score shape and add
    /// it. A local (sliding-window) layer of a windowed config (Gemma2) selects
    /// `mask_local`; every global layer, and every non-windowed config, uses the
    /// causal `mask` (same handle, so the emitted op is byte-identical for them).
    /// See [`layer_mask`].
    fn add_mask(&self, b: &mut Builder, c: &Config, li: usize, scores: &Val) -> Val {
        let (nq, nkv, g) = (c.n_q, c.n_kv, c.group());
        match self {
            AttnLayout::Single {
                mask, mask_local, ..
            } => {
                let m = layer_mask(mask, mask_local, c, li);
                let mb = b.broadcast(m, &[1], vec![nq, MAX_SEQ]);
                b.add(scores, &mb)
            }
            AttnLayout::Ragged {
                bsz,
                mask,
                mask_local,
                ..
            } => {
                let m = layer_mask(mask, mask_local, c, li);
                let mb = b.broadcast(m, &[0, 2], vec![*bsz, nq, MAX_SEQ]);
                b.add(scores, &mb)
            }
            AttnLayout::Prefill {
                lp,
                mask,
                mask_local,
                ..
            } => {
                let m = layer_mask(mask, mask_local, c, li);
                let mb = b.broadcast(m, &[1, 3], vec![nkv, *lp, g, *lp]);
                b.add(scores, &mb)
            }
        }
    }

    /// The softmax reduction axis (the key axis) in this kind's score shape.
    fn score_axis(&self) -> usize {
        match self {
            AttnLayout::Single { .. } => 1,
            AttnLayout::Ragged { .. } => 2,
            AttnLayout::Prefill { .. } => 3,
        }
    }

    /// The attention-weighted V context in `[.., nq*d]`, ready for o_proj.
    fn context(&self, b: &mut Builder, c: &Config, attn: &Val, vslab: &Val) -> Val {
        let d = c.head_dim;
        let (nq, nkv, g) = (c.n_q, c.n_kv, c.group());
        match self {
            AttnLayout::Single { .. } => {
                let attn_r = b.reshape(attn, vec![nkv, g, MAX_SEQ]);
                let o = b.dot_general(&attn_r, vslab, &[0], &[1], &[2], &[0], vec![nkv, g, d]);
                let o = b.reshape(&o, vec![nq, d]);
                b.reshape(&o, vec![nq * d])
            }
            AttnLayout::Ragged { bsz, .. } => {
                let attn_r = b.reshape(attn, vec![*bsz, nkv, g, MAX_SEQ]);
                let o = b.dot_general(
                    &attn_r,
                    vslab,
                    &[0, 1],
                    &[0, 2],
                    &[3],
                    &[1],
                    vec![*bsz, nkv, g, d],
                );
                let o = b.reshape(&o, vec![*bsz, nq, d]);
                b.reshape(&o, vec![*bsz, nq * d])
            }
            AttnLayout::Prefill { lp, .. } => {
                let o = b.dot_general(attn, vslab, &[0], &[1], &[3], &[0], vec![nkv, *lp, g, d]);
                let o = b.transpose(&o, &[1, 0, 2, 3]);
                b.reshape(&o, vec![*lp, nq * d])
            }
        }
    }

    /// The output projection at this kind's activation rank.
    fn o_proj(&self, b: &mut Builder, o: &Val, lw: &LayerW) -> Val {
        match self {
            AttnLayout::Single { .. } => b.linear(o, &lw.wo),
            _ => b.linear_seq(o, &lw.wo),
        }
    }
}

/// Select the additive key mask for layer `li` (issue #495): the local
/// (sliding-window) mask on a local layer of a windowed config (Gemma2, even
/// layers per [`Config::is_sliding_layer`]), otherwise the global causal `mask`.
/// For a non-windowed config `mask_local` is `None` and `is_sliding_layer` is
/// always false, so this returns `mask` and the emitted broadcast is
/// byte-for-byte identical to before.
fn layer_mask<'a>(mask: &'a Val, mask_local: &'a Option<Val>, c: &Config, li: usize) -> &'a Val {
    if c.is_sliding_layer(li) {
        mask_local.as_ref().unwrap_or(mask)
    } else {
        mask
    }
}

/// Numerically-stable softmax over `axis` of `scores` (max-subtract, exp,
/// sum-divide). The keep-dims for the max/sum broadcasts are every axis but
/// `axis`, so one helper serves the single (`axis 1`), ragged (`axis 2`), and
/// prefill (`axis 3`) score ranks identically.
fn attn_softmax(b: &mut Builder, k: &Consts, scores: &Val, axis: usize) -> Val {
    let shape = scores.ty.shape.clone();
    let keep: Vec<usize> = (0..shape.len()).filter(|&i| i != axis).collect();
    let m = b.reduce_max(scores, axis, &k.neg_inf);
    let m_b = b.broadcast(&m, &keep, shape.clone());
    let sh = b.subtract(scores, &m_b);
    let e = b.exponential(&sh);
    let s = b.reduce_add(&e, axis, &k.zero);
    let s_b = b.broadcast(&s, &keep, shape);
    b.divide(&e, &s_b)
}

/// Scale the raw scores by the attention scale and, for Gemma2, soft-cap them,
/// both before the mask. The scalar scale broadcasts to whatever score shape the
/// layout produced and the soft-cap is elementwise, so this is shape-agnostic.
fn apply_scale_and_softcap(b: &mut Builder, c: &Config, k: &Consts, scores: Val) -> Val {
    let scale_b = b.broadcast(&k.scale, &[], scores.ty.shape.clone());
    let scores = b.multiply(&scores, &scale_b);
    match c.attn_logit_softcap {
        Some(cap) => softcap(b, &scores, cap),
        None => scores,
    }
}

/// The architecture RMSNorm applied at a layout's rank: the Gemma2 `(1 + w)`
/// weight offset (a no-op handle-copy for Llama / Qwen2) followed by the layout's
/// rank-appropriate RMSNorm.
fn arch_norm(
    b: &mut Builder,
    c: &Config,
    k: &Consts,
    layout: &AttnLayout,
    x: &Val,
    w_raw: &Val,
) -> Val {
    let w = norm_w(b, w_raw, c, k, c.hidden);
    layout.norm(b, c, k, x, &w)
}

/// Gemma2's post-attention RMSNorm on the sublayer output before the residual (a
/// no-op for Llama / Qwen2, which have no such norm), applied at the layout's rank.
fn post_attn_norm(
    b: &mut Builder,
    c: &Config,
    k: &Consts,
    layout: &AttnLayout,
    attn_out: Val,
    lw: &LayerW,
) -> Val {
    if c.gemma2 {
        let w = norm_w(b, &lw.post_ln, c, k, c.hidden);
        layout.norm(b, c, k, &attn_out, &w)
    } else {
        attn_out
    }
}

/// Emit one layer's complete attention block for graph kind `layout`, returning
/// the residual stream after the attention residual add. The op sequence (input
/// norm, q/k/v projection + bias, RoPE, KV write/read, GQA scores, scale,
/// soft-cap, mask, softmax, context, o_proj, post-attn norm, residual) is the
/// architecture surface a new dense family customizes once; `layout` supplies the
/// per-graph-kind ranks, cache indexing, and dot shapes. Byte-for-byte identical
/// to the op sequence each path previously inlined.
#[allow(clippy::too_many_arguments)]
fn emit_attention(
    b: &mut Builder,
    c: &Config,
    k: &Consts,
    lw: &LayerW,
    li: usize,
    x: &Val,
    layout: &AttnLayout,
    kcache: &mut Val,
    vcache: &mut Val,
) -> Val {
    let hn = arch_norm(b, c, k, layout, x, &lw.in_ln);
    let (q, kk, vv) = layout.project_qkv(b, c, &hn, lw);
    // Reserved hook: a future per-head q/k normalization (e.g. Qwen3) is applied
    // to q and kk here, once, and reaches the single / ragged / prefill paths
    // together. No dense family the emitter serves emits it yet, so nothing is
    // emitted today and every existing graph is unchanged.
    let (q, kk) = layout.rope_qk(b, c, &q, &kk);
    let (kslab, vslab) = layout.write_read_kv(b, k, c, li, &kk, &vv, kcache, vcache);
    let scores = layout.raw_scores(b, c, &q, &kslab);
    let scores = apply_scale_and_softcap(b, c, k, scores);
    let scores = layout.add_mask(b, c, li, &scores);
    let attn = attn_softmax(b, k, &scores, layout.score_axis());
    let o = layout.context(b, c, &attn, &vslab);
    let attn_out = layout.o_proj(b, &o, lw);
    let attn_out = post_attn_norm(b, c, k, layout, attn_out, lw);
    b.add(x, &attn_out)
}

/// Emit the complete decode_step module text. With `sample`, the graph ends in
/// an on-device argmax and returns the next token id (`tensor<i32>`, the Phase
/// 2b pattern); otherwise it returns the raw `[V]` logits.
pub fn emit_decode(c: &Config, sample: bool) -> String {
    emit_decode_with(c, sample, precision_from_env())
}

pub fn emit_decode_with(c: &Config, sample: bool, precision: Precision) -> String {
    let (decls, a) = build_arg_schema(c);
    let mut b = Builder::new().with_precision(precision);
    let k = emit_consts(&mut b, c);

    let h = c.hidden;
    let d = c.head_dim;

    // --- head: embed gather, rope vectors, decode key mask ---
    let emb_row = b.dynamic_slice(&a.embed, &[&a.token, &k.c0], vec![1, h]);
    let mut x = b.reshape(&emb_row, vec![h]);
    // Gemma2 scales the input embeddings by sqrt(hidden).
    if c.gemma2 {
        let norm = b.const_f32(c.embed_normalizer());
        let nb = b.broadcast(&norm, &[], vec![h]);
        x = b.multiply(&x, &nb);
    }

    let cos_row = b.dynamic_slice(&k.cos_table, &[&a.pos, &k.c0], vec![1, d]);
    let cos_vec = b.reshape(&cos_row, vec![d]);
    let sin_row = b.dynamic_slice(&k.sin_table, &[&a.pos, &k.c0], vec![1, d]);
    let sin_vec = b.reshape(&sin_row, vec![d]);

    // mask: keys s valid iff s <= cache_len -> additive 0 / -1e30, shape [S]
    let ii = b.iota(MAX_SEQ);
    let clen_b = b.broadcast(&a.cache_len, &[], vec![MAX_SEQ]);
    let valid = b.compare("LE", &ii, &clen_b, "SIGNED");
    let zeros_s = b.broadcast(&k.zero, &[], vec![MAX_SEQ]);
    let negs_s = b.broadcast(&k.neg_big, &[], vec![MAX_SEQ]);
    let kmask = b.select(&valid, &zeros_s, &negs_s);
    // Gemma2 local (sliding-window) mask (issue #495): a local layer additionally
    // drops keys older than the window, keeping key s iff `cache_len - s < W`. The
    // global `kmask` already encodes causality, so anding the window in is one more
    // `select`: within-window -> keep `kmask`, else force -1e30. Built once and
    // reused by every local layer; emitted only for a config with a window
    // (Gemma2), so Llama / Qwen2 stay byte-identical. When `W >= MAX_SEQ` the
    // predicate is always true (`cache_len - s <= MAX_SEQ-1 < W`), so it is a value
    // no-op, which is why short-context Gemma2 output is unchanged.
    let kmask_local = c.sliding_window.map(|w| {
        let wc = b.const_i32(w as i32);
        let wb = b.broadcast(&wc, &[], vec![MAX_SEQ]);
        let age = b.subtract(&clen_b, &ii); // cache_len - s
        let within = b.compare("LT", &age, &wb, "SIGNED");
        b.select(&within, &kmask, &negs_s)
    });

    let layout = AttnLayout::Single {
        cos: cos_vec,
        sin: sin_vec,
        mask: kmask,
        mask_local: kmask_local,
        cache_len: a.cache_len.clone(),
    };

    let mut kcache = a.kcache.clone();
    let mut vcache = a.vcache.clone();

    for li in 0..c.n_layers {
        let lw = &a.layers[li];

        // attention block (shared per-layer core, issue #494)
        x = emit_attention(&mut b, c, &k, lw, li, &x, &layout, &mut kcache, &mut vcache);

        // MLP. Pre-MLP norm: Llama / Qwen2 use post_attention_layernorm; Gemma2
        // uses pre_feedforward_layernorm (post_attention_layernorm became the
        // post-attn norm above). Activation: SwiGLU (silu) vs Gemma2 GeGLU (gelu).
        let pre_mlp = if c.gemma2 {
            lw.pre_ff_ln.as_ref().expect("gemma2 pre_ff_ln")
        } else {
            &lw.post_ln
        };
        let pre_mlp_w = norm_w(&mut b, pre_mlp, c, &k, h);
        let hn2 = rms_norm(&mut b, &x, &pre_mlp_w, &k, h);
        let gate = b.linear(&hn2, &lw.gate);
        let up = b.linear(&hn2, &lw.up);
        let act = if c.gemma2 {
            gelu_tanh(&mut b, &gate)
        } else {
            // silu(gate) = gate * sigmoid(gate), sigmoid(z) = 1/(1+exp(-z))
            let neg = b.negate(&gate);
            let ex = b.exponential(&neg);
            let one_b = b.broadcast(&k.one, &[], vec![c.inter]);
            let denom = b.add(&one_b, &ex);
            let sig = b.divide(&one_b, &denom);
            b.multiply(&gate, &sig)
        };
        let act = b.multiply(&act, &up);
        let down = b.linear(&act, &lw.down);
        // Gemma2: post-MLP norm before the residual.
        let down = if c.gemma2 {
            let w = norm_w(
                &mut b,
                lw.post_ff_ln.as_ref().expect("gemma2 post_ff_ln"),
                c,
                &k,
                h,
            );
            rms_norm(&mut b, &down, &w, &k, h)
        } else {
            down
        };
        x = b.add(&x, &down);
    }

    // --- tail: final norm + LM head (tied embed or untied lm_head), Gemma2 final
    // logit soft-cap, then optional on-device argmax ---
    let final_w = norm_w(&mut b, &a.final_norm, c, &k, h);
    let xf = rms_norm(&mut b, &x, &final_w, &k, h);
    let logits = b.linear(&xf, head_weight(&a.embed, &a.lm_head)); // [V]
    let logits = match c.final_logit_softcap {
        Some(cap) => softcap(&mut b, &logits, cap),
        None => logits,
    };
    let (out_val, out_ty) = if sample {
        let tok = b.argmax(&logits);
        (tok.name, Ty::scalar("i32").render())
    } else {
        (logits.name, Ty::f32(vec![c.vocab]).render())
    };

    let sig = render_signature(&decls);
    let cache_ty = Ty::f32(vec![c.n_layers, MAX_SEQ, c.n_kv, c.head_dim]).render();
    format!(
        "module @decode_step {{\n  func.func public @main({sig}) -> ({out_ty}, {cache_ty}, {cache_ty}) {{\n{body}    return {l}, {kc}, {vc} : {out_ty}, {cache_ty}, {cache_ty}\n  }}\n}}\n",
        sig = sig,
        out_ty = out_ty,
        cache_ty = cache_ty,
        body = b.body(),
        l = out_val,
        kc = kcache.name,
        vc = vcache.name,
    )
}

// ===========================================================================
// batched decode: uniform-B (lockstep) static batched decode_step (#449 M3)
// ===========================================================================
//
// Stage 1 of the throughput milestone. All B sequences advance in lockstep at
// the SAME position, so `pos`, `cache_len`, and the key mask are shared scalars/
// vectors broadcast over the batch; only the token, the activations, and the KV
// cache carry a leading batch dim B. This turns each decode matmul from a
// batch-1 GEMV (bandwidth/launch-bound on the GPU) into a GEMM that reuses each
// weight across B rows. Signature mirrors `decode_step` with B prepended:
//   main(params..., token[B], pos, cache_len, kcache[B,L,S,nkv,d], vcache[...])
//       -> (token[B] | logits[B,V], kcache, vcache)
// Weights and their pytree-path locs are identical to the single-seq decode.

struct BatchedArgs {
    embed: Val,
    final_norm: Val,
    lm_head: Option<Val>,
    layers: Vec<LayerW>,
    token: Val,     // [B] i32
    pos: Val,       // scalar i32 (shared across the batch)
    cache_len: Val, // scalar i32 (shared across the batch)
    kcache: Val,    // [B, L, MAX_SEQ, nkv, d]
    vcache: Val,
}

fn build_batched_arg_schema(c: &Config, bsz: usize) -> (Vec<ArgDecl>, BatchedArgs) {
    let h = c.hidden;
    let v = c.vocab;

    let mut decls: Vec<ArgDecl> = Vec::new();
    let mut idx = 0usize;

    let embed = take_arg(
        &mut decls,
        &mut idx,
        Ty::f32(vec![v, h]),
        "params['embed']".into(),
    );
    let final_norm = take_arg(
        &mut decls,
        &mut idx,
        Ty::f32(vec![h]),
        "params['final_norm']".into(),
    );
    let lm_head = take_lm_head(&mut decls, &mut idx, c);

    let mut layers = Vec::with_capacity(c.n_layers);
    for li in 0..c.n_layers {
        layers.push(take_layer_weights(&mut decls, &mut idx, c, li));
    }

    let token = take_arg(
        &mut decls,
        &mut idx,
        Ty::new(vec![bsz], "i32"),
        "token".into(),
    );
    let pos = take_arg(&mut decls, &mut idx, Ty::scalar("i32"), "pos".into());
    let cache_len = take_arg(&mut decls, &mut idx, Ty::scalar("i32"), "cache_len".into());
    let kcache = take_arg(
        &mut decls,
        &mut idx,
        Ty::f32(vec![bsz, c.n_layers, MAX_SEQ, c.n_kv, c.head_dim]),
        "kcache".into(),
    );
    let vcache = take_arg(
        &mut decls,
        &mut idx,
        Ty::f32(vec![bsz, c.n_layers, MAX_SEQ, c.n_kv, c.head_dim]),
        "vcache".into(),
    );

    (
        decls,
        BatchedArgs {
            embed,
            final_norm,
            lm_head,
            layers,
            token,
            pos,
            cache_len,
            kcache,
            vcache,
        },
    )
}

/// HF half-split RoPE on x:[B, heads, d]; cos/sin are a single [d] vector for
/// the shared (lockstep) position, broadcast across the batch.
fn apply_rope_batched(
    b: &mut Builder,
    x: &Val,
    cos: &Val,
    sin: &Val,
    bsz: usize,
    heads: usize,
    d: usize,
) -> Val {
    let half = d / 2;
    let cos_b = b.broadcast(cos, &[2], vec![bsz, heads, d]); // [d] -> [B,heads,d]
    let sin_b = b.broadcast(sin, &[2], vec![bsz, heads, d]);
    let xc = b.multiply(x, &cos_b);
    let x1 = b.slice(x, &[(0, bsz), (0, heads), (0, half)]);
    let x2 = b.slice(x, &[(0, bsz), (0, heads), (half, d)]);
    let nx2 = b.negate(&x2);
    let rh = b.concatenate(&nx2, &x1, 2);
    let rs = b.multiply(&rh, &sin_b);
    b.add(&xc, &rs)
}

/// Emit the uniform-B batched `decode_step` module text for a static batch size
/// `bsz`. With `sample`, the graph ends in a per-row on-device argmax and
/// returns `[B]` token ids; otherwise it returns `[B, V]` logits.
pub fn emit_decode_batched(c: &Config, bsz: usize, sample: bool) -> String {
    emit_decode_batched_with(c, bsz, sample, precision_from_env())
}

pub fn emit_decode_batched_with(
    c: &Config,
    bsz: usize,
    sample: bool,
    precision: Precision,
) -> String {
    let (decls, a) = build_batched_arg_schema(c, bsz);
    let mut b = Builder::new().with_precision(precision);
    let k = emit_consts(&mut b, c);

    let h = c.hidden;
    let d = c.head_dim;
    let nq = c.n_q;
    let nkv = c.n_kv;
    let g = c.group();

    // --- head: per-row embed gather, shared rope vectors, shared key mask ---
    let tok_idx = b.reshape(&a.token, vec![bsz, 1]);
    let mut x = b.gather(&a.embed, &tok_idx); // [B, H]

    // pos is shared (lockstep), so cos/sin are one [d] vector for every row.
    let cos_row = b.dynamic_slice(&k.cos_table, &[&a.pos, &k.c0], vec![1, d]);
    let cos_vec = b.reshape(&cos_row, vec![d]);
    let sin_row = b.dynamic_slice(&k.sin_table, &[&a.pos, &k.c0], vec![1, d]);
    let sin_vec = b.reshape(&sin_row, vec![d]);

    // shared key mask [S]: key s valid iff s <= cache_len -> additive 0 / -1e30
    let ii = b.iota(MAX_SEQ);
    let clen_b = b.broadcast(&a.cache_len, &[], vec![MAX_SEQ]);
    let valid = b.compare("LE", &ii, &clen_b, "SIGNED");
    let zeros_s = b.broadcast(&k.zero, &[], vec![MAX_SEQ]);
    let negs_s = b.broadcast(&k.neg_big, &[], vec![MAX_SEQ]);
    let kmask = b.select(&valid, &zeros_s, &negs_s);

    let mut kcache = a.kcache.clone();
    let mut vcache = a.vcache.clone();

    for li in 0..c.n_layers {
        let lw = &a.layers[li];

        // attention block (RMSNorm over H reuses the [N,H] seq variant, N=B)
        let hn = rms_norm_seq(&mut b, &x, &lw.in_ln, &k, bsz, h); // [B, H]
        let q = b.linear_seq(&hn, &lw.wq); // [B, qd]
        let q = add_proj_bias_seq(&mut b, q, &lw.bq, bsz, nq * d);
        let q = b.reshape(&q, vec![bsz, nq, d]);
        let kk = b.linear_seq(&hn, &lw.wk); // [B, kv]
        let kk = add_proj_bias_seq(&mut b, kk, &lw.bk, bsz, nkv * d);
        let kk = b.reshape(&kk, vec![bsz, nkv, d]);
        let vv = b.linear_seq(&hn, &lw.wv); // [B, kv]
        let vv = add_proj_bias_seq(&mut b, vv, &lw.bv, bsz, nkv * d);
        let vv = b.reshape(&vv, vec![bsz, nkv, d]);

        let q = apply_rope_batched(&mut b, &q, &cos_vec, &sin_vec, bsz, nq, d);
        let kk = apply_rope_batched(&mut b, &kk, &cos_vec, &sin_vec, bsz, nkv, d);

        // write new K/V at [:, li, cache_len] across all B rows
        let k_upd = b.reshape(&kk, vec![bsz, 1, 1, nkv, d]);
        kcache = b.dynamic_update_slice(
            &kcache,
            &k_upd,
            &[&k.c0, &k.layer_idx[li], &a.cache_len, &k.c0, &k.c0],
        );
        let v_upd = b.reshape(&vv, vec![bsz, 1, 1, nkv, d]);
        vcache = b.dynamic_update_slice(
            &vcache,
            &v_upd,
            &[&k.c0, &k.layer_idx[li], &a.cache_len, &k.c0, &k.c0],
        );

        // read this layer's cache slabs [B, S, nkv, d]
        let kl = b.slice(
            &kcache,
            &[(0, bsz), (li, li + 1), (0, MAX_SEQ), (0, nkv), (0, d)],
        );
        let kl = b.reshape(&kl, vec![bsz, MAX_SEQ, nkv, d]);
        let vl = b.slice(
            &vcache,
            &[(0, bsz), (li, li + 1), (0, MAX_SEQ), (0, nkv), (0, d)],
        );
        let vl = b.reshape(&vl, vec![bsz, MAX_SEQ, nkv, d]);

        // GQA scores: batch over (B, kv head). q head kv*g+grp attends kv head
        // kv. Output [B, nkv, g, S] reshapes to [B, nq, S] (head = kv*g+grp).
        let q_r = b.reshape(&q, vec![bsz, nkv, g, d]);
        let scores = b.dot_general(
            &q_r,
            &kl,
            &[0, 1],
            &[0, 2],
            &[3],
            &[3],
            vec![bsz, nkv, g, MAX_SEQ],
        );
        let scores = b.reshape(&scores, vec![bsz, nq, MAX_SEQ]);
        let scale_b = b.broadcast(&k.scale, &[], vec![bsz, nq, MAX_SEQ]);
        let scores = b.multiply(&scores, &scale_b);
        let kmask_b = b.broadcast(&kmask, &[2], vec![bsz, nq, MAX_SEQ]);
        let scores = b.add(&scores, &kmask_b);

        // softmax over the key axis (dim 2)
        let m = b.reduce_max(&scores, 2, &k.neg_inf); // [B, nq]
        let m_b = b.broadcast(&m, &[0, 1], vec![bsz, nq, MAX_SEQ]);
        let sh = b.subtract(&scores, &m_b);
        let e = b.exponential(&sh);
        let s = b.reduce_add(&e, 2, &k.zero); // [B, nq]
        let s_b = b.broadcast(&s, &[0, 1], vec![bsz, nq, MAX_SEQ]);
        let attn = b.divide(&e, &s_b); // [B, nq, S]

        // context: o[b,h,d] = sum_s attn[b,h,s] * vl[b,s,h/g,d]
        let attn_r = b.reshape(&attn, vec![bsz, nkv, g, MAX_SEQ]);
        let o = b.dot_general(
            &attn_r,
            &vl,
            &[0, 1],
            &[0, 2],
            &[3],
            &[1],
            vec![bsz, nkv, g, d],
        );
        let o = b.reshape(&o, vec![bsz, nq, d]);
        let o = b.reshape(&o, vec![bsz, nq * d]);
        let attn_out = b.linear_seq(&o, &lw.wo); // [B, H]
        x = b.add(&x, &attn_out);

        // MLP: down( silu(x@gate^T) * (x@up^T) )
        let hn2 = rms_norm_seq(&mut b, &x, &lw.post_ln, &k, bsz, h);
        let gate = b.linear_seq(&hn2, &lw.gate); // [B, inter]
        let up = b.linear_seq(&hn2, &lw.up); // [B, inter]
        let neg = b.negate(&gate);
        let ex = b.exponential(&neg);
        let one_b = b.broadcast(&k.one, &[], vec![bsz, c.inter]);
        let denom = b.add(&one_b, &ex);
        let sig = b.divide(&one_b, &denom);
        let silu = b.multiply(&gate, &sig);
        let act = b.multiply(&silu, &up);
        let down = b.linear_seq(&act, &lw.down); // [B, H]
        x = b.add(&x, &down);
    }

    // --- tail: final norm + LM head (tied embed or untied lm_head) -> [B, V],
    // optional per-row argmax ---
    let xf = rms_norm_seq(&mut b, &x, &a.final_norm, &k, bsz, h); // [B, H]
    let logits = b.linear_seq(&xf, head_weight(&a.embed, &a.lm_head)); // [B, V]
    let (out_val, out_ty) = if sample {
        let tok = b.argmax_batched(&logits);
        (tok.name, Ty::new(vec![bsz], "i32").render())
    } else {
        (logits.name, Ty::f32(vec![bsz, c.vocab]).render())
    };

    let sig = render_signature(&decls);
    let cache_ty = Ty::f32(vec![bsz, c.n_layers, MAX_SEQ, c.n_kv, c.head_dim]).render();
    format!(
        "module @decode_step {{\n  func.func public @main({sig}) -> ({out_ty}, {cache_ty}, {cache_ty}) {{\n{body}    return {l}, {kc}, {vc} : {out_ty}, {cache_ty}, {cache_ty}\n  }}\n}}\n",
        sig = sig,
        out_ty = out_ty,
        cache_ty = cache_ty,
        body = b.body(),
        l = out_val,
        kc = kcache.name,
        vc = vcache.name,
    )
}

// ===========================================================================
// ragged decode: continuous-batching decode_step (#449 M3 Stage 2a)
// ===========================================================================
//
// Like the uniform-B graph, but each row carries its OWN position and length, so
// sequences of different lengths can share the batch (the continuous-batching
// requirement). Versus uniform-B: `pos` and `cache_len` are `[B]` (per row);
// RoPE cos/sin are a per-row gather `[B, d]` from the table by `pos[B]`; the key
// mask is per-row `[B, S]` (valid iff `s <= cache_len[b]`); and the KV write is
// unrolled per row, each row writing its new K/V at its own `pos[b]` (the shared-
// offset `dynamic_update_slice` no longer applies). The attention contractions
// and the LM head are identical to uniform-B; the per-row mask carries the
// raggedness.

struct RaggedArgs {
    embed: Val,
    final_norm: Val,
    lm_head: Option<Val>,
    layers: Vec<LayerW>,
    token: Val,     // [B] i32
    pos: Val,       // [B] i32 (per row)
    cache_len: Val, // [B] i32 (per row)
    kcache: Val,    // [B, L, MAX_SEQ, nkv, d]
    vcache: Val,
}

fn build_ragged_arg_schema(c: &Config, bsz: usize) -> (Vec<ArgDecl>, RaggedArgs) {
    let h = c.hidden;
    let v = c.vocab;

    let mut decls: Vec<ArgDecl> = Vec::new();
    let mut idx = 0usize;

    let embed = take_arg(
        &mut decls,
        &mut idx,
        Ty::f32(vec![v, h]),
        "params['embed']".into(),
    );
    let final_norm = take_arg(
        &mut decls,
        &mut idx,
        Ty::f32(vec![h]),
        "params['final_norm']".into(),
    );
    let lm_head = take_lm_head(&mut decls, &mut idx, c);

    let mut layers = Vec::with_capacity(c.n_layers);
    for li in 0..c.n_layers {
        layers.push(take_layer_weights(&mut decls, &mut idx, c, li));
    }

    let token = take_arg(
        &mut decls,
        &mut idx,
        Ty::new(vec![bsz], "i32"),
        "token".into(),
    );
    let pos = take_arg(
        &mut decls,
        &mut idx,
        Ty::new(vec![bsz], "i32"),
        "pos".into(),
    );
    let cache_len = take_arg(
        &mut decls,
        &mut idx,
        Ty::new(vec![bsz], "i32"),
        "cache_len".into(),
    );
    let kcache = take_arg(
        &mut decls,
        &mut idx,
        Ty::f32(vec![bsz, c.n_layers, MAX_SEQ, c.n_kv, c.head_dim]),
        "kcache".into(),
    );
    let vcache = take_arg(
        &mut decls,
        &mut idx,
        Ty::f32(vec![bsz, c.n_layers, MAX_SEQ, c.n_kv, c.head_dim]),
        "vcache".into(),
    );

    (
        decls,
        RaggedArgs {
            embed,
            final_norm,
            lm_head,
            layers,
            token,
            pos,
            cache_len,
            kcache,
            vcache,
        },
    )
}

/// HF half-split RoPE on x:[B, heads, d]; cos/sin are per-row `[B, d]` (each
/// row's own position), broadcast over the head axis.
fn apply_rope_ragged(
    b: &mut Builder,
    x: &Val,
    cos: &Val,
    sin: &Val,
    bsz: usize,
    heads: usize,
    d: usize,
) -> Val {
    let half = d / 2;
    let cos_b = b.broadcast(cos, &[0, 2], vec![bsz, heads, d]); // [B,d] -> [B,heads,d]
    let sin_b = b.broadcast(sin, &[0, 2], vec![bsz, heads, d]);
    let xc = b.multiply(x, &cos_b);
    let x1 = b.slice(x, &[(0, bsz), (0, heads), (0, half)]);
    let x2 = b.slice(x, &[(0, bsz), (0, heads), (half, d)]);
    let nx2 = b.negate(&x2);
    let rh = b.concatenate(&nx2, &x1, 2);
    let rs = b.multiply(&rh, &sin_b);
    b.add(&xc, &rs)
}

/// Emit the ragged (continuous-batching) `decode_step` module for a static batch
/// size `bsz`. With `sample`, ends in a per-row on-device argmax returning `[B]`
/// token ids; otherwise returns `[B, V]` logits.
pub fn emit_decode_ragged(c: &Config, bsz: usize, sample: bool) -> String {
    emit_decode_ragged_with(c, bsz, sample, precision_from_env())
}

pub fn emit_decode_ragged_with(
    c: &Config,
    bsz: usize,
    sample: bool,
    precision: Precision,
) -> String {
    let (decls, a) = build_ragged_arg_schema(c, bsz);
    let mut b = Builder::new().with_precision(precision);
    let k = emit_consts(&mut b, c);
    // Constant row indices 0..bsz for the per-row KV-write dim-0 offsets.
    let row_idx: Vec<Val> = (0..bsz).map(|i| b.const_i32(i as i32)).collect();

    let h = c.hidden;

    // --- head: per-row embed gather, per-row rope gather, per-row key mask ---
    let tok_idx = b.reshape(&a.token, vec![bsz, 1]);
    let mut x = b.gather(&a.embed, &tok_idx); // [B, H]
    // Gemma2 scales the input embeddings by sqrt(hidden).
    if c.gemma2 {
        let norm = b.const_f32(c.embed_normalizer());
        let nb = b.broadcast(&norm, &[], vec![bsz, h]);
        x = b.multiply(&x, &nb);
    }

    // each row's rope vectors come from its own position: gather [B,d] by pos[B]
    let pos_idx = b.reshape(&a.pos, vec![bsz, 1]);
    let cos = b.gather(&k.cos_table, &pos_idx); // [B, d]
    let sin = b.gather(&k.sin_table, &pos_idx); // [B, d]

    // per-row key mask [B,S]: key s valid for row b iff s <= cache_len[b]
    let ii = b.iota(MAX_SEQ); // [S]
    let ii_b = b.broadcast(&ii, &[1], vec![bsz, MAX_SEQ]); // entry[b,s] = s
    let clen_b = b.broadcast(&a.cache_len, &[0], vec![bsz, MAX_SEQ]); // entry[b,s] = cache_len[b]
    let valid = b.compare("LE", &ii_b, &clen_b, "SIGNED");
    let zeros = b.broadcast(&k.zero, &[], vec![bsz, MAX_SEQ]);
    let negs = b.broadcast(&k.neg_big, &[], vec![bsz, MAX_SEQ]);
    let kmask = b.select(&valid, &zeros, &negs); // [B, S]
    // Gemma2 local (sliding-window) mask (issue #495), per row: keep key s for row
    // b iff `cache_len[b] - s < W`. And it into the causal `kmask` with one more
    // `select`. Emitted only for a windowed config (Gemma2); reused by every local
    // layer. No-op on the value when `W >= MAX_SEQ` (short-context parity).
    let kmask_local = c.sliding_window.map(|w| {
        let wc = b.const_i32(w as i32);
        let wb = b.broadcast(&wc, &[], vec![bsz, MAX_SEQ]);
        let age = b.subtract(&clen_b, &ii_b); // cache_len[b] - s
        let within = b.compare("LT", &age, &wb, "SIGNED");
        b.select(&within, &kmask, &negs)
    });

    let layout = AttnLayout::Ragged {
        bsz,
        cos,
        sin,
        mask: kmask,
        mask_local: kmask_local,
        pos: a.pos.clone(),
        row_idx,
    };

    let mut kcache = a.kcache.clone();
    let mut vcache = a.vcache.clone();

    for li in 0..c.n_layers {
        let lw = &a.layers[li];

        // attention block (shared per-layer core, issue #494)
        x = emit_attention(&mut b, c, &k, lw, li, &x, &layout, &mut kcache, &mut vcache);

        // MLP + its norms (SwiGLU, or Gemma2 GeGLU with pre/post FF norms),
        // shared with the prefill graph.
        x = seq_mlp(&mut b, c, lw, &k, &x, bsz);
    }

    let final_w = norm_w(&mut b, &a.final_norm, c, &k, h);
    let xf = rms_norm_seq(&mut b, &x, &final_w, &k, bsz, h);
    let logits = b.linear_seq(&xf, head_weight(&a.embed, &a.lm_head)); // [B, V]
    // Gemma2 final logit soft-cap (per row; argmax-invariant but kept for exactness).
    let logits = match c.final_logit_softcap {
        Some(cap) => softcap(&mut b, &logits, cap),
        None => logits,
    };
    let (out_val, out_ty) = if sample {
        let tok = b.argmax_batched(&logits);
        (tok.name, Ty::new(vec![bsz], "i32").render())
    } else {
        (logits.name, Ty::f32(vec![bsz, c.vocab]).render())
    };

    let sig = render_signature(&decls);
    let cache_ty = Ty::f32(vec![bsz, c.n_layers, MAX_SEQ, c.n_kv, c.head_dim]).render();
    format!(
        "module @decode_step {{\n  func.func public @main({sig}) -> ({out_ty}, {cache_ty}, {cache_ty}) {{\n{body}    return {l}, {kc}, {vc} : {out_ty}, {cache_ty}, {cache_ty}\n  }}\n}}\n",
        sig = sig,
        out_ty = out_ty,
        cache_ty = cache_ty,
        body = b.body(),
        l = out_val,
        kc = kcache.name,
        vc = vcache.name,
    )
}

// ===========================================================================
// prefill: bucketed multi-token prompt processing
// ===========================================================================
//
// Signature mirrors spike/openxla/model_jax.py `prefill`:
//   main(params..., tokens[Lp], positions[Lp], real_len)
//       -> (last_logits[V], kcache, vcache)
// Unlike decode, prefill takes NO input caches: it zero-initializes them and
// returns the prompt's K/V written into the [0:Lp] block. The whole prompt is
// processed at once over an [Lp] sequence axis with an [Lp,Lp] causal mask, and
// the returned logit is the row at real_len-1 (the last real prompt token).

/// Prefill arg handles. Weights are identical to decode (same order/locs); the
/// trailing inputs are tokens/positions/real_len with no caches.
struct PrefillArgs {
    embed: Val,
    final_norm: Val,
    lm_head: Option<Val>,
    layers: Vec<LayerW>,
    tokens: Val,
    positions: Val,
    real_len: Val,
}

fn build_prefill_arg_schema(c: &Config, lp: usize) -> (Vec<ArgDecl>, PrefillArgs) {
    let h = c.hidden;
    let v = c.vocab;

    let mut decls: Vec<ArgDecl> = Vec::new();
    let mut idx = 0usize;

    let embed = take_arg(
        &mut decls,
        &mut idx,
        Ty::f32(vec![v, h]),
        "params['embed']".into(),
    );
    let final_norm = take_arg(
        &mut decls,
        &mut idx,
        Ty::f32(vec![h]),
        "params['final_norm']".into(),
    );
    let lm_head = take_lm_head(&mut decls, &mut idx, c);

    let mut layers = Vec::with_capacity(c.n_layers);
    for li in 0..c.n_layers {
        layers.push(take_layer_weights(&mut decls, &mut idx, c, li));
    }

    let tokens = take_arg(
        &mut decls,
        &mut idx,
        Ty::new(vec![lp], "i32"),
        "tokens".into(),
    );
    let positions = take_arg(
        &mut decls,
        &mut idx,
        Ty::new(vec![lp], "i32"),
        "positions".into(),
    );
    let real_len = take_arg(&mut decls, &mut idx, Ty::scalar("i32"), "real_len".into());

    (
        decls,
        PrefillArgs {
            embed,
            final_norm,
            lm_head,
            layers,
            tokens,
            positions,
            real_len,
        },
    )
}

/// RMSNorm over a sequence: x:[Lp, H] -> per-row x * rsqrt(mean(x*x)+eps) * w.
fn rms_norm_seq(b: &mut Builder, x: &Val, w: &Val, k: &Consts, lp: usize, hidden: usize) -> Val {
    let sq = b.multiply(x, x);
    let ssum = b.reduce_add(&sq, 1, &k.zero); // [Lp]
    let hb = b.broadcast(&k.hidden_f, &[], vec![lp]);
    let mean = b.divide(&ssum, &hb); // [Lp]
    let epsb = b.broadcast(&k.eps, &[], vec![lp]);
    let meps = b.add(&mean, &epsb);
    let r = b.rsqrt(&meps); // [Lp]
    let rb = b.broadcast(&r, &[0], vec![lp, hidden]); // [Lp, H]
    let xr = b.multiply(x, &rb);
    let wb = b.broadcast(w, &[1], vec![lp, hidden]); // [H] -> [Lp, H]
    b.multiply(&xr, &wb)
}

/// HF half-split RoPE on x:[Lp, heads, d]; cos/sin are [Lp, d] (per position).
fn apply_rope_seq(
    b: &mut Builder,
    x: &Val,
    cos: &Val,
    sin: &Val,
    lp: usize,
    heads: usize,
    d: usize,
) -> Val {
    let half = d / 2;
    let cos_b = b.broadcast(cos, &[0, 2], vec![lp, heads, d]); // [Lp,d] -> [Lp,heads,d]
    let sin_b = b.broadcast(sin, &[0, 2], vec![lp, heads, d]);
    let xc = b.multiply(x, &cos_b);
    let x1 = b.slice(x, &[(0, lp), (0, heads), (0, half)]);
    let x2 = b.slice(x, &[(0, lp), (0, heads), (half, d)]);
    let nx2 = b.negate(&x2);
    let rh = b.concatenate(&nx2, &x1, 2);
    let rs = b.multiply(&rh, &sin_b);
    b.add(&xc, &rs)
}

/// Emit the complete prefill module text. With `sample`, the graph ends in an
/// on-device argmax and returns the first token id (`tensor<i32>`); otherwise it
/// returns the raw `[V]` logits at `real_len-1`.
pub fn emit_prefill(c: &Config, sample: bool) -> String {
    emit_prefill_with(c, sample, precision_from_env())
}

pub fn emit_prefill_with(c: &Config, sample: bool, precision: Precision) -> String {
    let lp = PREFILL_LP;
    let (decls, a) = build_prefill_arg_schema(c, lp);
    let mut b = Builder::new().with_precision(precision);
    let k = emit_consts(&mut b, c);

    let h = c.hidden;
    let d = c.head_dim;
    let nkv = c.n_kv;

    // --- head: embed gather, per-position rope vectors, [Lp,Lp] causal mask ---
    let tok_idx = b.reshape(&a.tokens, vec![lp, 1]);
    let mut x = b.gather(&a.embed, &tok_idx); // [Lp, H]
    // Gemma2 scales the input embeddings by sqrt(hidden).
    if c.gemma2 {
        let norm = b.const_f32(c.embed_normalizer());
        let nb = b.broadcast(&norm, &[], vec![lp, h]);
        x = b.multiply(&x, &nb);
    }

    let pos_idx = b.reshape(&a.positions, vec![lp, 1]);
    let cos = b.gather(&k.cos_table, &pos_idx); // [Lp, d]
    let sin = b.gather(&k.sin_table, &pos_idx); // [Lp, d]

    // causal mask [Lp, Lp]: query i attends key j iff j <= i -> additive 0/-1e30
    let irow = b.iota(lp);
    let row = b.broadcast(&irow, &[0], vec![lp, lp]); // entry[i,j] = i
    let jcol = b.iota(lp);
    let col = b.broadcast(&jcol, &[1], vec![lp, lp]); // entry[i,j] = j
    let le = b.compare("LE", &col, &row, "SIGNED"); // j <= i
    let zeros = b.broadcast(&k.zero, &[], vec![lp, lp]);
    let negs = b.broadcast(&k.neg_big, &[], vec![lp, lp]);
    let cmask = b.select(&le, &zeros, &negs); // [Lp, Lp]
    // Gemma2 local (sliding-window) mask (issue #495): a local layer keeps key j
    // for query i iff `i - j < W` (prefill positions are 0..Lp, so the buffer
    // index equals the position). And it into the causal `cmask` with one more
    // `select`. Emitted only for a windowed config (Gemma2); reused by every local
    // layer. No-op on the value when `W >= Lp` (`i - j <= Lp-1 < W`), so a
    // short-prompt Gemma2 prefill is unchanged.
    let cmask_local = c.sliding_window.map(|w| {
        let wc = b.const_i32(w as i32);
        let wb = b.broadcast(&wc, &[], vec![lp, lp]);
        let age = b.subtract(&row, &col); // i - j
        let within = b.compare("LT", &age, &wb, "SIGNED");
        b.select(&within, &cmask, &negs)
    });

    let layout = AttnLayout::Prefill {
        lp,
        cos,
        sin,
        mask: cmask,
        mask_local: cmask_local,
    };

    // caches start as zeros; prefill writes the [0:Lp] block and returns them
    let mut kcache = b.broadcast(&k.zero, &[], vec![c.n_layers, MAX_SEQ, nkv, d]);
    let mut vcache = b.broadcast(&k.zero, &[], vec![c.n_layers, MAX_SEQ, nkv, d]);

    for li in 0..c.n_layers {
        let lw = &a.layers[li];

        // attention block (shared per-layer core, issue #494)
        x = emit_attention(&mut b, c, &k, lw, li, &x, &layout, &mut kcache, &mut vcache);

        // MLP + its norms (SwiGLU, or Gemma2 GeGLU with pre/post FF norms),
        // shared with the ragged-decode graph.
        x = seq_mlp(&mut b, c, lw, &k, &x, lp);
    }

    // --- tail: final norm, take the row at real_len-1, LM head (tied embed or
    // untied lm_head), Gemma2 final logit soft-cap ---
    let final_w = norm_w(&mut b, &a.final_norm, c, &k, h);
    let xf = rms_norm_seq(&mut b, &x, &final_w, &k, lp, h); // [Lp, H]
    let one_i = b.const_i32(1);
    let last_idx = b.subtract(&a.real_len, &one_i); // real_len - 1
    let last_row = b.dynamic_slice(&xf, &[&last_idx, &k.c0], vec![1, h]); // [1, H]
    let last = b.reshape(&last_row, vec![h]); // [H]
    let logits = b.linear(&last, head_weight(&a.embed, &a.lm_head)); // [V]
    let logits = match c.final_logit_softcap {
        Some(cap) => softcap(&mut b, &logits, cap),
        None => logits,
    };
    let (out_val, out_ty) = if sample {
        let tok = b.argmax(&logits);
        (tok.name, Ty::scalar("i32").render())
    } else {
        (logits.name, Ty::f32(vec![c.vocab]).render())
    };

    let sig = render_signature(&decls);
    let cache_ty = Ty::f32(vec![c.n_layers, MAX_SEQ, c.n_kv, c.head_dim]).render();
    format!(
        "module @prefill {{\n  func.func public @main({sig}) -> ({out_ty}, {cache_ty}, {cache_ty}) {{\n{body}    return {l}, {kc}, {vc} : {out_ty}, {cache_ty}, {cache_ty}\n  }}\n}}\n",
        sig = sig,
        out_ty = out_ty,
        cache_ty = cache_ty,
        body = b.body(),
        l = out_val,
        kc = kcache.name,
        vc = vcache.name,
    )
}
