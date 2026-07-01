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

use super::builder::{Builder, Ty, Val, precision_from_env};
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

/// Emit the complete decode_step module text. With `sample`, the graph ends in
/// an on-device argmax and returns the next token id (`tensor<i32>`, the Phase
/// 2b pattern); otherwise it returns the raw `[V]` logits.
pub fn emit_decode(c: &Config, sample: bool) -> String {
    let (decls, a) = build_arg_schema(c);
    let mut b = Builder::new().with_precision(precision_from_env());
    let k = emit_consts(&mut b, c);

    let h = c.hidden;
    let d = c.head_dim;
    let nq = c.n_q;
    let nkv = c.n_kv;
    let g = c.group();

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

    let mut kcache = a.kcache.clone();
    let mut vcache = a.vcache.clone();

    for li in 0..c.n_layers {
        let lw = &a.layers[li];

        // attention block
        let in_ln_w = norm_w(&mut b, &lw.in_ln, c, &k, h);
        let hn = rms_norm(&mut b, &x, &in_ln_w, &k, h);
        let q = b.linear(&hn, &lw.wq);
        let q = add_proj_bias(&mut b, q, &lw.bq);
        let q = b.reshape(&q, vec![nq, d]);
        let kk = b.linear(&hn, &lw.wk);
        let kk = add_proj_bias(&mut b, kk, &lw.bk);
        let kk = b.reshape(&kk, vec![nkv, d]);
        let vv = b.linear(&hn, &lw.wv);
        let vv = add_proj_bias(&mut b, vv, &lw.bv);
        let vv = b.reshape(&vv, vec![nkv, d]);

        let q = apply_rope(&mut b, &q, &cos_vec, &sin_vec, nq, d);
        let kk = apply_rope(&mut b, &kk, &cos_vec, &sin_vec, nkv, d);

        // write new K/V at [li, cache_len]
        let k_upd = b.reshape(&kk, vec![1, 1, nkv, d]);
        kcache = b.dynamic_update_slice(
            &kcache,
            &k_upd,
            &[&k.layer_idx[li], &a.cache_len, &k.c0, &k.c0],
        );
        let v_upd = b.reshape(&vv, vec![1, 1, nkv, d]);
        vcache = b.dynamic_update_slice(
            &vcache,
            &v_upd,
            &[&k.layer_idx[li], &a.cache_len, &k.c0, &k.c0],
        );

        // read this layer's cache slabs [S, nkv, d]
        let kl = b.slice(&kcache, &[(li, li + 1), (0, MAX_SEQ), (0, nkv), (0, d)]);
        let kl = b.reshape(&kl, vec![MAX_SEQ, nkv, d]);
        let vl = b.slice(&vcache, &[(li, li + 1), (0, MAX_SEQ), (0, nkv), (0, d)]);
        let vl = b.reshape(&vl, vec![MAX_SEQ, nkv, d]);

        // GQA scores via batched dot_general (q head h uses kv head h/g)
        let q_r = b.reshape(&q, vec![nkv, g, d]); // head h = kv*g + grp
        let scores = b.dot_general(&q_r, &kl, &[0], &[1], &[2], &[2], vec![nkv, g, MAX_SEQ]);
        let scores = b.reshape(&scores, vec![nq, MAX_SEQ]);
        let scale_b = b.broadcast(&k.scale, &[], vec![nq, MAX_SEQ]);
        let scores = b.multiply(&scores, &scale_b);
        // Gemma2 attention logit soft-cap, applied to the scaled scores before the mask.
        let scores = match c.attn_logit_softcap {
            Some(cap) => softcap(&mut b, &scores, cap),
            None => scores,
        };
        let kmask_b = b.broadcast(&kmask, &[1], vec![nq, MAX_SEQ]);
        let scores = b.add(&scores, &kmask_b);

        // softmax over the key axis
        let m = b.reduce_max(&scores, 1, &k.neg_inf);
        let m_b = b.broadcast(&m, &[0], vec![nq, MAX_SEQ]);
        let sh = b.subtract(&scores, &m_b);
        let e = b.exponential(&sh);
        let s = b.reduce_add(&e, 1, &k.zero);
        let s_b = b.broadcast(&s, &[0], vec![nq, MAX_SEQ]);
        let attn = b.divide(&e, &s_b);

        // context: out[h,d] = sum_s attn[h,s] * vl[s, h/g, d]
        let attn_r = b.reshape(&attn, vec![nkv, g, MAX_SEQ]);
        let o = b.dot_general(&attn_r, &vl, &[0], &[1], &[2], &[0], vec![nkv, g, d]);
        let o = b.reshape(&o, vec![nq, d]);
        let o = b.reshape(&o, vec![nq * d]);
        let attn_out = b.linear(&o, &lw.wo);
        // Gemma2: post-attention norm on the sublayer output before the residual.
        let attn_out = if c.gemma2 {
            let w = norm_w(&mut b, &lw.post_ln, c, &k, h);
            rms_norm(&mut b, &attn_out, &w, &k, h)
        } else {
            attn_out
        };
        x = b.add(&x, &attn_out);

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
    let (decls, a) = build_batched_arg_schema(c, bsz);
    let mut b = Builder::new().with_precision(precision_from_env());
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
    let (decls, a) = build_ragged_arg_schema(c, bsz);
    let mut b = Builder::new().with_precision(precision_from_env());
    let k = emit_consts(&mut b, c);
    // Constant row indices 0..bsz for the per-row KV-write dim-0 offsets.
    let row_idx: Vec<Val> = (0..bsz).map(|i| b.const_i32(i as i32)).collect();

    let h = c.hidden;
    let d = c.head_dim;
    let nq = c.n_q;
    let nkv = c.n_kv;
    let g = c.group();

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

    let mut kcache = a.kcache.clone();
    let mut vcache = a.vcache.clone();

    for li in 0..c.n_layers {
        let lw = &a.layers[li];

        let in_ln_w = norm_w(&mut b, &lw.in_ln, c, &k, h);
        let hn = rms_norm_seq(&mut b, &x, &in_ln_w, &k, bsz, h); // [B, H]
        let q = b.linear_seq(&hn, &lw.wq);
        let q = add_proj_bias_seq(&mut b, q, &lw.bq, bsz, nq * d);
        let q = b.reshape(&q, vec![bsz, nq, d]);
        let kk = b.linear_seq(&hn, &lw.wk);
        let kk = add_proj_bias_seq(&mut b, kk, &lw.bk, bsz, nkv * d);
        let kk = b.reshape(&kk, vec![bsz, nkv, d]);
        let vv = b.linear_seq(&hn, &lw.wv);
        let vv = add_proj_bias_seq(&mut b, vv, &lw.bv, bsz, nkv * d);
        let vv = b.reshape(&vv, vec![bsz, nkv, d]);

        let q = apply_rope_ragged(&mut b, &q, &cos, &sin, bsz, nq, d);
        let kk = apply_rope_ragged(&mut b, &kk, &cos, &sin, bsz, nkv, d);

        // per-row KV write: row r writes its [1,1,1,nkv,d] K/V at [r, li, pos[r]].
        // `r` indexes the row consts AND the slice ranges (r, r+1), so a plain
        // iterator does not fit; keep the range loop.
        #[allow(clippy::needless_range_loop)]
        for r in 0..bsz {
            let pos_r = b.slice(&a.pos, &[(r, r + 1)]); // [1]
            let pos_r = b.reshape(&pos_r, vec![]); // scalar i32 offset
            let kk_r = b.slice(&kk, &[(r, r + 1), (0, nkv), (0, d)]);
            let kk_upd = b.reshape(&kk_r, vec![1, 1, 1, nkv, d]);
            kcache = b.dynamic_update_slice(
                &kcache,
                &kk_upd,
                &[&row_idx[r], &k.layer_idx[li], &pos_r, &k.c0, &k.c0],
            );
            let vv_r = b.slice(&vv, &[(r, r + 1), (0, nkv), (0, d)]);
            let vv_upd = b.reshape(&vv_r, vec![1, 1, 1, nkv, d]);
            vcache = b.dynamic_update_slice(
                &vcache,
                &vv_upd,
                &[&row_idx[r], &k.layer_idx[li], &pos_r, &k.c0, &k.c0],
            );
        }

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

        // GQA scores (identical to uniform-B); only the mask below is per-row.
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
        // Gemma2 attention logit soft-cap on the scaled scores before the mask.
        let scores = match c.attn_logit_softcap {
            Some(cap) => softcap(&mut b, &scores, cap),
            None => scores,
        };
        let kmask_b = b.broadcast(&kmask, &[0, 2], vec![bsz, nq, MAX_SEQ]); // [B,S] -> [B,nq,S]
        let scores = b.add(&scores, &kmask_b);

        let m = b.reduce_max(&scores, 2, &k.neg_inf);
        let m_b = b.broadcast(&m, &[0, 1], vec![bsz, nq, MAX_SEQ]);
        let sh = b.subtract(&scores, &m_b);
        let e = b.exponential(&sh);
        let s = b.reduce_add(&e, 2, &k.zero);
        let s_b = b.broadcast(&s, &[0, 1], vec![bsz, nq, MAX_SEQ]);
        let attn = b.divide(&e, &s_b);

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
        let attn_out = b.linear_seq(&o, &lw.wo);
        // Gemma2: post-attention norm before the residual.
        let attn_out = if c.gemma2 {
            let w = norm_w(&mut b, &lw.post_ln, c, &k, h);
            rms_norm_seq(&mut b, &attn_out, &w, &k, bsz, h)
        } else {
            attn_out
        };
        x = b.add(&x, &attn_out);

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
    let lp = PREFILL_LP;
    let (decls, a) = build_prefill_arg_schema(c, lp);
    let mut b = Builder::new().with_precision(precision_from_env());
    let k = emit_consts(&mut b, c);

    let h = c.hidden;
    let d = c.head_dim;
    let nq = c.n_q;
    let nkv = c.n_kv;
    let g = c.group();

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

    // caches start as zeros; prefill writes the [0:Lp] block and returns them
    let mut kcache = b.broadcast(&k.zero, &[], vec![c.n_layers, MAX_SEQ, nkv, d]);
    let mut vcache = b.broadcast(&k.zero, &[], vec![c.n_layers, MAX_SEQ, nkv, d]);

    for li in 0..c.n_layers {
        let lw = &a.layers[li];

        // attention block
        let in_ln_w = norm_w(&mut b, &lw.in_ln, c, &k, h);
        let hn = rms_norm_seq(&mut b, &x, &in_ln_w, &k, lp, h); // [Lp, H]
        let q = b.linear_seq(&hn, &lw.wq); // [Lp, qd]
        let q = add_proj_bias_seq(&mut b, q, &lw.bq, lp, nq * d);
        let q = b.reshape(&q, vec![lp, nq, d]);
        let kk = b.linear_seq(&hn, &lw.wk); // [Lp, kv]
        let kk = add_proj_bias_seq(&mut b, kk, &lw.bk, lp, nkv * d);
        let kk = b.reshape(&kk, vec![lp, nkv, d]);
        let vv = b.linear_seq(&hn, &lw.wv); // [Lp, kv]
        let vv = add_proj_bias_seq(&mut b, vv, &lw.bv, lp, nkv * d);
        let vv = b.reshape(&vv, vec![lp, nkv, d]);

        let q = apply_rope_seq(&mut b, &q, &cos, &sin, lp, nq, d);
        let kk = apply_rope_seq(&mut b, &kk, &cos, &sin, lp, nkv, d);

        // write the whole [Lp] K/V block at [li, 0:Lp]
        let k_upd = b.reshape(&kk, vec![1, lp, nkv, d]);
        kcache = b.dynamic_update_slice(&kcache, &k_upd, &[&k.layer_idx[li], &k.c0, &k.c0, &k.c0]);
        let v_upd = b.reshape(&vv, vec![1, lp, nkv, d]);
        vcache = b.dynamic_update_slice(&vcache, &v_upd, &[&k.layer_idx[li], &k.c0, &k.c0, &k.c0]);

        // GQA scores: q head (kv*g+grp) attends kv head kv. Layout [nkv,Lp_i,g,Lp_j]
        // so it reshapes to [nq, Lp_i, Lp_j] without a transpose (head = kv*g+grp).
        let q4 = b.reshape(&q, vec![lp, nkv, g, d]);
        let scores = b.dot_general(&q4, &kk, &[1], &[1], &[3], &[2], vec![nkv, lp, g, lp]);
        let scale_b = b.broadcast(&k.scale, &[], vec![nkv, lp, g, lp]);
        let scores = b.multiply(&scores, &scale_b);
        // Gemma2 attention logit soft-cap on the scaled scores before the mask.
        let scores = match c.attn_logit_softcap {
            Some(cap) => softcap(&mut b, &scores, cap),
            None => scores,
        };
        let cmask_b = b.broadcast(&cmask, &[1, 3], vec![nkv, lp, g, lp]);
        let scores = b.add(&scores, &cmask_b);

        // softmax over the key axis (Lp_j, dim 3)
        let m = b.reduce_max(&scores, 3, &k.neg_inf); // [nkv, Lp, g]
        let m_b = b.broadcast(&m, &[0, 1, 2], vec![nkv, lp, g, lp]);
        let sh = b.subtract(&scores, &m_b);
        let e = b.exponential(&sh);
        let s = b.reduce_add(&e, 3, &k.zero); // [nkv, Lp, g]
        let s_b = b.broadcast(&s, &[0, 1, 2], vec![nkv, lp, g, lp]);
        let attn = b.divide(&e, &s_b); // [nkv, Lp, g, Lp]

        // context: o[kv,i,grp,d] = sum_j attn[kv,i,grp,j] * vv[j,kv,d]
        let o = b.dot_general(&attn, &vv, &[0], &[1], &[3], &[0], vec![nkv, lp, g, d]);
        let o = b.transpose(&o, &[1, 0, 2, 3]); // [Lp, nkv, g, d]
        let o = b.reshape(&o, vec![lp, nq * d]); // [Lp, nq*d], head-major
        let attn_out = b.linear_seq(&o, &lw.wo); // [Lp, H]
        // Gemma2: post-attention norm before the residual.
        let attn_out = if c.gemma2 {
            let w = norm_w(&mut b, &lw.post_ln, c, &k, h);
            rms_norm_seq(&mut b, &attn_out, &w, &k, lp, h)
        } else {
            attn_out
        };
        x = b.add(&x, &attn_out);

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
