//! Emits the Llama-3.2-1B `decode_step` StableHLO module from Rust.
//!
//! Signature mirrors spike/openxla/model_jax.py `decode_step`:
//!   main(params..., token, pos, cache_len, kcache, vcache)
//!       -> (logits[V], kcache, vcache)
//! Weights are individual tensor inputs in the same order JAX emitted
//! (alphabetical within each layer), each carrying its pytree-path loc so the
//! arg-to-weight mapping is self-documenting and reuses the JAX weight glue.

use super::builder::{Builder, Ty, Val};
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
/// post_ln, up, wk, wo, wq, wv).
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
}

struct Args {
    embed: Val,
    final_norm: Val,
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

fn build_arg_schema(c: &Config) -> (Vec<ArgDecl>, Args) {
    let h = c.hidden;
    let inter = c.inter;
    let kv = c.n_kv * c.head_dim; // 512
    let qd = c.n_q * c.head_dim; // 2048
    let v = c.vocab;

    let mut decls: Vec<ArgDecl> = Vec::new();
    let mut idx = 0usize;
    let mut take = |decls: &mut Vec<ArgDecl>, ty: Ty, loc: String| -> Val {
        let val = Builder::arg(idx, ty.clone());
        decls.push(ArgDecl { ty, loc });
        idx += 1;
        val
    };

    let embed = take(&mut decls, Ty::f32(vec![v, h]), "params['embed']".into());
    let final_norm = take(&mut decls, Ty::f32(vec![h]), "params['final_norm']".into());

    let mut layers = Vec::with_capacity(c.n_layers);
    for li in 0..c.n_layers {
        let p = |k: &str| format!("params['layers'][{}]['{}']", li, k);
        let down = take(&mut decls, Ty::f32(vec![h, inter]), p("down"));
        let gate = take(&mut decls, Ty::f32(vec![inter, h]), p("gate"));
        let in_ln = take(&mut decls, Ty::f32(vec![h]), p("in_ln"));
        let post_ln = take(&mut decls, Ty::f32(vec![h]), p("post_ln"));
        let up = take(&mut decls, Ty::f32(vec![inter, h]), p("up"));
        let wk = take(&mut decls, Ty::f32(vec![kv, h]), p("wk"));
        let wo = take(&mut decls, Ty::f32(vec![qd, h]), p("wo"));
        let wq = take(&mut decls, Ty::f32(vec![qd, h]), p("wq"));
        let wv = take(&mut decls, Ty::f32(vec![kv, h]), p("wv"));
        layers.push(LayerW {
            down,
            gate,
            in_ln,
            post_ln,
            up,
            wk,
            wo,
            wq,
            wv,
        });
    }

    let token = take(&mut decls, Ty::scalar("i32"), "token".into());
    let pos = take(&mut decls, Ty::scalar("i32"), "pos".into());
    let cache_len = take(&mut decls, Ty::scalar("i32"), "cache_len".into());
    let kcache = take(
        &mut decls,
        Ty::f32(vec![c.n_layers, MAX_SEQ, c.n_kv, c.head_dim]),
        "kcache".into(),
    );
    let vcache = take(
        &mut decls,
        Ty::f32(vec![c.n_layers, MAX_SEQ, c.n_kv, c.head_dim]),
        "vcache".into(),
    );

    (
        decls,
        Args {
            embed,
            final_norm,
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
    let mut b = Builder::new();
    let k = emit_consts(&mut b, c);

    let h = c.hidden;
    let d = c.head_dim;
    let nq = c.n_q;
    let nkv = c.n_kv;
    let g = c.group();

    // --- head: embed gather, rope vectors, decode key mask ---
    let emb_row = b.dynamic_slice(&a.embed, &[&a.token, &k.c0], vec![1, h]);
    let mut x = b.reshape(&emb_row, vec![h]);

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
        let hn = rms_norm(&mut b, &x, &lw.in_ln, &k, h);
        let q = b.linear(&hn, &lw.wq);
        let q = b.reshape(&q, vec![nq, d]);
        let kk = b.linear(&hn, &lw.wk);
        let kk = b.reshape(&kk, vec![nkv, d]);
        let vv = b.linear(&hn, &lw.wv);
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
        x = b.add(&x, &attn_out);

        // MLP: down( silu(x@gate^T) * (x@up^T) )
        let hn2 = rms_norm(&mut b, &x, &lw.post_ln, &k, h);
        let gate = b.linear(&hn2, &lw.gate);
        let up = b.linear(&hn2, &lw.up);
        // silu(gate) = gate * sigmoid(gate), sigmoid(z) = 1/(1+exp(-z))
        let neg = b.negate(&gate);
        let ex = b.exponential(&neg);
        let one_b = b.broadcast(&k.one, &[], vec![c.inter]);
        let denom = b.add(&one_b, &ex);
        let sig = b.divide(&one_b, &denom);
        let silu = b.multiply(&gate, &sig);
        let act = b.multiply(&silu, &up);
        let down = b.linear(&act, &lw.down);
        x = b.add(&x, &down);
    }

    // --- tail: final norm + tied LM head, then optional on-device argmax ---
    let xf = rms_norm(&mut b, &x, &a.final_norm, &k, h);
    let logits = b.linear(&xf, &a.embed); // [V]
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
    layers: Vec<LayerW>,
    token: Val,     // [B] i32
    pos: Val,       // scalar i32 (shared across the batch)
    cache_len: Val, // scalar i32 (shared across the batch)
    kcache: Val,    // [B, L, MAX_SEQ, nkv, d]
    vcache: Val,
}

fn build_batched_arg_schema(c: &Config, bsz: usize) -> (Vec<ArgDecl>, BatchedArgs) {
    let h = c.hidden;
    let inter = c.inter;
    let kv = c.n_kv * c.head_dim; // 512
    let qd = c.n_q * c.head_dim; // 2048
    let v = c.vocab;

    let mut decls: Vec<ArgDecl> = Vec::new();
    let mut idx = 0usize;
    let mut take = |decls: &mut Vec<ArgDecl>, ty: Ty, loc: String| -> Val {
        let val = Builder::arg(idx, ty.clone());
        decls.push(ArgDecl { ty, loc });
        idx += 1;
        val
    };

    let embed = take(&mut decls, Ty::f32(vec![v, h]), "params['embed']".into());
    let final_norm = take(&mut decls, Ty::f32(vec![h]), "params['final_norm']".into());

    let mut layers = Vec::with_capacity(c.n_layers);
    for li in 0..c.n_layers {
        let p = |k: &str| format!("params['layers'][{}]['{}']", li, k);
        let down = take(&mut decls, Ty::f32(vec![h, inter]), p("down"));
        let gate = take(&mut decls, Ty::f32(vec![inter, h]), p("gate"));
        let in_ln = take(&mut decls, Ty::f32(vec![h]), p("in_ln"));
        let post_ln = take(&mut decls, Ty::f32(vec![h]), p("post_ln"));
        let up = take(&mut decls, Ty::f32(vec![inter, h]), p("up"));
        let wk = take(&mut decls, Ty::f32(vec![kv, h]), p("wk"));
        let wo = take(&mut decls, Ty::f32(vec![qd, h]), p("wo"));
        let wq = take(&mut decls, Ty::f32(vec![qd, h]), p("wq"));
        let wv = take(&mut decls, Ty::f32(vec![kv, h]), p("wv"));
        layers.push(LayerW {
            down,
            gate,
            in_ln,
            post_ln,
            up,
            wk,
            wo,
            wq,
            wv,
        });
    }

    let token = take(&mut decls, Ty::new(vec![bsz], "i32"), "token".into());
    let pos = take(&mut decls, Ty::scalar("i32"), "pos".into());
    let cache_len = take(&mut decls, Ty::scalar("i32"), "cache_len".into());
    let kcache = take(
        &mut decls,
        Ty::f32(vec![bsz, c.n_layers, MAX_SEQ, c.n_kv, c.head_dim]),
        "kcache".into(),
    );
    let vcache = take(
        &mut decls,
        Ty::f32(vec![bsz, c.n_layers, MAX_SEQ, c.n_kv, c.head_dim]),
        "vcache".into(),
    );

    (
        decls,
        BatchedArgs {
            embed,
            final_norm,
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
    let mut b = Builder::new();
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
        let q = b.reshape(&q, vec![bsz, nq, d]);
        let kk = b.linear_seq(&hn, &lw.wk); // [B, kv]
        let kk = b.reshape(&kk, vec![bsz, nkv, d]);
        let vv = b.linear_seq(&hn, &lw.wv); // [B, kv]
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

    // --- tail: final norm + tied LM head -> [B, V], optional per-row argmax ---
    let xf = rms_norm_seq(&mut b, &x, &a.final_norm, &k, bsz, h); // [B, H]
    let logits = b.linear_seq(&xf, &a.embed); // [B, V]
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
    layers: Vec<LayerW>,
    token: Val,     // [B] i32
    pos: Val,       // [B] i32 (per row)
    cache_len: Val, // [B] i32 (per row)
    kcache: Val,    // [B, L, MAX_SEQ, nkv, d]
    vcache: Val,
}

fn build_ragged_arg_schema(c: &Config, bsz: usize) -> (Vec<ArgDecl>, RaggedArgs) {
    let h = c.hidden;
    let inter = c.inter;
    let kv = c.n_kv * c.head_dim; // 512
    let qd = c.n_q * c.head_dim; // 2048
    let v = c.vocab;

    let mut decls: Vec<ArgDecl> = Vec::new();
    let mut idx = 0usize;
    let mut take = |decls: &mut Vec<ArgDecl>, ty: Ty, loc: String| -> Val {
        let val = Builder::arg(idx, ty.clone());
        decls.push(ArgDecl { ty, loc });
        idx += 1;
        val
    };

    let embed = take(&mut decls, Ty::f32(vec![v, h]), "params['embed']".into());
    let final_norm = take(&mut decls, Ty::f32(vec![h]), "params['final_norm']".into());

    let mut layers = Vec::with_capacity(c.n_layers);
    for li in 0..c.n_layers {
        let p = |k: &str| format!("params['layers'][{}]['{}']", li, k);
        let down = take(&mut decls, Ty::f32(vec![h, inter]), p("down"));
        let gate = take(&mut decls, Ty::f32(vec![inter, h]), p("gate"));
        let in_ln = take(&mut decls, Ty::f32(vec![h]), p("in_ln"));
        let post_ln = take(&mut decls, Ty::f32(vec![h]), p("post_ln"));
        let up = take(&mut decls, Ty::f32(vec![inter, h]), p("up"));
        let wk = take(&mut decls, Ty::f32(vec![kv, h]), p("wk"));
        let wo = take(&mut decls, Ty::f32(vec![qd, h]), p("wo"));
        let wq = take(&mut decls, Ty::f32(vec![qd, h]), p("wq"));
        let wv = take(&mut decls, Ty::f32(vec![kv, h]), p("wv"));
        layers.push(LayerW {
            down,
            gate,
            in_ln,
            post_ln,
            up,
            wk,
            wo,
            wq,
            wv,
        });
    }

    let token = take(&mut decls, Ty::new(vec![bsz], "i32"), "token".into());
    let pos = take(&mut decls, Ty::new(vec![bsz], "i32"), "pos".into());
    let cache_len = take(&mut decls, Ty::new(vec![bsz], "i32"), "cache_len".into());
    let kcache = take(
        &mut decls,
        Ty::f32(vec![bsz, c.n_layers, MAX_SEQ, c.n_kv, c.head_dim]),
        "kcache".into(),
    );
    let vcache = take(
        &mut decls,
        Ty::f32(vec![bsz, c.n_layers, MAX_SEQ, c.n_kv, c.head_dim]),
        "vcache".into(),
    );

    (
        decls,
        RaggedArgs {
            embed,
            final_norm,
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
    let mut b = Builder::new();
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

        let hn = rms_norm_seq(&mut b, &x, &lw.in_ln, &k, bsz, h); // [B, H]
        let q = b.linear_seq(&hn, &lw.wq);
        let q = b.reshape(&q, vec![bsz, nq, d]);
        let kk = b.linear_seq(&hn, &lw.wk);
        let kk = b.reshape(&kk, vec![bsz, nkv, d]);
        let vv = b.linear_seq(&hn, &lw.wv);
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
        x = b.add(&x, &attn_out);

        let hn2 = rms_norm_seq(&mut b, &x, &lw.post_ln, &k, bsz, h);
        let gate = b.linear_seq(&hn2, &lw.gate);
        let up = b.linear_seq(&hn2, &lw.up);
        let neg = b.negate(&gate);
        let ex = b.exponential(&neg);
        let one_b = b.broadcast(&k.one, &[], vec![bsz, c.inter]);
        let denom = b.add(&one_b, &ex);
        let sig = b.divide(&one_b, &denom);
        let silu = b.multiply(&gate, &sig);
        let act = b.multiply(&silu, &up);
        let down = b.linear_seq(&act, &lw.down);
        x = b.add(&x, &down);
    }

    let xf = rms_norm_seq(&mut b, &x, &a.final_norm, &k, bsz, h);
    let logits = b.linear_seq(&xf, &a.embed); // [B, V]
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
    layers: Vec<LayerW>,
    tokens: Val,
    positions: Val,
    real_len: Val,
}

fn build_prefill_arg_schema(c: &Config, lp: usize) -> (Vec<ArgDecl>, PrefillArgs) {
    let h = c.hidden;
    let inter = c.inter;
    let kv = c.n_kv * c.head_dim; // 512
    let qd = c.n_q * c.head_dim; // 2048
    let v = c.vocab;

    let mut decls: Vec<ArgDecl> = Vec::new();
    let mut idx = 0usize;
    let mut take = |decls: &mut Vec<ArgDecl>, ty: Ty, loc: String| -> Val {
        let val = Builder::arg(idx, ty.clone());
        decls.push(ArgDecl { ty, loc });
        idx += 1;
        val
    };

    let embed = take(&mut decls, Ty::f32(vec![v, h]), "params['embed']".into());
    let final_norm = take(&mut decls, Ty::f32(vec![h]), "params['final_norm']".into());

    let mut layers = Vec::with_capacity(c.n_layers);
    for li in 0..c.n_layers {
        let p = |k: &str| format!("params['layers'][{}]['{}']", li, k);
        let down = take(&mut decls, Ty::f32(vec![h, inter]), p("down"));
        let gate = take(&mut decls, Ty::f32(vec![inter, h]), p("gate"));
        let in_ln = take(&mut decls, Ty::f32(vec![h]), p("in_ln"));
        let post_ln = take(&mut decls, Ty::f32(vec![h]), p("post_ln"));
        let up = take(&mut decls, Ty::f32(vec![inter, h]), p("up"));
        let wk = take(&mut decls, Ty::f32(vec![kv, h]), p("wk"));
        let wo = take(&mut decls, Ty::f32(vec![qd, h]), p("wo"));
        let wq = take(&mut decls, Ty::f32(vec![qd, h]), p("wq"));
        let wv = take(&mut decls, Ty::f32(vec![kv, h]), p("wv"));
        layers.push(LayerW {
            down,
            gate,
            in_ln,
            post_ln,
            up,
            wk,
            wo,
            wq,
            wv,
        });
    }

    let tokens = take(&mut decls, Ty::new(vec![lp], "i32"), "tokens".into());
    let positions = take(&mut decls, Ty::new(vec![lp], "i32"), "positions".into());
    let real_len = take(&mut decls, Ty::scalar("i32"), "real_len".into());

    (
        decls,
        PrefillArgs {
            embed,
            final_norm,
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
    let mut b = Builder::new();
    let k = emit_consts(&mut b, c);

    let h = c.hidden;
    let d = c.head_dim;
    let nq = c.n_q;
    let nkv = c.n_kv;
    let g = c.group();

    // --- head: embed gather, per-position rope vectors, [Lp,Lp] causal mask ---
    let tok_idx = b.reshape(&a.tokens, vec![lp, 1]);
    let mut x = b.gather(&a.embed, &tok_idx); // [Lp, H]

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
        let hn = rms_norm_seq(&mut b, &x, &lw.in_ln, &k, lp, h); // [Lp, H]
        let q = b.linear_seq(&hn, &lw.wq); // [Lp, qd]
        let q = b.reshape(&q, vec![lp, nq, d]);
        let kk = b.linear_seq(&hn, &lw.wk); // [Lp, kv]
        let kk = b.reshape(&kk, vec![lp, nkv, d]);
        let vv = b.linear_seq(&hn, &lw.wv); // [Lp, kv]
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
        x = b.add(&x, &attn_out);

        // MLP: down( silu(x@gate^T) * (x@up^T) )
        let hn2 = rms_norm_seq(&mut b, &x, &lw.post_ln, &k, lp, h);
        let gate = b.linear_seq(&hn2, &lw.gate); // [Lp, inter]
        let up = b.linear_seq(&hn2, &lw.up); // [Lp, inter]
        let neg = b.negate(&gate);
        let ex = b.exponential(&neg);
        let one_b = b.broadcast(&k.one, &[], vec![lp, c.inter]);
        let denom = b.add(&one_b, &ex);
        let sig = b.divide(&one_b, &denom);
        let silu = b.multiply(&gate, &sig);
        let act = b.multiply(&silu, &up);
        let down = b.linear_seq(&act, &lw.down); // [Lp, H]
        x = b.add(&x, &down);
    }

    // --- tail: final norm, take the row at real_len-1, tied LM head ---
    let xf = rms_norm_seq(&mut b, &x, &a.final_norm, &k, lp, h); // [Lp, H]
    let one_i = b.const_i32(1);
    let last_idx = b.subtract(&a.real_len, &one_i); // real_len - 1
    let last_row = b.dynamic_slice(&xf, &[&last_idx, &k.c0], vec![1, h]); // [1, H]
    let last = b.reshape(&last_row, vec![h]); // [H]
    let logits = b.linear(&last, &a.embed); // [V]
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
