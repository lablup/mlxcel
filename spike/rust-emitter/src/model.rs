//! Emits the Llama-3.2-1B `decode_step` StableHLO module from Rust.
//!
//! Signature mirrors spike/openxla/model_jax.py `decode_step`:
//!   main(params..., token, pos, cache_len, kcache, vcache)
//!       -> (logits[V], kcache, vcache)
//! Weights are individual tensor inputs in the same order JAX emitted
//! (alphabetical within each layer), each carrying its pytree-path loc so the
//! arg-to-weight mapping is self-documenting and reuses the JAX weight glue.

use crate::builder::{Builder, Ty, Val};
use crate::config::Config;
use crate::rope;

const MAX_SEQ: usize = 256;

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

/// Emit the complete decode_step module text.
pub fn emit_decode(c: &Config) -> String {
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

    // --- tail: final norm + tied LM head ---
    let xf = rms_norm(&mut b, &x, &a.final_norm, &k, h);
    let logits = b.linear(&xf, &a.embed); // [V]

    let sig = render_signature(&decls);
    let cache_ty = Ty::f32(vec![c.n_layers, MAX_SEQ, c.n_kv, c.head_dim]).render();
    let logits_ty = Ty::f32(vec![c.vocab]).render();
    format!(
        "module @decode_step {{\n  func.func public @main({sig}) -> ({logits_ty}, {cache_ty}, {cache_ty}) {{\n{body}    return {l}, {kc}, {vc} : {logits_ty}, {cache_ty}, {cache_ty}\n  }}\n}}\n",
        sig = sig,
        logits_ty = logits_ty,
        cache_ty = cache_ty,
        body = b.body(),
        l = logits.name,
        kc = kcache.name,
        vc = vcache.name,
    )
}
