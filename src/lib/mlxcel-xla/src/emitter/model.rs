//! Emits the `decode_step` / `prefill` StableHLO modules for the supported dense
//! architectures from Rust. One per-layer core ([`emit_transformer_layer`], built
//! on the shared [`emit_attention`] of issue #494) serves every family: the norm
//! placement, q/k norm, sliding windows, and dual RoPE of issue #497, and the
//! LayerNorm / parallel block / interleaved-and-partial RoPE / dense MLP / scalar
//! multipliers of issue #498, are flags on that one path. Llama / Qwen2 / Gemma2
//! emit byte-for-byte the op sequence the graphs carried before.
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

use super::builder::{Builder, Precision, Ty, Val, precision_from_env, quant_in_graph};
use super::config::{Config, MropeLayout, NormStyle};
use super::moe::{self, MoeLayerW, MoeSharedW};
use super::rope;

/// Canonical finite additive-attention value for a disallowed prefill edge.
///
/// The embeddings entry accepts only `0.0` (allowed) and this value (masked).
/// Negative infinity is deliberately rejected so every caller, compiler target,
/// and softmax lowering observes the same finite input convention as the existing
/// token prefill graph.
pub(crate) const PREFILL_EMBEDDINGS_MASKED_VALUE: f32 = -1e30;

/// Runtime tensor element types understood by the prefill-embeddings schema
/// validator. The graph currently consumes f32 hidden states and an f32 additive
/// bias; the other variants exist so callers get an actionable error before IREE
/// compilation/invocation instead of silently reinterpreting bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PrefillEmbeddingsDType {
    F32,
    F16,
    Bf16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PositionInputMode {
    OneD,
    Mrope3D,
}

/// Shape/dtype metadata validated before compiling or invoking
/// `prefill_embeddings.main`.
///
/// Argument order after the model weights is stable and deliberately documented
/// here: `embeddings`, `positions`, `real_len`, `attention_bias`. The explicit
/// `position_mode` fixes positions as either `[L]` or M-RoPE `[3,L]`; rank is
/// never inferred from the payload after the runtime boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PrefillEmbeddingsInputMetadata {
    pub embeddings_shape: [usize; 2],
    pub embeddings_dtype: PrefillEmbeddingsDType,
    pub position_mode: PositionInputMode,
    /// The first `positions_rank` dimensions are significant.
    pub positions_shape: [usize; 2],
    pub positions_rank: usize,
    pub attention_bias_shape: [usize; 2],
    pub attention_bias_dtype: PrefillEmbeddingsDType,
    pub real_len: usize,
}

impl PrefillEmbeddingsInputMetadata {
    /// Metadata for the static context-capacity schema emitted for `c`.
    pub(crate) fn canonical(c: &Config) -> Self {
        let lp = c.context_capacity;
        Self {
            embeddings_shape: [lp, c.hidden],
            embeddings_dtype: PrefillEmbeddingsDType::F32,
            position_mode: if c.uses_mrope() {
                PositionInputMode::Mrope3D
            } else {
                PositionInputMode::OneD
            },
            positions_shape: if c.uses_mrope() { [3, lp] } else { [lp, 0] },
            positions_rank: if c.uses_mrope() { 2 } else { 1 },
            attention_bias_shape: [lp, lp],
            attention_bias_dtype: PrefillEmbeddingsDType::F32,
            real_len: lp,
        }
    }
}

/// Validate the static prefill-from-embeddings input contract before compiling or
/// invoking the graph.
///
/// `real_len` selects the output row but does not alter padded-row computation:
/// callers provide a complete `[Lp, Lp]` bias for the whole static bucket. For
/// token-parity padding, padded rows therefore retain the same causal pattern as
/// ordinary rows, while real rows mask every future/padded key.
pub(crate) fn validate_prefill_embeddings_metadata(
    c: &Config,
    metadata: &PrefillEmbeddingsInputMetadata,
) -> Result<(), String> {
    let lp = c.context_capacity;
    let want_embeddings = [lp, c.hidden];
    if metadata.embeddings_shape != want_embeddings {
        return Err(format!(
            "prefill embeddings shape {:?} does not match required {:?}",
            metadata.embeddings_shape, want_embeddings
        ));
    }
    if metadata.embeddings_dtype != PrefillEmbeddingsDType::F32 {
        return Err(format!(
            "prefill embeddings dtype {:?} is unsupported; expected F32",
            metadata.embeddings_dtype
        ));
    }
    let expected_position_mode = if c.uses_mrope() {
        PositionInputMode::Mrope3D
    } else {
        PositionInputMode::OneD
    };
    let expected_positions_shape = if c.uses_mrope() { [3, lp] } else { [lp, 0] };
    let expected_positions_rank = if c.uses_mrope() { 2 } else { 1 };
    if metadata.position_mode != expected_position_mode
        || metadata.positions_shape != expected_positions_shape
        || metadata.positions_rank != expected_positions_rank
    {
        return Err(format!(
            "prefill position mode/shape {:?} {:?} rank {} does not match required {:?} {:?} rank {}",
            metadata.position_mode,
            metadata.positions_shape,
            metadata.positions_rank,
            expected_position_mode,
            expected_positions_shape,
            expected_positions_rank,
        ));
    }
    if metadata.attention_bias_shape != [lp, lp] {
        return Err(format!(
            "prefill attention-bias shape {:?} does not match required [{}, {}]",
            metadata.attention_bias_shape, lp, lp
        ));
    }
    if metadata.attention_bias_dtype != PrefillEmbeddingsDType::F32 {
        return Err(format!(
            "prefill attention-bias dtype {:?} is unsupported; expected F32",
            metadata.attention_bias_dtype
        ));
    }
    if !(1..=lp).contains(&metadata.real_len) {
        return Err(format!(
            "prefill real_len {} is outside the required range 1..={lp}",
            metadata.real_len,
        ));
    }
    Ok(())
}

/// Validate a canonical additive attention-bias payload. Only `0.0` (allowed)
/// and [`PREFILL_EMBEDDINGS_MASKED_VALUE`] (masked) are accepted; NaN, infinity,
/// and intermediate additive values are rejected to keep polarity unambiguous.
pub(crate) fn validate_prefill_embeddings_attention_bias(
    c: &Config,
    bias: &[f32],
) -> Result<(), String> {
    let expected = c
        .context_capacity
        .checked_mul(c.context_capacity)
        .ok_or_else(|| {
            format!(
                "prefill attention-bias size overflows for context capacity {}",
                c.context_capacity
            )
        })?;
    if bias.len() != expected {
        return Err(format!(
            "prefill attention bias has {} elements; expected {expected}",
            bias.len()
        ));
    }
    if let Some((index, value)) = bias
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| *value != 0.0 && *value != PREFILL_EMBEDDINGS_MASKED_VALUE)
    {
        return Err(format!(
            "prefill attention bias at flat index {index} is {value}; expected only 0 or {}",
            PREFILL_EMBEDDINGS_MASKED_VALUE
        ));
    }
    Ok(())
}
/// Per-layer weight handles (JAX alphabetical order: down, gate, in_ln,
/// post_ln, up, wk, wo, wq, wv). `bk`/`bq`/`bv` are the q/k/v projection biases,
/// present only for an architecture with `qkv_bias` (Qwen2); `None` for Llama,
/// where the bias add emits no op so the graph is byte-identical to before.
///
/// The dense arch pack (issue #498) adds three optionalities: `gate` is `None`
/// for a dense (non-gated) MLP (StarCoder2, which has only `up`=c_fc and
/// `down`=c_proj); `post_ln` is `None` for a parallel-block arch (Cohere/Cohere2,
/// which has no `post_attention_layernorm`); and the `*_bias` handles carry the
/// LayerNorm affine biases (`in_ln_bias`/`post_ln_bias`, StableLM/StarCoder2), the
/// output-projection bias (`wo_bias`, StarCoder2), and the MLP biases
/// (`down_bias`/`gate_bias`/`up_bias`, StarCoder2). All are `None` for the Llama
/// family, so its graphs are byte-identical.
struct LayerW {
    /// MLP down projection. `None` on a MoE layer (issue #500), whose FFN uses `moe`
    /// instead; `Some` for every dense layer (all non-MoE models, and the leading
    /// dense layers of a MoE model), so the dense-MLP op sequence is unchanged.
    down: Option<Val>,
    gate: Option<Val>,
    /// `input_layernorm` (`None` for OLMo2/3, whose reordered post-norm has no
    /// input norm; the attention projects the raw residual instead).
    in_ln: Option<Val>,
    /// `post_attention_layernorm` (`None` for a parallel-block arch, which has no
    /// post-attention norm; Plain uses it as the pre-MLP norm, Gemma2/3 and OLMo2/3
    /// as the post-attn norm).
    post_ln: Option<Val>,
    /// MLP up projection. `None` on a MoE layer (issue #500); `Some` for every dense
    /// layer, so the dense-MLP op sequence is unchanged.
    up: Option<Val>,
    wk: Val,
    wo: Val,
    wq: Val,
    wv: Val,
    bk: Option<Val>,
    bq: Option<Val>,
    bv: Option<Val>,
    /// q/k norm weights (`None` unless the arch has `qk_norm`). Per-head families
    /// (Qwen3 / Gemma3) size them `[head_dim]`; flat families (OLMo2/3) size them
    /// `[n_q*head_dim]` / `[n_kv*head_dim]`.
    q_norm: Option<Val>,
    k_norm: Option<Val>,
    /// Gemma2/3 pre/post feed-forward norms and the OLMo2/3 post-feedforward norm
    /// (`None` for the plain families). Gemma2/3 wrap each sublayer: `post_ln` is
    /// the POST-attn norm, `pre_ff_ln` the pre-MLP norm, `post_ff_ln` the post-MLP
    /// norm. OLMo2/3 have `post_ln` (post-attn) and `post_ff_ln` (post-MLP) only.
    pre_ff_ln: Option<Val>,
    post_ff_ln: Option<Val>,
    /// #498 LayerNorm affine biases (`in_ln_bias`/`post_ln_bias`, StableLM /
    /// StarCoder2), the o_proj bias (`wo_bias`, StarCoder2), and the MLP biases
    /// (`down_bias`/`gate_bias`/`up_bias`, StarCoder2). `None` for every arch that
    /// carries no such bias, so the Llama family is byte-identical.
    in_ln_bias: Option<Val>,
    post_ln_bias: Option<Val>,
    wo_bias: Option<Val>,
    down_bias: Option<Val>,
    gate_bias: Option<Val>,
    up_bias: Option<Val>,
    /// MoE FFN weight handles (issue #500), `Some` on a MoE layer (router, stacked
    /// experts, optional shared expert); `None` on a dense layer.
    moe: Option<MoeLayerW>,
}

impl LayerW {
    /// The dense MLP weight `w`, present on every dense layer. Panics only if a MoE
    /// layer reaches a dense-MLP emit path, which the `is_moe_layer` branch guards
    /// against, so the dense paths are never taken for a MoE layer.
    fn dense(w: &Option<Val>) -> &Val {
        w.as_ref()
            .expect("dense MLP weight taken on a dense layer (MoE layers use `moe`)")
    }
}

struct Args {
    embed: Val,
    final_norm: Val,
    /// Final-norm affine bias (`Some` for a LayerNorm arch, #498).
    final_norm_bias: Option<Val>,
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

/// Take one linear-projection weight `[out, in_]`, returning an `[out, in_]` f32
/// handle the forward consumes uniformly. For an unquantized checkpoint (or with
/// the packed path off) this is a single f32 arg, byte-identical to before. For an
/// MLX affine-quantized checkpoint with `MLXCEL_XLA_QUANT=packed` (issue #516) it is
/// THREE args, the packed `[out, in_/(32/bits)]` `ui32` weight and its
/// `[out, in_/group_size]` f16 `scales` / `biases`, reconstructed to the `[out, in_]`
/// f32 weight IN THE GRAPH via [`Builder::dequant_affine`]. That reconstruction is
/// bit-identical to the host `dequantize_affine`, so the downstream forward (and its
/// optional f16 contraction demotion) is unchanged and stays token-exact. The three
/// args are declared packed, scales, biases, the order [`weight_specs`]
/// (`weights.rs`) mirrors so the loader's uploaded buffers line up with the graph.
fn take_weight(
    b: &mut Builder,
    decls: &mut Vec<ArgDecl>,
    idx: &mut usize,
    c: &Config,
    out: usize,
    in_: usize,
    loc: String,
) -> Val {
    match c.quantization {
        Some(qc) if quant_in_graph() && c.supports_packed_quant() => {
            let in_packed = in_ * qc.bits / 32;
            let n_groups = in_ / qc.group_size;
            let packed = take_arg(
                decls,
                idx,
                Ty::new(vec![out, in_packed], "ui32"),
                loc.clone(),
            );
            let scales = take_arg(
                decls,
                idx,
                Ty::new(vec![out, n_groups], "f16"),
                format!("{loc}.scales"),
            );
            let biases = take_arg(
                decls,
                idx,
                Ty::new(vec![out, n_groups], "f16"),
                format!("{loc}.biases"),
            );
            b.dequant_affine(&packed, &scales, &biases, qc.bits, qc.group_size)
        }
        // issue #572: on the f16 GPU path, declare the resident projection weight as
        // an f16 arg (uploaded f16-resident by the loader) instead of an f32 arg the
        // dot demotes in-graph. `dot_general` sees the same f16 operand either way
        // (its f32->f16 convert is skipped for an already-f16 weight), so this stays
        // token-exact while halving the weight's per-step DRAM read. f32 / bf16
        // precision and the fused layouts keep the f32 arg (byte-identical goldens).
        _ => {
            let elt = if b.precision() == Precision::F16 && c.supports_f16_resident() {
                "f16"
            } else {
                "f32"
            };
            take_arg(decls, idx, Ty::new(vec![out, in_], elt), loc)
        }
    }
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

/// Take the final-norm affine bias `params['final_norm_bias']` (`[H]`) for a
/// LayerNorm arch (`norm_bias`, StableLM/StarCoder2), or `None` for an RMSNorm
/// arch (which emits no such arg, keeping its graph byte-identical). Taken right
/// after `final_norm` and before `lm_head`, matching `weight_specs` in
/// `weights.rs` (invoked from `iree.rs`).
fn take_final_norm_bias(decls: &mut Vec<ArgDecl>, idx: &mut usize, c: &Config) -> Option<Val> {
    if c.norm_bias {
        Some(take_arg(
            decls,
            idx,
            Ty::f32(vec![c.hidden]),
            "params['final_norm_bias']".into(),
        ))
    } else {
        None
    }
}

/// Append layer `li`'s weights in the one canonical order every graph kind shares,
/// so the emitted arg order matches `weight_names` in `iree.rs` exactly. The base
/// order is the JAX-alphabetical down, gate, in_ln, post_ln, up, wk, wo, wq, wv;
/// then, conditionally, the k/q/v projection biases (`qkv_bias`, Qwen2), the q/k
/// norm weights (`qk_norm`, Qwen3 / Gemma3 / OLMo2/3), and the feed-forward norms.
/// `in_ln` is skipped for the OLMo post-norm style (no input norm). The new
/// conditional weights are inserted after the biases and before the FF norms, so a
/// config that has none of them (Llama / Qwen2 / Gemma2) is byte-identical.
fn take_layer_weights(
    b: &mut Builder,
    decls: &mut Vec<ArgDecl>,
    idx: &mut usize,
    c: &Config,
    li: usize,
) -> LayerW {
    let h = c.hidden;
    let inter = c.inter;
    let kv = c.n_kv * c.head_dim;
    let qd = c.n_q * c.head_dim;
    let p = |k: &str| format!("params['layers'][{}]['{}']", li, k);
    // A dense (non-gated) MLP has no gate projection (StarCoder2); a parallel-block
    // arch has no post-attention layernorm (Cohere/Cohere2). Both are `true` for the
    // Llama family, so its arg order is unchanged. A MoE layer (issue #500) takes no
    // dense MLP weights (down/gate/up) here; its expert bank is appended last.
    let gated = !c.dense_mlp;
    let has_post = !c.parallel_block;
    let moe_layer = c.is_moe_layer(li);
    // The big linear projections take the packed-quantized path when enabled
    // (issue #516: `take_weight` declares packed+scales+biases and dequants in the
    // graph); the norms below stay f32. Byte-identical when unquantized / packed off.
    let down = (!moe_layer).then(|| take_weight(b, decls, idx, c, h, inter, p("down")));
    let gate = (gated && !moe_layer).then(|| take_weight(b, decls, idx, c, inter, h, p("gate")));
    // input_layernorm: present unless the reordered (OLMo) post-norm drops it.
    let in_ln = c
        .has_input_norm()
        .then(|| take_arg(decls, idx, Ty::f32(vec![h]), p("in_ln")));
    let post_ln = has_post.then(|| take_arg(decls, idx, Ty::f32(vec![h]), p("post_ln")));
    let up = (!moe_layer).then(|| take_weight(b, decls, idx, c, inter, h, p("up")));
    let wk = take_weight(b, decls, idx, c, kv, h, p("wk"));
    // o_proj maps `[n_q*head_dim]` -> `[hidden]`, so its weight is `[h, qd]` (HF's
    // `[out, in]`). For Llama / Qwen2 `qd == h`, so this renders the same square
    // type as before (byte-identical); Gemma is genuinely non-square.
    let wo = take_weight(b, decls, idx, c, h, qd, p("wo"));
    let wq = take_weight(b, decls, idx, c, qd, h, p("wq"));
    let wv = take_weight(b, decls, idx, c, kv, h, p("wv"));
    let (bk, bq, bv) = if c.qkv_bias {
        let bk = take_arg(decls, idx, Ty::f32(vec![kv]), p("bk"));
        let bq = take_arg(decls, idx, Ty::f32(vec![qd]), p("bq"));
        let bv = take_arg(decls, idx, Ty::f32(vec![kv]), p("bv"));
        (Some(bk), Some(bq), Some(bv))
    } else {
        (None, None, None)
    };
    // q/k norm weights, after the biases. Per-head (Qwen3 / Gemma3) sizes them
    // `[head_dim]`; flat (OLMo2/3) sizes them `[n_q*head_dim]` / `[n_kv*head_dim]`.
    let (q_norm, k_norm) = match c.qk_norm {
        Some(qn) => {
            let (qsz, ksz) = if qn.per_head {
                (c.head_dim, c.head_dim)
            } else {
                (qd, kv)
            };
            let qn_w = take_arg(decls, idx, Ty::f32(vec![qsz]), p("q_norm"));
            let kn_w = take_arg(decls, idx, Ty::f32(vec![ksz]), p("k_norm"));
            (Some(qn_w), Some(kn_w))
        }
        None => (None, None),
    };
    // Feed-forward norms. Gemma2/3 add pre AND post; OLMo2/3 add post only.
    let pre_ff_ln = c
        .has_pre_ff_norm()
        .then(|| take_arg(decls, idx, Ty::f32(vec![h]), p("pre_ff_ln")));
    let post_ff_ln = c
        .has_post_ff_norm()
        .then(|| take_arg(decls, idx, Ty::f32(vec![h]), p("post_ff_ln")));
    // #498 optional handles, appended after the feed-forward norms in the canonical
    // order `weight_specs` (`weights.rs`) mirrors: the LayerNorm affine biases (in_ln
    // then, when present, post_ln), the o_proj bias, then the MLP biases (down, gate,
    // up). Each is absent for the Llama family, so nothing is appended and its args
    // are unchanged.
    let in_ln_bias = c
        .norm_bias
        .then(|| take_arg(decls, idx, Ty::f32(vec![h]), p("in_ln_bias")));
    let post_ln_bias = (c.norm_bias && has_post)
        .then(|| take_arg(decls, idx, Ty::f32(vec![h]), p("post_ln_bias")));
    let wo_bias = c
        .attn_o_bias
        .then(|| take_arg(decls, idx, Ty::f32(vec![h]), p("wo_bias")));
    let down_bias = c
        .mlp_bias
        .then(|| take_arg(decls, idx, Ty::f32(vec![h]), p("down_bias")));
    let gate_bias =
        (c.mlp_bias && gated).then(|| take_arg(decls, idx, Ty::f32(vec![inter]), p("gate_bias")));
    let up_bias = c
        .mlp_bias
        .then(|| take_arg(decls, idx, Ty::f32(vec![inter]), p("up_bias")));
    // MoE expert bank (issue #500): router, stacked experts, optional shared expert,
    // appended after the attention weights / biases / FF norms on a MoE layer, in the
    // same order `weight_specs` (`weights.rs`) lists them so the args line up.
    let moe = moe_layer.then(|| take_moe_weights(decls, idx, c, li));
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
        q_norm,
        k_norm,
        pre_ff_ln,
        post_ff_ln,
        in_ln_bias,
        post_ln_bias,
        wo_bias,
        down_bias,
        gate_bias,
        up_bias,
        moe,
    }
}

/// Append a MoE layer's expert-bank args (issue #500): the router `[E, H]`, the
/// three stacked expert projections `[E, I, H]` / `[E, I, H]` / `[E, H, I]`, and,
/// when the family has a shared expert, its `[Is, H]` / `[Is, H]` / `[H, Is]` SwiGLU
/// plus, for a gated shared expert (Qwen2-MoE), its `[1, H]` gate. The order mirrors
/// `weight_specs` (`weights.rs`) so the loaded buffers line up with the emitted args.
fn take_moe_weights(decls: &mut Vec<ArgDecl>, idx: &mut usize, c: &Config, li: usize) -> MoeLayerW {
    let m = c.moe.as_ref().expect("a MoE layer has a MoeConfig");
    let h = c.hidden;
    let e = m.n_experts;
    let i = m.intermediate;
    let p = |k: &str| format!("params['layers'][{}]['{}']", li, k);
    let router = take_arg(decls, idx, Ty::f32(vec![e, h]), p("moe_router"));
    let w_gate = take_arg(decls, idx, Ty::f32(vec![e, i, h]), p("moe_gate"));
    let w_up = take_arg(decls, idx, Ty::f32(vec![e, i, h]), p("moe_up"));
    let w_down = take_arg(decls, idx, Ty::f32(vec![e, h, i]), p("moe_down"));
    let shared = if let Some(sh) = m.shared {
        let is = sh.intermediate;
        let gate = take_arg(decls, idx, Ty::f32(vec![is, h]), p("moe_shared_gate"));
        let up = take_arg(decls, idx, Ty::f32(vec![is, h]), p("moe_shared_up"));
        let down = take_arg(decls, idx, Ty::f32(vec![h, is]), p("moe_shared_down"));
        let expert_gate = if sh.gated {
            Some(take_arg(
                decls,
                idx,
                Ty::f32(vec![1, h]),
                p("moe_shared_expert_gate"),
            ))
        } else {
            None
        };
        Some(MoeSharedW {
            gate,
            up,
            down,
            expert_gate,
        })
    } else {
        None
    };
    MoeLayerW {
        router,
        w_gate,
        w_up,
        w_down,
        shared,
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

fn build_arg_schema(b: &mut Builder, c: &Config) -> (Vec<ArgDecl>, Args) {
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
    let final_norm_bias = take_final_norm_bias(&mut decls, &mut idx, c);
    let lm_head = take_lm_head(&mut decls, &mut idx, c);

    let mut layers = Vec::with_capacity(c.n_layers);
    for li in 0..c.n_layers {
        layers.push(take_layer_weights(b, &mut decls, &mut idx, c, li));
    }

    let token = take_arg(&mut decls, &mut idx, Ty::scalar("i32"), "token".into());
    let pos = take_arg(
        &mut decls,
        &mut idx,
        Ty::new(if c.uses_mrope() { vec![3] } else { vec![] }, "i32"),
        "pos".into(),
    );
    let cache_len = take_arg(&mut decls, &mut idx, Ty::scalar("i32"), "cache_len".into());
    let kcache = take_arg(
        &mut decls,
        &mut idx,
        Ty::f32(vec![c.n_layers, c.context_capacity, c.n_kv, c.head_dim]),
        "kcache".into(),
    );
    let vcache = take_arg(
        &mut decls,
        &mut idx,
        Ty::f32(vec![c.n_layers, c.context_capacity, c.n_kv, c.head_dim]),
        "vcache".into(),
    );

    (
        decls,
        Args {
            embed,
            final_norm,
            final_norm_bias,
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

/// Shared scalar/table constants, emitted once at the top of the body. Crate-
/// visible so the sibling `moe` module (issue #500) reads the shared scalars
/// (`zero`, `one`, `neg_inf`) it needs for the router softmax / top-k masking.
pub(crate) struct Consts {
    cos_table: Val,
    sin_table: Val,
    /// Gemma3 / OLMo3 local RoPE tables (`Some` only when the config has a local
    /// base), used by the sliding (local) layers; the global layers keep the
    /// `cos_table` / `sin_table`. Emitted only when present, so single-RoPE
    /// families are byte-identical.
    cos_local: Option<Val>,
    sin_local: Option<Val>,
    pub(crate) zero: Val,
    pub(crate) one: Val,
    pub(crate) neg_inf: Val,
    neg_big: Val,
    eps: Val,
    hidden_f: Val,
    scale: Val,
    c0: Val,
    layer_idx: Vec<Val>,
}

fn emit_consts(b: &mut Builder, c: &Config) -> Consts {
    let seq = c.context_capacity;
    let (cos, sin) = rope::rope_tables(c, seq);
    // The rotary width (`head_dim`, or the smaller `rotary_dim` for a partial-RoPE
    // arch like StableLM). The Llama family has `rot == head_dim`, so its tables are
    // byte-identical.
    let rot = c.rotary_width();
    let cos_table = b.const_tensor_f32(&cos, vec![seq, rot]);
    let sin_table = b.const_tensor_f32(&sin, vec![seq, rot]);
    // Gemma3 / OLMo3 local RoPE tables (distinct base for the sliding layers).
    let (cos_local, sin_local) = match c.rope_local_base {
        Some(base) => {
            let (cl, sl) = rope::rope_tables_local(c, seq, base);
            (
                Some(b.const_tensor_f32(&cl, vec![seq, rot])),
                Some(b.const_tensor_f32(&sl, vec![seq, rot])),
            )
        }
        None => (None, None),
    };
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
        cos_local,
        sin_local,
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

/// Gemma `(1 + weight)` norm scale (`weight + 1` over a `[dim]` feature axis).
/// Gemma stores the RMSNorm weight offset by one, so the Gemma paths pass
/// `gemma_norm_w(...)` where the other families pass the raw weight.
fn gemma_norm_w(b: &mut Builder, w: &Val, k: &Consts, dim: usize) -> Val {
    let one = b.broadcast(&k.one, &[], vec![dim]);
    b.add(w, &one)
}

/// The RMSNorm weight to feed `rms_norm`: `1 + w` for the Gemma family, the raw
/// `w` otherwise. A `Val` clone is just a handle copy (no emitted op), so the
/// non-Gemma graphs are unchanged.
fn norm_w(b: &mut Builder, w: &Val, c: &Config, k: &Consts, hidden: usize) -> Val {
    if c.norm_one_plus {
        gemma_norm_w(b, w, k, hidden)
    } else {
        w.clone()
    }
}

/// Single-token (`[hidden]`) normalization: RMSNorm (the Llama family, byte-exact
/// `rms_norm`) or, for a LayerNorm arch (`c.layernorm`, issue #498), mean-subtract
/// LayerNorm `w * (x - mean) * rsqrt(var + eps)` with an optional affine `bias`.
/// The RMSNorm branch never carries a bias, so it delegates to `rms_norm` and is
/// unchanged.
fn normalize(
    b: &mut Builder,
    c: &Config,
    k: &Consts,
    x: &Val,
    w: &Val,
    bias: Option<&Val>,
    hidden: usize,
) -> Val {
    if !c.layernorm {
        return rms_norm(b, x, w, k, hidden);
    }
    let sum = b.reduce_add(x, 0, &k.zero); // scalar
    let mean = b.divide(&sum, &k.hidden_f);
    let mean_b = b.broadcast(&mean, &[], vec![hidden]);
    let xc = b.subtract(x, &mean_b);
    let sq = b.multiply(&xc, &xc);
    let ssum = b.reduce_add(&sq, 0, &k.zero);
    let var = b.divide(&ssum, &k.hidden_f);
    let veps = b.add(&var, &k.eps);
    let r = b.rsqrt(&veps);
    let rb = b.broadcast(&r, &[], vec![hidden]);
    let xn = b.multiply(&xc, &rb);
    let out = b.multiply(&xn, w);
    match bias {
        Some(bb) => b.add(&out, bb),
        None => out,
    }
}

/// Per-row (`[n, hidden]`) normalization: RMSNorm (byte-exact `rms_norm_seq`) or
/// mean-subtract LayerNorm with an optional affine `bias` (issue #498).
#[allow(clippy::too_many_arguments)]
fn normalize_seq(
    b: &mut Builder,
    c: &Config,
    k: &Consts,
    x: &Val,
    w: &Val,
    bias: Option<&Val>,
    n: usize,
    hidden: usize,
) -> Val {
    if !c.layernorm {
        return rms_norm_seq(b, x, w, k, n, hidden);
    }
    let sum = b.reduce_add(x, 1, &k.zero); // [n]
    let hb = b.broadcast(&k.hidden_f, &[], vec![n]);
    let mean = b.divide(&sum, &hb);
    let mean_b = b.broadcast(&mean, &[0], vec![n, hidden]);
    let xc = b.subtract(x, &mean_b);
    let sq = b.multiply(&xc, &xc);
    let ssum = b.reduce_add(&sq, 1, &k.zero); // [n]
    let var = b.divide(&ssum, &hb);
    let epsb = b.broadcast(&k.eps, &[], vec![n]);
    let veps = b.add(&var, &epsb);
    let r = b.rsqrt(&veps); // [n]
    let rb = b.broadcast(&r, &[0], vec![n, hidden]);
    let xn = b.multiply(&xc, &rb);
    let wb = b.broadcast(w, &[1], vec![n, hidden]);
    let out = b.multiply(&xn, &wb);
    match bias {
        Some(bb) => {
            let bbc = b.broadcast(bb, &[1], vec![n, hidden]);
            b.add(&out, &bbc)
        }
        None => out,
    }
}

/// RMSNorm over the LAST axis of `x` (any rank), with `w` broadcast over that axis
/// and the optional Gemma `(1+w)` offset. The one helper serves both q/k-norm
/// flavors: per-head norm passes the head-shaped tensor `[.., heads, d]` (reduce
/// over `d`, weight `[d]`), while flat norm passes the folded `[.., heads*d]`
/// (reduce over `heads*d`, weight `[heads*d]`). Emitted only on a `qk_norm` arch,
/// so every existing graph is unchanged.
fn last_axis_rms_norm(b: &mut Builder, x: &Val, w: &Val, k: &Consts, one_plus: bool) -> Val {
    let shape = x.ty.shape.clone();
    let last = shape.len() - 1;
    let d = shape[last];
    let lead: Vec<usize> = (0..last).collect();
    let df = b.const_f32(d as f32);
    let sq = b.multiply(x, x);
    let ssum = b.reduce_add(&sq, last, &k.zero); // [..lead..]
    let red_shape = ssum.ty.shape.clone();
    let dfb = b.broadcast(&df, &[], red_shape.clone());
    let mean = b.divide(&ssum, &dfb);
    let epsb = b.broadcast(&k.eps, &[], red_shape);
    let meps = b.add(&mean, &epsb);
    let r = b.rsqrt(&meps);
    let rb = b.broadcast(&r, &lead, shape.clone());
    let xr = b.multiply(x, &rb);
    let wv = if one_plus {
        gemma_norm_w(b, w, k, d)
    } else {
        w.clone()
    };
    let wb = b.broadcast(&wv, &[last], shape);
    b.multiply(&xr, &wb)
}

/// Fold the last two axes of a head-shaped tensor `[.., heads, d]` into
/// `[.., heads*d]` (the flat q/k-norm feature layout).
fn fold_heads(b: &mut Builder, x: &Val) -> Val {
    let mut shape = x.ty.shape.clone();
    let d = shape.pop().expect("head dim");
    let heads = shape.pop().expect("head count");
    shape.push(heads * d);
    b.reshape(x, shape)
}

/// Apply the reserved q/k normalization (issue #494 hook), if the arch has one, to
/// the projected q / k before RoPE. Per-head (Qwen3 / Gemma3) norms each head over
/// `head_dim`; flat (OLMo2/3) folds the heads into the feature axis, norms the
/// whole `[.., heads*d]`, and unfolds. `q` is `[.., n_q, d]` and `kk` is
/// `[.., n_kv, d]` (single decode has no leading axis; the seq paths carry `[N,
/// ...]`). No-op (and no emitted op) for a config without `qk_norm`.
fn apply_qk_norm(
    b: &mut Builder,
    c: &Config,
    k: &Consts,
    lw: &LayerW,
    q: Val,
    kk: Val,
) -> (Val, Val) {
    let Some(qn) = c.qk_norm else {
        return (q, kk);
    };
    let qw = lw
        .q_norm
        .as_ref()
        .expect("qk_norm arch has a q_norm weight");
    let kw = lw
        .k_norm
        .as_ref()
        .expect("qk_norm arch has a k_norm weight");
    if qn.per_head {
        let q = last_axis_rms_norm(b, &q, qw, k, qn.one_plus);
        let kk = last_axis_rms_norm(b, &kk, kw, k, qn.one_plus);
        (q, kk)
    } else {
        // Flat: fold heads into the feature axis, norm, unfold back to head layout.
        let q_shape = q.ty.shape.clone();
        let k_shape = kk.ty.shape.clone();
        let q_folded = fold_heads(b, &q);
        let q_normed = last_axis_rms_norm(b, &q_folded, qw, k, qn.one_plus);
        let q = b.reshape(&q_normed, q_shape);
        let k_folded = fold_heads(b, &kk);
        let k_normed = last_axis_rms_norm(b, &k_folded, kw, k, qn.one_plus);
        let kk = b.reshape(&k_normed, k_shape);
        (q, kk)
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

/// `silu(x) = x * sigmoid(x)`, `sigmoid(z) = 1/(1+exp(-z))`, at any rank (the
/// `one` broadcast uses `x`'s own shape, so this emits the exact op sequence the
/// single-token and seq MLP paths inlined). Byte-identical to those inlines.
fn silu(b: &mut Builder, k: &Consts, x: &Val) -> Val {
    let neg = b.negate(x);
    let ex = b.exponential(&neg);
    let one_b = b.broadcast(&k.one, &[], x.ty.shape.clone());
    let denom = b.add(&one_b, &ex);
    let sig = b.divide(&one_b, &denom);
    b.multiply(x, &sig)
}

/// Scale a sublayer output by the arch's per-residual multiplier before its
/// residual add (Granite `residual_multiplier`, MiniCPM `scale_depth/sqrt(L)`,
/// issue #498). A no-op (and no emitted op) for the Llama family (`None`), so its
/// graphs are unchanged. Shape-agnostic (broadcasts to the output's own shape).
fn scale_residual(b: &mut Builder, c: &Config, x: Val) -> Val {
    match c.residual_multiplier {
        Some(m) => {
            let mc = b.const_f32(m);
            let mb = b.broadcast(&mc, &[], x.ty.shape.clone());
            b.multiply(&x, &mb)
        }
        None => x,
    }
}

/// Apply the arch's final-logit scaling to `[V]` (single/prefill) or `[B, V]`
/// (ragged) logits: the Gemma2 soft-cap, then a Cohere `logit_scale` multiply,
/// then a Granite / MiniCPM divide. Each is per-arch exclusive and `None` for the
/// Llama family, so its logits are unchanged.
fn apply_logit_scale(b: &mut Builder, c: &Config, logits: Val) -> Val {
    let logits = match c.final_logit_softcap {
        Some(cap) => softcap(b, &logits, cap),
        None => logits,
    };
    let logits = match c.logit_mul {
        Some(m) => {
            let mc = b.const_f32(m);
            let mb = b.broadcast(&mc, &[], logits.ty.shape.clone());
            b.multiply(&logits, &mb)
        }
        None => logits,
    };
    match c.logit_div {
        Some(d) => {
            let dc = b.const_f32(d);
            let db = b.broadcast(&dc, &[], logits.ty.shape.clone());
            b.divide(&logits, &db)
        }
        None => logits,
    }
}

/// Scale the gathered input embeddings: the Gemma family by `sqrt(hidden)`
/// (`embed_scale`), and Granite / MiniCPM by `embedding_multiplier` / `scale_emb`
/// (issue #498). `shape` is the activation shape at the graph kind's rank (`[H]`
/// single, `[N, H]` seq). Both are no-ops (no emitted op) for the Llama family, so
/// its graphs are unchanged.
fn scale_embedding(b: &mut Builder, c: &Config, x: Val, shape: Vec<usize>) -> Val {
    let mut x = x;
    if c.embed_scale {
        let norm = b.const_f32(c.embed_normalizer());
        let nb = b.broadcast(&norm, &[], shape.clone());
        x = b.multiply(&x, &nb);
    }
    if let Some(em) = c.embedding_multiplier {
        let emc = b.const_f32(em);
        let emb = b.broadcast(&emc, &[], shape);
        x = b.multiply(&x, &emb);
    }
    x
}

/// The per-layer MLP compute given a pre-normed input `hn`, returning the MLP
/// output (the `down` projection) WITHOUT the residual add. Handles the gated
/// SwiGLU (Llama family: `down(silu(gate(hn)) * up(hn))`), the Gemma GeGLU
/// (`gelu_tanh`, driven by `mlp_geglu`) with the Gemma2/3 and OLMo2/3
/// `post_feedforward_layernorm` (`has_post_ff_norm`), and the dense (non-gated)
/// StarCoder2 MLP (`c_proj(gelu_tanh(c_fc(hn)))`), plus the optional MLP biases
/// (issue #498). `layout` supplies the rank (single vs seq). For the Llama family
/// this emits the exact `gate, up, silu, mul, down` op sequence the paths inlined,
/// byte-identical.
fn emit_mlp_body(
    b: &mut Builder,
    c: &Config,
    k: &Consts,
    layout: &AttnLayout,
    hn: &Val,
    lw: &LayerW,
) -> Val {
    if c.dense_mlp {
        // Dense (StarCoder2): up == c_fc, then gelu, then down == c_proj. No gate.
        let up = layout.linear(b, hn, LayerW::dense(&lw.up));
        let up = layout.add_bias(b, up, lw.up_bias.as_ref());
        let act = gelu_tanh(b, &up);
        let down = layout.linear(b, &act, LayerW::dense(&lw.down));
        return layout.add_bias(b, down, lw.down_bias.as_ref());
    }
    let gate = layout.linear(b, hn, lw.gate.as_ref().expect("gated MLP has a gate"));
    let gate = layout.add_bias(b, gate, lw.gate_bias.as_ref());
    let up = layout.linear(b, hn, LayerW::dense(&lw.up));
    let up = layout.add_bias(b, up, lw.up_bias.as_ref());
    let act = if c.mlp_geglu {
        gelu_tanh(b, &gate)
    } else {
        silu(b, k, &gate)
    };
    let act = b.multiply(&act, &up);
    let down = layout.linear(b, &act, LayerW::dense(&lw.down));
    let down = layout.add_bias(b, down, lw.down_bias.as_ref());
    if c.has_post_ff_norm() {
        let w = norm_w(
            b,
            lw.post_ff_ln.as_ref().expect("post_ff_ln"),
            c,
            k,
            c.hidden,
        );
        layout.norm(b, c, k, &down, &w, None)
    } else {
        down
    }
}

/// The per-layer FFN body over the pre-normed hidden `hn`, returning the FFN output
/// WITHOUT the residual add (the transformer layer owns the residual): the MoE FFN
/// block (issue #500, `moe_block` over the routed + optional shared experts) on a
/// MoE layer, else the dense SwiGLU / GeGLU / dense-MLP body ([`emit_mlp_body`]). On
/// a dense layer it forwards straight to `emit_mlp_body`, so every dense graph stays
/// byte-for-byte identical. The MoE primitive is seq-only (`[N, H]`); the single-
/// token decode rank is `[H]`, so this reshapes `[H]` to `[1, H]` and back there.
#[allow(clippy::too_many_arguments)]
fn emit_ffn_body(
    b: &mut Builder,
    c: &Config,
    k: &Consts,
    layout: &AttnLayout,
    hn: &Val,
    lw: &LayerW,
    li: usize,
) -> Val {
    if !c.is_moe_layer(li) {
        return emit_mlp_body(b, c, k, layout, hn, lw);
    }
    let m = c.moe.as_ref().expect("a MoE layer has a MoeConfig");
    let mw = lw.moe.as_ref().expect("a MoE layer has expert weights");
    let h = c.hidden;
    match layout {
        AttnLayout::Single { .. } => {
            let hn1 = b.reshape(hn, vec![1, h]);
            let out = moe::moe_block(b, c, m, mw, k, &hn1, 1);
            b.reshape(&out, vec![h])
        }
        AttnLayout::Ragged { bsz, .. } => moe::moe_block(b, c, m, mw, k, hn, *bsz),
        AttnLayout::Prefill { lp, .. } => moe::moe_block(b, c, m, mw, k, hn, *lp),
    }
}

/// The sequential MLP's pre-norm at the layout's rank: the RAW residual for the
/// OLMo reordered post-norm (whose MLP consumes the residual and norms its OUTPUT),
/// the `pre_feedforward_layernorm` for Gemma2/3, else the `post_attention_layernorm`
/// (with its LayerNorm bias for a LayerNorm arch). Only the sequential block calls
/// this; a parallel-block arch reuses the shared `input_layernorm` output instead.
/// Byte-identical to the inlined pre-MLP norm for the Llama family.
fn mlp_pre_norm(
    b: &mut Builder,
    c: &Config,
    k: &Consts,
    layout: &AttnLayout,
    x: &Val,
    lw: &LayerW,
) -> Val {
    match c.norm_style {
        NormStyle::OlmoPost => x.clone(),
        NormStyle::GemmaFf => arch_norm(b, c, k, layout, x, &lw.pre_ff_ln, None),
        NormStyle::Plain => arch_norm(b, c, k, layout, x, &lw.post_ln, lw.post_ln_bias.as_ref()),
    }
}

/// Select the RoPE (cos, sin) tensors for a layer: the local-base pair on a local
/// (sliding) layer of a dual-RoPE config, else the global pair. `local` is
/// [`Config::local_rope_layer`]; a single-RoPE config carries `None` locals, so
/// this returns the global pair and the emit is byte-identical.
fn pick_rope<'a>(
    local: bool,
    cos: &'a Val,
    sin: &'a Val,
    cos_local: &'a Option<Val>,
    sin_local: &'a Option<Val>,
) -> (&'a Val, &'a Val) {
    if local {
        (
            cos_local.as_ref().unwrap_or(cos),
            sin_local.as_ref().unwrap_or(sin),
        )
    } else {
        (cos, sin)
    }
}

/// HF RoPE on x:[heads, d]; cos/sin are [rd] for the position (`rd` = rotary
/// width). Full RoPE (`rd == d`) rotates the whole head; partial RoPE (StableLM,
/// `rd < d`) rotates only the first `rd` of each head and passes the rest through.
/// The rotation convention (half-split / interleaved) comes from `c` (issue #498).
/// For a full-rope, half-split arch (Llama family) this emits the exact op sequence
/// the path inlined, byte-identical.
fn apply_rope(b: &mut Builder, c: &Config, x: &Val, cos: &Val, sin: &Val, heads: usize) -> Val {
    let d = c.head_dim;
    let rd = c.rotary_width();
    if rd == d {
        return rotate(b, c, x, cos, sin, heads, rd);
    }
    let x_rot = b.slice(x, &[(0, heads), (0, rd)]);
    let x_pass = b.slice(x, &[(0, heads), (rd, d)]);
    let rotated = rotate(b, c, &x_rot, cos, sin, heads, rd);
    b.concatenate(&rotated, &x_pass, 1)
}

/// The core RoPE rotation on x:[heads, rd] with cos/sin [rd]. Half-split
/// (`concat(-x2, x1)`, Llama) or interleaved (per adjacent pair `(2i, 2i+1)`:
/// `[-x_odd, x_even]`, Cohere). Byte-identical to the inlined half-split for the
/// Llama family.
fn rotate(
    b: &mut Builder,
    c: &Config,
    x: &Val,
    cos: &Val,
    sin: &Val,
    heads: usize,
    rd: usize,
) -> Val {
    let cos_b = b.broadcast(cos, &[1], vec![heads, rd]);
    let sin_b = b.broadcast(sin, &[1], vec![heads, rd]);
    let xc = b.multiply(x, &cos_b);
    let rh = if c.rope_interleaved {
        let xr = b.reshape(x, vec![heads, rd / 2, 2]);
        let even = b.slice(&xr, &[(0, heads), (0, rd / 2), (0, 1)]);
        let odd = b.slice(&xr, &[(0, heads), (0, rd / 2), (1, 2)]);
        let neg_odd = b.negate(&odd);
        let st = b.concatenate(&neg_odd, &even, 2);
        b.reshape(&st, vec![heads, rd])
    } else {
        let half = rd / 2;
        let x1 = b.slice(x, &[(0, heads), (0, half)]);
        let x2 = b.slice(x, &[(0, heads), (half, rd)]);
        let nx2 = b.negate(&x2);
        b.concatenate(&nx2, &x1, 1)
    };
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
        /// Gemma3 / OLMo3 local-base RoPE vectors for the sliding layers (`Some`
        /// only for a dual-RoPE config); the global layers use `cos` / `sin`.
        cos_local: Option<Val>,
        sin_local: Option<Val>,
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
        cos_local: Option<Val>,
        sin_local: Option<Val>,
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
        cos_local: Option<Val>,
        sin_local: Option<Val>,
        mask: Val,
        mask_local: Option<Val>,
    },
}

impl AttnLayout {
    /// Normalization at this kind's activation rank (rank-reduced for single
    /// decode, per-row otherwise): RMSNorm for the Llama family, mean-subtract
    /// LayerNorm with the optional `bias` for a LayerNorm arch (issue #498).
    fn norm(
        &self,
        b: &mut Builder,
        c: &Config,
        k: &Consts,
        x: &Val,
        w: &Val,
        bias: Option<&Val>,
    ) -> Val {
        match self {
            AttnLayout::Single { .. } => normalize(b, c, k, x, w, bias, c.hidden),
            AttnLayout::Ragged { bsz, .. } => normalize_seq(b, c, k, x, w, bias, *bsz, c.hidden),
            AttnLayout::Prefill { lp, .. } => normalize_seq(b, c, k, x, w, bias, *lp, c.hidden),
        }
    }

    /// A rank-aware linear `x @ W^T`: `[K] -> [N]` for single decode, `[L, K] ->
    /// [L, N]` for the seq kinds. Lets the shared MLP body emit at any rank.
    fn linear(&self, b: &mut Builder, x: &Val, w: &Val) -> Val {
        match self {
            AttnLayout::Single { .. } => b.linear(x, w),
            _ => b.linear_seq(x, w),
        }
    }

    /// Add an optional projection/MLP bias at this kind's rank: a plain `[K]` add
    /// for single decode, a `[K] -> [N, K]` broadcast-add for the seq kinds. A
    /// no-op (no emitted op) when the bias is absent, keeping the Llama family
    /// byte-identical.
    fn add_bias(&self, b: &mut Builder, x: Val, bias: Option<&Val>) -> Val {
        match bias {
            None => x,
            Some(bb) => match self {
                AttnLayout::Single { .. } => b.add(&x, bb),
                _ => {
                    let n = x.ty.shape[0];
                    let kdim = x.ty.shape[1];
                    let bbc = b.broadcast(bb, &[1], vec![n, kdim]);
                    b.add(&x, &bbc)
                }
            },
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

    /// Apply this kind's RoPE to q and k for layer `li` (v is never rotated). A
    /// dual-RoPE config (Gemma3 / OLMo3) selects the local-base table on a sliding
    /// layer and the global table on a full layer; single-RoPE configs always use
    /// the global table (byte-identical to before). `li` is unused for the latter.
    fn rope_qk(&self, b: &mut Builder, c: &Config, li: usize, q: &Val, kk: &Val) -> (Val, Val) {
        let (nq, nkv) = (c.n_q, c.n_kv);
        let local = c.local_rope_layer(li);
        match self {
            AttnLayout::Single {
                cos,
                sin,
                cos_local,
                sin_local,
                ..
            } => {
                let (cos, sin) = pick_rope(local, cos, sin, cos_local, sin_local);
                let q = apply_rope(b, c, q, cos, sin, nq);
                let kk = apply_rope(b, c, kk, cos, sin, nkv);
                (q, kk)
            }
            AttnLayout::Ragged {
                bsz,
                cos,
                sin,
                cos_local,
                sin_local,
                ..
            } => {
                let (cos, sin) = pick_rope(local, cos, sin, cos_local, sin_local);
                let q = apply_rope_seq(b, c, q, cos, sin, *bsz, nq);
                let kk = apply_rope_seq(b, c, kk, cos, sin, *bsz, nkv);
                (q, kk)
            }
            AttnLayout::Prefill {
                lp,
                cos,
                sin,
                cos_local,
                sin_local,
                ..
            } => {
                let (cos, sin) = pick_rope(local, cos, sin, cos_local, sin_local);
                let q = apply_rope_seq(b, c, q, cos, sin, *lp, nq);
                let kk = apply_rope_seq(b, c, kk, cos, sin, *lp, nkv);
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
        let seq = c.context_capacity;
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
                let kl = b.slice(&*kcache, &[(li, li + 1), (0, seq), (0, nkv), (0, d)]);
                let kl = b.reshape(&kl, vec![seq, nkv, d]);
                let vl = b.slice(&*vcache, &[(li, li + 1), (0, seq), (0, nkv), (0, d)]);
                let vl = b.reshape(&vl, vec![seq, nkv, d]);
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
                    &[(0, *bsz), (li, li + 1), (0, seq), (0, nkv), (0, d)],
                );
                let kl = b.reshape(&kl, vec![*bsz, seq, nkv, d]);
                let vl = b.slice(
                    &*vcache,
                    &[(0, *bsz), (li, li + 1), (0, seq), (0, nkv), (0, d)],
                );
                let vl = b.reshape(&vl, vec![*bsz, seq, nkv, d]);
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
        let seq = c.context_capacity;
        match self {
            AttnLayout::Single { .. } => {
                let q_r = b.reshape(q, vec![nkv, g, d]);
                let scores = b.dot_general(&q_r, kslab, &[0], &[1], &[2], &[2], vec![nkv, g, seq]);
                b.reshape(&scores, vec![nq, seq])
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
                    vec![*bsz, nkv, g, seq],
                );
                b.reshape(&scores, vec![*bsz, nq, seq])
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
        let seq = c.context_capacity;
        match self {
            AttnLayout::Single {
                mask, mask_local, ..
            } => {
                let m = layer_mask(mask, mask_local, c, li);
                let mb = b.broadcast(m, &[1], vec![nq, seq]);
                b.add(scores, &mb)
            }
            AttnLayout::Ragged {
                bsz,
                mask,
                mask_local,
                ..
            } => {
                let m = layer_mask(mask, mask_local, c, li);
                let mb = b.broadcast(m, &[0, 2], vec![*bsz, nq, seq]);
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
        let seq = c.context_capacity;
        match self {
            AttnLayout::Single { .. } => {
                let attn_r = b.reshape(attn, vec![nkv, g, seq]);
                let o = b.dot_general(&attn_r, vslab, &[0], &[1], &[2], &[0], vec![nkv, g, d]);
                let o = b.reshape(&o, vec![nq, d]);
                b.reshape(&o, vec![nq * d])
            }
            AttnLayout::Ragged { bsz, .. } => {
                let attn_r = b.reshape(attn, vec![*bsz, nkv, g, seq]);
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

    /// The output projection at this kind's activation rank, plus the optional
    /// `o_proj` bias (StarCoder2 `use_bias`, issue #498; a no-op for the Llama
    /// family, byte-identical).
    fn o_proj(&self, b: &mut Builder, o: &Val, lw: &LayerW) -> Val {
        let out = match self {
            AttnLayout::Single { .. } => b.linear(o, &lw.wo),
            _ => b.linear_seq(o, &lw.wo),
        };
        self.add_bias(b, out, lw.wo_bias.as_ref())
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
/// prefill (`axis 3`) score ranks identically. Crate-visible: the `moe` module
/// (issue #500) reuses it for the router softmax over the expert axis.
pub(crate) fn attn_softmax(b: &mut Builder, k: &Consts, scores: &Val, axis: usize) -> Val {
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

/// The input RMSNorm applied at a layout's rank: the Gemma `(1 + w)` weight offset
/// (a no-op handle-copy for the non-Gemma families) followed by the layout's
/// rank-appropriate RMSNorm. The OLMo reordered post-norm has NO input norm
/// (`w_raw` is `None`), so the attention projects the raw residual unchanged.
fn arch_norm(
    b: &mut Builder,
    c: &Config,
    k: &Consts,
    layout: &AttnLayout,
    x: &Val,
    w_raw: &Option<Val>,
    bias: Option<&Val>,
) -> Val {
    match w_raw {
        Some(w_raw) => {
            let w = norm_w(b, w_raw, c, k, c.hidden);
            layout.norm(b, c, k, x, &w, bias)
        }
        None => x.clone(),
    }
}

/// The post-attention RMSNorm on the attention output before the residual, for the
/// families that have one (`post_attention_layernorm` in Gemma2/3 and OLMo2/3),
/// applied at the layout's rank. A no-op (handle copy) for the plain pre-norm
/// families (Llama / Qwen2/3 / Gemma1 / SmolLM3), which have no such norm.
fn post_attn_norm(
    b: &mut Builder,
    c: &Config,
    k: &Consts,
    layout: &AttnLayout,
    attn_out: Val,
    lw: &LayerW,
) -> Val {
    if c.has_post_attn_norm() {
        let post_ln = lw
            .post_ln
            .as_ref()
            .expect("post-attn norm arch has post_ln");
        let w = norm_w(b, post_ln, c, k, c.hidden);
        layout.norm(b, c, k, &attn_out, &w, None)
    } else {
        attn_out
    }
}

/// Emit one layer's attention block for graph kind `layout`, returning both the
/// shared normed input `hn` (`input_layernorm(x)`, which a parallel-block arch
/// feeds to the MLP too) and `attn_out` (the o_proj output after any post-attn
/// norm), WITHOUT the residual add: the caller ([`emit_transformer_layer`]) owns
/// the residual, so it can combine the attention and MLP outputs sequentially or
/// in parallel. The op sequence (input norm + bias, q/k/v projection + bias, q/k
/// norm, RoPE, KV write/read, GQA scores, scale, soft-cap, mask, softmax, context,
/// o_proj + bias, post-attn norm) is the architecture surface a new dense family
/// customizes once; `layout` supplies the per-graph-kind ranks, cache indexing, and
/// dot shapes. For the Llama family this emits the exact op sequence the paths
/// inlined (the caller's `add(x, attn_out)` lands at the same point), byte-identical.
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
) -> (Val, Val) {
    let hn = arch_norm(b, c, k, layout, x, &lw.in_ln, lw.in_ln_bias.as_ref());
    let (q, kk, vv) = layout.project_qkv(b, c, &hn, lw);
    // q/k-norm hook (issue #494): Qwen3 / Gemma3 norm each head over head_dim,
    // OLMo2/3 norm the whole flat projection; both before RoPE, applied once here so
    // they reach the single / ragged / prefill paths together. A config without
    // `qk_norm` emits nothing, so every existing graph is unchanged.
    let (q, kk) = apply_qk_norm(b, c, k, lw, q, kk);
    // RoPE, unless this layer skips it: a SmolLM3 NoPE-masked layer, or a Cohere2
    // full-attention (rope_on_sliding_only) layer. On the layer's own RoPE base
    // (dual-RoPE local/global for Gemma3 / OLMo3), with the arch's rotation
    // convention and width (interleaved / partial for Cohere / StableLM).
    let (q, kk) = if c.layer_uses_rope(li) {
        layout.rope_qk(b, c, li, &q, &kk)
    } else {
        (q, kk)
    };
    let (kslab, vslab) = layout.write_read_kv(b, k, c, li, &kk, &vv, kcache, vcache);
    let scores = layout.raw_scores(b, c, &q, &kslab);
    let scores = apply_scale_and_softcap(b, c, k, scores);
    let scores = layout.add_mask(b, c, li, &scores);
    let attn = attn_softmax(b, k, &scores, layout.score_axis());
    let o = layout.context(b, c, &attn, &vslab);
    let attn_out = layout.o_proj(b, &o, lw);
    let attn_out = post_attn_norm(b, c, k, layout, attn_out, lw);
    (hn, attn_out)
}

/// Emit one complete transformer layer (attention + MLP) for graph kind `layout`,
/// returning the residual stream after the layer. Sequential (Llama family, the
/// two-residual pre-norm block, with the optional Granite / MiniCPM per-residual
/// multiplier) or parallel (Cohere/Cohere2: `x + attn(ln(x)) + mlp(ln(x))`, one
/// shared norm and a single residual). Shared by the single-token decode, ragged
/// decode, and prefill graphs so a family's block structure is authored once and
/// reaches all three. For the Llama family this emits the exact op sequence each
/// path inlined.
#[allow(clippy::too_many_arguments)]
fn emit_transformer_layer(
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
    if c.parallel_block {
        // Parallel: attention and MLP both read the one input_layernorm output;
        // their results are summed into a single residual (Cohere/Cohere2).
        let (hn, attn_out) = emit_attention(b, c, k, lw, li, x, layout, kcache, vcache);
        let mlp_out = emit_ffn_body(b, c, k, layout, &hn, lw, li);
        let s1 = b.add(x, &attn_out);
        b.add(&s1, &mlp_out)
    } else {
        // Sequential: attention residual, then a re-normed MLP residual, each
        // optionally scaled by the arch's residual multiplier (Granite / MiniCPM).
        let (_, attn_out) = emit_attention(b, c, k, lw, li, x, layout, kcache, vcache);
        let attn_out = scale_residual(b, c, attn_out);
        let x1 = b.add(x, &attn_out);
        let hn2 = mlp_pre_norm(b, c, k, layout, &x1, lw);
        let down = emit_ffn_body(b, c, k, layout, &hn2, lw, li);
        let down = scale_residual(b, c, down);
        b.add(&x1, &down)
    }
}

/// Emit the complete decode_step module text. With `sample`, the graph ends in
/// an on-device argmax and returns the next token id (`tensor<i32>`, the Phase
/// 2b pattern); otherwise it returns the raw `[V]` logits.
pub fn emit_decode(c: &Config, sample: bool) -> String {
    emit_decode_with(c, sample, precision_from_env())
}

pub fn emit_decode_with(c: &Config, sample: bool, precision: Precision) -> String {
    let mut b = Builder::new().with_precision(precision);
    let (decls, a) = build_arg_schema(&mut b, c);
    let k = emit_consts(&mut b, c);

    let h = c.hidden;
    let seq = c.context_capacity;
    // RoPE cos/sin vectors span the rotary width (`head_dim` for a full-rope arch,
    // the smaller `rotary_dim` for partial RoPE; equal for the Llama family).
    let rot = c.rotary_width();

    // --- head: embed gather, rope vectors, decode key mask ---
    let emb_row = b.dynamic_slice(&a.embed, &[&a.token, &k.c0], vec![1, h]);
    let emb = b.reshape(&emb_row, vec![h]);
    let mut x = scale_embedding(&mut b, c, emb, vec![h]);

    let (cos_vec, sin_vec, cos_local, sin_local) = if c.uses_mrope() {
        let positions = b.reshape(&a.pos, vec![3, 1]);
        let (cos, sin) = mrope_cos_sin(&mut b, c, &positions, 1);
        (
            b.reshape(&cos, vec![rot]),
            b.reshape(&sin, vec![rot]),
            None,
            None,
        )
    } else {
        let cos_row = b.dynamic_slice(&k.cos_table, &[&a.pos, &k.c0], vec![1, rot]);
        let cos_vec = b.reshape(&cos_row, vec![rot]);
        let sin_row = b.dynamic_slice(&k.sin_table, &[&a.pos, &k.c0], vec![1, rot]);
        let sin_vec = b.reshape(&sin_row, vec![rot]);
        // Dual-RoPE (Gemma3 / OLMo3): the local-base [rot] vectors for the sliding layers.
        let (cos_local, sin_local) = match (&k.cos_local, &k.sin_local) {
            (Some(ct), Some(st)) => {
                let cr = b.dynamic_slice(ct, &[&a.pos, &k.c0], vec![1, rot]);
                let cl = b.reshape(&cr, vec![rot]);
                let sr = b.dynamic_slice(st, &[&a.pos, &k.c0], vec![1, rot]);
                let sl = b.reshape(&sr, vec![rot]);
                (Some(cl), Some(sl))
            }
            _ => (None, None),
        };
        (cos_vec, sin_vec, cos_local, sin_local)
    };

    // mask: keys s valid iff s <= cache_len -> additive 0 / -1e30, shape [S]
    let ii = b.iota(seq);
    let clen_b = b.broadcast(&a.cache_len, &[], vec![seq]);
    let valid = b.compare("LE", &ii, &clen_b, "SIGNED");
    let zeros_s = b.broadcast(&k.zero, &[], vec![seq]);
    let negs_s = b.broadcast(&k.neg_big, &[], vec![seq]);
    let kmask = b.select(&valid, &zeros_s, &negs_s);
    // Gemma2 local (sliding-window) mask (issue #495): a local layer additionally
    // drops keys older than the window, keeping key s iff `cache_len - s < W`. The
    // global `kmask` already encodes causality, so anding the window in is one more
    // `select`: within-window -> keep `kmask`, else force -1e30. Built once and
    // reused by every local layer; emitted only for a config with a window
    // (Gemma2), so Llama / Qwen2 stay byte-identical. When `W >= context_capacity`
    // the predicate is always true, so it is a value
    // no-op, which is why short-context Gemma2 output is unchanged.
    let kmask_local = c.sliding_window.map(|w| {
        let wc = b.const_i32(w as i32);
        let wb = b.broadcast(&wc, &[], vec![seq]);
        let age = b.subtract(&clen_b, &ii); // cache_len - s
        let within = b.compare("LT", &age, &wb, "SIGNED");
        b.select(&within, &kmask, &negs_s)
    });

    let layout = AttnLayout::Single {
        cos: cos_vec,
        sin: sin_vec,
        cos_local,
        sin_local,
        mask: kmask,
        mask_local: kmask_local,
        cache_len: a.cache_len.clone(),
    };

    let mut kcache = a.kcache.clone();
    let mut vcache = a.vcache.clone();

    for li in 0..c.n_layers {
        let lw = &a.layers[li];
        // Full transformer layer (attention + MLP, sequential or parallel), shared
        // with the ragged / prefill graphs (issues #494 / #498).
        x = emit_transformer_layer(&mut b, c, &k, lw, li, &x, &layout, &mut kcache, &mut vcache);
    }

    // --- tail: final norm (+ LayerNorm bias) + LM head (tied embed or untied
    // lm_head), the arch's final logit scaling (Gemma soft-cap / Cohere multiply /
    // Granite / MiniCPM divide), then optional on-device argmax ---
    let final_w = norm_w(&mut b, &a.final_norm, c, &k, h);
    let xf = normalize(&mut b, c, &k, &x, &final_w, a.final_norm_bias.as_ref(), h);
    let logits = b.linear(&xf, head_weight(&a.embed, &a.lm_head)); // [V]
    let logits = apply_logit_scale(&mut b, c, logits);
    let (out_val, out_ty) = if sample {
        let tok = b.argmax(&logits);
        (tok.name, Ty::scalar("i32").render())
    } else {
        (logits.name, Ty::f32(vec![c.vocab]).render())
    };

    let sig = render_signature(&decls);
    let cache_ty = Ty::f32(vec![c.n_layers, seq, c.n_kv, c.head_dim]).render();
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
    final_norm_bias: Option<Val>,
    lm_head: Option<Val>,
    layers: Vec<LayerW>,
    token: Val,     // [B] i32
    pos: Val,       // scalar i32 (shared across the batch)
    cache_len: Val, // scalar i32 (shared across the batch)
    kcache: Val,    // [B, L, context_capacity, nkv, d]
    vcache: Val,
}

fn build_batched_arg_schema(
    b: &mut Builder,
    c: &Config,
    bsz: usize,
) -> (Vec<ArgDecl>, BatchedArgs) {
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
    let final_norm_bias = take_final_norm_bias(&mut decls, &mut idx, c);
    let lm_head = take_lm_head(&mut decls, &mut idx, c);

    let mut layers = Vec::with_capacity(c.n_layers);
    for li in 0..c.n_layers {
        layers.push(take_layer_weights(b, &mut decls, &mut idx, c, li));
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
        Ty::new(if c.uses_mrope() { vec![3] } else { vec![] }, "i32"),
        "pos".into(),
    );
    let cache_len = take_arg(&mut decls, &mut idx, Ty::scalar("i32"), "cache_len".into());
    let kcache = take_arg(
        &mut decls,
        &mut idx,
        Ty::f32(vec![
            bsz,
            c.n_layers,
            c.context_capacity,
            c.n_kv,
            c.head_dim,
        ]),
        "kcache".into(),
    );
    let vcache = take_arg(
        &mut decls,
        &mut idx,
        Ty::f32(vec![
            bsz,
            c.n_layers,
            c.context_capacity,
            c.n_kv,
            c.head_dim,
        ]),
        "vcache".into(),
    );

    (
        decls,
        BatchedArgs {
            embed,
            final_norm,
            final_norm_bias,
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
    let mut b = Builder::new().with_precision(precision);
    let (decls, a) = build_batched_arg_schema(&mut b, c, bsz);
    let k = emit_consts(&mut b, c);

    let h = c.hidden;
    let d = c.head_dim;
    let nq = c.n_q;
    let nkv = c.n_kv;
    let g = c.group();
    let seq = c.context_capacity;

    // --- head: per-row embed gather, shared rope vectors, shared key mask ---
    let tok_idx = b.reshape(&a.token, vec![bsz, 1]);
    let mut x = b.gather(&a.embed, &tok_idx); // [B, H]

    // pos is shared (lockstep), so cos/sin are one [d] vector for every row.
    let (cos_vec, sin_vec) = if c.uses_mrope() {
        let positions = b.reshape(&a.pos, vec![3, 1]);
        let (cos, sin) = mrope_cos_sin(&mut b, c, &positions, 1);
        (b.reshape(&cos, vec![d]), b.reshape(&sin, vec![d]))
    } else {
        let cos_row = b.dynamic_slice(&k.cos_table, &[&a.pos, &k.c0], vec![1, d]);
        let cos_vec = b.reshape(&cos_row, vec![d]);
        let sin_row = b.dynamic_slice(&k.sin_table, &[&a.pos, &k.c0], vec![1, d]);
        let sin_vec = b.reshape(&sin_row, vec![d]);
        (cos_vec, sin_vec)
    };

    // shared key mask [S]: key s valid iff s <= cache_len -> additive 0 / -1e30
    let ii = b.iota(seq);
    let clen_b = b.broadcast(&a.cache_len, &[], vec![seq]);
    let valid = b.compare("LE", &ii, &clen_b, "SIGNED");
    let zeros_s = b.broadcast(&k.zero, &[], vec![seq]);
    let negs_s = b.broadcast(&k.neg_big, &[], vec![seq]);
    let kmask = b.select(&valid, &zeros_s, &negs_s);

    let mut kcache = a.kcache.clone();
    let mut vcache = a.vcache.clone();

    for li in 0..c.n_layers {
        let lw = &a.layers[li];

        // attention block (RMSNorm over H reuses the [N,H] seq variant, N=B). The
        // superseded uniform-B batched graph serves only the pre-norm families
        // (Llama / Qwen2), which always carry `input_layernorm`.
        let in_ln = lw
            .in_ln
            .as_ref()
            .expect("uniform-B batched decode is emitted only for pre-norm archs");
        let hn = rms_norm_seq(&mut b, &x, in_ln, &k, bsz, h); // [B, H]
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
            &[(0, bsz), (li, li + 1), (0, seq), (0, nkv), (0, d)],
        );
        let kl = b.reshape(&kl, vec![bsz, seq, nkv, d]);
        let vl = b.slice(
            &vcache,
            &[(0, bsz), (li, li + 1), (0, seq), (0, nkv), (0, d)],
        );
        let vl = b.reshape(&vl, vec![bsz, seq, nkv, d]);

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
            vec![bsz, nkv, g, seq],
        );
        let scores = b.reshape(&scores, vec![bsz, nq, seq]);
        let scale_b = b.broadcast(&k.scale, &[], vec![bsz, nq, seq]);
        let scores = b.multiply(&scores, &scale_b);
        let kmask_b = b.broadcast(&kmask, &[2], vec![bsz, nq, seq]);
        let scores = b.add(&scores, &kmask_b);

        // softmax over the key axis (dim 2)
        let m = b.reduce_max(&scores, 2, &k.neg_inf); // [B, nq]
        let m_b = b.broadcast(&m, &[0, 1], vec![bsz, nq, seq]);
        let sh = b.subtract(&scores, &m_b);
        let e = b.exponential(&sh);
        let s = b.reduce_add(&e, 2, &k.zero); // [B, nq]
        let s_b = b.broadcast(&s, &[0, 1], vec![bsz, nq, seq]);
        let attn = b.divide(&e, &s_b); // [B, nq, S]

        // context: o[b,h,d] = sum_s attn[b,h,s] * vl[b,s,h/g,d]
        let attn_r = b.reshape(&attn, vec![bsz, nkv, g, seq]);
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

        // FFN: the MoE block (issue #500) on a MoE layer, else the dense SwiGLU MLP
        // `down( silu(x@gate^T) * (x@up^T) )`. The superseded uniform-B graph serves
        // only the gated, sequential families, so `post_ln` / `gate` are always
        // present (the dense-arch-pack graphs use the ragged serve path); the dense
        // branch emits the exact op sequence it did before, so it stays byte-for-byte
        // identical.
        if c.is_moe_layer(li) {
            let m = c.moe.as_ref().expect("a MoE layer has a MoeConfig");
            let mw = lw.moe.as_ref().expect("a MoE layer has expert weights");
            let hn2 = rms_norm_seq(
                &mut b,
                &x,
                lw.post_ln.as_ref().expect("batched decode: post_ln"),
                &k,
                bsz,
                h,
            );
            let moe_out = moe::moe_block(&mut b, c, m, mw, &k, &hn2, bsz);
            x = b.add(&x, &moe_out);
        } else {
            let hn2 = rms_norm_seq(
                &mut b,
                &x,
                lw.post_ln.as_ref().expect("batched decode: post_ln"),
                &k,
                bsz,
                h,
            );
            let gate = b.linear_seq(&hn2, lw.gate.as_ref().expect("batched decode: gate")); // [B, inter]
            let up = b.linear_seq(&hn2, LayerW::dense(&lw.up)); // [B, inter]
            let neg = b.negate(&gate);
            let ex = b.exponential(&neg);
            let one_b = b.broadcast(&k.one, &[], vec![bsz, c.inter]);
            let denom = b.add(&one_b, &ex);
            let sig = b.divide(&one_b, &denom);
            let silu = b.multiply(&gate, &sig);
            let act = b.multiply(&silu, &up);
            let down = b.linear_seq(&act, LayerW::dense(&lw.down)); // [B, H]
            x = b.add(&x, &down);
        }
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
    let cache_ty = Ty::f32(vec![bsz, c.n_layers, seq, c.n_kv, c.head_dim]).render();
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
    final_norm_bias: Option<Val>,
    lm_head: Option<Val>,
    layers: Vec<LayerW>,
    token: Val,     // [B] i32
    pos: Val,       // [B] i32 (per row)
    cache_len: Val, // [B] i32 (per row)
    kcache: Val,    // [B, L, context_capacity, nkv, d]
    vcache: Val,
}

fn build_ragged_arg_schema(b: &mut Builder, c: &Config, bsz: usize) -> (Vec<ArgDecl>, RaggedArgs) {
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
    let final_norm_bias = take_final_norm_bias(&mut decls, &mut idx, c);
    let lm_head = take_lm_head(&mut decls, &mut idx, c);

    let mut layers = Vec::with_capacity(c.n_layers);
    for li in 0..c.n_layers {
        layers.push(take_layer_weights(b, &mut decls, &mut idx, c, li));
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
        Ty::new(
            if c.uses_mrope() {
                vec![bsz, 3]
            } else {
                vec![bsz]
            },
            "i32",
        ),
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
        Ty::f32(vec![
            bsz,
            c.n_layers,
            c.context_capacity,
            c.n_kv,
            c.head_dim,
        ]),
        "kcache".into(),
    );
    let vcache = take_arg(
        &mut decls,
        &mut idx,
        Ty::f32(vec![
            bsz,
            c.n_layers,
            c.context_capacity,
            c.n_kv,
            c.head_dim,
        ]),
        "vcache".into(),
    );

    (
        decls,
        RaggedArgs {
            embed,
            final_norm,
            final_norm_bias,
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
    let mut b = Builder::new().with_precision(precision);
    let (decls, a) = build_ragged_arg_schema(&mut b, c, bsz);
    let k = emit_consts(&mut b, c);
    // Constant row indices 0..bsz for the per-row KV-write dim-0 offsets.
    let row_idx: Vec<Val> = (0..bsz).map(|i| b.const_i32(i as i32)).collect();

    let h = c.hidden;
    let seq = c.context_capacity;

    // --- head: per-row embed gather, per-row rope gather, per-row key mask ---
    let tok_idx = b.reshape(&a.token, vec![bsz, 1]);
    let emb = b.gather(&a.embed, &tok_idx); // [B, H]
    let mut x = scale_embedding(&mut b, c, emb, vec![bsz, h]);

    // each row's rope vectors come from its own position: gather [B,d] by pos[B]
    let (cos, sin, cos_local, sin_local) = if c.uses_mrope() {
        let positions = b.transpose(&a.pos, &[1, 0]);
        let (cos, sin) = mrope_cos_sin(&mut b, c, &positions, bsz);
        (cos, sin, None, None)
    } else {
        let pos_idx = b.reshape(&a.pos, vec![bsz, 1]);
        let cos = b.gather(&k.cos_table, &pos_idx); // [B, d]
        let sin = b.gather(&k.sin_table, &pos_idx); // [B, d]
        // Dual-RoPE (Gemma3 / OLMo3): per-row local-base gathers for the sliding layers.
        let (cos_local, sin_local) = match (&k.cos_local, &k.sin_local) {
            (Some(ct), Some(st)) => (Some(b.gather(ct, &pos_idx)), Some(b.gather(st, &pos_idx))),
            _ => (None, None),
        };
        (cos, sin, cos_local, sin_local)
    };

    // per-row key mask [B,S]: key s valid for row b iff s <= cache_len[b]
    let ii = b.iota(seq); // [S]
    let ii_b = b.broadcast(&ii, &[1], vec![bsz, seq]); // entry[b,s] = s
    let clen_b = b.broadcast(&a.cache_len, &[0], vec![bsz, seq]); // entry[b,s] = cache_len[b]
    let valid = b.compare("LE", &ii_b, &clen_b, "SIGNED");
    let zeros = b.broadcast(&k.zero, &[], vec![bsz, seq]);
    let negs = b.broadcast(&k.neg_big, &[], vec![bsz, seq]);
    let kmask = b.select(&valid, &zeros, &negs); // [B, S]
    // Gemma2 local (sliding-window) mask (issue #495), per row: keep key s for row
    // b iff `cache_len[b] - s < W`. And it into the causal `kmask` with one more
    // `select`. Emitted only for a windowed config (Gemma2); reused by every local
    // layer. No-op on the value when `W >= context_capacity` (short-context parity).
    let kmask_local = c.sliding_window.map(|w| {
        let wc = b.const_i32(w as i32);
        let wb = b.broadcast(&wc, &[], vec![bsz, seq]);
        let age = b.subtract(&clen_b, &ii_b); // cache_len[b] - s
        let within = b.compare("LT", &age, &wb, "SIGNED");
        b.select(&within, &kmask, &negs)
    });

    let layout = AttnLayout::Ragged {
        bsz,
        cos,
        sin,
        cos_local,
        sin_local,
        mask: kmask,
        mask_local: kmask_local,
        // M-RoPE KV writes are indexed by physical cache length while the
        // distinct position input may include a signed per-slot delta. Preserve
        // the byte-exact legacy 1D graph, whose ABI validates pos == cache_len.
        pos: if c.uses_mrope() {
            a.cache_len.clone()
        } else {
            a.pos.clone()
        },
        row_idx,
    };

    let mut kcache = a.kcache.clone();
    let mut vcache = a.vcache.clone();

    for li in 0..c.n_layers {
        let lw = &a.layers[li];
        // Full transformer layer, shared with the single / prefill graphs.
        x = emit_transformer_layer(&mut b, c, &k, lw, li, &x, &layout, &mut kcache, &mut vcache);
    }

    let final_w = norm_w(&mut b, &a.final_norm, c, &k, h);
    let xf = normalize_seq(
        &mut b,
        c,
        &k,
        &x,
        &final_w,
        a.final_norm_bias.as_ref(),
        bsz,
        h,
    );
    let logits = b.linear_seq(&xf, head_weight(&a.embed, &a.lm_head)); // [B, V]
    // Final logit scaling (Gemma soft-cap / Cohere multiply / Granite / MiniCPM
    // divide), per row; argmax-invariant but kept for exactness.
    let logits = apply_logit_scale(&mut b, c, logits);
    let (out_val, out_ty) = if sample {
        let tok = b.argmax_batched(&logits);
        (tok.name, Ty::new(vec![bsz], "i32").render())
    } else {
        (logits.name, Ty::f32(vec![bsz, c.vocab]).render())
    };

    let sig = render_signature(&decls);
    let cache_ty = Ty::f32(vec![bsz, c.n_layers, seq, c.n_kv, c.head_dim]).render();
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

/// Which stable prefill entry schema to build. The token entry remains unchanged;
/// the embeddings entry replaces `tokens` with post-scale hidden states and adds
/// an explicit additive attention bias.
#[derive(Clone, Copy)]
enum PrefillInputKind {
    Tokens,
    Embeddings,
}

enum PrefillInput {
    Tokens(Val),
    Embeddings { hidden: Val, attention_bias: Val },
}

/// Prefill arg handles. Weights are identical to decode (same order/locs), which
/// intentionally retains `params['embed']`: token prefill gathers from it and a
/// tied embeddings prefill reuses it as the LM head. The trailing schema is:
///
/// - token: `tokens`, `positions`, `real_len`;
/// - embeddings: `embeddings`, `positions`, `real_len`, `attention_bias`.
struct PrefillArgs {
    embed: Val,
    final_norm: Val,
    final_norm_bias: Option<Val>,
    lm_head: Option<Val>,
    layers: Vec<LayerW>,
    input: PrefillInput,
    positions: Val,
    real_len: Val,
}

fn build_prefill_arg_schema(
    b: &mut Builder,
    c: &Config,
    lp: usize,
    input_kind: PrefillInputKind,
) -> (Vec<ArgDecl>, PrefillArgs) {
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
    let final_norm_bias = take_final_norm_bias(&mut decls, &mut idx, c);
    let lm_head = take_lm_head(&mut decls, &mut idx, c);

    let mut layers = Vec::with_capacity(c.n_layers);
    for li in 0..c.n_layers {
        layers.push(take_layer_weights(b, &mut decls, &mut idx, c, li));
    }

    let (tokens, embeddings) = match input_kind {
        PrefillInputKind::Tokens => (
            Some(take_arg(
                &mut decls,
                &mut idx,
                Ty::new(vec![lp], "i32"),
                "tokens".into(),
            )),
            None,
        ),
        PrefillInputKind::Embeddings => (
            None,
            Some(take_arg(
                &mut decls,
                &mut idx,
                Ty::f32(vec![lp, h]),
                "embeddings".into(),
            )),
        ),
    };
    let positions = take_arg(
        &mut decls,
        &mut idx,
        Ty::new(
            if c.uses_mrope() {
                vec![3, lp]
            } else {
                vec![lp]
            },
            "i32",
        ),
        "positions".into(),
    );
    let real_len = take_arg(&mut decls, &mut idx, Ty::scalar("i32"), "real_len".into());
    let input = match (tokens, embeddings) {
        (None, Some(hidden)) => PrefillInput::Embeddings {
            hidden,
            attention_bias: take_arg(
                &mut decls,
                &mut idx,
                Ty::f32(vec![lp, lp]),
                "attention_bias".into(),
            ),
        },
        (Some(tokens), None) => PrefillInput::Tokens(tokens),
        _ => unreachable!("each prefill schema has exactly one input representation"),
    };

    (
        decls,
        PrefillArgs {
            embed,
            final_norm,
            final_norm_bias,
            lm_head,
            layers,
            input,
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

/// HF RoPE on the seq / ragged activations x:[N, heads, d] (N = Lp for prefill,
/// bsz for ragged decode); cos/sin are per-row `[N, rd]` (rotary width). Shared by
/// both graph kinds (they differ only in the row count). Full or partial RoPE,
/// half-split or interleaved, per `c`. Byte-identical to the inlined half-split for
/// the Llama family.
fn apply_rope_seq(
    b: &mut Builder,
    c: &Config,
    x: &Val,
    cos: &Val,
    sin: &Val,
    n: usize,
    heads: usize,
) -> Val {
    let d = c.head_dim;
    let rd = c.rotary_width();
    if rd == d {
        return rotate_seq(b, c, x, cos, sin, n, heads, rd);
    }
    let x_rot = b.slice(x, &[(0, n), (0, heads), (0, rd)]);
    let x_pass = b.slice(x, &[(0, n), (0, heads), (rd, d)]);
    let rotated = rotate_seq(b, c, &x_rot, cos, sin, n, heads, rd);
    b.concatenate(&rotated, &x_pass, 2)
}

/// The MLX Qwen axis source for each half-rotary frequency column.
pub(crate) fn mrope_axis_selector(c: &Config) -> Result<Vec<usize>, String> {
    let mrope = c
        .mrope
        .as_ref()
        .ok_or_else(|| "M-RoPE axis selection requires an M-RoPE config".to_string())?;
    let half = c.rotary_width() / 2;
    let selector = match mrope.layout {
        MropeLayout::Chunked => {
            let mut selector = Vec::with_capacity(half);
            for (axis, &width) in mrope.sections.iter().enumerate() {
                selector.extend(std::iter::repeat_n(axis, width));
            }
            selector
        }
        MropeLayout::Interleaved => {
            // Match `InterleavedMRoPE::apply_interleaved_mrope` in the MLX Qwen3
            // implementation: temporal is the default, and H/W overwrite the
            // step-3 columns within their configured windows.
            let mut selector = vec![0usize; half];
            for (offset, &section_len) in mrope.sections[1..].iter().enumerate() {
                let axis = offset + 1;
                let mut index = axis;
                while index < section_len * 3 {
                    if index < selector.len() {
                        selector[index] = axis;
                    }
                    index += 3;
                }
            }
            selector
        }
    };
    if selector.len() != half {
        return Err(format!(
            "M-RoPE axis selector has {} columns, expected {half}",
            selector.len()
        ));
    }
    Ok(selector)
}

/// Build dynamic M-RoPE cos/sin values for explicit signed coordinates.
///
/// `positions` is `[3, N]` ordered temporal/height/width. Unlike the ordinary
/// one-dimensional path, this intentionally computes trig in the graph instead
/// of indexing a context-sized table: decode coordinates are physical KV length
/// plus a signed per-sequence delta and therefore are not bounded by the KV
/// table's index range.
fn mrope_cos_sin(b: &mut Builder, c: &Config, positions: &Val, n: usize) -> (Val, Val) {
    assert_eq!(positions.ty.elt, "i32");
    assert_eq!(positions.ty.shape, vec![3, n]);
    let selector =
        mrope_axis_selector(c).unwrap_or_else(|error| panic!("invalid M-RoPE config: {error}"));
    let axes: Vec<Val> = (0..3)
        .map(|axis| {
            let row = b.slice(positions, &[(axis, axis + 1), (0, n)]);
            b.reshape(&row, vec![n])
        })
        .collect();
    let columns: Vec<Val> = selector
        .into_iter()
        .map(|axis| b.broadcast(&axes[axis], &[0], vec![n, 1]))
        .collect();
    let mut columns = columns.into_iter();
    let mut selected = columns.next().expect("positive M-RoPE rotary width");
    for column in columns {
        selected = b.concatenate(&selected, &column, 1);
    }
    let selected = b.convert(&selected, "f32");
    let inv = rope::inv_freq(c)
        .into_iter()
        .map(|value| value as f32)
        .collect::<Vec<_>>();
    let inv = b.const_tensor_f32(&inv, vec![c.rotary_width() / 2]);
    let inv = b.broadcast(&inv, &[1], vec![n, c.rotary_width() / 2]);
    let freqs = b.multiply(&selected, &inv);
    let angles = b.concatenate(&freqs, &freqs, 1);
    (b.cosine(&angles), b.sine(&angles))
}

/// The core RoPE rotation on x:[N, heads, rd] with per-row cos/sin [N, rd].
/// Half-split or interleaved (issue #498). Byte-identical to the inlined
/// half-split for the Llama family.
#[allow(clippy::too_many_arguments)]
fn rotate_seq(
    b: &mut Builder,
    c: &Config,
    x: &Val,
    cos: &Val,
    sin: &Val,
    n: usize,
    heads: usize,
    rd: usize,
) -> Val {
    let cos_b = b.broadcast(cos, &[0, 2], vec![n, heads, rd]); // [N,rd] -> [N,heads,rd]
    let sin_b = b.broadcast(sin, &[0, 2], vec![n, heads, rd]);
    let xc = b.multiply(x, &cos_b);
    let rh = if c.rope_interleaved {
        let xr = b.reshape(x, vec![n, heads, rd / 2, 2]);
        let even = b.slice(&xr, &[(0, n), (0, heads), (0, rd / 2), (0, 1)]);
        let odd = b.slice(&xr, &[(0, n), (0, heads), (0, rd / 2), (1, 2)]);
        let neg_odd = b.negate(&odd);
        let st = b.concatenate(&neg_odd, &even, 3);
        b.reshape(&st, vec![n, heads, rd])
    } else {
        let half = rd / 2;
        let x1 = b.slice(x, &[(0, n), (0, heads), (0, half)]);
        let x2 = b.slice(x, &[(0, n), (0, heads), (half, rd)]);
        let nx2 = b.negate(&x2);
        b.concatenate(&nx2, &x1, 2)
    };
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
    emit_prefill_module(c, sample, precision, PrefillInputKind::Tokens)
}

/// Emit the distinct prefill-from-embeddings StableHLO module at the ambient
/// precision. The input hidden states are already post-token-embedding-scale;
/// this entry performs neither a token gather nor [`scale_embedding`].
pub(crate) fn emit_prefill_embeddings(c: &Config, sample: bool) -> String {
    emit_prefill_embeddings_with(c, sample, precision_from_env())
}

/// Emit `prefill_embeddings.main` at an explicit contraction precision.
pub(crate) fn emit_prefill_embeddings_with(
    c: &Config,
    sample: bool,
    precision: Precision,
) -> String {
    emit_prefill_module(c, sample, precision, PrefillInputKind::Embeddings)
}

fn emit_prefill_module(
    c: &Config,
    sample: bool,
    precision: Precision,
    input_kind: PrefillInputKind,
) -> String {
    let lp = c.context_capacity;
    let mut b = Builder::new().with_precision(precision);
    let (decls, a) = build_prefill_arg_schema(&mut b, c, lp, input_kind);
    let k = emit_consts(&mut b, c);

    let h = c.hidden;
    let d = c.head_dim;
    let nkv = c.n_kv;

    // --- input head ---
    // Token prefill gathers + applies the architecture's embedding scale. The
    // embeddings entry consumes already-merged, post-scale hidden states as-is.
    let mut x = match &a.input {
        PrefillInput::Tokens(tokens) => {
            let tok_idx = b.reshape(tokens, vec![lp, 1]);
            let emb = b.gather(&a.embed, &tok_idx); // [Lp, H]
            scale_embedding(&mut b, c, emb, vec![lp, h])
        }
        PrefillInput::Embeddings { hidden, .. } => hidden.clone(),
    };

    // --- shared position preparation ---
    let (cos, sin, cos_local, sin_local) = if c.uses_mrope() {
        let (cos, sin) = mrope_cos_sin(&mut b, c, &a.positions, lp);
        (cos, sin, None, None)
    } else {
        let pos_idx = b.reshape(&a.positions, vec![lp, 1]);
        let cos = b.gather(&k.cos_table, &pos_idx); // [Lp, d]
        let sin = b.gather(&k.sin_table, &pos_idx); // [Lp, d]
        // Dual-RoPE (Gemma3 / OLMo3): per-position local-base gathers for sliding layers.
        let (cos_local, sin_local) = match (&k.cos_local, &k.sin_local) {
            (Some(ct), Some(st)) => (Some(b.gather(ct, &pos_idx)), Some(b.gather(st, &pos_idx))),
            _ => (None, None),
        };
        (cos, sin, cos_local, sin_local)
    };

    // --- attention-bias preparation ---
    // Token prefill retains the exact internal causal-mask construction. The
    // embeddings entry uses the caller's canonical f32 additive bias directly.
    // For a sliding architecture, the local-layer window is intersected with
    // either base mask, preserving the architecture invariant without changing
    // multimodal/global visibility.
    let (cmask, cmask_local) = match &a.input {
        PrefillInput::Tokens(_) => {
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
            (cmask, cmask_local)
        }
        PrefillInput::Embeddings { attention_bias, .. } => {
            let cmask = attention_bias.clone();
            let cmask_local = c.sliding_window.map(|w| {
                let irow = b.iota(lp);
                let row = b.broadcast(&irow, &[0], vec![lp, lp]);
                let jcol = b.iota(lp);
                let col = b.broadcast(&jcol, &[1], vec![lp, lp]);
                let wc = b.const_i32(w as i32);
                let wb = b.broadcast(&wc, &[], vec![lp, lp]);
                let age = b.subtract(&row, &col);
                let within = b.compare("LT", &age, &wb, "SIGNED");
                let negs = b.broadcast(&k.neg_big, &[], vec![lp, lp]);
                b.select(&within, &cmask, &negs)
            });
            (cmask, cmask_local)
        }
    };

    let layout = AttnLayout::Prefill {
        lp,
        cos,
        sin,
        cos_local,
        sin_local,
        mask: cmask,
        mask_local: cmask_local,
    };

    // caches start as zeros; prefill writes the [0:Lp] block and returns them
    let mut kcache = b.broadcast(&k.zero, &[], vec![c.n_layers, lp, nkv, d]);
    let mut vcache = b.broadcast(&k.zero, &[], vec![c.n_layers, lp, nkv, d]);

    for li in 0..c.n_layers {
        let lw = &a.layers[li];
        // Full transformer layer, shared with the single / ragged graphs.
        x = emit_transformer_layer(&mut b, c, &k, lw, li, &x, &layout, &mut kcache, &mut vcache);
    }

    // --- tail: final norm (+ LayerNorm bias), take the row at real_len-1, LM head
    // (tied embed or untied lm_head), then the arch's final logit scaling ---
    let final_w = norm_w(&mut b, &a.final_norm, c, &k, h);
    let xf = normalize_seq(
        &mut b,
        c,
        &k,
        &x,
        &final_w,
        a.final_norm_bias.as_ref(),
        lp,
        h,
    ); // [Lp, H]
    let one_i = b.const_i32(1);
    let last_idx = b.subtract(&a.real_len, &one_i); // real_len - 1
    let last_row = b.dynamic_slice(&xf, &[&last_idx, &k.c0], vec![1, h]); // [1, H]
    let last = b.reshape(&last_row, vec![h]); // [H]
    let logits = b.linear(&last, head_weight(&a.embed, &a.lm_head)); // [V]
    let logits = apply_logit_scale(&mut b, c, logits);
    let (out_val, out_ty) = if sample {
        let tok = b.argmax(&logits);
        (tok.name, Ty::scalar("i32").render())
    } else {
        (logits.name, Ty::f32(vec![c.vocab]).render())
    };

    let sig = render_signature(&decls);
    let cache_ty = Ty::f32(vec![c.n_layers, lp, c.n_kv, c.head_dim]).render();
    let module_name = match input_kind {
        PrefillInputKind::Tokens => "prefill",
        PrefillInputKind::Embeddings => "prefill_embeddings",
    };
    format!(
        "module @{module_name} {{\n  func.func public @main({sig}) -> ({out_ty}, {cache_ty}, {cache_ty}) {{\n{body}    return {l}, {kc}, {vc} : {out_ty}, {cache_ty}, {cache_ty}\n  }}\n}}\n",
        module_name = module_name,
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
// MoE FFN block probe (issue #500): a standalone module for the execution check
// ===========================================================================
//
// `main(moe weights..., hn[N, H]) -> out[N, H]` runs ONLY the MoE FFN block
// ([`moe::moe_block`]) on an already-normed hidden `hn`, with no attention, no
// pre-norm, and no residual, so an out-of-crate check (spike/openxla/moe_oracle.py)
// can compile it with IREE and compare it directly to an HF MoE block's fp32
// forward. Isolating the block makes the routing / dispatch math the only variable,
// so the execution check proves it without needing a full model or a real
// (unsupported-attention) MoE checkpoint. The arg order is the per-layer MoE order
// `take_moe_weights` / `weight_specs` share, so the probe doubles as a check that
// the emitted expert-arg schema is what the loader feeds.

/// Emit the MoE FFN block probe for `c` over `n` input rows (default precision).
pub(crate) fn emit_moe_probe(c: &Config, n: usize) -> String {
    emit_moe_probe_with(c, n, precision_from_env())
}

/// Emit the MoE FFN block probe at an explicit contraction precision.
pub(crate) fn emit_moe_probe_with(c: &Config, n: usize, precision: Precision) -> String {
    let m = c
        .moe
        .as_ref()
        .expect("emit_moe_probe requires a MoE config");
    let h = c.hidden;

    let mut decls: Vec<ArgDecl> = Vec::new();
    let mut idx = 0usize;
    // The expert bank in the canonical per-layer MoE arg order, then the input.
    let mw = take_moe_weights(&mut decls, &mut idx, c, 0);
    let hn = take_arg(&mut decls, &mut idx, Ty::f32(vec![n, h]), "hn".into());

    let mut b = Builder::new().with_precision(precision);
    let k = emit_consts(&mut b, c);
    let out = moe::moe_block(&mut b, c, m, &mw, &k, &hn, n);

    let sig = render_signature(&decls);
    let out_ty = Ty::f32(vec![n, h]).render();
    format!(
        "module @moe_probe {{\n  func.func public @main({sig}) -> {out_ty} {{\n{body}    return {o} : {out_ty}\n  }}\n}}\n",
        sig = sig,
        out_ty = out_ty,
        body = b.body(),
        o = out.name,
    )
}
