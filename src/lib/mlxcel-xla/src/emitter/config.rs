//! Emitter config for the dense architectures the OpenXLA backend serves.
//! The hard-coded [`Config::llama_3_2_1b`] matches spike/openxla/model_jax.py;
//! [`Config::from_json`] reads the same shape from a checkpoint's `config.json`.
//!
//! The config carries orthogonal architecture switches the emitter branches on,
//! so a new dense family is a combination of flags rather than a new code path:
//! the RoPE kind (llama3 vs plain, plus an optional per-layer local base for
//! Gemma3, an interleaved layout and a partial width for Cohere / StableLM),
//! whether q/k/v projections carry a bias (Qwen2), the LM-head tie, MLX
//! quantization, the Gemma embedding scale / `(1+w)` RMSNorm / GeGLU MLP, the
//! per-layer norm placement ([`NormStyle`]), an optional q/k normalization
//! ([`QkNorm`]: per-head for Qwen3 / Gemma3, flat for OLMo2 / OLMo3), the
//! sliding-window schedule (window + pattern period), and a per-layer NoPE mask
//! (SmolLM3). Llama / Qwen2 / Gemma2 keep their exact previous flag combinations,
//! so their emitted graphs are byte-for-byte unchanged.
//!
//! A second dense pack (issue #499) rides the same flags with a config / naming
//! delta and no new emit: Seed-OSS / MiMo (the Qwen2 bias forward), InternLM3
//! (plain / in-context `dynamic` RoPE), and ExaOne 3.x (llama3 RoPE with GPT-2-style
//! tensor names, selected by [`WeightScheme`]). A third pack (issue #498) adds the
//! per-family deltas the shared core needs for the parallel-block and norm-variant
//! families: a mean-subtract LayerNorm (with an optional affine bias), a parallel
//! attention + MLP block (Cohere/Cohere2), interleaved / partial RoPE, a dense
//! (non-gated) MLP (StarCoder2), the o_proj / MLP biases, and the Granite / MiniCPM
//! scalar multipliers, plus the Phi3 fused q/k/v and gate/up projections split at
//! load. Out-of-scope deltas an emit would get wrong (interleaved RoPE where the
//! family is not validated, an unsupported activation, yarn RoPE, MiniCPM3's MLA)
//! are rejected rather than mis-emitted.

/// How the RoPE inverse-frequency table is computed. Both kinds share the
/// `outer(pos, inv_freq)` table build (see [`rope`](super::rope)); they differ
/// only in `inv_freq`.
#[derive(Clone, Debug, PartialEq)]
pub enum RopeScaling {
    /// Plain RoPE: `inv_freq[i] = 1 / theta^(2i/head_dim)` (Qwen2, Qwen3, Gemma,
    /// SmolLM3, OLMo2, Cohere, and plain-RoPE Llama without a `rope_scaling` block).
    Plain,
    /// Llama3 RoPE scaling, byte-for-byte with HF `_compute_llama3_parameters`.
    Llama3 {
        factor: f64,
        low_freq_factor: f64,
        high_freq_factor: f64,
        orig_ctx: usize,
    },
}

/// Per-layer norm placement. The three dense patterns differ in where the
/// RMSNorms sit relative to the attention / MLP sublayers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormStyle {
    /// Llama / Qwen2 / Qwen3 / Gemma1 / SmolLM3 / Cohere / Phi3 / StableLM /
    /// StarCoder2 / Granite: pre-norm. `input_layernorm` normalizes the residual
    /// before attention; `post_attention_layernorm` normalizes it before the MLP.
    /// Two norms per layer, both on the input side.
    Plain,
    /// Gemma2 / Gemma3: pre-norm wrapped by post-norms. `input_layernorm` before
    /// attention, then `post_attention_layernorm` on the attention output before
    /// the residual; `pre_feedforward_layernorm` before the MLP, then
    /// `post_feedforward_layernorm` on the MLP output before the residual. Four
    /// norms per layer.
    GemmaFf,
    /// OLMo2 / OLMo3: reordered (post) norm. No `input_layernorm`; attention and
    /// the MLP consume the raw residual, and `post_attention_layernorm` /
    /// `post_feedforward_layernorm` normalize each sublayer's OUTPUT before its
    /// residual add. Two norms per layer, both on the output side.
    OlmoPost,
}

/// Optional q/k normalization applied to the projected query / key before RoPE.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QkNorm {
    /// `true` normalizes each head independently over `head_dim` (Qwen3, Gemma3;
    /// weight shape `[head_dim]`). `false` normalizes the whole flattened
    /// projection over `n_q*head_dim` / `n_kv*head_dim` (OLMo2, OLMo3; weight
    /// shapes `[n_q*head_dim]` / `[n_kv*head_dim]`).
    pub per_head: bool,
    /// `true` uses Gemma's `(1 + weight)` RMSNorm (Gemma3); `false` the raw weight
    /// (Qwen3, OLMo2, OLMo3).
    pub one_plus: bool,
}

/// The checkpoint tensor-naming scheme (issue #499). Almost every Llama-family
/// checkpoint uses the standard HF layout
/// (`model.layers.{i}.self_attn.q_proj.weight`, `model.embed_tokens.weight`,
/// `model.norm.weight`); ExaOne 3.x instead keeps the original GPT-2-style names
/// (`transformer.h.{i}.attn.attention.q_proj.weight`, `transformer.wte.weight`,
/// `transformer.ln_f.weight`, and a `c_fc_0` / `c_fc_1` / `c_proj` gated MLP). The
/// scheme is a loader-only concern: it maps the emitter's fixed arg order to the
/// checkpoint's tensor names (`weight_names` in [`weight_names`](crate::weight_names)),
/// so it never changes an emitted graph and two configs that differ only in scheme
/// emit byte-for-byte identical StableHLO.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum WeightScheme {
    /// Standard HF Llama-family names (Llama, Qwen2, Gemma2, ERNIE-4.5, Seed-OSS,
    /// MiMo, InternLM3, Cohere, Phi3, StableLM, StarCoder2, Granite, MiniCPM, ...).
    #[default]
    Llama,
    /// ExaOne 3.x GPT-2-style names (`transformer.h.{i}...`, gated MLP `c_fc_0` /
    /// `c_fc_1` / `c_proj`, `out_proj` attention output).
    Exaone,
}

/// MLX affine weight quantization (`config.json` `quantization`). The linear /
/// embedding `*.weight` tensors are stored packed as `U32` with companion
/// `*.scales` / `*.biases`; the loader dequantizes them to f32 as
/// `q * scale + bias` per group of `group_size` input columns.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuantConfig {
    pub bits: usize,
    pub group_size: usize,
}

/// A Mixture-of-Experts FFN's always-on shared-expert branch (issue #500). A plain
/// SwiGLU MLP of `intermediate` hidden width, run on every token in parallel with
/// the routed experts and added to the routed output. Qwen2-MoE additionally gates
/// it by `sigmoid(x @ Wg^T)` (`gated = true`, a per-token scalar); DeepSeek adds it
/// ungated (`gated = false`). Mixtral has no shared expert.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SharedExpertConfig {
    /// The shared expert's SwiGLU intermediate width.
    pub intermediate: usize,
    /// Multiply the shared-expert output by `sigmoid(x @ Wg^T)` (Qwen2-MoE). When
    /// `false` the branch is added ungated (DeepSeek), and no gate weight is taken.
    pub gated: bool,
}

/// Mixture-of-Experts FFN parameters (issue #500). The router linear scores the
/// `n_experts`, a softmax over ALL experts forms routing probabilities, the top
/// `top_k` are selected, their probabilities are renormalized to sum to one when
/// `norm_topk_prob`, scaled by `routed_scaling_factor`, and used to combine the
/// selected experts' SwiGLU outputs (Mixtral / Qwen2-MoE / DeepSeek
/// `scoring_func = "softmax"`, softmax-before-top-k). Experts are a stacked SwiGLU
/// of `intermediate` width (mlx-lm `switch_mlp`, one `[n_experts, out, in]` tensor
/// per projection). `shared` is the shared-expert branch when the family has one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MoeConfig {
    /// Total number of routed experts (`num_experts` / `num_local_experts`).
    pub n_experts: usize,
    /// Experts selected per token (`num_experts_per_tok`).
    pub top_k: usize,
    /// Each routed expert's SwiGLU intermediate width (`moe_intermediate_size`;
    /// Mixtral reuses `intermediate_size`).
    pub intermediate: usize,
    /// Renormalize the selected top-k routing probabilities to sum to one
    /// (`norm_topk_prob`). Mixtral always does; Qwen2-MoE reads the flag.
    pub norm_topk_prob: bool,
    /// Scale applied to the routed combine weights (`routed_scaling_factor`, 1.0
    /// for Mixtral / Qwen2-MoE). Emitted only when not exactly 1.0.
    pub routed_scaling_factor: f64,
    /// The shared-expert branch, when the family has one (Qwen2-MoE / DeepSeek).
    pub shared: Option<SharedExpertConfig>,
    /// Layers with index `< first_k_dense` are ordinary dense SwiGLU MLP layers
    /// (DeepSeek `first_k_dense_replace`); 0 for an all-MoE model (Mixtral,
    /// Qwen2-MoE). A dense layer takes the dense `mlp.{gate,up,down}_proj` weights.
    pub first_k_dense: usize,
    /// The checkpoint's per-layer MoE weight-name prefix: `block_sparse_moe`
    /// (Mixtral) or `mlp` (Qwen2-MoE / DeepSeek). Read by the weight loader only.
    pub weight_prefix: &'static str,
}

#[derive(Clone, Debug)]
pub struct Config {
    pub hidden: usize,
    pub inter: usize,
    pub n_layers: usize,
    pub n_q: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    pub eps: f32,
    pub rope_theta: f64,
    pub vocab: usize,
    /// RoPE inverse-frequency scheme for the global (full-attention) layers.
    pub rope: RopeScaling,
    /// q/k/v projections carry a bias (Qwen2 hard-codes it; Cohere / Granite /
    /// StableLM / StarCoder2 / Seed-OSS / MiMo / InternLM3 read it from config).
    /// Qwen3 drops the bias.
    pub qkv_bias: bool,
    /// The LM head shares the token-embedding matrix (HF `tie_word_embeddings`).
    /// `true` reuses `params['embed']` for the final projection; `false` adds a
    /// separate `params['lm_head']` weight (Llama-3.1-8B, larger Qwen2.5, OLMo2/3,
    /// Phi3, StableLM, MiniCPM).
    pub tie_word_embeddings: bool,
    /// MLX affine weight quantization, if the checkpoint is quantized (`None` for
    /// an unquantized bf16/f16/f32 checkpoint). The graph runs in f32; the loader
    /// dequantizes the packed weights at load.
    pub quantization: Option<QuantConfig>,
    /// Scale the input embeddings by `sqrt(hidden)` (the Gemma family).
    pub embed_scale: bool,
    /// Use Gemma's `(1 + weight)` RMSNorm on the layer / final norms (the Gemma
    /// family). The q/k norm has its own `one_plus` flag in [`QkNorm`].
    pub norm_one_plus: bool,
    /// GeGLU (`gelu_pytorch_tanh`) MLP activation instead of SwiGLU (silu) (the
    /// Gemma family).
    pub mlp_geglu: bool,
    /// Per-layer RMSNorm placement (see [`NormStyle`]).
    pub norm_style: NormStyle,
    /// Optional q/k normalization before RoPE (Qwen3 / Gemma3 per-head, OLMo2 /
    /// OLMo3 flat). `None` for Llama / Qwen2 / Gemma1/2 / SmolLM3 / the #498 pack.
    pub qk_norm: Option<QkNorm>,
    /// Gemma3 (and OLMo3) local RoPE base for the sliding (local) layers: those
    /// layers build their RoPE table from this base while the global layers use
    /// `rope_theta`. `None` means every layer shares the single `rope` table.
    pub rope_local_base: Option<f64>,
    /// Gemma2 query pre-attention scale base: the attention score scale is
    /// `query_pre_attn_scalar^-0.5`. `None` uses `head_dim^-0.5` (unless
    /// `attention_multiplier` overrides it).
    pub query_pre_attn_scalar: Option<f64>,
    /// Gemma2 attention logit soft-cap: `softcap * tanh(scores / softcap)` on the
    /// pre-mask scores. `None` for the other families (Gemma3's is null).
    pub attn_logit_softcap: Option<f32>,
    /// Gemma2 final logit soft-cap on the LM-head logits. `None` otherwise.
    pub final_logit_softcap: Option<f32>,
    /// Sliding-window attention size: `Some(window)` makes the local layers attend
    /// only to the last `window` keys, while the global layers keep full context.
    /// `None` means every layer is global. The local/global schedule is set by
    /// [`Config::is_sliding_layer`] via [`sliding_pattern`](Self::sliding_pattern).
    pub sliding_window: Option<usize>,
    /// Sliding-window schedule period: layer `li` is global iff `(li+1) %
    /// sliding_pattern == 0`, otherwise local. Gemma2 uses 2 (even layers local);
    /// Gemma3 uses `sliding_window_pattern` (6, i.e. 5 local : 1 global); OLMo3
    /// uses 4; Cohere2 uses `sliding_window_pattern` (4). Only meaningful when
    /// `sliding_window` is `Some`.
    pub sliding_pattern: usize,
    /// Per-layer NoPE mask (SmolLM3): `use_rope_layers[li] == false` skips RoPE on
    /// that layer (`no_rope_layers`). `None` applies RoPE on every layer.
    pub use_rope_layers: Option<Vec<bool>>,
    /// Mixture-of-Experts FFN parameters (issues #500 / #501). `Some` for a MoE
    /// architecture (Mixtral, Qwen2-MoE, Qwen3-MoE, OLMoE); the FFN then routes the
    /// top-k of N experts instead of a dense MLP. `None` for a dense model, whose
    /// graphs are byte-for-byte unchanged (no MoE op is emitted).
    pub moe: Option<MoeConfig>,
    /// Checkpoint tensor-naming scheme (issue #499). Loader-only: it selects how
    /// [`weight_names`](crate::weight_names) maps the emitter's arg order onto the
    /// checkpoint tensors, so it never affects the emitted graph. `Llama` (the
    /// default) is the standard HF layout; `Exaone` is ExaOne 3.x's GPT-2-style
    /// names.
    pub weight_scheme: WeightScheme,

    // --- dense arch pack (issue #498): per-family deltas on the shared core ---
    /// The per-layer / final norms subtract the mean (true LayerNorm) rather than
    /// the RMSNorm the Llama family uses. `true` for Cohere/Cohere2
    /// (`CohereLayerNorm`) and StableLM/StarCoder2 (`nn.LayerNorm`). Llama / Qwen2 /
    /// Gemma keep RMSNorm (`false`), so their graphs are byte-identical.
    pub layernorm: bool,
    /// The per-layer / final norms carry an affine bias (`nn.LayerNorm` with
    /// `bias=True`). `true` for StableLM and StarCoder2; the emitter then takes and
    /// adds a per-norm bias arg. `false` (Cohere's bias-free `CohereLayerNorm`, and
    /// every RMSNorm arch) emits no bias op, so those graphs are unchanged.
    pub norm_bias: bool,
    /// Parallel attention + MLP block (Cohere/Cohere2): both sublayers read the one
    /// `input_layernorm` output and their results are summed into a single residual
    /// (`x + attn(ln(x)) + mlp(ln(x))`), so there is no `post_attention_layernorm`.
    /// `false` keeps the sequential two-residual Llama structure (byte-identical).
    pub parallel_block: bool,
    /// The `o_proj` output projection carries a bias (StarCoder2 `use_bias`).
    pub attn_o_bias: bool,
    /// The MLP projections carry biases (StarCoder2 `use_bias`; Granite `mlp_bias`).
    pub mlp_bias: bool,
    /// Dense (non-gated) MLP: `c_proj(act(c_fc(x)))` with a `gelu_tanh` activation
    /// and no gate projection (StarCoder2). `false` keeps the SwiGLU/GeGLU gated MLP.
    pub dense_mlp: bool,
    /// Interleaved ("traditional" / GPT-J) RoPE: adjacent dims `(2i, 2i+1)` rotate
    /// together (Cohere/Cohere2, `position_embedding_type = rope_gptj`). `false` is
    /// the half-split (GPT-NeoX / Llama) convention, so the Llama family is unchanged.
    pub rope_interleaved: bool,
    /// Partial-RoPE width: only the first `rotary_dim` of each head is rotated, the
    /// rest passes through (StableLM `partial_rotary_factor`). `None` rotates the
    /// full `head_dim` (Llama family), byte-identical.
    pub rotary_dim: Option<usize>,
    /// Apply RoPE only on the sliding-window (local) layers, leaving the
    /// full-attention layers position-free (Cohere2 NoPE on its every-`pattern`-th
    /// full layer). `false` applies RoPE on every layer (Llama family, Cohere v1).
    pub rope_on_sliding_only: bool,
    /// Attention score scale override: the raw multiplier applied to the scores
    /// (Granite `attention_multiplier`, which replaces `head_dim^-0.5`). `None`
    /// uses [`Config::scale`]'s default. See [`Config::scale`].
    pub attention_multiplier: Option<f64>,
    /// Input-embedding scalar multiply (Granite `embedding_multiplier`, MiniCPM
    /// `scale_emb`). `None` leaves the embeddings unscaled by a scalar. Distinct
    /// from the Gemma `embed_scale` `sqrt(hidden)` normalizer (both can apply).
    pub embedding_multiplier: Option<f32>,
    /// Per-sublayer residual scalar: each attention / MLP output is multiplied by
    /// this before its residual add (Granite `residual_multiplier`, MiniCPM
    /// `scale_depth / sqrt(num_layers)`). `None` adds the raw output (Llama family).
    /// Only applies to the sequential block (parallel-block archs carry no scalar).
    pub residual_multiplier: Option<f32>,
    /// Final-logit scalar multiply (Cohere `logit_scale`). `None` leaves the logits
    /// unscaled.
    pub logit_mul: Option<f32>,
    /// Final-logit scalar divide (Granite `logits_scaling`; MiniCPM's pre-head
    /// `hidden / dim_model_base` divide, equivalent since the head is bias-free).
    /// `None` leaves the logits unscaled.
    pub logit_div: Option<f32>,
    /// The checkpoint fuses q/k/v into one `qkv_proj` weight (Phi3): the loader
    /// splits it into the emitter's separate `wq`/`wk`/`wv` args, so the emitted
    /// graph is the standard separate-projection shape. `false` for every arch that
    /// ships separate projections. Consumed by the weight loader (`iree.rs`); the
    /// emitter graph is unaffected.
    pub fused_qkv: bool,
    /// The checkpoint fuses gate/up into one `gate_up_proj` weight (Phi3): the
    /// loader splits it (gate first, up second) into the emitter's `gate`/`up` args.
    /// Consumed by the weight loader (`iree.rs`); the emitter graph is unaffected.
    pub fused_gate_up: bool,
}

impl Config {
    /// Hard-coded Llama-3.2-1B-Instruct values (config.json of the spike model).
    pub fn llama_3_2_1b() -> Self {
        Config {
            hidden: 2048,
            inter: 8192,
            n_layers: 16,
            n_q: 32,
            n_kv: 8,
            head_dim: 64,
            eps: 1e-5,
            rope_theta: 500000.0,
            vocab: 128256,
            rope: RopeScaling::Llama3 {
                factor: 32.0,
                low_freq_factor: 1.0,
                high_freq_factor: 4.0,
                orig_ctx: 8192,
            },
            qkv_bias: false,
            tie_word_embeddings: true,
            quantization: None,
            embed_scale: false,
            norm_one_plus: false,
            mlp_geglu: false,
            norm_style: NormStyle::Plain,
            qk_norm: None,
            rope_local_base: None,
            query_pre_attn_scalar: None,
            attn_logit_softcap: None,
            final_logit_softcap: None,
            sliding_window: None,
            sliding_pattern: 2,
            use_rope_layers: None,
            moe: None,
            weight_scheme: WeightScheme::Llama,
            layernorm: false,
            norm_bias: false,
            parallel_block: false,
            attn_o_bias: false,
            mlp_bias: false,
            dense_mlp: false,
            rope_interleaved: false,
            rotary_dim: None,
            rope_on_sliding_only: false,
            attention_multiplier: None,
            embedding_multiplier: None,
            residual_multiplier: None,
            logit_mul: None,
            logit_div: None,
            fused_qkv: false,
            fused_gate_up: false,
        }
    }

    /// Build a [`Config`] from a model's `config.json` text.
    ///
    /// Scope: the dense architectures Llama, Qwen2, Qwen3, Gemma1, Gemma2, Gemma3,
    /// SmolLM3, OLMo2/3, Seed-OSS, MiMo, InternLM3, ExaOne, Cohere, Cohere2, Phi3,
    /// StableLM, StarCoder2, Granite, and MiniCPM (RMSNorm / LayerNorm variants,
    /// SwiGLU / GeGLU / dense MLP, GQA/MHA, tied or untied embeddings, optional q/k
    /// norm, sliding windows, NoPE, parallel blocks, interleaved / partial RoPE,
    /// scalar multipliers, fused projections). Configs the emitter cannot yet
    /// reproduce are rejected with a clear error rather than silently mis-emitted:
    /// an unsupported `model_type`, a `llama` checkpoint with `attention_bias`, an
    /// unsupported activation, an o_proj / MLP bias on an arch with no emit for it,
    /// a `rope_scaling` whose `rope_type` is not `llama3` (e.g. yarn), or MiniCPM3's
    /// MLA attention.
    pub fn from_json_str(s: &str) -> Result<Self, String> {
        let v: serde_json::Value =
            serde_json::from_str(s).map_err(|e| format!("parse config.json: {e}"))?;

        let model_type = v.get("model_type").and_then(serde_json::Value::as_str);

        // ExaOne 3.x keeps GPT-2-style tensor names; every other supported family
        // uses the standard HF Llama layout. Loader-only (see [`WeightScheme`]), so
        // it never changes the emitted graph.
        let weight_scheme = if model_type == Some("exaone") {
            WeightScheme::Exaone
        } else {
            WeightScheme::Llama
        };

        // Interleaved (GPT-J-style) RoPE reaches the supported families only through
        // the validated Cohere path. ERNIE-4.5 (`rotate_half` over `x[..., 0::2]` /
        // `x[..., 1::2]`) looks like a plain-RoPE Llama in config.json but is such an
        // arch, and it is not a validated target here, so it is rejected rather than
        // mis-emitted (a half-split emit is close but wrong).
        if model_type == Some("ernie4_5") {
            return Err(
                "the OpenXLA emitter uses half-split RoPE; ERNIE-4.5 (model_type \
                 ernie4_5) uses interleaved (GPT-J-style) RoPE, which is a follow-up \
                 (an interleaved-RoPE emit variant or a load-time q/k permutation)"
                    .to_string(),
            );
        }

        // MLX affine quantization: an optional `{bits, group_size}` block. The
        // loader dequantizes the packed weights; the emitted graph is unchanged.
        let quantization = match v.get("quantization") {
            None | Some(serde_json::Value::Null) => None,
            Some(q) => {
                let qu = |k: &str| -> Result<usize, String> {
                    q.get(k)
                        .and_then(serde_json::Value::as_u64)
                        .map(|x| x as usize)
                        .ok_or_else(|| format!("config.json quantization missing integer `{k}`"))
                };
                Some(QuantConfig {
                    bits: qu("bits")?,
                    group_size: qu("group_size")?,
                })
            }
        };

        let u = |k: &str| -> Result<usize, String> {
            v.get(k)
                .and_then(serde_json::Value::as_u64)
                .map(|x| x as usize)
                .ok_or_else(|| format!("config.json missing integer `{k}`"))
        };
        let f = |k: &str| -> Result<f64, String> {
            v.get(k)
                .and_then(serde_json::Value::as_f64)
                .ok_or_else(|| format!("config.json missing number `{k}`"))
        };
        let ob = |k: &str| -> Option<bool> { v.get(k).and_then(serde_json::Value::as_bool) };
        let of = |k: &str| -> Option<f64> { v.get(k).and_then(serde_json::Value::as_f64) };
        let ou = |k: &str| -> Option<usize> {
            v.get(k)
                .and_then(serde_json::Value::as_u64)
                .map(|x| x as usize)
        };
        // Some arches use alternate field names (ExaOne 3.x: `num_layers`), so this
        // reads the first present of a key list.
        let u_any = |keys: &[&str]| -> Result<usize, String> {
            keys.iter()
                .find_map(|k| v.get(*k).and_then(serde_json::Value::as_u64))
                .map(|x| x as usize)
                .ok_or_else(|| format!("config.json missing integer among {keys:?}"))
        };

        let hidden = u("hidden_size")?;
        let n_q = u("num_attention_heads")?;
        // head_dim is explicit in recent configs; otherwise it is hidden / heads.
        let head_dim = ou("head_dim").unwrap_or(hidden / n_q.max(1));
        // ExaOne 3.x uses `num_layers` in place of `num_hidden_layers`.
        let n_layers = u_any(&["num_hidden_layers", "num_layers"])?;

        // Norm epsilon: `rms_norm_eps` (RMSNorm archs), else `layer_norm_eps`
        // (Cohere / StableLM LayerNorm), else `layer_norm_epsilon` (ExaOne), else
        // `norm_epsilon` (StarCoder2).
        let eps = of("rms_norm_eps")
            .or_else(|| of("layer_norm_eps"))
            .or_else(|| of("layer_norm_epsilon"))
            .or_else(|| of("norm_epsilon"))
            .ok_or(
                "config.json missing a norm epsilon (rms_norm_eps / layer_norm_eps / \
                 layer_norm_epsilon / norm_epsilon)",
            )? as f32;

        // Architecture-family flags, defaulted to the Llama baseline and overridden
        // per model_type below. `tie_default` is the arch's `tie_word_embeddings`
        // default (an explicit config field always wins).
        let mut qkv_bias = false;
        let mut tie_default = true;
        let mut embed_scale = false;
        let mut norm_one_plus = false;
        let mut mlp_geglu = false;
        let mut norm_style = NormStyle::Plain;
        let mut qk_norm: Option<QkNorm> = None;
        let mut rope_local_base: Option<f64> = None;
        let mut query_pre_attn_scalar: Option<f64> = None;
        let mut attn_logit_softcap: Option<f32> = None;
        let mut final_logit_softcap: Option<f32> = None;
        let mut sliding_window: Option<usize> = None;
        let mut sliding_pattern = 2usize;
        let mut use_rope_layers: Option<Vec<bool>> = None;
        // issue #498 dense arch pack flags.
        let mut layernorm = false;
        let mut norm_bias = false;
        let mut parallel_block = false;
        let mut attn_o_bias = false;
        let mut mlp_bias = false;
        let mut dense_mlp = false;
        let mut rope_interleaved = false;
        let mut rotary_dim: Option<usize> = None;
        let mut rope_on_sliding_only = false;
        let mut attention_multiplier: Option<f64> = None;
        let mut embedding_multiplier: Option<f32> = None;
        let mut residual_multiplier: Option<f32> = None;
        let mut logit_mul: Option<f32> = None;
        let mut logit_div: Option<f32> = None;
        let mut fused_qkv = false;
        let mut fused_gate_up = false;

        // Read a Gemma family's soft-caps + query scale (shared by gemma2 / gemma3).
        // Gemma3's soft-caps are null, so this yields `None` there.
        let read_gemma_common =
            |qpa: &mut Option<f64>, asc: &mut Option<f32>, fsc: &mut Option<f32>| {
                *qpa = Some(of("query_pre_attn_scalar").unwrap_or(head_dim as f64));
                *asc = of("attn_logit_softcapping").map(|x| x as f32);
                *fsc = of("final_logit_softcapping").map(|x| x as f32);
            };

        // Partial-RoPE width from `partial_rotary_factor` (only when < 1).
        let partial_rotary = |default: f64| -> Option<usize> {
            let prf = of("partial_rotary_factor").unwrap_or(default);
            (prf < 1.0).then_some((head_dim as f64 * prf) as usize)
        };

        match model_type {
            Some("llama") | Some("minicpm") => {
                // A `llama` checkpoint with attention bias would need the Qwen2 bias
                // emit, untested here; reject rather than emit an unvalidated graph.
                if ob("attention_bias") == Some(true) {
                    return Err(
                        "the OpenXLA emitter does not support a `llama` checkpoint with \
                         attention_bias = true (only the bias-bearing dense arches carry a \
                         q/k/v bias here)"
                            .to_string(),
                    );
                }
                if model_type == Some("minicpm") {
                    tie_default = false;
                }
                // MiniCPM scalars (`scale_emb` / `scale_depth` / `dim_model_base`):
                // some MiniCPM checkpoints ship as `model_type = "llama"` but keep
                // these fields, so detect them by presence rather than model_type.
                if let Some(se) = of("scale_emb") {
                    embedding_multiplier = Some(se as f32);
                    if let Some(sd) = of("scale_depth") {
                        residual_multiplier = Some((sd / (n_layers as f64).sqrt()) as f32);
                    }
                    if let Some(dmb) = of("dim_model_base") {
                        // Dividing the pre-head hidden by hidden/dim_model_base is a
                        // logit divide (the LM head is bias-free).
                        logit_div = Some((hidden as f64 / dmb) as f32);
                    }
                }
            }
            Some("qwen2") => {
                qkv_bias = true;
            }
            Some("qwen3") => {
                // Qwen3 drops the Qwen2 bias and adds a per-head q/k RMSNorm (raw
                // weight, over head_dim) before RoPE.
                qk_norm = Some(QkNorm {
                    per_head: true,
                    one_plus: false,
                });
            }
            Some("gemma") => {
                // Gemma1: Llama-shaped norm placement, but embedding scale, `(1+w)`
                // RMSNorm, and a GeGLU MLP. No soft-caps, sliding, or q/k norm.
                embed_scale = true;
                norm_one_plus = true;
                mlp_geglu = true;
            }
            Some("gemma2") => {
                embed_scale = true;
                norm_one_plus = true;
                mlp_geglu = true;
                norm_style = NormStyle::GemmaFf;
                sliding_pattern = 2;
                sliding_window = Some(ou("sliding_window").unwrap_or(4096));
                read_gemma_common(
                    &mut query_pre_attn_scalar,
                    &mut attn_logit_softcap,
                    &mut final_logit_softcap,
                );
            }
            Some("gemma3") | Some("gemma3_text") => {
                embed_scale = true;
                norm_one_plus = true;
                mlp_geglu = true;
                norm_style = NormStyle::GemmaFf;
                // Gemma3: per-head `(1+w)` q/k norm, a 5:1 local:global schedule
                // (`sliding_window_pattern` = 6), and a local RoPE base for the
                // sliding layers (`rope_local_base_freq`) distinct from `rope_theta`.
                qk_norm = Some(QkNorm {
                    per_head: true,
                    one_plus: true,
                });
                sliding_pattern = ou("sliding_window_pattern").unwrap_or(6).max(1);
                sliding_window = Some(ou("sliding_window").unwrap_or(4096));
                rope_local_base = Some(of("rope_local_base_freq").unwrap_or(10000.0));
                read_gemma_common(
                    &mut query_pre_attn_scalar,
                    &mut attn_logit_softcap,
                    &mut final_logit_softcap,
                );
            }
            Some("smollm3") => {
                // SmolLM3: Llama-shaped, with a per-layer NoPE mask. HF stores
                // `no_rope_layers[li]` as 1 = use RoPE, 0 = NoPE, so it maps directly
                // to `use_rope_layers`.
                if let Some(arr) = v
                    .get("no_rope_layers")
                    .and_then(serde_json::Value::as_array)
                {
                    let flags: Vec<bool> = arr
                        .iter()
                        .map(|x| x.as_i64().map(|n| n != 0).unwrap_or(true))
                        .collect();
                    if flags.iter().any(|&b| !b) {
                        use_rope_layers = Some(flags);
                    }
                }
            }
            Some("olmo2") => {
                // OLMo2: reordered (post) norm and a FLAT q/k RMSNorm over the whole
                // projection (raw weight). No input_layernorm.
                norm_style = NormStyle::OlmoPost;
                qk_norm = Some(QkNorm {
                    per_head: false,
                    one_plus: false,
                });
            }
            Some("olmo3") => {
                // OLMo3: OLMo2 plus a sliding-window schedule. The full-size
                // checkpoint additionally uses yarn RoPE scaling, which the rope
                // block below rejects (a documented follow-up); a plain-RoPE OLMo3
                // config exercises the norm/qk/sliding structure.
                norm_style = NormStyle::OlmoPost;
                qk_norm = Some(QkNorm {
                    per_head: false,
                    one_plus: false,
                });
                if let Some(w) = ou("sliding_window") {
                    sliding_window = Some(w);
                    // OLMo3 marks every `sliding_window_pattern`-th layer global;
                    // the layer_types list (3 sliding : 1 full) implies a period 4.
                    sliding_pattern = ou("sliding_window_pattern").unwrap_or(4).max(1);
                }
                rope_local_base = of("rope_local_base_freq");
            }
            // issue #499 dense pack: each maps to a proven Llama / Qwen2 forward with
            // a config / naming delta and no new emit.
            Some("seed_oss") | Some("mimo") | Some("internlm3") => {
                // Bias-bearing dense forwards: Seed-OSS / MiMo expose the q/k/v bias
                // as `attention_bias`, InternLM3 as `qkv_bias`. MiMo's config
                // `sliding_window` is deliberately not read (served globally, as for
                // Qwen2); the rope block serves Seed-OSS `default` / InternLM3
                // `dynamic` as plain RoPE.
                qkv_bias = ob("attention_bias") == Some(true) || ob("qkv_bias") == Some(true);
            }
            Some("exaone") => {
                // ExaOne 3.x: llama3-RoPE Llama with GPT-2-style tensor names (see
                // `weight_scheme`) and the `num_layers` / `layer_norm_epsilon`
                // alternate config field names read elsewhere.
            }
            // issue #498 dense pack: the parallel-block and norm-variant families.
            Some("cohere") => {
                // LayerNorm (bias-free), parallel block, interleaved RoPE, tied,
                // final logit multiply. `attention_bias` (default false) applies to
                // q/k/v and o_proj alike.
                layernorm = true;
                parallel_block = true;
                rope_interleaved = true;
                let ab = ob("attention_bias").unwrap_or(false);
                qkv_bias = ab;
                attn_o_bias = ab;
                if ob("use_qk_norm") == Some(true) {
                    return Err(
                        "the OpenXLA emitter does not yet support Cohere `use_qk_norm = true` \
                         (per-head q/k LayerNorm is a follow-up)"
                            .to_string(),
                    );
                }
                logit_mul = Some(of("logit_scale").unwrap_or(0.0625) as f32);
            }
            Some("cohere2") => {
                layernorm = true;
                parallel_block = true;
                rope_interleaved = true;
                rope_on_sliding_only = true;
                let ab = ob("attention_bias").unwrap_or(false);
                qkv_bias = ab;
                attn_o_bias = ab;
                sliding_window = Some(ou("sliding_window").unwrap_or(4096));
                sliding_pattern = ou("sliding_window_pattern").unwrap_or(4).max(1);
                logit_mul = Some(of("logit_scale").unwrap_or(0.0625) as f32);
            }
            Some("phi3") => {
                // Fused qkv_proj / gate_up_proj (split at load); RMSNorm; untied.
                fused_qkv = true;
                fused_gate_up = true;
                tie_default = false;
                rotary_dim = partial_rotary(1.0);
            }
            Some("stablelm") => {
                // LayerNorm with bias, partial RoPE, optional q/k/v bias, untied.
                layernorm = true;
                norm_bias = true;
                tie_default = false;
                qkv_bias = ob("use_qkv_bias").unwrap_or(false);
                rotary_dim = partial_rotary(0.25);
                if ob("qk_layernorm") == Some(true) {
                    return Err("the OpenXLA emitter does not yet support StableLM \
                         `qk_layernorm = true` (per-head q/k LayerNorm is a follow-up)"
                        .to_string());
                }
                if ob("use_parallel_residual") == Some(true) {
                    parallel_block = true;
                }
            }
            Some("starcoder2") => {
                // LayerNorm with bias, biases on q/k/v/o and the dense (non-gated)
                // GELU MLP, tied.
                layernorm = true;
                norm_bias = true;
                dense_mlp = true;
                let ub = ob("use_bias").unwrap_or(true);
                qkv_bias = ub;
                attn_o_bias = ub;
                mlp_bias = ub;
            }
            Some("granite") => {
                // Llama shape + four scalar multipliers.
                let ab = ob("attention_bias").unwrap_or(false);
                qkv_bias = ab;
                attn_o_bias = ab;
                mlp_bias = ob("mlp_bias").unwrap_or(false);
                attention_multiplier = of("attention_multiplier");
                embedding_multiplier = of("embedding_multiplier").map(|x| x as f32);
                residual_multiplier = of("residual_multiplier").map(|x| x as f32);
                logit_div = of("logits_scaling").map(|x| x as f32);
            }
            Some("minicpm3") => {
                return Err(
                    "the OpenXLA emitter does not yet support MiniCPM3: its MLA \
                     attention (q/kv LoRA latent projections with separate nope/rope \
                     head dims) and LongRoPE are a follow-up to this dense arch pack \
                     (issue #498)"
                        .to_string(),
                );
            }
            // MoE families (issues #500 / #501). The attention flags are set here; the
            // MoE FFN itself is built by the `moe` block below. Mixtral is Llama-style
            // attention (no bias, plain RoPE); Qwen2-MoE is Qwen2 attention (q/k/v
            // bias); Qwen3-MoE is Qwen3 attention (per-head q/k RMSNorm, no bias);
            // OLMoE is standard pre-norm attention with a flat q/k RMSNorm. Families
            // whose attention or routing the shared core / MoE primitive does not yet
            // reproduce (DeepSeek MLA, PhiMoE sparsemixer, GLM4-MoE / dots1 grouped
            // sigmoid routing, ERNIE-4.5-MoE interleaved RoPE, gpt-oss attention sinks)
            // are deferred with a specific message rather than mis-emitted.
            Some("mixtral") => {}
            Some("qwen2_moe") => {
                qkv_bias = true;
            }
            Some("qwen3_moe") => {
                // Qwen3 attention: a per-head q/k RMSNorm (raw weight, over head_dim)
                // before RoPE, and no q/k/v bias. The MoE FFN is built below.
                qk_norm = Some(QkNorm {
                    per_head: true,
                    one_plus: false,
                });
            }
            Some("olmoe") => {
                // OLMoE attention: a FLAT q/k RMSNorm (raw weight, over the whole
                // projection, like OLMo2) before RoPE, on the standard pre-norm block.
                // `clip_qkv` (q/k/v projection clamping) is a follow-up: reject it
                // rather than silently drop the clamp.
                qk_norm = Some(QkNorm {
                    per_head: false,
                    one_plus: false,
                });
                if of("clip_qkv").is_some() {
                    return Err("the OpenXLA emitter does not yet support OLMoE \
                                `clip_qkv` (q/k/v projection clamping); it is a \
                                follow-up (#501)"
                        .to_string());
                }
            }
            Some("deepseek_v2") | Some("deepseek_v3") => {
                return Err(
                    "DeepSeek-V2/V3 use multi-head latent attention (compressed \
                            q/kv LoRA latent projections with a decoupled RoPE head), \
                            which the shared attention core does not reproduce; the MoE \
                            routing is supported but the MLA attention is a follow-up \
                            (#501)"
                        .to_string(),
                );
            }
            Some("phimoe") => {
                return Err("PhiMoE (Phi-3.5-MoE) routes with sparsemixer (a two-step \
                            masked top-2 selection), not the softmax-before-top-k the \
                            shared MoE FFN primitive emits; it is a follow-up (#501)"
                    .to_string());
            }
            Some("glm4_moe") => {
                return Err("GLM-4.5-MoE routes with sigmoid expert scores, grouped \
                            (n_group / topk_group) top-k, and a routed score-correction \
                            bias, not the softmax-before-top-k the shared MoE FFN \
                            primitive emits; it is a follow-up (#501)"
                    .to_string());
            }
            Some("dots1") => {
                return Err(
                    "dots.llm1 routes with sigmoid expert scores, grouped top-k, \
                            and a score-correction bias (DeepSeek-style), not the \
                            softmax-before-top-k the shared MoE FFN primitive emits; it \
                            is a follow-up (#501)"
                        .to_string(),
                );
            }
            Some("ernie4_5_moe") => {
                return Err("ERNIE-4.5-MoE uses interleaved (GPT-J-style) RoPE, which \
                            the half-split RoPE emit would get wrong, plus a routed \
                            score-correction bias; it is a follow-up (#501)"
                    .to_string());
            }
            Some("gpt_oss") => {
                return Err("gpt-oss uses attention sinks and a clamped, alpha-scaled \
                            gated-SwiGLU expert activation, neither in the shared \
                            attention core nor the MoE FFN primitive; it is a follow-up \
                            (#501)"
                    .to_string());
            }
            other => {
                return Err(format!(
                    "the OpenXLA emitter supports the dense architectures Llama, Qwen2, \
                     Qwen3, Gemma1/2/3, SmolLM3, OLMo2/3, Seed-OSS, MiMo, InternLM3, ExaOne, \
                     Cohere, Cohere2, Phi3, StableLM, StarCoder2, Granite, and MiniCPM, plus \
                     the Mixtral, Qwen2-MoE, Qwen3-MoE, and OLMoE mixture-of-experts \
                     architectures; config.json model_type = {other:?} (other MoE / MLA / \
                     novel-activation variants are follow-ups)"
                ));
            }
        }

        // Out-of-scope deltas an emit would get wrong are rejected rather than
        // silently dropped: an o_proj / MLP bias on an arch with no emit for it (the
        // #498 bias-bearing arches set the flag themselves and are exempt), and a
        // non-SwiGLU activation on a non-GeGLU, non-dense family (Gemma drives its
        // GeGLU with `mlp_geglu`, StarCoder2 its dense GELU with `dense_mlp`).
        if ob("attention_out_bias") == Some(true) && !attn_o_bias {
            return Err(
                "the OpenXLA emitter has no attention output (o_proj) bias for this \
                 architecture; config.json attention_out_bias = true is a follow-up"
                    .to_string(),
            );
        }
        if ob("mlp_bias") == Some(true) && !mlp_bias {
            return Err(
                "the OpenXLA emitter has no MLP bias for this architecture; \
                 config.json mlp_bias = true is a follow-up"
                    .to_string(),
            );
        }
        if !mlp_geglu && !dense_mlp {
            let act = v
                .get("hidden_act")
                .or_else(|| v.get("activation_function"))
                .and_then(serde_json::Value::as_str);
            if let Some(a) = act
                && a != "silu"
            {
                return Err(format!(
                    "the OpenXLA emitter emits a SwiGLU (silu) MLP for this architecture; \
                     config.json activation = {a:?} is unsupported"
                ));
            }
        }

        // rope_scaling is optional: absent -> plain RoPE (Qwen2.5, plain Llama).
        // When present, the supported schemes are llama3 (scaled) and, served as
        // plain RoPE, the `default` (identity) and in-context `dynamic` types;
        // anything else (e.g. yarn, which OLMo3 uses at full size) is a follow-up.
        let rope = match v.get("rope_scaling") {
            None | Some(serde_json::Value::Null) => RopeScaling::Plain,
            Some(scaling) => {
                let rope_type = scaling
                    .get("rope_type")
                    .or_else(|| scaling.get("type"))
                    .and_then(serde_json::Value::as_str);
                match rope_type {
                    // `default` is HF's identity rope (Seed-OSS). `dynamic` NTK
                    // (InternLM2/3) is identity within the original context and only
                    // rescales beyond it, so short / in-context generation is served
                    // as plain RoPE here (both use `rope_theta`); the long-context
                    // NTK rescale is a follow-up.
                    Some("default") | Some("dynamic") => RopeScaling::Plain,
                    Some("llama3") => {
                        let sf = |k: &str| -> Result<f64, String> {
                            scaling
                                .get(k)
                                .and_then(serde_json::Value::as_f64)
                                .ok_or_else(|| {
                                    format!("config.json rope_scaling missing number `{k}`")
                                })
                        };
                        let orig_ctx = scaling
                            .get("original_max_position_embeddings")
                            .and_then(serde_json::Value::as_u64)
                            .map(|x| x as usize)
                            .ok_or_else(|| {
                                "config.json rope_scaling missing \
                                 `original_max_position_embeddings`"
                                    .to_string()
                            })?;
                        RopeScaling::Llama3 {
                            factor: sf("factor")?,
                            low_freq_factor: sf("low_freq_factor")?,
                            high_freq_factor: sf("high_freq_factor")?,
                            orig_ctx,
                        }
                    }
                    other => {
                        return Err(format!(
                            "the OpenXLA emitter supports plain / default / (in-context) dynamic \
                             RoPE and llama3 RoPE scaling; config.json rope_scaling.rope_type = \
                             {other:?} (e.g. yarn is a follow-up)"
                        ));
                    }
                }
            }
        };

        // MoE FFN (issues #500 / #501). A recognized MoE `model_type` builds a
        // `MoeConfig`; the FFN then routes the top-k of N experts. The router does a
        // softmax over ALL experts BEFORE the top-k (`scoring_func = "softmax"`), so
        // the primitive is shared across Mixtral / Qwen2-MoE / Qwen3-MoE / OLMoE. Field
        // names differ per family (Mixtral `num_local_experts` + `intermediate_size`;
        // Qwen2/3-MoE `num_experts` + `moe_intermediate_size`; OLMoE `num_experts` +
        // `intermediate_size`), so each is read explicitly rather than guessed.
        let moe = match model_type {
            Some("mixtral") => Some(MoeConfig {
                n_experts: u("num_local_experts")?,
                top_k: u("num_experts_per_tok")?,
                // Mixtral's experts use `intermediate_size` (no `moe_intermediate_size`).
                intermediate: u("intermediate_size")?,
                // Mixtral always renormalizes the selected top-k routing weights.
                norm_topk_prob: true,
                routed_scaling_factor: 1.0,
                shared: None,
                first_k_dense: 0,
                weight_prefix: "block_sparse_moe",
            }),
            Some("qwen2_moe") => Some(MoeConfig {
                n_experts: u("num_experts")?,
                top_k: u("num_experts_per_tok")?,
                intermediate: u("moe_intermediate_size")?,
                norm_topk_prob: v
                    .get("norm_topk_prob")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false),
                routed_scaling_factor: 1.0,
                shared: Some(SharedExpertConfig {
                    intermediate: u("shared_expert_intermediate_size")?,
                    // Qwen2-MoE gates the shared expert by sigmoid(x @ Wg^T).
                    gated: true,
                }),
                first_k_dense: 0,
                weight_prefix: "mlp",
            }),
            Some("qwen3_moe") => {
                // The shipped Qwen3-MoE checkpoints are all-MoE (`decoder_sparse_step`
                // = 1, empty `mlp_only_layers`). An interleaved dense/MoE schedule is
                // not expressible by the leading-dense `first_k_dense` prefix, so it is
                // rejected here rather than mis-routed.
                let step = ou("decoder_sparse_step").unwrap_or(1);
                let mlp_only = v
                    .get("mlp_only_layers")
                    .and_then(serde_json::Value::as_array)
                    .is_some_and(|a| !a.is_empty());
                if step != 1 || mlp_only {
                    return Err("the OpenXLA emitter supports all-MoE Qwen3-MoE \
                                (decoder_sparse_step = 1, empty mlp_only_layers); an \
                                interleaved dense/MoE layer schedule is a follow-up \
                                (#501)"
                        .to_string());
                }
                Some(MoeConfig {
                    n_experts: u("num_experts")?,
                    top_k: u("num_experts_per_tok")?,
                    intermediate: u("moe_intermediate_size")?,
                    // Qwen3-MoE reads the renormalization flag (softmax-before-top-k).
                    norm_topk_prob: v
                        .get("norm_topk_prob")
                        .and_then(serde_json::Value::as_bool)
                        .unwrap_or(false),
                    routed_scaling_factor: 1.0,
                    // Qwen3-MoE has no shared expert.
                    shared: None,
                    first_k_dense: 0,
                    weight_prefix: "mlp",
                })
            }
            Some("olmoe") => Some(MoeConfig {
                n_experts: u("num_experts")?,
                top_k: u("num_experts_per_tok")?,
                // OLMoE's experts use `intermediate_size` (no `moe_intermediate_size`).
                intermediate: u("intermediate_size")?,
                norm_topk_prob: v
                    .get("norm_topk_prob")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false),
                routed_scaling_factor: 1.0,
                // OLMoE has no shared expert.
                shared: None,
                first_k_dense: 0,
                weight_prefix: "mlp",
            }),
            _ => None,
        };
        // Tied (share `embed` for the head) vs untied (separate `lm_head.weight`).
        // HF `PretrainedConfig` defaults this to `true`, but some arches default
        // untied (`tie_default`); an explicit config field always wins.
        let tie_word_embeddings = ob("tie_word_embeddings").unwrap_or(tie_default);

        Ok(Config {
            hidden,
            inter: u("intermediate_size")?,
            n_layers,
            n_q,
            n_kv: u("num_key_value_heads")?,
            head_dim,
            eps,
            rope_theta: f("rope_theta")?,
            vocab: u("vocab_size")?,
            rope,
            qkv_bias,
            tie_word_embeddings,
            quantization,
            embed_scale,
            norm_one_plus,
            mlp_geglu,
            norm_style,
            qk_norm,
            rope_local_base,
            query_pre_attn_scalar,
            attn_logit_softcap,
            final_logit_softcap,
            sliding_window,
            sliding_pattern,
            use_rope_layers,
            moe,
            weight_scheme,
            layernorm,
            norm_bias,
            parallel_block,
            attn_o_bias,
            mlp_bias,
            dense_mlp,
            rope_interleaved,
            rotary_dim,
            rope_on_sliding_only,
            attention_multiplier,
            embedding_multiplier,
            residual_multiplier,
            logit_mul,
            logit_div,
            fused_qkv,
            fused_gate_up,
        })
    }

    /// Read and parse a model's `config.json` from its directory.
    pub fn from_json(model_dir: &std::path::Path) -> Result<Self, String> {
        let p = model_dir.join("config.json");
        let s = std::fs::read_to_string(&p).map_err(|e| format!("read {}: {e}", p.display()))?;
        Self::from_json_str(&s).map_err(|e| format!("{}: {e}", p.display()))
    }

    pub fn group(&self) -> usize {
        self.n_q / self.n_kv
    }

    /// Whether the issue #516 packed in-graph dequant path can apply: an MLX
    /// affine-quantized checkpoint in the standard (non-fused-qkv, non-fused-gate-up,
    /// non-dense-MLP, non-MoE) Llama layout the v1 packed path supports. Combined
    /// with the `MLXCEL_XLA_QUANT=packed` opt-in (`builder::quant_in_graph`) by BOTH
    /// the emitter (`take_weight`) and the loader (`weights::weight_specs`), so they
    /// agree on which projections carry packed args and never diverge.
    pub(crate) fn supports_packed_quant(&self) -> bool {
        self.quantization.is_some()
            && !self.fused_qkv
            && !self.fused_gate_up
            && !self.dense_mlp
            && self.moe.is_none()
    }

    /// Attention score scale. Granite supplies the raw multiplier directly
    /// (`attention_multiplier`, which replaces `head_dim^-0.5`); Gemma2/3 use
    /// `query_pre_attn_scalar^-0.5` (computed in f64 to match HF, since it can
    /// differ from `head_dim`); most families use `head_dim^-0.5`. The Llama /
    /// Qwen2 branch is unchanged.
    pub fn scale(&self) -> f32 {
        if let Some(am) = self.attention_multiplier {
            return am as f32;
        }
        match self.query_pre_attn_scalar {
            Some(q) => q.powf(-0.5) as f32,
            None => (self.head_dim as f32).powf(-0.5),
        }
    }

    /// The RoPE rotation width: `rotary_dim` for a partial-RoPE arch (StableLM),
    /// else the full `head_dim` (Llama family). Always even.
    pub fn rotary_width(&self) -> usize {
        self.rotary_dim.unwrap_or(self.head_dim)
    }

    /// Gemma input-embedding normalizer `sqrt(hidden)` (computed in f64 then
    /// narrowed, matching HF's `hidden_size**0.5` cast to the activation dtype).
    pub fn embed_normalizer(&self) -> f32 {
        (self.hidden as f64).sqrt() as f32
    }

    /// Whether attention layer `li` uses sliding-window (local) attention. A
    /// windowed config marks layer `li` global iff `(li+1) % sliding_pattern == 0`,
    /// otherwise local (Gemma2 period 2 = even local; Gemma3 period 6 = 5 local : 1
    /// global; OLMo3 period 4; Cohere2 period `sliding_window_pattern`). A
    /// non-windowed config has no local layer, so its emitted graphs are unchanged.
    pub fn is_sliding_layer(&self, li: usize) -> bool {
        self.sliding_window.is_some() && !(li + 1).is_multiple_of(self.sliding_pattern.max(1))
    }

    /// Whether attention layer `li` applies RoPE at all. Every layer does unless the
    /// config carries a NoPE mask (SmolLM3) that clears it, or the arch rotates only
    /// its sliding (local) layers (Cohere2 NoPE on its full-attention layers).
    pub fn layer_uses_rope(&self, li: usize) -> bool {
        let masked = self
            .use_rope_layers
            .as_ref()
            .and_then(|v| v.get(li).copied())
            .unwrap_or(true);
        let sliding_ok = if self.rope_on_sliding_only {
            self.is_sliding_layer(li)
        } else {
            true
        };
        masked && sliding_ok
    }

    /// The local RoPE table base for the sliding (local) layers, when the config
    /// has a distinct one (Gemma3 / OLMo3). `None` means every layer shares the
    /// single global `rope` table.
    pub fn local_rope_layer(&self, li: usize) -> bool {
        self.rope_local_base.is_some() && self.is_sliding_layer(li)
    }

    /// The layer has an `input_layernorm` applied to the residual before attention
    /// (all styles except OLMo2/3's reordered post-norm).
    pub fn has_input_norm(&self) -> bool {
        self.norm_style != NormStyle::OlmoPost
    }

    /// The layer normalizes the attention OUTPUT before the residual add
    /// (`post_attention_layernorm` in Gemma2/3 and OLMo2/3).
    pub fn has_post_attn_norm(&self) -> bool {
        matches!(self.norm_style, NormStyle::GemmaFf | NormStyle::OlmoPost)
    }

    /// The layer has a `pre_feedforward_layernorm` before the MLP (Gemma2/3).
    pub fn has_pre_ff_norm(&self) -> bool {
        self.norm_style == NormStyle::GemmaFf
    }

    /// The layer normalizes the MLP OUTPUT before the residual add
    /// (`post_feedforward_layernorm` in Gemma2/3 and OLMo2/3).
    pub fn has_post_ff_norm(&self) -> bool {
        matches!(self.norm_style, NormStyle::GemmaFf | NormStyle::OlmoPost)
    }

    /// Whether layer `li` is a MoE FFN layer (issue #500). True for a MoE config on
    /// every layer at or past `first_k_dense` (the leading dense layers a family
    /// like DeepSeek keeps as ordinary MLP). A dense config returns `false` for
    /// every layer, so its graphs emit no MoE op and stay byte-for-byte unchanged.
    pub fn is_moe_layer(&self, li: usize) -> bool {
        self.moe.as_ref().is_some_and(|m| li >= m.first_k_dense)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ERNIE-4.5 is rejected with a message naming its interleaved (GPT-J-style)
    /// RoPE: it looks like a plain-RoPE Llama in config.json but its `rotate_half`
    /// rotates the (2i, 2i+1) pairs, not the (i, i+d/2) halves the Llama emit uses,
    /// so a half-split emit would be wrong. Deferred (it is not a validated target).
    #[test]
    fn rejects_ernie4_5_interleaved_rope() {
        let j = r#"{"model_type":"ernie4_5","hidden_size":1024,"intermediate_size":3072,
            "num_hidden_layers":18,"num_attention_heads":16,"num_key_value_heads":2,
            "head_dim":128,"rms_norm_eps":1e-5,"rope_theta":500000,"vocab_size":103424,
            "tie_word_embeddings":true,"hidden_act":"silu","use_bias":false}"#;
        let err = Config::from_json_str(j).expect_err("ernie4_5 is deferred");
        assert!(
            err.contains("interleaved"),
            "names the interleaved-RoPE reason: {err}"
        );
    }

    /// Seed-OSS parses to a Qwen2-style bias forward: `attention_bias = true` turns
    /// on the q/k/v bias, `rope_type = "default"` is served as plain RoPE, and it is
    /// untied. `attention_out_bias = false` is accepted (only `true` is rejected).
    #[test]
    fn parses_seed_oss_as_qkv_bias_default_rope() {
        let j = r#"{"model_type":"seed_oss","hidden_size":5120,"intermediate_size":27648,
            "num_hidden_layers":64,"num_attention_heads":80,"num_key_value_heads":8,
            "head_dim":128,"rms_norm_eps":1e-6,"rope_theta":1e7,"vocab_size":155136,
            "tie_word_embeddings":false,"attention_bias":true,"attention_out_bias":false,
            "rope_scaling":{"rope_type":"default"},"hidden_act":"silu"}"#;
        let c = Config::from_json_str(j).expect("seed_oss parses");
        assert!(c.qkv_bias, "attention_bias=true -> q/k/v bias");
        assert_eq!(c.rope, RopeScaling::Plain, "rope_type default -> plain");
        assert!(!c.tie_word_embeddings, "seed_oss is untied");
    }

    /// MiMo parses to a Qwen2-style bias forward, and its config `sliding_window`
    /// is ignored (served globally, as for Qwen2), so it parses to `None`.
    #[test]
    fn parses_mimo_qkv_bias_ignores_sliding_window() {
        let j = r#"{"model_type":"mimo","hidden_size":4096,"intermediate_size":11008,
            "num_hidden_layers":36,"num_attention_heads":32,"num_key_value_heads":8,
            "head_dim":128,"rms_norm_eps":1e-5,"rope_theta":640000,"vocab_size":151680,
            "tie_word_embeddings":false,"attention_bias":true,"sliding_window":32768,
            "use_sliding_window":true,"hidden_act":"silu"}"#;
        let c = Config::from_json_str(j).expect("mimo parses");
        assert!(c.qkv_bias);
        assert_eq!(c.rope, RopeScaling::Plain);
        assert_eq!(
            c.sliding_window, None,
            "non-gemma2 sliding_window is ignored"
        );
    }

    /// InternLM3 parses to a plain-RoPE untied Llama: `rope_type = "dynamic"` is
    /// served as plain (in-context), and `qkv_bias` drives the bias (false here).
    #[test]
    fn parses_internlm3_dynamic_rope_as_plain() {
        let j = r#"{"model_type":"internlm3","hidden_size":4096,"intermediate_size":10240,
            "num_hidden_layers":48,"num_attention_heads":32,"num_key_value_heads":2,
            "head_dim":128,"rms_norm_eps":1e-5,"rope_theta":50000000,"vocab_size":128512,
            "tie_word_embeddings":false,"qkv_bias":false,
            "rope_scaling":{"rope_type":"dynamic","factor":6.0},"hidden_act":"silu"}"#;
        let c = Config::from_json_str(j).expect("internlm3 parses");
        assert_eq!(c.rope, RopeScaling::Plain, "dynamic -> plain (in-context)");
        assert!(!c.qkv_bias, "qkv_bias=false");
        assert!(!c.tie_word_embeddings);
        // A `qkv_bias = true` internlm3 turns the bias on.
        let biased = j.replace("\"qkv_bias\":false", "\"qkv_bias\":true");
        assert!(Config::from_json_str(&biased).unwrap().qkv_bias);
    }

    /// ExaOne 3.x parses to a llama3-RoPE tied Llama with the ExaOne weight scheme,
    /// reading the alternate field names (`num_layers`, `layer_norm_epsilon`).
    #[test]
    fn parses_exaone_alt_fields_and_scheme() {
        let j = r#"{"model_type":"exaone","hidden_size":2560,"intermediate_size":7168,
            "num_layers":30,"num_attention_heads":32,"num_key_value_heads":8,"head_dim":80,
            "layer_norm_epsilon":1e-5,"rope_theta":1000000,"vocab_size":102400,
            "tie_word_embeddings":true,"activation_function":"silu",
            "rope_scaling":{"rope_type":"llama3","factor":8.0,"low_freq_factor":1.0,
            "high_freq_factor":4.0,"original_max_position_embeddings":8192}}"#;
        let c = Config::from_json_str(j).expect("exaone parses");
        assert_eq!(c.weight_scheme, WeightScheme::Exaone);
        assert_eq!(c.n_layers, 30, "num_layers -> n_layers");
        assert_eq!(c.eps, 1e-5, "layer_norm_epsilon -> eps");
        assert_eq!(c.head_dim, 80);
        assert!(c.tie_word_embeddings);
        assert!(matches!(c.rope, RopeScaling::Llama3 { factor, .. } if factor == 8.0));
    }

    /// Unsupported deltas are rejected with a clear message rather than mis-emitted:
    /// an attention output bias, an MLP bias, a non-SwiGLU activation, an
    /// unsupported rope type (yarn), and an unsupported `model_type`.
    #[test]
    fn rejects_out_of_scope_dense_deltas() {
        let base = |extra: &str| {
            format!(
                r#"{{"model_type":"seed_oss","hidden_size":8,"intermediate_size":16,
                "num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":1,
                "rms_norm_eps":1e-6,"rope_theta":1e4,"vocab_size":10{extra}}}"#
            )
        };
        assert!(
            Config::from_json_str(&base(",\"attention_out_bias\":true"))
                .unwrap_err()
                .contains("o_proj"),
            "o_proj bias rejected"
        );
        assert!(
            Config::from_json_str(&base(",\"mlp_bias\":true"))
                .unwrap_err()
                .contains("MLP bias"),
            "mlp bias rejected"
        );
        assert!(
            Config::from_json_str(&base(",\"hidden_act\":\"gelu\""))
                .unwrap_err()
                .contains("SwiGLU"),
            "non-silu activation rejected"
        );
        assert!(
            Config::from_json_str(&base(",\"rope_scaling\":{\"rope_type\":\"yarn\"}"))
                .unwrap_err()
                .contains("yarn"),
            "yarn rope rejected"
        );
        // An architecture the emitter cannot reproduce (MoE / MLA glm4 variant).
        let glm = r#"{"model_type":"glm4_moe_lite","hidden_size":8,"intermediate_size":16,
            "num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":1,
            "rms_norm_eps":1e-5,"rope_theta":1e4,"vocab_size":10}"#;
        assert!(
            Config::from_json_str(glm)
                .unwrap_err()
                .contains("model_type"),
            "unsupported model_type rejected"
        );
    }
}
