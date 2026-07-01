//! Emitter config for the Llama-family architectures the OpenXLA backend serves.
//! The hard-coded [`Config::llama_3_2_1b`] matches spike/openxla/model_jax.py;
//! [`Config::from_json`] reads the same shape from a checkpoint's `config.json`
//! (issue #449 M3 Stage 2d). Stage A covered the Llama architecture (llama3 RoPE,
//! no attention bias); Stage B adds Qwen2 (plain RoPE + QKV bias), so the config
//! carries the architecture switches the emitter branches on: the RoPE kind,
//! whether q/k/v projections have a bias, and whether the LM head is tied to the
//! token embedding (tied) or a separate `lm_head.weight` (untied, e.g.
//! Llama-3.1-8B and the larger Qwen2.5 checkpoints).

/// How the RoPE inverse-frequency table is computed. Both kinds share the
/// `outer(pos, inv_freq)` table build (see [`rope`](super::rope)); they differ
/// only in `inv_freq`.
#[derive(Clone, Debug, PartialEq)]
pub enum RopeScaling {
    /// Plain RoPE: `inv_freq[i] = 1 / theta^(2i/head_dim)` (Qwen2, and plain-RoPE
    /// Llama without a `rope_scaling` block).
    Plain,
    /// Llama3 RoPE scaling, byte-for-byte with HF `_compute_llama3_parameters`.
    Llama3 {
        factor: f64,
        low_freq_factor: f64,
        high_freq_factor: f64,
        orig_ctx: usize,
    },
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
    /// RoPE inverse-frequency scheme (Stage B: `Plain` for Qwen2).
    pub rope: RopeScaling,
    /// q/k/v projections carry a bias (Qwen2). `o_proj` never does, and the MLP
    /// projections never do, so this single switch covers the architecture delta.
    pub qkv_bias: bool,
    /// The LM head shares the token-embedding matrix (HF `tie_word_embeddings`).
    /// `true` (Llama-3.2-1B, Qwen2.5-0.5B) reuses `params['embed']` for the final
    /// projection; `false` adds a separate `params['lm_head']` weight the tail
    /// projects through instead (Llama-3.1-8B, larger Qwen2.5 sizes).
    pub tie_word_embeddings: bool,
    /// MLX affine weight quantization, if the checkpoint is quantized (`None` for
    /// an unquantized bf16/f16/f32 checkpoint). The graph itself is unchanged (it
    /// runs in f32); the loader dequantizes the packed weights at load.
    pub quantization: Option<QuantConfig>,
    /// Gemma2 architecture switch. When true the emitter scales the input
    /// embeddings by `sqrt(hidden)`, uses `(1 + weight)` RMSNorm, a GeGLU
    /// (`gelu_tanh`) MLP, a post-norm on each sublayer (four norms per layer), and
    /// attention / final logit soft-capping; `o_proj` is non-square
    /// (`n_q*head_dim != hidden`). Llama / Qwen2 keep their existing path.
    pub gemma2: bool,
    /// Gemma2 query pre-attention scale base: the attention score scale is
    /// `query_pre_attn_scalar^-0.5` (Gemma2; can differ from `head_dim`). `None`
    /// uses `head_dim^-0.5` (Llama / Qwen2).
    pub query_pre_attn_scalar: Option<f64>,
    /// Gemma2 attention logit soft-cap: `softcap * tanh(scores / softcap)` on the
    /// pre-mask scores. `None` for Llama / Qwen2.
    pub attn_logit_softcap: Option<f32>,
    /// Gemma2 final logit soft-cap on the LM-head logits. `None` for Llama / Qwen2.
    pub final_logit_softcap: Option<f32>,
    /// Gemma2 sliding-window attention (issue #495): `Some(window)` makes the
    /// local (even) layers attend only to the last `window` keys, while the
    /// global (odd) layers keep full-context attention. Read from `config.json`'s
    /// `sliding_window` (HF Gemma2 default 4096) for a gemma2 checkpoint. `None`
    /// for Llama / Qwen2, whose every layer is global; the emitter then emits no
    /// window ops, so those graphs are byte-identical. (Qwen2's own
    /// `sliding_window` field is deliberately ignored: the emitter serves Qwen2
    /// with `use_sliding_window = false` semantics.)
    pub sliding_window: Option<usize>,

    // --- dense arch pack (issue #498): per-family deltas on the shared core ---
    /// The per-layer norms subtract the mean (true LayerNorm) rather than the
    /// RMSNorm the Llama family uses. `true` for Cohere/Cohere2 (`CohereLayerNorm`)
    /// and StableLM/StarCoder2 (`nn.LayerNorm`). Llama / Qwen2 / Gemma2 keep
    /// RMSNorm (`false`), so their graphs are byte-identical.
    pub layernorm: bool,
    /// The per-layer norms carry an affine bias (`nn.LayerNorm` with `bias=True`).
    /// `true` for StableLM and StarCoder2; the emitter then takes and adds a
    /// per-norm bias arg. `false` (Cohere's bias-free `CohereLayerNorm`, and every
    /// RMSNorm arch) emits no bias op, so those graphs are unchanged.
    pub norm_bias: bool,
    /// Parallel attention + MLP block (Cohere/Cohere2): both sublayers read the one
    /// `input_layernorm` output and their results are summed into a single residual
    /// (`x + attn(ln(x)) + mlp(ln(x))`), so there is no `post_attention_layernorm`.
    /// `false` keeps the sequential two-residual Llama structure (byte-identical).
    pub parallel_block: bool,
    /// The `o_proj` output projection carries a bias (StarCoder2 `use_bias`).
    pub attn_o_bias: bool,
    /// The MLP projections carry biases (StarCoder2 `use_bias`; Granite `mlp_bias`
    /// is false).
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
    /// Sliding-window layer schedule period: a layer is local (sliding) iff
    /// `(li + 1) % pattern != 0` (Cohere2 `sliding_window_pattern`). `None` with a
    /// window uses Gemma2's even-local alternation; without a window there are no
    /// local layers.
    pub sliding_pattern: Option<usize>,
    /// Attention score scale override: the raw multiplier applied to the scores
    /// (Granite `attention_multiplier`, which replaces `head_dim^-0.5`). `None`
    /// uses [`Config::scale`]'s default. See [`Config::scale`].
    pub attention_multiplier: Option<f64>,
    /// Input-embedding scalar multiply (Granite `embedding_multiplier`, MiniCPM
    /// `scale_emb`). `None` leaves the embeddings unscaled (Llama family).
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
            gemma2: false,
            query_pre_attn_scalar: None,
            attn_logit_softcap: None,
            final_logit_softcap: None,
            sliding_window: None,
            layernorm: false,
            norm_bias: false,
            parallel_block: false,
            attn_o_bias: false,
            mlp_bias: false,
            dense_mlp: false,
            rope_interleaved: false,
            rotary_dim: None,
            rope_on_sliding_only: false,
            sliding_pattern: None,
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
    /// Scope: the Llama and Qwen2 architectures (RMSNorm, SwiGLU MLP, GQA, tied or
    /// untied embeddings). Llama uses llama3 RoPE scaling and no attention bias;
    /// Qwen2 uses plain RoPE and a q/k/v projection bias; either may tie its LM
    /// head to the token embedding or carry a separate `lm_head.weight`. Configs
    /// the emitter cannot yet reproduce are rejected with a clear error rather than
    /// silently mis-emitted: an unsupported `model_type`, a `llama` checkpoint with
    /// `attention_bias`, or a `rope_scaling` whose `rope_type` is not `llama3`.
    pub fn from_json_str(s: &str) -> Result<Self, String> {
        let v: serde_json::Value =
            serde_json::from_str(s).map_err(|e| format!("parse config.json: {e}"))?;

        let model_type = v.get("model_type").and_then(serde_json::Value::as_str);

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

        // Required and optional field readers.
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

        let hidden = u("hidden_size")?;
        let n_q = u("num_attention_heads")?;
        let n_layers = u("num_hidden_layers")?;
        // head_dim is explicit in recent configs; otherwise it is hidden / heads.
        let head_dim = ou("head_dim").unwrap_or(hidden / n_q.max(1));

        // Norm epsilon: `rms_norm_eps` (RMSNorm archs), else `layer_norm_eps`
        // (Cohere / StableLM LayerNorm), else `norm_epsilon` (StarCoder2).
        let eps = of("rms_norm_eps")
            .or_else(|| of("layer_norm_eps"))
            .or_else(|| of("norm_epsilon"))
            .ok_or(
                "config.json missing a norm epsilon (rms_norm_eps / layer_norm_eps / norm_epsilon)",
            )? as f32;

        // rope_scaling is optional: absent -> plain RoPE (Qwen2.5, plain Llama);
        // present -> only the llama3 scheme is supported (Stage A).
        let rope = match v.get("rope_scaling") {
            None | Some(serde_json::Value::Null) => RopeScaling::Plain,
            Some(scaling) => {
                let rope_type = scaling
                    .get("rope_type")
                    .or_else(|| scaling.get("type"))
                    .and_then(serde_json::Value::as_str);
                if rope_type != Some("llama3") {
                    return Err(format!(
                        "the OpenXLA emitter supports plain RoPE and llama3 RoPE scaling; \
                         config.json rope_scaling.rope_type = {rope_type:?} (e.g. yarn / \
                         longrope are a follow-up)"
                    ));
                }
                let sf = |k: &str| -> Result<f64, String> {
                    scaling
                        .get(k)
                        .and_then(serde_json::Value::as_f64)
                        .ok_or_else(|| format!("config.json rope_scaling missing number `{k}`"))
                };
                let orig_ctx = scaling
                    .get("original_max_position_embeddings")
                    .and_then(serde_json::Value::as_u64)
                    .map(|x| x as usize)
                    .ok_or_else(|| {
                        "config.json rope_scaling missing `original_max_position_embeddings`"
                            .to_string()
                    })?;
                RopeScaling::Llama3 {
                    factor: sf("factor")?,
                    low_freq_factor: sf("low_freq_factor")?,
                    high_freq_factor: sf("high_freq_factor")?,
                    orig_ctx,
                }
            }
        };

        // Per-architecture deltas start at the Llama defaults and are overridden by
        // the arch arm. `tie_default` is the arch's `tie_word_embeddings` default
        // (an explicit config field always wins).
        let mut qkv_bias = false;
        let mut tie_default = true;
        let mut gemma2 = false;
        let mut query_pre_attn_scalar = None;
        let mut attn_logit_softcap = None;
        let mut final_logit_softcap = None;
        let mut sliding_window = None;
        let mut layernorm = false;
        let mut norm_bias = false;
        let mut parallel_block = false;
        let mut attn_o_bias = false;
        let mut mlp_bias = false;
        let mut dense_mlp = false;
        let mut rope_interleaved = false;
        let mut rotary_dim = None;
        let mut rope_on_sliding_only = false;
        let mut sliding_pattern = None;
        let mut attention_multiplier = None;
        let mut embedding_multiplier = None;
        let mut residual_multiplier = None;
        let mut logit_mul = None;
        let mut logit_div = None;
        let mut fused_qkv = false;
        let mut fused_gate_up = false;

        // Partial-RoPE width from `partial_rotary_factor` (only when < 1).
        let partial_rotary = |default: f64| -> Option<usize> {
            let prf = of("partial_rotary_factor").unwrap_or(default);
            (prf < 1.0).then_some((head_dim as f64 * prf) as usize)
        };

        match model_type {
            Some("llama") | Some("minicpm") => {
                // A `llama` checkpoint with attention bias would need the Qwen2
                // bias emit, untested here, so reject it rather than mis-emit.
                if ob("attention_bias") == Some(true) {
                    return Err(
                        "the OpenXLA emitter does not support a `llama` checkpoint with \
                         attention_bias = true (only Qwen2 carries a q/k/v bias here)"
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
                // Qwen2 hard-codes a q/k/v projection bias (HF `Qwen2Attention`).
                qkv_bias = true;
            }
            Some("gemma2") => {
                gemma2 = true;
                query_pre_attn_scalar =
                    Some(of("query_pre_attn_scalar").unwrap_or(head_dim as f64));
                attn_logit_softcap = of("attn_logit_softcapping").map(|x| x as f32);
                final_logit_softcap = of("final_logit_softcapping").map(|x| x as f32);
                sliding_window = Some(ou("sliding_window").unwrap_or(4096));
            }
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
                sliding_pattern = Some(ou("sliding_window_pattern").unwrap_or(4));
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
            other => {
                return Err(format!(
                    "the OpenXLA emitter supports the Llama, Qwen2, Gemma2, Cohere, Cohere2, \
                     Phi3, StableLM, StarCoder2, Granite, and MiniCPM architectures; \
                     config.json model_type = {other:?}"
                ));
            }
        }

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
            gemma2,
            query_pre_attn_scalar,
            attn_logit_softcap,
            final_logit_softcap,
            sliding_window,
            layernorm,
            norm_bias,
            parallel_block,
            attn_o_bias,
            mlp_bias,
            dense_mlp,
            rope_interleaved,
            rotary_dim,
            rope_on_sliding_only,
            sliding_pattern,
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

    /// Attention score scale. Granite supplies the raw multiplier directly
    /// (`attention_multiplier`, which replaces `head_dim^-0.5`); Gemma2 uses
    /// `query_pre_attn_scalar^-0.5` (computed in f64 to match HF, since it can
    /// differ from `head_dim`); Llama / Qwen2 / Cohere / others use `head_dim^-0.5`.
    /// The Llama / Qwen2 branch is unchanged.
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

    /// Whether RoPE is applied on attention layer `li`. Cohere2 leaves its
    /// full-attention layers position-free (NoPE), rotating only the sliding
    /// (local) layers; every other arch rotates every layer.
    pub fn rope_applies_layer(&self, li: usize) -> bool {
        if self.rope_on_sliding_only {
            self.is_sliding_layer(li)
        } else {
            true
        }
    }

    /// Gemma2 input-embedding normalizer `sqrt(hidden)` (computed in f64 then
    /// narrowed, matching HF's `hidden_size**0.5` cast to the activation dtype).
    pub fn embed_normalizer(&self) -> f32 {
        (self.hidden as f64).sqrt() as f32
    }

    /// Whether attention layer `li` uses sliding-window (local) attention (issue
    /// #495). Gemma2 alternates local and global starting local, so even layers
    /// (0, 2, 4, …) are local (`is_sliding = not bool(layer_idx % 2)`). Cohere2
    /// (issue #498) instead makes every `sliding_pattern`-th layer full-attention:
    /// a layer is local iff `(li + 1) % pattern != 0`, matching HF
    /// `Cohere2Config.layer_types`. Only a config with a sliding window has local
    /// layers; Llama / Qwen2 return `false` for every layer, unchanged.
    pub fn is_sliding_layer(&self, li: usize) -> bool {
        if self.sliding_window.is_none() {
            return false;
        }
        match self.sliding_pattern {
            Some(p) => !(li + 1).is_multiple_of(p),
            None => li.is_multiple_of(2),
        }
    }
}
