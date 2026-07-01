//! Emitter config for the dense architectures the OpenXLA backend serves.
//! The hard-coded [`Config::llama_3_2_1b`] matches spike/openxla/model_jax.py;
//! [`Config::from_json`] reads the same shape from a checkpoint's `config.json`.
//!
//! The config carries orthogonal architecture switches the emitter branches on,
//! so a new dense family is a combination of flags rather than a new code path:
//! the RoPE kind (llama3 vs plain, plus an optional per-layer local base for
//! Gemma3), whether q/k/v projections carry a bias (Qwen2), the LM-head tie, MLX
//! quantization, the Gemma embedding scale / `(1+w)` RMSNorm / GeGLU MLP, the
//! per-layer norm placement ([`NormStyle`]), an optional q/k normalization
//! ([`QkNorm`]: per-head for Qwen3 / Gemma3, flat for OLMo2 / OLMo3), the
//! sliding-window schedule (window + pattern period), and a per-layer NoPE mask
//! (SmolLM3). Llama / Qwen2 / Gemma2 keep their exact previous flag combinations,
//! so their emitted graphs are byte-for-byte unchanged.

/// How the RoPE inverse-frequency table is computed. Both kinds share the
/// `outer(pos, inv_freq)` table build (see [`rope`](super::rope)); they differ
/// only in `inv_freq`.
#[derive(Clone, Debug, PartialEq)]
pub enum RopeScaling {
    /// Plain RoPE: `inv_freq[i] = 1 / theta^(2i/head_dim)` (Qwen2, Qwen3, Gemma,
    /// SmolLM3, OLMo2, and plain-RoPE Llama without a `rope_scaling` block).
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
    /// Llama / Qwen2 / Qwen3 / Gemma1 / SmolLM3: pre-norm. `input_layernorm`
    /// normalizes the residual before attention; `post_attention_layernorm`
    /// normalizes it before the MLP. Two norms per layer, both on the input side.
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
    /// RoPE inverse-frequency scheme for the global (full-attention) layers.
    pub rope: RopeScaling,
    /// q/k/v projections carry a bias (Qwen2 only; the HF `Qwen2Attention` hard-
    /// codes `bias=True`). `o_proj` and the MLP projections never do. Qwen3 drops
    /// the bias.
    pub qkv_bias: bool,
    /// The LM head shares the token-embedding matrix (HF `tie_word_embeddings`).
    /// `true` reuses `params['embed']` for the final projection; `false` adds a
    /// separate `params['lm_head']` weight (Llama-3.1-8B, larger Qwen2.5, OLMo2/3).
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
    /// OLMo3 flat). `None` for Llama / Qwen2 / Gemma1/2 / SmolLM3.
    pub qk_norm: Option<QkNorm>,
    /// Gemma3 (and OLMo3) local RoPE base for the sliding (local) layers: those
    /// layers build their RoPE table from this base while the global layers use
    /// `rope_theta`. `None` means every layer shares the single `rope` table.
    pub rope_local_base: Option<f64>,
    /// Gemma2 query pre-attention scale base: the attention score scale is
    /// `query_pre_attn_scalar^-0.5`. `None` uses `head_dim^-0.5` (Llama / Qwen2 /
    /// Qwen3 / Gemma1 / SmolLM3 / OLMo2/3).
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
    /// uses 4. Only meaningful when `sliding_window` is `Some`.
    pub sliding_pattern: usize,
    /// Per-layer NoPE mask (SmolLM3): `use_rope_layers[li] == false` skips RoPE on
    /// that layer (`no_rope_layers`). `None` applies RoPE on every layer.
    pub use_rope_layers: Option<Vec<bool>>,
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
        }
    }

    /// Build a [`Config`] from a model's `config.json` text.
    ///
    /// Scope: the dense architectures Llama, Qwen2, Qwen3, Gemma1, Gemma2, Gemma3,
    /// SmolLM3, and OLMo2/3 (RMSNorm variants, SwiGLU / GeGLU MLP, GQA/MHA, tied or
    /// untied embeddings, optional q/k norm, sliding windows, NoPE). Configs the
    /// emitter cannot yet reproduce are rejected with a clear error rather than
    /// silently mis-emitted: an unsupported `model_type`, a `llama` checkpoint with
    /// `attention_bias`, or a `rope_scaling` whose `rope_type` is not `llama3`
    /// (e.g. yarn, which OLMo3 uses at full size).
    pub fn from_json_str(s: &str) -> Result<Self, String> {
        let v: serde_json::Value =
            serde_json::from_str(s).map_err(|e| format!("parse config.json: {e}"))?;

        let model_type = v.get("model_type").and_then(serde_json::Value::as_str);

        // Tied (share `embed` for the head) vs untied (separate `lm_head.weight`).
        // HF `PretrainedConfig` defaults this to `true`, so an absent field means
        // tied; the emitter and the weight loader branch on it.
        let tie_word_embeddings = v
            .get("tie_word_embeddings")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(true);

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
        let of64 = |k: &str| -> Option<f64> { v.get(k).and_then(serde_json::Value::as_f64) };
        let ou = |k: &str| -> Option<usize> {
            v.get(k)
                .and_then(serde_json::Value::as_u64)
                .map(|x| x as usize)
        };

        let hidden = u("hidden_size")?;
        let n_q = u("num_attention_heads")?;
        // head_dim is explicit in recent configs; otherwise it is hidden / heads.
        let head_dim = ou("head_dim").unwrap_or(hidden / n_q.max(1));
        let n_layers = u("num_hidden_layers")?;

        // Architecture-family flags, defaulted to the Llama baseline and overridden
        // per model_type below.
        let mut qkv_bias = false;
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

        // Read a Gemma family's soft-caps + query scale + sliding window (shared by
        // gemma2 / gemma3). Gemma3's soft-caps are null, so this yields `None` there.
        let read_gemma_common =
            |qpa: &mut Option<f64>, asc: &mut Option<f32>, fsc: &mut Option<f32>| {
                *qpa = Some(of64("query_pre_attn_scalar").unwrap_or(head_dim as f64));
                *asc = of64("attn_logit_softcapping").map(|x| x as f32);
                *fsc = of64("final_logit_softcapping").map(|x| x as f32);
            };

        match model_type {
            Some("llama") => {
                // A `llama` checkpoint with attention bias would need the Qwen2 bias
                // emit, untested here; reject rather than emit an unvalidated graph.
                if v.get("attention_bias").and_then(serde_json::Value::as_bool) == Some(true) {
                    return Err(
                        "the OpenXLA emitter does not support a `llama` checkpoint with \
                         attention_bias = true (only Qwen2 carries a q/k/v bias here)"
                            .to_string(),
                    );
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
                rope_local_base = Some(of64("rope_local_base_freq").unwrap_or(10000.0));
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
                rope_local_base = of64("rope_local_base_freq");
            }
            other => {
                return Err(format!(
                    "the OpenXLA emitter supports the Llama, Qwen2, Qwen3, Gemma1/2/3, \
                     SmolLM3, and OLMo2/3 architectures; config.json model_type = {other:?}"
                ));
            }
        }

        // rope_scaling is optional: absent -> plain RoPE; present -> only the llama3
        // scheme is supported (yarn, e.g. OLMo3 at full size, is a follow-up).
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
                         config.json rope_scaling.rope_type = {rope_type:?} (e.g. yarn is a \
                         follow-up)"
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

        Ok(Config {
            hidden,
            inter: u("intermediate_size")?,
            n_layers,
            n_q,
            n_kv: u("num_key_value_heads")?,
            head_dim,
            eps: f("rms_norm_eps")? as f32,
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

    /// Attention score scale. Most families use `head_dim^-0.5`; Gemma2/3 use
    /// `query_pre_attn_scalar^-0.5` (computed in f64 to match HF, since it can
    /// differ from `head_dim`).
    pub fn scale(&self) -> f32 {
        match self.query_pre_attn_scalar {
            Some(q) => q.powf(-0.5) as f32,
            None => (self.head_dim as f32).powf(-0.5),
        }
    }

    /// Gemma input-embedding normalizer `sqrt(hidden)` (computed in f64 then
    /// narrowed, matching HF's `hidden_size**0.5` cast to the activation dtype).
    pub fn embed_normalizer(&self) -> f32 {
        (self.hidden as f64).sqrt() as f32
    }

    /// Whether attention layer `li` uses sliding-window (local) attention. A
    /// windowed config marks layer `li` global iff `(li+1) % sliding_pattern == 0`,
    /// otherwise local (Gemma2 period 2 = even local; Gemma3 period 6 = 5 local : 1
    /// global; OLMo3 period 4). A non-windowed config has no local layer, so its
    /// emitted graphs are unchanged.
    pub fn is_sliding_layer(&self, li: usize) -> bool {
        self.sliding_window.is_some() && !(li + 1).is_multiple_of(self.sliding_pattern.max(1))
    }

    /// Whether attention layer `li` applies RoPE. Every layer does unless the
    /// config carries a NoPE mask (SmolLM3) that clears it.
    pub fn layer_uses_rope(&self, li: usize) -> bool {
        self.use_rope_layers
            .as_ref()
            .and_then(|v| v.get(li).copied())
            .unwrap_or(true)
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
}
