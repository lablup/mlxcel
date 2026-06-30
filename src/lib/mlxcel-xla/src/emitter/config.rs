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
        let gemma2 = model_type == Some("gemma2");
        // Qwen2 always has a q/k/v projection bias (the HF `Qwen2Attention` hard-
        // codes `bias=True`), and it is not a `config.json` field, so it is keyed
        // off the architecture rather than read.
        let qkv_bias = match model_type {
            Some("llama") => {
                // A `llama` checkpoint with attention bias would need the same bias
                // emit Qwen2 uses, but that pairing is untested here, so reject it
                // rather than emit an unvalidated graph.
                if v.get("attention_bias").and_then(serde_json::Value::as_bool) == Some(true) {
                    return Err(
                        "the OpenXLA emitter does not support a `llama` checkpoint with \
                         attention_bias = true (only Qwen2 carries a q/k/v bias here)"
                            .to_string(),
                    );
                }
                false
            }
            Some("qwen2") => true,
            Some("gemma2") => false,
            other => {
                return Err(format!(
                    "the OpenXLA emitter supports the Llama, Qwen2, and Gemma2 architectures; \
                     config.json model_type = {other:?} (other Gemma variants are a follow-up)"
                ));
            }
        };

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

        let hidden = u("hidden_size")?;
        let n_q = u("num_attention_heads")?;
        // head_dim is explicit in recent configs; otherwise it is hidden / heads.
        let head_dim = v
            .get("head_dim")
            .and_then(serde_json::Value::as_u64)
            .map(|x| x as usize)
            .unwrap_or(hidden / n_q.max(1));

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

        // Gemma2 logit soft-caps and the query pre-attention scale base (read only
        // for a gemma2 checkpoint; the scale defaults to `head_dim` if absent).
        let (query_pre_attn_scalar, attn_logit_softcap, final_logit_softcap) = if gemma2 {
            (
                Some(
                    v.get("query_pre_attn_scalar")
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(head_dim as f64),
                ),
                v.get("attn_logit_softcapping")
                    .and_then(serde_json::Value::as_f64)
                    .map(|x| x as f32),
                v.get("final_logit_softcapping")
                    .and_then(serde_json::Value::as_f64)
                    .map(|x| x as f32),
            )
        } else {
            (None, None, None)
        };

        Ok(Config {
            hidden,
            inter: u("intermediate_size")?,
            n_layers: u("num_hidden_layers")?,
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
            gemma2,
            query_pre_attn_scalar,
            attn_logit_softcap,
            final_logit_softcap,
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

    /// Attention score scale. Llama / Qwen2 use `head_dim^-0.5`; Gemma2 uses
    /// `query_pre_attn_scalar^-0.5` (computed in f64 to match HF, since it can
    /// differ from `head_dim`). The Llama / Qwen2 branch is unchanged.
    pub fn scale(&self) -> f32 {
        match self.query_pre_attn_scalar {
            Some(q) => q.powf(-0.5) as f32,
            None => (self.head_dim as f32).powf(-0.5),
        }
    }

    /// Gemma2 input-embedding normalizer `sqrt(hidden)` (computed in f64 then
    /// narrowed, matching HF's `hidden_size**0.5` cast to the activation dtype).
    pub fn embed_normalizer(&self) -> f32 {
        (self.hidden as f64).sqrt() as f32
    }
}
