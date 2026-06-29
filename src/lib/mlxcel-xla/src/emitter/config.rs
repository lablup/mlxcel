//! Emitter config for the Llama-family architectures the OpenXLA backend serves.
//! The hard-coded [`Config::llama_3_2_1b`] matches spike/openxla/model_jax.py;
//! [`Config::from_json`] reads the same shape from a checkpoint's `config.json`
//! (issue #449 M3 Stage 2d). Stage A covered the Llama architecture (llama3 RoPE,
//! no attention bias); Stage B adds Qwen2 (plain RoPE + QKV bias), so the config
//! carries the two architecture switches the emitter branches on: the RoPE kind
//! and whether q/k/v projections have a bias.

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
        }
    }

    /// Build a [`Config`] from a model's `config.json` text.
    ///
    /// Scope: the Llama and Qwen2 architectures (RMSNorm, SwiGLU MLP, GQA, tied
    /// embeddings). Llama uses llama3 RoPE scaling and no attention bias; Qwen2
    /// uses plain RoPE and a q/k/v projection bias. Configs the emitter cannot yet
    /// reproduce are rejected with a clear error rather than silently mis-emitted:
    /// an unsupported `model_type`, untied embeddings, a `llama` checkpoint with
    /// `attention_bias`, or a `rope_scaling` whose `rope_type` is not `llama3`.
    pub fn from_json_str(s: &str) -> Result<Self, String> {
        let v: serde_json::Value =
            serde_json::from_str(s).map_err(|e| format!("parse config.json: {e}"))?;

        let model_type = v.get("model_type").and_then(serde_json::Value::as_str);
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
            other => {
                return Err(format!(
                    "the OpenXLA emitter supports the Llama and Qwen2 architectures; \
                     config.json model_type = {other:?} (Gemma / others are a follow-up)"
                ));
            }
        };

        if v.get("tie_word_embeddings")
            .and_then(serde_json::Value::as_bool)
            != Some(true)
        {
            return Err("the OpenXLA emitter assumes tied word embeddings; \
                 config.json tie_word_embeddings != true (untied LM head is a follow-up)"
                .to_string());
        }

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

    /// Attention scale head_dim^-0.5.
    pub fn scale(&self) -> f32 {
        (self.head_dim as f32).powf(-0.5)
    }
}
