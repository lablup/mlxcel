//! Llama-architecture emitter config. The hard-coded [`Config::llama_3_2_1b`]
//! matches spike/openxla/model_jax.py; [`Config::from_json`] reads the same shape
//! from any Llama checkpoint's `config.json` (issue #449 M3 Stage 2d).

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
    // llama3 rope scaling
    pub factor: f64,
    pub low_freq_factor: f64,
    pub high_freq_factor: f64,
    pub orig_ctx: usize,
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
            factor: 32.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            orig_ctx: 8192,
        }
    }

    /// Build a [`Config`] from a model's `config.json` text.
    ///
    /// Scope (Stage A): the Llama architecture with llama3 RoPE scaling, which is
    /// what the emitter encodes. Configs the emitter cannot yet reproduce are
    /// rejected with a clear error rather than silently mis-emitted: a non-`llama`
    /// `model_type`, attention bias (e.g. Qwen2), untied embeddings, or a missing
    /// / non-`llama3` `rope_scaling`.
    pub fn from_json_str(s: &str) -> Result<Self, String> {
        let v: serde_json::Value =
            serde_json::from_str(s).map_err(|e| format!("parse config.json: {e}"))?;

        let model_type = v.get("model_type").and_then(serde_json::Value::as_str);
        if model_type != Some("llama") {
            return Err(format!(
                "the OpenXLA emitter currently supports the Llama architecture only; \
                 config.json model_type = {model_type:?} (Qwen / Gemma / others are a follow-up)"
            ));
        }
        if v.get("attention_bias").and_then(serde_json::Value::as_bool) == Some(true) {
            return Err(
                "the OpenXLA emitter does not yet support attention bias (e.g. Qwen2); \
                 config.json attention_bias = true"
                    .to_string(),
            );
        }
        if v.get("tie_word_embeddings")
            .and_then(serde_json::Value::as_bool)
            != Some(true)
        {
            return Err("the OpenXLA emitter assumes tied word embeddings; \
                 config.json tie_word_embeddings != true"
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

        let scaling = v.get("rope_scaling").ok_or_else(|| {
            "the OpenXLA emitter encodes llama3 RoPE scaling; config.json has no \
             `rope_scaling` (plain-RoPE Llama is a follow-up)"
                .to_string()
        })?;
        let rope_type = scaling
            .get("rope_type")
            .or_else(|| scaling.get("type"))
            .and_then(serde_json::Value::as_str);
        if rope_type != Some("llama3") {
            return Err(format!(
                "the OpenXLA emitter encodes llama3 RoPE scaling; config.json \
                 rope_scaling.rope_type = {rope_type:?} (only `llama3` is supported)"
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
                "config.json rope_scaling missing `original_max_position_embeddings`".to_string()
            })?;

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
            factor: sf("factor")?,
            low_freq_factor: sf("low_freq_factor")?,
            high_freq_factor: sf("high_freq_factor")?,
            orig_ctx,
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
