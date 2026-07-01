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
    /// MiMo, InternLM3, ...).
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
    /// Checkpoint tensor-naming scheme (issue #499). Loader-only: it selects how
    /// [`weight_names`](crate::weight_names) maps the emitter's arg order onto the
    /// checkpoint tensors, so it never affects the emitted graph. `Llama` (the
    /// default) is the standard HF layout; `Exaone` is ExaOne 3.x's GPT-2-style
    /// names.
    pub weight_scheme: WeightScheme,
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
            weight_scheme: WeightScheme::Llama,
        }
    }

    /// Build a [`Config`] from a model's `config.json` text.
    ///
    /// Scope (issue #499): the dense Llama-family architectures whose forward is
    /// Llama or Qwen2 up to config-level deltas. The switches the emitter branches
    /// on are read here: the RoPE scheme (llama3 scaling, or plain, which also
    /// covers the `default` and in-context `dynamic` rope types), whether the q/k/v
    /// projections carry a bias, whether the LM head is tied or a separate
    /// `lm_head.weight`, the `gemma2` structural flag, and the tensor-naming
    /// scheme. Supported `model_type`s: `llama`, `qwen2`, `gemma2`, and the dense
    /// pack `seed_oss`, `mimo`, `internlm3`, `exaone` (each is one of the two proven
    /// forwards with a config / naming delta; all verified from their modeling code
    /// to use the emitter's half-split RoPE and `head_dim^-0.5` scaling). Configs
    /// the emitter cannot reproduce are rejected with a clear error rather than
    /// mis-emitted: an interleaved-RoPE arch (e.g. `ernie4_5`, GPT-J-style pairs),
    /// an unsupported `model_type`, an unsupported `rope_type` (e.g. `yarn`), an
    /// attention output bias, an MLP bias, or a non-SwiGLU activation.
    pub fn from_json_str(s: &str) -> Result<Self, String> {
        let v: serde_json::Value =
            serde_json::from_str(s).map_err(|e| format!("parse config.json: {e}"))?;

        let model_type = v.get("model_type").and_then(serde_json::Value::as_str);
        let bool_field = |k: &str| v.get(k).and_then(serde_json::Value::as_bool);
        let gemma2 = model_type == Some("gemma2");
        // ExaOne 3.x keeps GPT-2-style tensor names; everything else uses the
        // standard HF Llama layout. Loader-only, so it never changes the graph.
        let weight_scheme = if model_type == Some("exaone") {
            WeightScheme::Exaone
        } else {
            WeightScheme::Llama
        };

        // Interleaved (GPT-J-style) RoPE is a distinct emitter delta from the
        // half-split RoPE Llama / Qwen2 use: the pairs rotated are (2i, 2i+1), not
        // (i, i+d/2). ERNIE-4.5 (`rotate_half` over `x[..., 0::2]` / `x[..., 1::2]`)
        // is such an arch, so its forward is NOT the plain-RoPE Llama it looks like
        // from config.json; it is rejected here rather than mis-emitted (a
        // half-split emit is close but wrong) pending an interleaved-RoPE variant.
        if model_type == Some("ernie4_5") {
            return Err(
                "the OpenXLA emitter uses half-split RoPE; ERNIE-4.5 (model_type \
                 ernie4_5) uses interleaved (GPT-J-style) RoPE, which is a follow-up \
                 (an interleaved-RoPE emit variant or a load-time q/k permutation)"
                    .to_string(),
            );
        }

        // Supported dense Llama-family architectures (issue #499). Each maps to one
        // of the two proven forwards (Llama or Qwen2) plus a config / naming delta;
        // anything else (MoE, MLA, cross-layer attention, fused QKV, interleaved
        // RoPE, novel activations, ...) is a follow-up and rejected rather than
        // mis-emitted.
        const SUPPORTED: &[&str] = &[
            "llama",
            "qwen2",
            "gemma2",
            "seed_oss",
            "mimo",
            "internlm3",
            "exaone",
        ];
        match model_type {
            Some(mt) if SUPPORTED.contains(&mt) => {}
            other => {
                return Err(format!(
                    "the OpenXLA emitter supports the dense Llama-family architectures \
                     {SUPPORTED:?}; config.json model_type = {other:?} (MoE / MLA / \
                     fused-QKV / interleaved-RoPE / novel-activation variants are follow-ups)"
                ));
            }
        }

        // q/k/v projection bias. Qwen2 hard-codes `bias=True` in `Qwen2Attention`
        // (not a config field); the other bias-bearing dense arches expose it as
        // `attention_bias` (Seed-OSS, MiMo) or `qkv_bias` (InternLM3). A `llama`
        // checkpoint with attention_bias stays rejected (that pairing is untested
        // here).
        let attention_bias = bool_field("attention_bias");
        let qkv_bias = match model_type {
            Some("llama") => {
                if attention_bias == Some(true) {
                    return Err(
                        "the OpenXLA emitter does not support a `llama` checkpoint with \
                         attention_bias = true (only the bias-bearing dense arches carry a \
                         q/k/v bias here)"
                            .to_string(),
                    );
                }
                false
            }
            Some("qwen2") => true,
            _ => attention_bias == Some(true) || bool_field("qkv_bias") == Some(true),
        };
        // The o_proj bias and the MLP bias have no emit, and a non-SwiGLU
        // activation would be mis-emitted, so a config that sets them is rejected
        // rather than silently dropped. Gemma2's GeGLU is handled by its own flag.
        if bool_field("attention_out_bias") == Some(true) {
            return Err(
                "the OpenXLA emitter has no attention output (o_proj) bias; \
                 config.json attention_out_bias = true is a follow-up"
                    .to_string(),
            );
        }
        if bool_field("mlp_bias") == Some(true) {
            return Err(
                "the OpenXLA emitter has no MLP bias; config.json mlp_bias = true is a follow-up"
                    .to_string(),
            );
        }
        if !gemma2 {
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
        // Some arches use alternate field names (ExaOne 3.x: `num_layers`,
        // `layer_norm_epsilon`), so these read the first present of a key list.
        let u_any = |keys: &[&str]| -> Result<usize, String> {
            keys.iter()
                .find_map(|k| v.get(*k).and_then(serde_json::Value::as_u64))
                .map(|x| x as usize)
                .ok_or_else(|| format!("config.json missing integer among {keys:?}"))
        };
        let f_any = |keys: &[&str]| -> Result<f64, String> {
            keys.iter()
                .find_map(|k| v.get(*k).and_then(serde_json::Value::as_f64))
                .ok_or_else(|| format!("config.json missing number among {keys:?}"))
        };

        let hidden = u("hidden_size")?;
        let n_q = u("num_attention_heads")?;
        // head_dim is explicit in recent configs; otherwise it is hidden / heads.
        let head_dim = v
            .get("head_dim")
            .and_then(serde_json::Value::as_u64)
            .map(|x| x as usize)
            .unwrap_or(hidden / n_q.max(1));

        // rope_scaling is optional: absent -> plain RoPE (Qwen2.5, plain Llama).
        // When present, the supported schemes are llama3 (scaled) and, served as
        // plain RoPE, the `default` (identity) and `dynamic` (NTK-by-parts) types.
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

        // Gemma2 sliding-window size (issue #495). Read only for a gemma2
        // checkpoint; an absent field falls back to the HF Gemma2 default of 4096.
        // Non-gemma2 architectures get `None` (global attention on every layer),
        // even if their config carries a `sliding_window` (e.g. Qwen2.5, which the
        // emitter serves without sliding-window attention).
        let sliding_window = if gemma2 {
            Some(
                v.get("sliding_window")
                    .and_then(serde_json::Value::as_u64)
                    .map(|x| x as usize)
                    .unwrap_or(4096),
            )
        } else {
            None
        };

        Ok(Config {
            hidden,
            inter: u("intermediate_size")?,
            // ExaOne 3.x uses `num_layers` in place of `num_hidden_layers`.
            n_layers: u_any(&["num_hidden_layers", "num_layers"])?,
            n_q,
            n_kv: u("num_key_value_heads")?,
            head_dim,
            // ExaOne 3.x uses `layer_norm_epsilon` in place of `rms_norm_eps`.
            eps: f_any(&["rms_norm_eps", "layer_norm_epsilon"])? as f32,
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
            weight_scheme,
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

    /// Whether attention layer `li` uses sliding-window (local) attention (issue
    /// #495). Gemma2 alternates local and global attention starting local, so the
    /// even layers (0, 2, 4, …) are local and the odd layers are global, matching
    /// HF `Gemma2DecoderLayer` (`is_sliding = not bool(layer_idx % 2)`). Only a
    /// config with a sliding window (Gemma2) has local layers; Llama / Qwen2
    /// return `false` for every layer, so their emitted graphs are unchanged.
    pub fn is_sliding_layer(&self, li: usize) -> bool {
        self.sliding_window.is_some() && li.is_multiple_of(2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ERNIE-4.5 is rejected with a message naming its interleaved (GPT-J-style)
    /// RoPE: it looks like a plain-RoPE Llama in config.json but its `rotate_half`
    /// rotates the (2i, 2i+1) pairs, not the (i, i+d/2) halves the emitter uses, so
    /// a half-split emit would be wrong. Deferred to an interleaved-RoPE follow-up.
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
