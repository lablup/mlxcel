//! Gemma3n audio graph configuration and artifact identity.
//!
//! Waveform decoding, resampling, and log-mel extraction stay on the bounded
//! host stage. This module defines the exact static shape and numeric policy
//! compiled into the split `audio.encode` and `audio.merge_ple` artifacts.

use std::path::Path;

use serde::Deserialize;

pub const GEMMA3N_AUDIO_GRAPH_ABI: &str = "gemma3n-audio-split-v2";
pub const GEMMA3N_AUDIO_MODALITY_FAMILY: &str = "gemma3n_audio";
pub const GEMMA3N_AUDIO_MEL_BINS: usize = 128;
pub const GEMMA3N_AUDIO_MAX_FRAMES: usize = 2_997;
pub const GEMMA3N_AUDIO_SOFT_TOKENS: usize = 188;
pub const GEMMA3N_AUDIO_MAX_CLIPS: usize = 4;
pub const GEMMA3N_AUDIO_FRAME_BUCKETS: &[usize] = &[8, 32, 128, 512, 1_024, 2_048, 2_997];

fn default_hidden_size() -> usize {
    1_536
}

fn default_input_feat_size() -> usize {
    GEMMA3N_AUDIO_MEL_BINS
}

fn default_vocab_size() -> usize {
    128
}

fn default_vocab_offset() -> i32 {
    262_272
}

fn default_rms_norm_eps() -> f32 {
    1e-6
}

fn default_gradient_clipping() -> f32 {
    1e10
}

fn default_chunk_size() -> usize {
    12
}

fn default_context_left() -> usize {
    13
}

fn default_logit_cap() -> f32 {
    50.0
}

fn default_attention_heads() -> usize {
    8
}

fn default_hidden_layers() -> usize {
    12
}

fn default_conv_kernel_size() -> usize {
    5
}

fn default_reduction_factor() -> usize {
    4
}

fn default_residual_weight() -> f32 {
    0.5
}

fn default_conv_channels() -> Vec<usize> {
    vec![128, 32]
}

fn default_group_norm_eps() -> f32 {
    1e-3
}

fn default_conv_kernels() -> Vec<[usize; 2]> {
    vec![[3, 3], [3, 3]]
}

fn default_conv_strides() -> Vec<[usize; 2]> {
    vec![[2, 2], [2, 2]]
}

/// Audio sub-configuration consumed by the StableHLO emitter.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(default)]
pub struct Gemma3nXlaAudioConfig {
    pub input_feat_size: usize,
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub vocab_offset: i32,
    pub rms_norm_eps: f32,
    pub gradient_clipping: f32,
    pub conf_attention_chunk_size: usize,
    pub conf_attention_context_left: usize,
    pub conf_attention_context_right: usize,
    pub conf_attention_logit_cap: f32,
    pub conf_num_attention_heads: usize,
    pub conf_num_hidden_layers: usize,
    pub conf_conv_kernel_size: usize,
    pub conf_reduction_factor: usize,
    pub conf_residual_weight: f32,
    pub sscp_conv_channel_size: Vec<usize>,
    pub sscp_conv_group_norm_eps: f32,
    pub sscp_conv_kernel_size: Vec<[usize; 2]>,
    pub sscp_conv_stride_size: Vec<[usize; 2]>,
}

impl Default for Gemma3nXlaAudioConfig {
    fn default() -> Self {
        Self {
            input_feat_size: default_input_feat_size(),
            hidden_size: default_hidden_size(),
            vocab_size: default_vocab_size(),
            vocab_offset: default_vocab_offset(),
            rms_norm_eps: default_rms_norm_eps(),
            gradient_clipping: default_gradient_clipping(),
            conf_attention_chunk_size: default_chunk_size(),
            conf_attention_context_left: default_context_left(),
            conf_attention_context_right: 0,
            conf_attention_logit_cap: default_logit_cap(),
            conf_num_attention_heads: default_attention_heads(),
            conf_num_hidden_layers: default_hidden_layers(),
            conf_conv_kernel_size: default_conv_kernel_size(),
            conf_reduction_factor: default_reduction_factor(),
            conf_residual_weight: default_residual_weight(),
            sscp_conv_channel_size: default_conv_channels(),
            sscp_conv_group_norm_eps: default_group_norm_eps(),
            sscp_conv_kernel_size: default_conv_kernels(),
            sscp_conv_stride_size: default_conv_strides(),
        }
    }
}

impl Gemma3nXlaAudioConfig {
    pub fn from_model_dir(model_dir: &Path) -> Result<Option<Self>, String> {
        let path = model_dir.join("config.json");
        let text = std::fs::read_to_string(&path)
            .map_err(|error| format!("read {}: {error}", path.display()))?;
        Self::from_json_str(&text).map_err(|error| format!("{}: {error}", path.display()))
    }

    pub fn from_json_str(text: &str) -> Result<Option<Self>, String> {
        let root: serde_json::Value =
            serde_json::from_str(text).map_err(|error| format!("parse config JSON: {error}"))?;
        let Some(value) = root.get("audio_config") else {
            return Ok(None);
        };
        if value.is_null() {
            return Ok(None);
        }
        if value.get("model_type").and_then(serde_json::Value::as_str) != Some("gemma3n_audio") {
            return Err("Gemma3n audio_config.model_type must be `gemma3n_audio`".into());
        }
        let config: Self = serde_json::from_value(value.clone())
            .map_err(|error| format!("parse Gemma3n audio_config: {error}"))?;
        config.validate()?;
        Ok(Some(config))
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.input_feat_size != GEMMA3N_AUDIO_MEL_BINS {
            return Err(format!(
                "Gemma3n audio input_feat_size={} is unsupported; expected {GEMMA3N_AUDIO_MEL_BINS}",
                self.input_feat_size
            ));
        }
        if self.hidden_size == 0
            || self.conf_num_attention_heads == 0
            || !self
                .hidden_size
                .is_multiple_of(self.conf_num_attention_heads)
        {
            return Err("Gemma3n audio hidden size must be divisible by the head count".into());
        }
        if self.conf_attention_chunk_size == 0
            || self.conf_reduction_factor == 0
            || self.conf_conv_kernel_size == 0
            || self.conf_num_hidden_layers == 0
        {
            return Err(
                "Gemma3n audio chunk, reduction, convolution, and layer counts must be positive"
                    .into(),
            );
        }
        if self.sscp_conv_channel_size.len() != 2
            || self.sscp_conv_kernel_size.len() != 2
            || self.sscp_conv_stride_size.len() != 2
        {
            return Err("Gemma3n audio SSCP requires exactly two convolution stages".into());
        }
        if self.sscp_conv_channel_size.contains(&0)
            || self
                .sscp_conv_kernel_size
                .iter()
                .flatten()
                .any(|&value| value == 0)
            || self
                .sscp_conv_stride_size
                .iter()
                .flatten()
                .any(|&value| value == 0)
        {
            return Err("Gemma3n audio SSCP dimensions and strides must be positive".into());
        }
        if self.vocab_size == 0
            || self.vocab_offset < 0
            || !self.rms_norm_eps.is_finite()
            || self.rms_norm_eps <= 0.0
            || !self.sscp_conv_group_norm_eps.is_finite()
            || self.sscp_conv_group_norm_eps <= 0.0
            || !self.gradient_clipping.is_finite()
            || self.gradient_clipping <= 0.0
            || !self.conf_attention_logit_cap.is_finite()
            || self.conf_attention_logit_cap <= 0.0
            || !self.conf_residual_weight.is_finite()
        {
            return Err("Gemma3n audio scalar configuration is invalid".into());
        }
        let output_frequency = self.output_frequency()?;
        self.sscp_conv_channel_size[1]
            .checked_mul(output_frequency)
            .ok_or_else(|| "Gemma3n audio SSCP projection width overflows".to_string())?;
        Ok(())
    }

    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.conf_num_attention_heads
    }

    #[must_use]
    pub fn context_size(&self) -> usize {
        self.conf_attention_context_left.saturating_sub(1)
            + self.conf_attention_chunk_size
            + self.conf_attention_context_right
    }

    pub fn time_stride_product(&self) -> Result<usize, String> {
        self.sscp_conv_stride_size
            .iter()
            .try_fold(1usize, |product, stride| {
                product
                    .checked_mul(stride[0])
                    .ok_or_else(|| "Gemma3n audio time stride product overflows".to_string())
            })
    }

    pub fn output_frequency(&self) -> Result<usize, String> {
        let mut frequency = self.input_feat_size;
        for (kernel, stride) in self
            .sscp_conv_kernel_size
            .iter()
            .zip(&self.sscp_conv_stride_size)
        {
            frequency = frequency
                .checked_add(2)
                .filter(|&padded| padded >= kernel[1])
                .ok_or_else(|| {
                    "Gemma3n audio SSCP frequency kernel exceeds padded input".to_string()
                })?;
            frequency = (frequency - kernel[1]) / stride[1] + 1;
        }
        Ok(frequency)
    }

    pub fn encoded_frames(&self, frame_bucket: usize) -> Result<usize, String> {
        validate_gemma3n_audio_frame_bucket(frame_bucket)?;
        let mut time = frame_bucket;
        for (kernel, stride) in self
            .sscp_conv_kernel_size
            .iter()
            .zip(&self.sscp_conv_stride_size)
        {
            time = time
                .checked_add(2)
                .filter(|&padded| padded >= kernel[0])
                .ok_or_else(|| "Gemma3n audio SSCP time kernel exceeds padded input".to_string())?;
            time = (time - kernel[0]) / stride[0] + 1;
        }
        Ok(time)
    }

    pub fn projected_frames(&self, frame_bucket: usize) -> Result<usize, String> {
        Ok(self
            .encoded_frames(frame_bucket)?
            .div_ceil(self.conf_reduction_factor))
    }

    pub fn artifact_identity(
        &self,
        frame_bucket: usize,
        clips: usize,
        text_hidden: usize,
        text_layers: usize,
        text_ple_hidden: usize,
    ) -> Result<String, String> {
        validate_gemma3n_audio_batch(frame_bucket, clips)?;
        if text_hidden == 0 || text_layers == 0 || text_ple_hidden == 0 {
            return Err("Gemma3n audio language dimensions must be positive".into());
        }
        Ok(format!(
            "{GEMMA3N_AUDIO_GRAPH_ABI}:dtype=bf16-accum-f32:mel={}:bucket={frame_bucket}:clips={clips}:\
             sscp-kernels={:?}:strides={:?}:channels={:?}:group-eps={:08x}:hidden={}:heads={}:layers={}:\
             chunk={}:left={}:right={}:softcap={:08x}:conv={}:reduction={}:residual={:08x}:\
             clip={:08x}:rms={:08x}:audio-vocab={}:offset={}:soft-tokens={}:text={}:{}/{}",
            self.input_feat_size,
            self.sscp_conv_kernel_size,
            self.sscp_conv_stride_size,
            self.sscp_conv_channel_size,
            self.sscp_conv_group_norm_eps.to_bits(),
            self.hidden_size,
            self.conf_num_attention_heads,
            self.conf_num_hidden_layers,
            self.conf_attention_chunk_size,
            self.conf_attention_context_left,
            self.conf_attention_context_right,
            self.conf_attention_logit_cap.to_bits(),
            self.conf_conv_kernel_size,
            self.conf_reduction_factor,
            self.conf_residual_weight.to_bits(),
            self.gradient_clipping.to_bits(),
            self.rms_norm_eps.to_bits(),
            self.vocab_size,
            self.vocab_offset,
            GEMMA3N_AUDIO_SOFT_TOKENS,
            text_hidden,
            text_layers,
            text_ple_hidden,
        ))
    }
}

pub(crate) fn validate_gemma3n_audio_frame_bucket(frame_bucket: usize) -> Result<(), String> {
    if GEMMA3N_AUDIO_FRAME_BUCKETS.contains(&frame_bucket) {
        Ok(())
    } else {
        Err(format!(
            "Gemma3n audio frame bucket {frame_bucket} is not one of {GEMMA3N_AUDIO_FRAME_BUCKETS:?}"
        ))
    }
}

pub(crate) fn validate_gemma3n_audio_batch(
    frame_bucket: usize,
    clips: usize,
) -> Result<(), String> {
    validate_gemma3n_audio_frame_bucket(frame_bucket)?;
    if !(1..=GEMMA3N_AUDIO_MAX_CLIPS).contains(&clips) {
        return Err(format!(
            "Gemma3n audio clip count {clips} is outside 1..={GEMMA3N_AUDIO_MAX_CLIPS}"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config_json(audio: serde_json::Value) -> String {
        serde_json::json!({
            "model_type": "gemma3n",
            "audio_config": audio,
        })
        .to_string()
    }

    #[test]
    fn config_identity_covers_every_graph_shape_and_numeric_policy() {
        let mut config = Gemma3nXlaAudioConfig::default();
        config.validate().unwrap();
        assert_eq!(config.encoded_frames(2_997).unwrap(), 750);
        assert_eq!(config.projected_frames(2_997).unwrap(), 188);
        assert_eq!(config.output_frequency().unwrap(), 32);
        let first = config.artifact_identity(2_997, 1, 2_048, 30, 256).unwrap();
        config.conf_attention_logit_cap = 49.0;
        let second = config.artifact_identity(2_997, 1, 2_048, 30, 256).unwrap();
        assert_ne!(first, second);
        assert!(first.contains(GEMMA3N_AUDIO_GRAPH_ABI));
        assert!(first.contains("bucket=2997"));
        assert!(first.contains("dtype=bf16-accum-f32"));
    }

    #[test]
    fn capability_requires_gemma3n_audio_config() {
        assert!(
            Gemma3nXlaAudioConfig::from_json_str(
                &serde_json::json!({"model_type": "gemma3n"}).to_string()
            )
            .unwrap()
            .is_none()
        );
        assert!(
            Gemma3nXlaAudioConfig::from_json_str(&config_json(serde_json::json!({
                "model_type": "gemma4_audio"
            })))
            .unwrap_err()
            .contains("gemma3n_audio")
        );
        assert!(
            Gemma3nXlaAudioConfig::from_json_str(&config_json(serde_json::json!({
                "model_type": "gemma3n_audio"
            })))
            .unwrap()
            .is_some()
        );
    }
}
