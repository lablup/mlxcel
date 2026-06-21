// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Kokoro-82M text-to-speech model (StyleTTS2 acoustic model + built-in
//! iSTFTNet vocoder), ported to MLX.
//!
//! The forward pass turns a phoneme string into a `24 kHz` mono waveform:
//! phoneme ids -> PLBert -> duration predictor (per-token frame counts) ->
//! alignment-matrix expansion -> F0/N prosody + acoustic text features ->
//! iSTFTNet decoder. The g2p front-end (text -> phonemes) lives in
//! [`crate::models::g2p`]; this module consumes phonemes via the checkpoint's
//! vocab.
//!
//! Detection: the Kokoro `config.json` has no top-level `model_type`, so
//! [`is_kokoro_checkpoint`] keys off the `istftnet` config block or the
//! `kokoro-v1_0.safetensors` weight filename. All MLX load and evaluation run on
//! the audio worker thread (see [`crate::server::kokoro_tts`]); the model holds
//! MLX handles directly and is neither `Send` nor `Sync`.

mod bert;
mod blocks;
mod decoder;
mod lstm;
mod ops;
mod predictor;
mod stft;
mod text_encoder;
mod voices;
mod weights;

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, anyhow};
use serde_json::Value;

use bert::PlBert;
use decoder::Decoder;
use predictor::Predictor;
use text_encoder::TextEncoder;
use voices::VoicePack;
use weights::Weights;

use voices::resolve_voice;

/// The canonical Kokoro acoustic-model weight filename.
pub const KOKORO_WEIGHT_FILE: &str = "kokoro-v1_0.safetensors";

/// Largest phoneme token count the model context allows (`512 - 2` boundary
/// tokens).
const MAX_TOKENS: usize = 510;

/// Detect a Kokoro checkpoint without a `model_type` field.
///
/// Returns `true` when the directory contains the `kokoro-v1_0.safetensors`
/// weight file, or when `config.json` carries an `istftnet` block (the
/// architecture signature). Either signal is sufficient; both are checked so
/// renamed weight files still resolve via the config.
pub fn is_kokoro_checkpoint(model_path: &Path, config: &Value) -> bool {
    if config.get("istftnet").is_some() {
        return true;
    }
    model_path.join(KOKORO_WEIGHT_FILE).exists()
}

/// Parsed Kokoro architecture configuration.
#[derive(Debug, Clone)]
struct KokoroConfig {
    n_layers: usize,
}

impl KokoroConfig {
    fn from_value(config: &Value) -> Self {
        // plbert.num_hidden_layers drives the ALBERT layer reuse count.
        let n_layers = config
            .get("plbert")
            .and_then(|p| p.get("num_hidden_layers"))
            .and_then(Value::as_u64)
            .unwrap_or(12) as usize;
        Self { n_layers }
    }
}

/// The Kokoro TTS model: weights, sub-modules, and the phoneme vocab.
pub struct KokoroModel {
    bert: PlBert,
    bert_encoder_w: mlxcel_core::UniquePtr<mlxcel_core::MlxArray>,
    bert_encoder_b: mlxcel_core::UniquePtr<mlxcel_core::MlxArray>,
    predictor: Predictor,
    text_encoder: TextEncoder,
    decoder: Decoder,
    vocab: HashMap<String, i32>,
    model_path: std::path::PathBuf,
}

impl KokoroModel {
    /// Load a Kokoro checkpoint from a directory.
    ///
    /// Reads `config.json` (for the vocab and layer count) and
    /// `kokoro-v1_0.safetensors`, then builds every sub-module. Must run on the
    /// audio worker thread because MLX arrays are thread-affine.
    pub fn load(model_path: &Path) -> Result<Self> {
        let config_path = model_path.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .with_context(|| format!("reading {config_path:?}"))?;
        let config_str = super::sanitize_config_json(&config_str);
        let config: Value =
            serde_json::from_str(&config_str).context("parsing Kokoro config.json")?;

        let cfg = KokoroConfig::from_value(&config);
        let vocab = parse_vocab(&config)?;

        let weight_path = model_path.join(KOKORO_WEIGHT_FILE);
        let map = mlxcel_core::weights::load_safetensors(&weight_path)
            .map_err(|e| anyhow!("loading {weight_path:?}: {e}"))?;
        let w = Weights::new(&map);

        let bert = PlBert::load(&w, cfg.n_layers).map_err(|e| anyhow!(e))?;
        let (bert_encoder_w, bert_encoder_b) = w
            .linear("bert_encoder")
            .map_err(|e| anyhow!(e))
            .and_then(|(weight, bias)| {
                bias.map(|b| (weight, b))
                    .ok_or_else(|| anyhow!("kokoro: bert_encoder.bias missing"))
            })?;
        let predictor = Predictor::load(&w).map_err(|e| anyhow!(e))?;
        let text_encoder = TextEncoder::load(&w).map_err(|e| anyhow!(e))?;
        let decoder = Decoder::load(&w).map_err(|e| anyhow!(e))?;

        Ok(Self {
            bert,
            bert_encoder_w,
            bert_encoder_b,
            predictor,
            text_encoder,
            decoder,
            vocab,
            model_path: model_path.to_path_buf(),
        })
    }

    /// Map a phoneme string to vocab token ids, dropping symbols not in the
    /// vocab (matching the upstream `filter`).
    pub fn phonemes_to_ids(&self, phonemes: &str) -> Vec<i32> {
        phonemes
            .chars()
            .filter_map(|c| {
                let s = c.to_string();
                self.vocab.get(&s).copied()
            })
            .collect()
    }

    /// Synthesize a waveform from a phoneme string.
    ///
    /// `voice` selects a `voices/` pack (validated, defaulting to `af_heart`);
    /// `speed` scales the predicted durations (larger = faster/shorter). Returns
    /// `(samples, sample_rate)` with mono `f32` PCM in `[-1, 1]`.
    pub fn synthesize(
        &self,
        phonemes: &str,
        voice: Option<&str>,
        speed: f32,
    ) -> Result<(Vec<f32>, u32)> {
        let mut ids = self.phonemes_to_ids(phonemes);
        if ids.is_empty() {
            return Err(anyhow!(
                "kokoro: no recognizable phonemes produced from input"
            ));
        }
        if ids.len() > MAX_TOKENS {
            ids.truncate(MAX_TOKENS);
        }
        let n_tokens = ids.len();
        let speed = if speed.is_finite() && speed > 0.0 {
            speed
        } else {
            1.0
        };

        // Resolve and load the voice pack, take the style row for this length.
        let voice_name = resolve_voice(&self.model_path, voice);
        let pack = VoicePack::load(&self.model_path, &voice_name).map_err(|e| anyhow!(e))?;
        let ref_s = pack.row(n_tokens); // (1, 256)
        let s_pred = ops::slice(&ref_s, &[0, 128], &[1, 256]); // (1,128) predictor
        let s_dec = ops::slice(&ref_s, &[0, 0], &[1, 128]); // (1,128) decoder

        // Pad with boundary token 0 on each side.
        let mut padded = Vec::with_capacity(n_tokens + 2);
        padded.push(0);
        padded.extend_from_slice(&ids);
        padded.push(0);
        let l = padded.len();

        // PLBert -> bert_encoder projection (768 -> 512), transposed to (512, L).
        let bert_dur = self.bert.forward(&padded); // (L, 768)
        let d_en = ops::linear(&bert_dur, &self.bert_encoder_w, Some(&self.bert_encoder_b));
        let d_en = ops::swap_axes(&d_en, 0, 1); // (512, L)

        // Duration predictor.
        let (d, dur_logits) = self.predictor.durations(&d_en, &s_pred, l); // d:(L,640) dur:(L,50)
        let durations = duration_frames(&dur_logits, speed)?; // per-token frame counts

        // Alignment matrix (L, T) and expanded features.
        let total_frames: usize = durations.iter().map(|&c| c as usize).sum();
        if total_frames == 0 {
            return Err(anyhow!("kokoro: predicted zero output frames"));
        }
        let aln = alignment_matrix(&durations, l, total_frames);

        // These two matmuls expand per-token features to per-frame against the
        // alignment matrix, whose `T` (frame) dimension is derived from the
        // runtime duration prediction. They are the data-dependent construction
        // ops on the synthesis path, so route them through the fallible variant:
        // a malformed graph returns Err here instead of aborting the process via
        // an uncaught MLX C++ exception (MLX validates matmul shapes eagerly at
        // graph-build time).
        let d_t = ops::swap_axes(&d, 0, 1); // (640, L)
        let en = ops::try_matmul(&d_t, &aln).map_err(|e| anyhow!(e))?; // (640, T)
        let (f0_pred, n_pred) = self.predictor.f0n(&en, &s_pred, total_frames);

        let t_en = self.text_encoder.forward(&padded); // (512, L)
        let asr = ops::try_matmul(&t_en, &aln).map_err(|e| anyhow!(e))?; // (512, T)

        let samples = self
            .decoder
            .forward(&asr, &f0_pred, &n_pred, &s_dec)
            .map_err(|e| anyhow!(e))?;

        Ok((samples, stft::SAMPLE_RATE))
    }
}

/// Compute per-token frame counts from the `(L, 50)` duration logits:
/// `round(sigmoid(logits).sum(-1) / speed)`, clamped to `[1, 100]` with NaN
/// mapped to 1.
fn duration_frames(
    dur_logits: &mlxcel_core::UniquePtr<mlxcel_core::MlxArray>,
    speed: f32,
) -> Result<Vec<i64>> {
    let summed = ops::sum_axis(&ops::sigmoid(dur_logits), -1, false); // (L,)
    let scaled = ops::div_scalar(&summed, speed);
    let host = ops::to_vec_f32(&scaled).map_err(|e| anyhow!(e))?;
    Ok(host
        .into_iter()
        .map(|v| {
            let v = if v.is_finite() { v } else { 1.0 };
            (v.round() as i64).clamp(1, 100)
        })
        .collect())
}

/// Build the `(L, T)` alignment matrix that expands per-token features to
/// per-frame: row `i` is 1 across the frame span of token `i`.
fn alignment_matrix(
    durations: &[i64],
    l: usize,
    total_frames: usize,
) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let mut data = vec![0.0_f32; l * total_frames];
    let mut frame = 0usize;
    for (i, &count) in durations.iter().enumerate() {
        for _ in 0..count {
            if frame < total_frames {
                data[i * total_frames + frame] = 1.0;
                frame += 1;
            }
        }
    }
    mlxcel_core::from_slice_f32(&data, &[l as i32, total_frames as i32])
}

/// Parse the `vocab` (phoneme -> id) map from `config.json`.
fn parse_vocab(config: &Value) -> Result<HashMap<String, i32>> {
    let obj = config
        .get("vocab")
        .and_then(Value::as_object)
        .ok_or_else(|| anyhow!("kokoro: config.json missing 'vocab' map"))?;
    let mut vocab = HashMap::with_capacity(obj.len());
    for (k, v) in obj {
        if let Some(id) = v.as_i64() {
            vocab.insert(k.clone(), id as i32);
        }
    }
    if vocab.is_empty() {
        return Err(anyhow!("kokoro: empty vocab"));
    }
    Ok(vocab)
}

#[cfg(test)]
mod tests {
    use super::stft::SAMPLE_RATE;
    use super::voices::{self, DEFAULT_VOICE};
    use super::*;

    /// Run a closure with a thread-local MLX stream installed, the same setup
    /// the audio worker performs. MLX evaluation is thread-affine, so any test
    /// that reads an array back must own a stream on its own thread.
    fn with_stream<R>(f: impl FnOnce() -> R) -> R {
        let stream = mlxcel_core::streams::new_thread_local_generation_stream();
        mlxcel_core::streams::install_thread_local_default_stream(stream.as_ref());
        f()
    }

    #[test]
    fn alignment_matrix_places_ones_per_span() {
        with_stream(|| {
            // tokens with durations [2, 3]; total frames = 5.
            let aln = alignment_matrix(&[2, 3], 2, 5);
            let host = ops::to_vec_f32(&aln).expect("readback");
            // row 0: frames 0,1 ; row 1: frames 2,3,4
            assert_eq!(host, vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        });
    }

    #[test]
    fn detection_keys_off_istftnet_block() {
        let cfg: Value = serde_json::json!({ "istftnet": { "gen_istft_n_fft": 20 } });
        assert!(is_kokoro_checkpoint(Path::new("/nonexistent"), &cfg));

        let cfg_no: Value = serde_json::json!({ "model_type": "llama" });
        assert!(!is_kokoro_checkpoint(Path::new("/nonexistent"), &cfg_no));
    }

    #[test]
    fn detection_falls_back_to_weight_filename() {
        // No `istftnet` block, but the canonical weight file exists in a tmpdir.
        let dir = std::env::temp_dir().join("kokoro_detection_test");
        let _ = std::fs::create_dir_all(&dir);
        let weight_path = dir.join(KOKORO_WEIGHT_FILE);
        std::fs::write(&weight_path, b"fake").expect("create fake weight file");

        let cfg_no_istftnet: Value = serde_json::json!({ "some_other_key": 1 });
        assert!(
            is_kokoro_checkpoint(&dir, &cfg_no_istftnet),
            "weight file presence triggers detection even without istftnet block"
        );

        // Clean up (best-effort).
        let _ = std::fs::remove_file(&weight_path);
    }

    /// Resolve the local Kokoro checkpoint used for development, if present.
    /// Returns `None` (skipping the test) when the asset is unavailable, so the
    /// suite stays green on machines without the weights.
    fn checkpoint_dir() -> Option<std::path::PathBuf> {
        [
            std::env::var("KOKORO_MODEL_DIR").ok().map(Into::into),
            Some(std::path::PathBuf::from("models/kokoro-82m")),
            std::env::var("HOME")
                .ok()
                .map(|h| std::path::PathBuf::from(h).join("models/kokoro-82m")),
        ]
        .into_iter()
        .flatten()
        .find(|cand| cand.join(KOKORO_WEIGHT_FILE).exists())
    }

    #[test]
    fn loads_checkpoint_and_synthesizes_audio() {
        let Some(dir) = checkpoint_dir() else {
            eprintln!("skipping: Kokoro checkpoint not found");
            return;
        };
        with_stream(|| {
            let model = KokoroModel::load(&dir).expect("load Kokoro checkpoint");
            // A short phoneme string drawn from the vocab; bypasses g2p so the
            // acoustic path is exercised deterministically.
            let phonemes = "hˈɛloʊ";
            let (samples, sr) = model
                .synthesize(phonemes, Some(DEFAULT_VOICE), 1.0)
                .expect("synthesize");
            assert_eq!(sr, SAMPLE_RATE, "sample rate is 24 kHz");
            assert!(
                samples.len() > SAMPLE_RATE as usize / 10,
                "expected a non-trivial waveform (> 0.1s), got {} samples",
                samples.len()
            );
            let rms = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();
            assert!(
                rms.is_finite() && rms > 1e-4,
                "waveform has signal (rms {rms})"
            );
            assert!(
                samples.iter().all(|s| s.is_finite() && s.abs() <= 1.5),
                "samples are finite and roughly bounded in [-1.5, 1.5]"
            );
        });
    }

    #[test]
    fn voice_pack_loads_and_indexes() {
        let Some(dir) = checkpoint_dir() else {
            eprintln!("skipping: Kokoro checkpoint not found");
            return;
        };
        with_stream(|| {
            let pack = voices::VoicePack::load(&dir, DEFAULT_VOICE).expect("load voice pack");
            let row = pack.row(6); // 6-token sequence -> row index 5
            assert_eq!(ops::shape(&row), vec![1, 256], "style row is (1, 256)");
            let host = ops::to_vec_f32(&row).expect("readback");
            assert_eq!(host.len(), 256);
            assert!(host.iter().any(|&v| v != 0.0), "style row is non-zero");
        });
    }

    #[test]
    fn speed_scales_duration() {
        let Some(dir) = checkpoint_dir() else {
            eprintln!("skipping: Kokoro checkpoint not found");
            return;
        };
        with_stream(|| {
            let model = KokoroModel::load(&dir).expect("load");
            let phonemes = "hˈɛloʊ wˈɜːld";
            let (slow, _) = model
                .synthesize(phonemes, Some(DEFAULT_VOICE), 0.7)
                .expect("slow");
            let (fast, _) = model
                .synthesize(phonemes, Some(DEFAULT_VOICE), 1.5)
                .expect("fast");
            // Faster speed divides the duration, so it must yield fewer samples.
            assert!(
                fast.len() < slow.len(),
                "speed 1.5 ({} samples) should be shorter than speed 0.7 ({} samples)",
                fast.len(),
                slow.len()
            );
        });
    }
}
