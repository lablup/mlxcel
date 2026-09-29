// Copyright 2025-2026 Lablup Inc.
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

//! Random-weight unit tests for the VoiceChat codec port.

use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::cache::{CacheKey, CausalConv1dCache};
use super::{CodecConfig, NemotronCodec, stft};

fn small_config() -> CodecConfig {
    CodecConfig {
        sample_rate: 24_000,
        base_channels: 8,
        channel_multipliers: vec![1, 2],
        downsample_rates: vec![2, 3],
        blocks_per_stage: 1,
        block_kernel_size: 3,
        latent_dim: 6,
        n_fft: 16,
        hop_length: 4,
        num_quantizers: 3,
        codebook_size: 8,
    }
}

struct Rng(u32);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (self.0 >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
    }

    fn array(&mut self, shape: &[i32], scale: f32, offset: f32) -> UniquePtr<MlxArray> {
        let total: usize = shape.iter().map(|&d| d as usize).product();
        let data: Vec<f32> = (0..total).map(|_| offset + scale * self.next()).collect();
        mlxcel_core::from_slice_f32(&data, shape)
    }
}

fn to_vec(a: &MlxArray) -> Vec<f32> {
    mlxcel_core::utils::array_to_vec_f32(a)
}

fn put_block(w: &mut WeightMap, rng: &mut Rng, p: &str, c: i32, k: i32) {
    w.insert(
        format!("{p}.dwconv.weight"),
        rng.array(&[c, 1, k], 0.3, 0.0),
    );
    w.insert(format!("{p}.dwconv.bias"), rng.array(&[c], 0.05, 0.0));
    w.insert(format!("{p}.norm.weight"), rng.array(&[c], 0.1, 1.0));
    w.insert(format!("{p}.norm.bias"), rng.array(&[c], 0.05, 0.0));
    w.insert(
        format!("{p}.pwconv1.weight"),
        rng.array(&[4 * c, c, 1], 0.3, 0.0),
    );
    w.insert(format!("{p}.pwconv1.bias"), rng.array(&[4 * c], 0.05, 0.0));
    w.insert(
        format!("{p}.pwconv2.weight"),
        rng.array(&[c, 4 * c, 1], 0.2, 0.0),
    );
    w.insert(format!("{p}.pwconv2.bias"), rng.array(&[c], 0.05, 0.0));
}

/// Random weights in the checkpoint's PyTorch conv layout.
fn torch_weights(cfg: &CodecConfig, prefix: &str) -> WeightMap {
    let mut rng = Rng(7);
    let mut w = WeightMap::new();
    let ch: Vec<i32> = cfg.stage_channels().iter().map(|&c| c as i32).collect();
    let (k, stft_ch, latent) = (
        cfg.block_kernel_size as i32,
        cfg.stft_channels() as i32,
        cfg.latent_dim as i32,
    );
    let mut idx = 0;
    let mut key = |kind: &str| {
        let s = format!("{prefix}.{kind}.layers.{idx}");
        idx += 1;
        s
    };
    w.insert(
        format!("{}.weight", key("encoder")),
        rng.array(&[ch[0], stft_ch, 1], 0.3, 0.0),
    );
    for (s, &rate) in cfg.downsample_rates.iter().enumerate() {
        for _ in 0..cfg.blocks_per_stage {
            put_block(&mut w, &mut rng, &key("encoder"), ch[s], k);
        }
        let next = ch.get(s + 1).copied().unwrap_or(latent);
        let shape = [next, ch[s], rate as i32];
        w.insert(
            format!("{}.weight", key("encoder")),
            rng.array(&shape, 0.3, 0.0),
        );
    }
    let mut idx = 0;
    let mut key = |kind: &str| {
        let s = format!("{prefix}.{kind}.layers.{idx}");
        idx += 1;
        s
    };
    let mut source = latent;
    for (s, &rate) in cfg.downsample_rates.iter().enumerate().rev() {
        let shape = [source, ch[s], rate as i32];
        w.insert(
            format!("{}.weight", key("decoder")),
            rng.array(&shape, 0.3, 0.0),
        );
        for _ in 0..cfg.blocks_per_stage {
            put_block(&mut w, &mut rng, &key("decoder"), ch[s], k);
        }
        source = ch[s];
    }
    w.insert(
        format!("{}.weight", key("decoder")),
        rng.array(&[stft_ch, ch[0], 1], 0.3, 0.0),
    );
    for q in 0..cfg.num_quantizers {
        let shape = [cfg.codebook_size as i32, latent];
        w.insert(
            format!("{prefix}.prvq.mus_list.{q}"),
            rng.array(&shape, 0.5, 0.0),
        );
    }
    w
}

/// Convert to MLX layout with the reference `sanitize` rule.
fn to_mlx_layout(cfg: &CodecConfig, torch: &WeightMap) -> WeightMap {
    let stage_width = cfg.blocks_per_stage + 1;
    let region = stage_width * cfg.downsample_rates.len();
    torch
        .iter()
        .map(|(key, value)| {
            let is_weight = key.ends_with(".weight") && mlxcel_core::array_shape(value).len() == 3;
            let out = if !is_weight {
                mlxcel_core::copy(value)
            } else if let Some(rest) = key.split(".decoder.layers.").nth(1) {
                let layer: usize = rest.split('.').next().unwrap().parse().unwrap();
                if layer < region && layer.is_multiple_of(stage_width) && !key.contains("dwconv") {
                    mlxcel_core::transpose_axes(value, &[1, 2, 0])
                } else {
                    mlxcel_core::transpose_axes(value, &[0, 2, 1])
                }
            } else {
                mlxcel_core::transpose_axes(value, &[0, 2, 1])
            };
            (key.clone(), mlxcel_core::contiguous(&out, false))
        })
        .collect()
}

fn small_codec() -> NemotronCodec {
    let cfg = small_config();
    NemotronCodec::from_weights(
        &torch_weights(&cfg, "tts_model.audio_codec"),
        "tts_model.audio_codec",
        &cfg,
    )
    .unwrap()
}

fn wave(n: usize) -> UniquePtr<MlxArray> {
    let data: Vec<f32> = (0..n).map(|i| 0.5 * (i as f32 * 0.37).sin()).collect();
    mlxcel_core::from_slice_f32(&data, &[1, 1, n as i32])
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
fn default_config_matches_checkpoint() {
    let cfg: CodecConfig = serde_json::from_str(r#"{"latent_dim": 512}"#).unwrap();
    assert_eq!(cfg, CodecConfig::default());
    assert_eq!(cfg.waveform_to_token_ratio(), 1764);
    assert_eq!(cfg.stft_channels(), 18);
    assert!((cfg.frame_rate() - 12.5).abs() < 1e-9);
    cfg.validate().unwrap();
    let bad = CodecConfig {
        hop_length: 5,
        ..CodecConfig::default()
    };
    assert!(bad.validate().is_err());
}

#[test]
fn encode_decode_shapes() {
    let codec = small_codec();
    let ratio = codec.waveform_to_token_ratio();
    assert_eq!(ratio, 24);
    let codes = codec.encode(&wave(ratio * 5 + 7)).unwrap();
    assert_eq!(mlxcel_core::array_shape(&codes), vec![1, 3, 5]);
    assert_eq!(mlxcel_core::array_dtype(&codes), mlxcel_core::dtype::INT32);
    assert!(to_vec(&codes).iter().all(|&c| (0.0..8.0).contains(&c)));
    // [B, N] input gives the same codes.
    let flat = mlxcel_core::reshape(&wave(ratio * 5 + 7), &[1, (ratio * 5 + 7) as i32]);
    assert_eq!(to_vec(&codec.encode(&flat).unwrap()), to_vec(&codes));
    let audio = codec.decode(&codes).unwrap();
    assert_eq!(
        mlxcel_core::array_shape(&audio),
        vec![1, 1, 5 * ratio as i32]
    );
    assert_eq!(
        mlxcel_core::array_dtype(&audio),
        mlxcel_core::dtype::FLOAT32
    );
    assert!(to_vec(&audio).iter().all(|v| v.is_finite()));
    let stereo = mlxcel_core::zeros(&[1, 2, 64], mlxcel_core::dtype::FLOAT32);
    assert!(codec.encode(&stereo).is_err());
}

#[test]
fn layout_gate_is_idempotent() {
    let cfg = small_config();
    let torch = torch_weights(&cfg, "c");
    let mlx = to_mlx_layout(&cfg, &torch);
    let a = NemotronCodec::from_weights(&torch, "c", &cfg).unwrap();
    let b = NemotronCodec::from_weights(&mlx, "c.", &cfg).unwrap();
    let codes = a.encode(&wave(24 * 4)).unwrap();
    assert_eq!(to_vec(&codes), to_vec(&b.encode(&wave(24 * 4)).unwrap()));
    let (da, db) = (
        to_vec(&a.decode(&codes).unwrap()),
        to_vec(&b.decode(&codes).unwrap()),
    );
    assert!(max_abs(&da, &db) < 1e-6);
    let mut broken = to_mlx_layout(&cfg, &torch);
    broken.insert(
        "c.decoder.layers.0.weight".into(),
        mlxcel_core::zeros(&[3, 3, 3], 0),
    );
    assert!(NemotronCodec::from_weights(&broken, "c", &cfg).is_err());
}

#[test]
fn decode_step_stream_matches_full_decode_after_delay() {
    let codec = small_codec();
    let (ratio, delay) = (
        codec.waveform_to_token_ratio(),
        codec.stream_delay_samples(),
    );
    assert_eq!(delay, 8);
    let frames = 6;
    let ids: Vec<i32> = (0..3 * frames).map(|i| (i * 5 + 3) % 8).collect();
    let codes = mlxcel_core::from_slice_i32(&ids, &[1, 3, frames]);
    let full = to_vec(&codec.decode(&codes).unwrap());
    let mut cache = CausalConv1dCache::new();
    let mut stream = Vec::new();
    for t in 0..frames {
        let step = mlxcel_core::slice(&codes, &[0, 0, t], &[1, 3, t + 1]);
        let flush = t == frames - 1;
        let out = to_vec(&codec.decode_step(&step, &mut cache, flush).unwrap());
        assert_eq!(out.len(), if flush { ratio + delay } else { ratio });
        stream.extend(out);
    }
    assert!(cache.is_empty(), "flush must drop every cache entry");
    assert_eq!(stream.len(), full.len() + delay);
    // The reference guarantees equality away from the two ends: the first and
    // last (n_fft - hop) / 2 = 6 samples of the full decode use a thinner
    // overlap-add window envelope than the stream (zero pre-roll / flush tail).
    let edge = 6;
    let stream = &stream[delay..];
    let err = max_abs(
        &stream[edge..full.len() - edge],
        &full[edge..full.len() - edge],
    );
    assert!(err < 1e-5, "stream vs full interior max abs {err}");
}

#[test]
fn stft_istft_round_trip() {
    let n = 96usize;
    let data: Vec<f32> = (0..n).map(|i| 0.8 * (i as f32 * 0.21).sin()).collect();
    let x = mlxcel_core::from_slice_f32(&data, &[1, n as i32]);
    let (re, im) = stft::spectrogram(&x, 16, 4).unwrap();
    assert_eq!(mlxcel_core::array_shape(&re), vec![1, 24, 9]);
    let y = stft::istft(&re, &im, 16, 4).unwrap();
    let y = to_vec(&y);
    assert_eq!(y.len(), n + 12);
    assert!(max_abs(&y[6..6 + n], &data) < 1e-5);
}

#[test]
fn invalid_codes_are_rejected() {
    let codec = small_codec();
    let over = mlxcel_core::from_slice_i32(&[0, 1, 8], &[1, 3, 1]);
    assert!(codec.decode(&over).is_err());
    let rank2 = mlxcel_core::from_slice_i32(&[0, 1, 2], &[3, 1]);
    assert!(codec.decode(&rank2).is_err());
    let too_many = mlxcel_core::from_slice_i32(&[0; 4], &[1, 4, 1]);
    assert!(codec.decode(&too_many).is_err());
}

#[test]
fn cache_update_semantics() {
    let mut cache = CausalConv1dCache::new();
    let x = mlxcel_core::from_slice_f32(&[1.0, 2.0, 3.0], &[1, 3, 1]);
    let padded = cache.update(&x, CacheKey::Block(0), 2, false).unwrap();
    assert_eq!(to_vec(&padded), vec![0.0, 0.0, 1.0, 2.0, 3.0]);
    let y = mlxcel_core::from_slice_f32(&[4.0], &[1, 1, 1]);
    let padded = cache.update(&y, CacheKey::Block(0), 2, true).unwrap();
    assert_eq!(to_vec(&padded), vec![2.0, 3.0, 4.0]);
    assert!(cache.is_empty());
    cache.update(&x, CacheKey::Block(1), 2, false).unwrap();
    let wide = mlxcel_core::zeros(&[1, 1, 2], mlxcel_core::dtype::FLOAT32);
    assert!(cache.update(&wide, CacheKey::Block(1), 2, false).is_err());
}
