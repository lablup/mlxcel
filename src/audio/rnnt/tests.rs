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

use mlxcel_core::utils::array_to_vec_f32;
use mlxcel_core::weights::WeightMap;

use super::*;

const H: usize = 4;
pub(super) const ENC: usize = 6;
pub(super) const VOCAB: usize = 5; // blank = 5

fn rand_vec(n: usize, seed: &mut u64, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|_| {
            *seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (((*seed >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0) * scale
        })
        .collect()
}

fn put_vec(w: &mut WeightMap, key: &str, data: &[f32], shape: &[i32]) {
    w.insert(key.to_string(), mlxcel_core::from_slice_f32(data, shape));
}

fn put(w: &mut WeightMap, key: &str, shape: &[i32], seed: &mut u64) {
    let n = shape.iter().product::<i32>() as usize;
    put_vec(w, key, &rand_vec(n, seed, 0.5), shape);
}

fn predict_args() -> PredictArgs {
    PredictArgs {
        pred_hidden: H,
        pred_rnn_layers: 2,
        vocab_size: VOCAB,
        blank_as_pad: true,
    }
}

fn joint_args() -> JointArgs {
    JointArgs {
        joint_hidden: 3,
        encoder_hidden: ENC,
        pred_hidden: H,
        num_classes: VOCAB,
        ..JointArgs::default()
    }
}

/// Torch-named weights; the output layer is zero so logits equal `out_bias`.
fn weights(out_bias: &[f32; VOCAB + 1]) -> WeightMap {
    let mut s = 11u64;
    let mut w = WeightMap::new();
    let (h, g) = (H as i32, 4 * H as i32);
    put(
        &mut w,
        "dec.prediction.embed.weight",
        &[VOCAB as i32 + 1, h],
        &mut s,
    );
    for n in 0..2 {
        let p = "dec.prediction.dec_rnn.lstm";
        put(&mut w, &format!("{p}.weight_ih_l{n}"), &[g, h], &mut s);
        put(&mut w, &format!("{p}.weight_hh_l{n}"), &[g, h], &mut s);
        put(&mut w, &format!("{p}.bias_ih_l{n}"), &[g], &mut s);
        put(&mut w, &format!("{p}.bias_hh_l{n}"), &[g], &mut s);
    }
    put(&mut w, "joint.enc.weight", &[3, ENC as i32], &mut s);
    put(&mut w, "joint.enc.bias", &[3], &mut s);
    put(&mut w, "joint.pred.weight", &[3, h], &mut s);
    put(&mut w, "joint.pred.bias", &[3], &mut s);
    put_vec(
        &mut w,
        "joint.joint_net.2.weight",
        &[0.0; 3 * (VOCAB + 1)],
        &[VOCAB as i32 + 1, 3],
    );
    put_vec(
        &mut w,
        "joint.joint_net.2.bias",
        out_bias,
        &[VOCAB as i32 + 1],
    );
    w
}

pub(super) fn decoder(out_bias: &[f32; VOCAB + 1]) -> RnntDecoder {
    RnntDecoder::from_weights(
        &weights(out_bias),
        "dec",
        "joint",
        &predict_args(),
        &joint_args(),
    )
    .unwrap()
}

pub(super) fn encoded(frames: i32) -> UniquePtr<MlxArray> {
    let data = rand_vec(frames as usize * ENC, &mut 3u64, 1.0);
    mlxcel_core::from_slice_f32(&data, &[1, frames, ENC as i32])
}

#[test]
fn blank_only_frame_emits_nothing() {
    let dec = decoder(&[0.0, 0.0, 0.0, 0.0, 0.0, 10.0]);
    assert_eq!(dec.blank_id(), 5);
    let enc = encoded(3);
    let mut state = dec.initial_state();
    let frame = mlxcel_core::slice(&enc, &[0, 0, 0], &[1, 1, ENC as i32]);
    assert!(dec.step_frame(&frame, &mut state, 10).unwrap().is_empty());
    // The state only advances on a non-blank symbol.
    assert_eq!(state.last_token(), 5);
    assert!(state.hidden.is_none());
    assert!(dec.greedy_decode(&enc, 3, 10).unwrap().is_empty());
}

#[test]
fn max_symbols_bounds_loop() {
    let dec = decoder(&[0.0, 0.0, 10.0, 0.0, 0.0, 0.0]);
    let enc = encoded(2);
    assert_eq!(dec.greedy_decode(&enc, 2, 3).unwrap(), vec![2; 6]);
    // `max_symbols = 0` still emits one symbol per frame, like the reference.
    assert_eq!(dec.greedy_decode(&enc, 2, 0).unwrap(), vec![2; 2]);
    // `length` caps the frames decoded.
    assert_eq!(dec.greedy_decode(&enc, 1, 2).unwrap(), vec![2; 2]);
    let mut state = dec.initial_state();
    let frame = mlxcel_core::slice(&enc, &[0, 0, 0], &[1, 1, ENC as i32]);
    dec.step_frame(&frame, &mut state, 1).unwrap();
    assert_eq!(state.last_token(), 2);
    assert!(state.hidden.is_some());
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// CPU LSTM step with torch-layout `w_ih [4H, in]`, `w_hh [4H, H]`.
fn cpu_step(
    x: &[f32],
    state: Option<(&[f32], &[f32])>,
    w_ih: &[f32],
    w_hh: &[f32],
    bias: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let gates: Vec<f32> = (0..4 * H)
        .map(|r| {
            let mut v = bias[r] + (0..H).map(|k| w_ih[r * H + k] * x[k]).sum::<f32>();
            if let Some((h, _)) = state {
                v += (0..H).map(|k| w_hh[r * H + k] * h[k]).sum::<f32>();
            }
            v
        })
        .collect();
    let mut h = vec![0.0; H];
    let mut c = vec![0.0; H];
    for j in 0..H {
        let (i, f, g, o) = (
            sigmoid(gates[j]),
            sigmoid(gates[H + j]),
            gates[2 * H + j].tanh(),
            sigmoid(gates[3 * H + j]),
        );
        c[j] = i * g + state.map_or(0.0, |(_, cp)| f * cp[j]);
        h[j] = o * c[j].tanh();
    }
    (h, c)
}

#[test]
fn stacked_lstm_matches_cpu_reference_in_both_layouts() {
    let torch = weights(&[0.0; VOCAB + 1]);
    let p = "dec.prediction.dec_rnn.lstm";
    let read = |k: &str| array_to_vec_f32(torch.get(k).unwrap());
    let mut mlx = WeightMap::new();
    for n in 0..2 {
        let bias: Vec<f32> = read(&format!("{p}.bias_ih_l{n}"))
            .iter()
            .zip(read(&format!("{p}.bias_hh_l{n}")))
            .map(|(a, b)| a + b)
            .collect();
        let g = 4 * H as i32;
        put_vec(
            &mut mlx,
            &format!("{p}.{n}.Wx"),
            &read(&format!("{p}.weight_ih_l{n}")),
            &[g, H as i32],
        );
        put_vec(
            &mut mlx,
            &format!("{p}.{n}.Wh"),
            &read(&format!("{p}.weight_hh_l{n}")),
            &[g, H as i32],
        );
        put_vec(&mut mlx, &format!("{p}.{n}.bias"), &bias, &[g]);
    }
    let lstm_t = StackedLstm::from_weights(&torch, p, H, H, 2).unwrap();
    let lstm_m = StackedLstm::from_weights(&mlx, p, H, H, 2).unwrap();
    assert_eq!(lstm_t.num_layers(), 2);

    let x1 = [0.3f32, -0.2, 0.5, 0.1];
    let x2 = [-0.4f32, 0.6, 0.0, 0.2];
    let input = |x: &[f32]| mlxcel_core::from_slice_f32(x, &[1, H as i32]);
    let (out_a, state_a) = lstm_t.step(&input(&x1), None);
    let (out_b, _) = lstm_m.step(&input(&x1), None);
    let (out2, _) = lstm_t.step(&input(&x2), Some(&state_a));

    // CPU reference over both layers and both steps.
    let layer = |n: usize| {
        (
            read(&format!("{p}.weight_ih_l{n}")),
            read(&format!("{p}.weight_hh_l{n}")),
            read(&format!("{p}.bias_ih_l{n}"))
                .iter()
                .zip(read(&format!("{p}.bias_hh_l{n}")))
                .map(|(a, b)| a + b)
                .collect::<Vec<f32>>(),
        )
    };
    let (l0, l1) = (layer(0), layer(1));
    let (h0, c0) = cpu_step(&x1, None, &l0.0, &l0.1, &l0.2);
    let (h1, c1) = cpu_step(&h0, None, &l1.0, &l1.1, &l1.2);
    let (h0b, _) = cpu_step(&x2, Some((&h0, &c0)), &l0.0, &l0.1, &l0.2);
    let (h1b, _) = cpu_step(&h0b, Some((&h1, &c1)), &l1.0, &l1.1, &l1.2);

    let close = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-5);
    assert!(close(&array_to_vec_f32(&out_a), &h1));
    assert!(close(&array_to_vec_f32(&out_b), &h1));
    assert!(close(&array_to_vec_f32(&out2), &h1b));
}

#[test]
fn piece_decode_maps_underscore_to_space_and_drops_lang_tags() {
    let vocab: Vec<String> = [
        "<unk>",
        "\u{2581}What",
        "\u{2581}is",
        "<en-US>",
        "s",
        "<pad>",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    assert_eq!(
        decode_pieces(&[3, 1, 2, 4, 0, 5, 99, -1], &vocab),
        " What iss"
    );
    assert!(is_special_token(3, &vocab));
    assert!(is_special_token(0, &vocab));
    assert!(!is_special_token(1, &vocab));
    assert!(!is_special_token(99, &vocab));
    for tag in ["<en-US>", "<zh-Hans>", "<eng-GB>", "<de-de>"] {
        assert!(is_lang_tag(tag), "{tag}");
    }
    for not_tag in ["<EN-us>", "<e-US>", "<en-USAAA>", "en-US", "<unk>", "<s>"] {
        assert!(!is_lang_tag(not_tag), "{not_tag}");
    }
    assert!(is_special_piece("</s>") && is_special_piece("<s>"));

    let dec = decoder(&[0.0, 10.0, 0.0, 0.0, 0.0, 0.0]);
    let text = dec.transcribe(&encoded(2), 2, 1, &vocab).unwrap();
    assert_eq!(text, "What What");
}

#[test]
fn args_deserialize_checkpoint_objects() {
    let predict: PredictArgs = serde_json::from_str(
        r#"{"pred_hidden":640,"pred_rnn_layers":2,"t_max":null,"dropout":0.2,
            "vocab_size":1024,"blank_as_pad":true}"#,
    )
    .unwrap();
    assert_eq!(predict, PredictArgs::default());
    let joint: JointArgs = serde_json::from_str(
        r#"{"joint_hidden":640,"activation":"relu","dropout":0.2,"encoder_hidden":1024,
            "pred_hidden":640,"num_classes":1024}"#,
    )
    .unwrap();
    assert_eq!(joint, JointArgs::default());
    let bad = JointArgs {
        activation: "gelu".to_string(),
        ..joint_args()
    };
    assert!(
        RnntDecoder::from_weights(&weights(&[0.0; 6]), "dec", "joint", &predict_args(), &bad)
            .is_err()
    );
}
