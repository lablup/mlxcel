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

use super::*;

const FLOAT16: i32 = 9;
const FLOAT32: i32 = 10;
const BFLOAT16: i32 = 12;

/// A `[1, 3, 4]` activation-shaped tensor in `dtype`, with values that are
/// not exactly representable in half precision so a widening or narrowing
/// cast anywhere on the wire path would change the bytes.
fn activation(dtype: i32) -> UniquePtr<MlxArray> {
    let values: Vec<f32> = (0..12).map(|i| (i as f32 - 5.3) * 0.731).collect();
    let base = mlxcel_core::from_slice_f32(&values, &[1, 3, 4]);
    let arr = mlxcel_core::astype(&base, dtype);
    mlxcel_core::eval(&arr);
    arr
}

fn assert_round_trip_is_exact(original: &MlxArray) {
    let wire = serialize_mlx_array(original).expect("serialize");
    let restored = deserialize_wire_tensor(&wire).expect("deserialize");
    assert_eq!(
        mlxcel_core::array_dtype(&restored),
        mlxcel_core::array_dtype(original),
        "a stage boundary must not change the activation dtype"
    );
    assert_eq!(
        mlxcel_core::array_shape(&restored),
        mlxcel_core::array_shape(original)
    );
    assert_eq!(
        mlxcel_core::array_to_raw_bytes(&restored),
        mlxcel_core::array_to_raw_bytes(original),
        "a stage boundary must hand the next stage the exact bytes"
    );
}

/// Issue #2654: on CUDA the float16 branch went through the checkpoint
/// loader `from_bytes_f16`, which widens to float32, so a float16 model's
/// downstream stage computed in float32 and its logits left the dense ones.
#[test]
fn float16_activation_round_trips_in_float16() {
    assert_round_trip_is_exact(&activation(FLOAT16));
}

#[test]
fn bfloat16_activation_round_trips_in_bfloat16() {
    assert_round_trip_is_exact(&activation(BFLOAT16));
}

#[test]
fn float32_activation_round_trips_in_float32() {
    assert_round_trip_is_exact(&activation(FLOAT32));
}

#[test]
fn token_ids_round_trip_in_int32() {
    let ids = mlxcel_core::from_slice_i32(&[128000, 9906, 13], &[1, 3]);
    assert_round_trip_is_exact(&ids);
}

/// `from_bytes` used to build every 2-byte integer dtype from a `uint8_t*`,
/// one element per byte, so an int16 tensor crossed a stage boundary as the
/// first half of its buffer widened byte by byte.
#[test]
fn int16_tensor_round_trips_bit_exactly() {
    let values = mlxcel_core::from_slice_i32(&[-32768, -1, 0, 1, 255, 256, 32767, 7], &[2, 4]);
    let arr = mlxcel_core::astype(&values, mlxcel_core::dtype::INT16);
    mlxcel_core::eval(&arr);
    assert_round_trip_is_exact(&arr);
}

/// Build an activation wire buffer by hand so the header shape and the
/// payload length can disagree, which `serialize_activation` refuses to emit.
fn hand_built_wire(dtype: TensorDtype, shape: &[u64], payload: &[u8]) -> Vec<u8> {
    use crate::distributed::tensor_protocol::{
        PROTOCOL_VERSION, TensorFlags, TensorHeader, TensorKind,
    };

    let header = TensorHeader {
        version: PROTOCOL_VERSION,
        flags: TensorFlags::new(),
        kind: TensorKind::Activation,
        dtype,
        shape: shape.to_vec(),
        metadata: Vec::new(),
        data_len: payload.len() as u64,
    };
    let mut wire = header.encode();
    wire.extend_from_slice(payload);
    wire
}

fn expect_size_mismatch(wire: &[u8]) {
    let err = match deserialize_wire_tensor(wire) {
        Ok(_) => panic!("a payload shorter than its shape must be rejected before from_bytes"),
        Err(err) => err.to_string(),
    };
    assert!(
        err.contains("size mismatch"),
        "expected a size-mismatch error, got: {err}"
    );
}

/// A peer that claims a large float16 activation but ships two bytes must
/// not make `from_bytes` read past the payload.
#[test]
fn float16_short_payload_is_rejected() {
    expect_size_mismatch(&hand_built_wire(
        TensorDtype::Float16,
        &[1, 64, 64],
        &[0u8; 2],
    ));
}

#[test]
fn float32_short_payload_is_rejected() {
    // `[1, 3, 4]` float32 needs 48 bytes; send one element short.
    expect_size_mismatch(&hand_built_wire(
        TensorDtype::Float32,
        &[1, 3, 4],
        &[0u8; 44],
    ));
}

/// An empty shape is a one-element MLX scalar, so an empty payload must be
/// rejected.
#[test]
fn scalar_shape_with_empty_payload_is_rejected() {
    expect_size_mismatch(&hand_built_wire(TensorDtype::Float32, &[], &[]));
}

/// The hand-built header is a valid frame: a correctly sized bfloat16
/// payload still decodes to the same bytes in bfloat16.
#[test]
fn hand_built_bfloat16_buffer_round_trips() {
    let original = activation(BFLOAT16);
    let bytes = mlxcel_core::array_to_raw_bytes(&original);
    let wire = hand_built_wire(TensorDtype::BFloat16, &[1, 3, 4], &bytes);
    let restored = deserialize_wire_tensor(&wire).expect("deserialize");
    assert_eq!(mlxcel_core::array_dtype(&restored), BFLOAT16);
    assert_eq!(mlxcel_core::array_shape(&restored), vec![1, 3, 4]);
    assert_eq!(mlxcel_core::array_to_raw_bytes(&restored), bytes);
}

// --- Receiver-side limits (issue #2716) ---

const WIDTH: u64 = 8;

fn stage_limits() -> StageWireLimits {
    StageWireLimits {
        activation_width: Some(WIDTH),
        vocab_size: Some(32),
    }
}

/// A header whose `data_len` matches the shape but with no payload bytes
/// behind it.
fn header_only(flags: u8, dtype: TensorDtype, shape: &[u64]) -> Vec<u8> {
    use crate::distributed::tensor_protocol::{
        PROTOCOL_VERSION, TensorFlags, TensorHeader, TensorKind,
    };

    let elements = shape.iter().product::<u64>();
    TensorHeader {
        version: PROTOCOL_VERSION,
        flags: TensorFlags::from_bits(flags),
        kind: TensorKind::Activation,
        dtype,
        shape: shape.to_vec(),
        metadata: Vec::new(),
        data_len: elements * dtype.element_size().max(1) as u64,
    }
    .encode()
}

fn wire_err(wire: &[u8], limits: &TensorDecodeLimits) -> String {
    match deserialize_wire_tensor_with_limits(wire, limits) {
        Ok(_) => panic!("the frame must be rejected"),
        Err(err) => format!("{err:#}"),
    }
}

/// Hidden states of the wrong width, batch, or token count are refused on
/// the header alone: the error is the shape, not "buffer too short", so the
/// check runs before the payload is touched.
#[test]
fn hidden_states_shape_is_checked_before_the_payload() {
    let limits = stage_limits().hidden_states();
    for (shape, needle) in [
        ([1, 8, WIDTH + 1], "axis 2 is 9, expected exactly 8"),
        ([2, 8, WIDTH], "axis 0 is 2, expected exactly 1"),
        (
            [1, MAX_STAGE_TOKENS_PER_FRAME + 1, WIDTH],
            "axis 1 is 1048577, expected between 1 and 1048576",
        ),
    ] {
        let err = wire_err(&header_only(0, TensorDtype::Float16, &shape), &limits);
        assert!(err.contains(needle), "{shape:?}: unexpected error: {err}");
    }
    // Control: an admissible header reaches the payload check.
    let err = wire_err(
        &header_only(0, TensorDtype::Float16, &[1, 8, WIDTH]),
        &limits,
    );
    assert!(err.contains("buffer too short"), "unexpected error: {err}");
}

/// The activation decoder refuses COMPRESSED and QUANTIZED, which honest
/// senders never set, before `decompress` or `dequantize_*` runs. The
/// 13-byte LZ4 payload declares about 4 GiB under a shape that allows it.
#[test]
fn activation_decode_refuses_compressed_and_quantized_frames() {
    use crate::distributed::tensor_protocol::{
        PROTOCOL_VERSION, TensorFlags, TensorHeader, TensorKind,
    };

    let declared = u32::MAX;
    let mut payload = u64::from(declared).to_le_bytes().to_vec();
    payload.extend_from_slice(&declared.to_le_bytes());
    payload.push(0x00);
    assert_eq!(payload.len(), 13);
    for (flag, name, dtype, shape) in [
        (
            TensorFlags::COMPRESSED,
            "COMPRESSED",
            TensorDtype::Float32,
            vec![1, 1 << 20, 1024],
        ),
        (
            TensorFlags::QUANTIZED,
            "QUANTIZED",
            TensorDtype::Int8,
            vec![1, 1 << 20, 2048],
        ),
    ] {
        let mut wire = TensorHeader {
            version: PROTOCOL_VERSION,
            flags: TensorFlags::from_bits(flag),
            kind: TensorKind::Activation,
            dtype,
            shape,
            metadata: Vec::new(),
            data_len: payload.len() as u64,
        }
        .encode();
        wire.extend_from_slice(&payload);
        let err = match ActivationMessage::deserialize_activation(&wire) {
            Ok(tensor) => panic!("{name} must be refused, got {} bytes", tensor.data.len()),
            Err(err) => format!("{err:#}"),
        };
        assert!(
            err.contains(&format!("flag {name}")),
            "{name}: unexpected error: {err}"
        );
        // Permissive limits do not re-enable either flag.
        let err = match ActivationMessage::deserialize_activation_with_limits(
            &wire,
            &TensorDecodeLimits::unrestricted(),
        ) {
            Ok(tensor) => panic!("{name} must be refused, got {} bytes", tensor.data.len()),
            Err(err) => format!("{err:#}"),
        };
        assert!(err.contains(&format!("flag {name}")), "{name}: {err}");
    }
}

#[test]
fn entry_tensor_limits_bound_ids_and_masks() {
    let limits = stage_limits();
    let ids = limits.input_ids();
    let err = wire_err(&header_only(0, TensorDtype::Float32, &[1, 4]), &ids);
    assert!(err.contains("dtype float32"), "unexpected error: {err}");
    let err = wire_err(&header_only(0, TensorDtype::Int32, &[1, 4, 1]), &ids);
    assert!(err.contains("rank 3"), "unexpected error: {err}");

    let mask = limits.attention_mask();
    let err = wire_err(&header_only(0, TensorDtype::Bool, &[1, 1, 1, 4, 4]), &mask);
    assert!(err.contains("rank 2..=4"), "unexpected error: {err}");
    let err = wire_err(&header_only(0, TensorDtype::Int32, &[1, 1, 4, 4]), &mask);
    assert!(err.contains("dtype int32"), "unexpected error: {err}");
    let err = wire_err(
        &header_only(
            0,
            TensorDtype::Bool,
            &[1, 1, 4, MAX_STAGE_TOKENS_PER_FRAME + 1],
        ),
        &mask,
    );
    assert!(err.contains("axis 3"), "unexpected error: {err}");
}

#[test]
fn logits_limits_bound_sequence_and_vocab() {
    let limits = stage_limits().logits(4);
    let err = wire_err(&header_only(0, TensorDtype::Float32, &[1, 5, 32]), &limits);
    assert!(
        err.contains("axis 1 is 5, expected between 1 and 4"),
        "{err}"
    );
    let err = wire_err(&header_only(0, TensorDtype::Float32, &[1, 4, 33]), &limits);
    assert!(err.contains("axis 2 is 33, expected exactly 32"), "{err}");
    for shape in [[1, 4, 32], [1, 1, 32]] {
        let err = wire_err(&header_only(0, TensorDtype::BFloat16, &shape), &limits);
        assert!(err.contains("buffer too short"), "{shape:?}: {err}");
    }
}

/// A `None` width or vocabulary leaves that axis unbounded.
#[test]
fn unknown_width_and_vocab_are_unbounded() {
    let limits = StageWireLimits::default();
    for (wire, limits) in [
        (
            header_only(0, TensorDtype::Float16, &[1, 4, 123_456]),
            limits.hidden_states(),
        ),
        (
            header_only(0, TensorDtype::Float32, &[1, 4, 654_321]),
            limits.logits(4),
        ),
    ] {
        assert!(wire_err(&wire, &limits).contains("buffer too short"));
    }
}

/// An honest activation still decodes under the stage limits.
#[test]
fn honest_activation_round_trips_under_stage_limits() {
    let values: Vec<f32> = (0..16).map(|i| i as f32 * 0.25).collect();
    let arr = mlxcel_core::astype(
        &mlxcel_core::from_slice_f32(&values, &[1, 2, WIDTH as i32]),
        BFLOAT16,
    );
    mlxcel_core::eval(&arr);
    let wire = serialize_mlx_array(&arr).expect("serialize");
    let restored = deserialize_wire_tensor_with_limits(&wire, &stage_limits().hidden_states())
        .expect("an honest frame must decode");
    assert_eq!(
        mlxcel_core::array_to_raw_bytes(&restored),
        mlxcel_core::array_to_raw_bytes(&arr)
    );
}

/// #2726 item 4: a zero-length sequence axis (hidden states, token ids,
/// logits) or mask axis is refused on the header, with a message naming the
/// axis, instead of reaching the executor as an empty tensor.
#[test]
fn zero_length_axes_are_refused() {
    let limits = stage_limits();
    for (wire, limits, needle) in [
        (
            header_only(0, TensorDtype::Float16, &[1, 0, WIDTH]),
            limits.hidden_states(),
            "axis 1 is 0",
        ),
        (
            header_only(0, TensorDtype::Int32, &[1, 0]),
            limits.input_ids(),
            "axis 1 is 0",
        ),
        (
            header_only(0, TensorDtype::Float32, &[1, 0, 32]),
            limits.logits(4),
            "axis 1 is 0",
        ),
        (
            header_only(0, TensorDtype::Bool, &[1, 1, 0, 4]),
            limits.attention_mask(),
            "axis 2 is 0",
        ),
    ] {
        let err = match deserialize_wire_tensor_with_limits(&wire, &limits) {
            Ok(arr) => panic!(
                "{needle}: a zero-length axis must be refused, got shape {:?}",
                mlxcel_core::array_shape(arr.as_ref().unwrap())
            ),
            Err(err) => format!("{err:#}"),
        };
        assert!(err.contains(needle), "{needle}: unexpected error: {err}");
        assert!(err.contains("expected between 1 and"), "{err}");
    }
}
