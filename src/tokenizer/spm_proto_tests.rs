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

//! Tests for the SentencePiece `ModelProto` `add_dummy_prefix` rewrite (#1357).

use super::*;
use sentencepiece::SentencePieceProcessor;
use std::path::Path;

/// The IQuest-Coder checkpoint the end-to-end assertions read. Absent on
/// machines that never downloaded it, in which case the shared pinned gate
/// skips (or fails under `MLXCEL_REQUIRE_PINNED_CHECKPOINTS=1`).
const IQUEST_MODEL_DIR: &str = "models/mlx/iquest-coder-v1-7b-instruct-8bit";

fn varint(mut value: u64) -> Vec<u8> {
    let mut out = Vec::new();
    loop {
        let byte = (value & 0x7f) as u8;
        value >>= 7;
        if value == 0 {
            out.push(byte);
            return out;
        }
        out.push(byte | 0x80);
    }
}

fn len_field(number: u32, payload: &[u8]) -> Vec<u8> {
    let mut out = varint(u64::from(number) << 3 | 2);
    out.extend(varint(payload.len() as u64));
    out.extend_from_slice(payload);
    out
}

fn varint_field(number: u32, value: u64) -> Vec<u8> {
    let mut out = varint(u64::from(number) << 3);
    out.extend(varint(value));
    out
}

/// Return the payload of every `normalizer_spec` field, in wire order.
fn normalizer_bodies(proto: &[u8]) -> Vec<Vec<u8>> {
    scan_fields(proto)
        .expect("scan")
        .into_iter()
        .filter(|field| field.number == MODEL_NORMALIZER_SPEC_FIELD)
        .map(|field| {
            let (start, end) = field.payload.expect("normalizer_spec is length-delimited");
            proto[start..end].to_vec()
        })
        .collect()
}

/// Return the payload of the single `normalizer_spec` field, or `None`.
fn normalizer_body(proto: &[u8]) -> Option<Vec<u8>> {
    let mut bodies = normalizer_bodies(proto);
    assert!(bodies.len() <= 1, "more than one normalizer_spec emitted");
    bodies.pop()
}

/// Read every varint field with `number` out of a message body.
fn varint_values(body: &[u8], number: u32) -> Vec<u64> {
    let mut values = Vec::new();
    let mut pos = 0usize;
    while pos < body.len() {
        let tag = read_varint(body, &mut pos).expect("tag");
        let field_number = u32::try_from(tag >> 3).unwrap();
        let wire = tag & 0x7;
        match wire {
            0 => {
                let value = read_varint(body, &mut pos).expect("varint value");
                if field_number == number {
                    values.push(value);
                }
            }
            2 => {
                let len = read_varint(body, &mut pos).expect("len") as usize;
                pos += len;
            }
            other => panic!("unexpected wire type {other}"),
        }
    }
    values
}

/// A `normalizer_spec` body shaped like the one a real `tokenizer.model`
/// carries: a name, a (short, stand-in) charsmap, and the three booleans.
fn sample_normalizer_body(add_dummy_prefix: bool) -> Vec<u8> {
    let mut body = len_field(1, b"identity");
    body.extend(len_field(2, &[0xde, 0xad, 0xbe, 0xef]));
    body.extend(varint_field(3, u64::from(add_dummy_prefix)));
    body.extend(varint_field(4, 0));
    body.extend(varint_field(5, 1));
    body
}

/// A stand-in `ModelProto`: two pieces, a trainer spec, a normalizer spec and
/// a trailing self-test message, in the order sentencepiece emits them.
fn sample_model_proto(normalizer: Option<&[u8]>) -> Vec<u8> {
    let mut proto = len_field(1, b"\x0a\x05<unk>");
    proto.extend(len_field(1, b"\x0a\x03<s>"));
    proto.extend(len_field(2, &varint_field(3, 2)));
    if let Some(body) = normalizer {
        proto.extend(len_field(MODEL_NORMALIZER_SPEC_FIELD, body));
    }
    proto.extend(len_field(4, b""));
    proto
}

#[test]
fn a_present_add_dummy_prefix_is_rewritten_to_false() {
    let proto = sample_model_proto(Some(&sample_normalizer_body(true)));
    let patched = disable_add_dummy_prefix(&proto).expect("patch");
    let body = normalizer_body(&patched).expect("normalizer_spec survives");
    assert_eq!(varint_values(&body, 3), vec![0]);
}

#[test]
fn an_already_false_add_dummy_prefix_stays_false() {
    let proto = sample_model_proto(Some(&sample_normalizer_body(false)));
    let patched = disable_add_dummy_prefix(&proto).expect("patch");
    let body = normalizer_body(&patched).expect("normalizer_spec survives");
    assert_eq!(varint_values(&body, 3), vec![0]);
}

#[test]
fn the_other_normalizer_fields_survive_the_rewrite() {
    let proto = sample_model_proto(Some(&sample_normalizer_body(true)));
    let patched = disable_add_dummy_prefix(&proto).expect("patch");
    let body = normalizer_body(&patched).expect("normalizer_spec survives");

    // `remove_extra_whitespaces` (4) and `escape_whitespaces` (5) keep their
    // values; dropping either would change tokenization as surely as the
    // dummy prefix does.
    assert_eq!(varint_values(&body, 4), vec![0]);
    assert_eq!(varint_values(&body, 5), vec![1]);
    // `name` and `precompiled_charsmap` are copied byte for byte.
    assert!(body.windows(8).any(|w| w == b"identity"));
    assert!(body.windows(4).any(|w| w == [0xde, 0xad, 0xbe, 0xef]));
}

#[test]
fn every_top_level_field_and_its_order_survives_the_rewrite() {
    let proto = sample_model_proto(Some(&sample_normalizer_body(true)));
    let patched = disable_add_dummy_prefix(&proto).expect("patch");

    let before: Vec<u32> = scan_fields(&proto)
        .expect("scan")
        .iter()
        .map(|f| f.number)
        .collect();
    let after: Vec<u32> = scan_fields(&patched)
        .expect("scan")
        .iter()
        .map(|f| f.number)
        .collect();
    assert_eq!(before, after, "field order or membership changed");

    // The pieces are the vocabulary; a rewrite that perturbs them would
    // silently change every id in the model.
    let pieces_before: Vec<&[u8]> = scan_fields(&proto)
        .expect("scan")
        .iter()
        .filter(|f| f.number == 1)
        .map(|f| {
            let (s, e) = f.payload.unwrap();
            &proto[s..e]
        })
        .collect();
    let pieces_after: Vec<&[u8]> = scan_fields(&patched)
        .expect("scan")
        .iter()
        .filter(|f| f.number == 1)
        .map(|f| {
            let (s, e) = f.payload.unwrap();
            &patched[s..e]
        })
        .collect();
    assert_eq!(pieces_before, pieces_after);
}

#[test]
fn a_repeated_normalizer_spec_is_rewritten_in_every_occurrence() {
    // `normalizer_spec` is a singular field, but the wire format permits
    // repeats and protobuf merges them, so a later occurrence carrying
    // `add_dummy_prefix = true` would undo a patch applied only to the first.
    let mut proto = len_field(1, b"\x0a\x05<unk>");
    proto.extend(len_field(
        MODEL_NORMALIZER_SPEC_FIELD,
        &sample_normalizer_body(true),
    ));
    proto.extend(len_field(2, &varint_field(3, 2)));
    proto.extend(len_field(
        MODEL_NORMALIZER_SPEC_FIELD,
        &sample_normalizer_body(true),
    ));

    let patched = disable_add_dummy_prefix(&proto).expect("patch");
    let bodies = normalizer_bodies(&patched);
    assert_eq!(bodies.len(), 2, "both occurrences must survive");
    for body in bodies {
        assert_eq!(varint_values(&body, 3), vec![0]);
    }
}

#[test]
fn a_missing_normalizer_spec_gains_one_that_disables_the_prefix() {
    let proto = sample_model_proto(None);
    assert!(normalizer_body(&proto).is_none());

    let patched = disable_add_dummy_prefix(&proto).expect("patch");
    let body = normalizer_body(&patched).expect("normalizer_spec appended");
    assert_eq!(varint_values(&body, 3), vec![0]);
}

#[test]
fn duplicate_add_dummy_prefix_entries_collapse_to_a_single_false() {
    let mut body = varint_field(3, 1);
    body.extend(varint_field(3, 1));
    body.extend(len_field(1, b"identity"));
    let proto = sample_model_proto(Some(&body));

    let patched = disable_add_dummy_prefix(&proto).expect("patch");
    let patched_body = normalizer_body(&patched).expect("normalizer_spec survives");
    assert_eq!(varint_values(&patched_body, 3), vec![0]);
}

#[test]
fn a_truncated_message_is_rejected_rather_than_silently_repaired() {
    let proto = sample_model_proto(Some(&sample_normalizer_body(true)));
    let truncated = &proto[..proto.len() - 1];
    assert!(disable_add_dummy_prefix(truncated).is_err());

    // A length prefix that runs past the end of the buffer is the shape a
    // partially-downloaded tokenizer.model takes.
    let mut overlong = len_field(3, b"body");
    overlong[1] = 0x40;
    assert!(disable_add_dummy_prefix(&overlong).is_err());
}

#[test]
fn a_group_wire_type_is_rejected_rather_than_mis_parsed() {
    // Wire types 3 and 4 are the deprecated group encoding. Skipping them
    // needs a matching end-group tag, which this editor does not track, so
    // they must error rather than be walked as if they were varints.
    let proto = varint(u64::from(7u32) << 3 | 3);
    assert!(disable_add_dummy_prefix(&proto).is_err());
}

/// The end-to-end claim: on the IQuest-Coder checkpoint, the rewritten proto
/// tokenizes the first word with no leading word boundary, and every later
/// word keeps its `U+2581` marker.
#[test]
fn the_iquest_coder_model_loses_only_its_leading_word_boundary() {
    let model_path = Path::new(IQUEST_MODEL_DIR).join("tokenizer.model");
    let Ok(raw) = std::fs::read(&model_path) else {
        crate::test_support::pinned_checkpoint::skip_or_fail_pinned_checkpoint(
            "the_iquest_coder_model_loses_only_its_leading_word_boundary",
            &format!(
                "IQuest-Coder tokenizer.model not present at {}",
                model_path.display()
            ),
        );
        return;
    };

    let plain = SentencePieceProcessor::from_serialized_proto(&raw).expect("load original");
    let patched_bytes = disable_add_dummy_prefix(&raw).expect("patch");
    let patched =
        SentencePieceProcessor::from_serialized_proto(&patched_bytes).expect("load patched");

    assert_eq!(
        plain.len(),
        patched.len(),
        "the rewrite must not change the vocabulary size"
    );

    let ids = |sp: &SentencePieceProcessor, text: &str| -> Vec<u32> {
        sp.encode(text)
            .expect("encode")
            .iter()
            .map(|piece| piece.id)
            .collect()
    };

    // Both prompts are the ones the checkpoint was validated against. The
    // second is byte-identical to what the transformers fast conversion of
    // this same `tokenizer.model` produces; the first differs only in how
    // "Fibonacci" is split, because that conversion reconstructs BPE merges
    // from piece scores and does not always recover the trained order.
    assert_eq!(
        ids(&plain, "The Fibonacci sequence begins with"),
        vec![477, 56411, 6161, 43714, 7420, 13712, 409]
    );
    assert_eq!(
        ids(&patched, "The Fibonacci sequence begins with"),
        vec![1545, 56411, 6161, 43714, 7420, 13712, 409]
    );
    assert_eq!(
        ids(
            &plain,
            "In distributed systems, consensus protocols such as Raft"
        ),
        vec![615, 3594, 4785, 66560, 29900, 32343, 1442, 382, 421, 4121]
    );
    assert_eq!(
        ids(
            &patched,
            "In distributed systems, consensus protocols such as Raft"
        ),
        vec![578, 3594, 4785, 66560, 29900, 32343, 1442, 382, 421, 4121]
    );

    // Text that already begins with a space keeps that space as a word
    // boundary: the rewrite removes the phantom prefix, not real whitespace.
    assert_eq!(
        ids(&patched, " In distributed"),
        ids(&plain, "In distributed")[..2].to_vec()
    );
}
