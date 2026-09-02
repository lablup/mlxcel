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

//! Minimal editor for the one SentencePiece `ModelProto` field mlxcel has to
//! override at load time: `normalizer_spec.add_dummy_prefix`.
//!
//! # Why this exists
//!
//! A SentencePiece model carries its normalizer settings inside
//! `tokenizer.model`, and `add_dummy_prefix` (default `true`) makes every
//! `encode` behave as if the text began with a space. A checkpoint can
//! contradict that from `tokenizer_config.json` by declaring
//! `"add_prefix_space": false`, which is what the HuggingFace fast-tokenizer
//! conversion honors: `LlamaConverter.normalizer` emits the `Prepend("_")`
//! normalizer only when `add_prefix_space` is true, so a checkpoint that turns
//! it off tokenizes the first word with no leading word-boundary marker. See
//! <https://github.com/huggingface/transformers/blob/main/src/transformers/convert_slow_tokenizer.py>.
//!
//! The `sentencepiece` crate exposes no setter for the normalizer, but it does
//! expose [`SentencePieceProcessor::from_serialized_proto`], so mlxcel rewrites
//! the single field and re-loads. Everything else about the model, the pieces,
//! their scores, the BPE merge order, byte fallback, is left byte-for-byte
//! alone, which is the point: the checkpoint's own segmentation is preserved
//! and only the phantom leading space is dropped.
//!
//! # Encoding
//!
//! The relevant slice of `sentencepiece_model.proto`
//! (<https://github.com/google/sentencepiece/blob/master/src/sentencepiece_model.proto>):
//!
//! ```text
//! message ModelProto {
//!   repeated SentencePiece pieces           = 1;
//!   optional TrainerSpec    trainer_spec    = 2;
//!   optional NormalizerSpec normalizer_spec = 3;
//!   optional SelfTestData   self_test_data  = 4;
//!   optional NormalizerSpec denormalizer_spec = 5;
//! }
//! message NormalizerSpec {
//!   optional string name                   = 1;
//!   optional bytes  precompiled_charsmap   = 2;
//!   optional bool   add_dummy_prefix        = 3 [default = true];
//!   optional bool   remove_extra_whitespaces = 4 [default = true];
//!   optional bool   escape_whitespaces      = 5 [default = true];
//!   optional string normalization_rule_tsv  = 6;
//! }
//! ```
//!
//! So the edit is: find top-level field 3, drop every occurrence of field 3
//! inside it, append `add_dummy_prefix = false`, and re-emit the outer message
//! with a recomputed length prefix. A model that carries no `normalizer_spec`
//! at all gets one appended, because an absent submessage means the proto2
//! default `true` applies.

use anyhow::{Result, bail};

/// Protobuf field number of `ModelProto.normalizer_spec`.
const MODEL_NORMALIZER_SPEC_FIELD: u32 = 3;
/// Protobuf field number of `NormalizerSpec.add_dummy_prefix`.
const NORMALIZER_ADD_DUMMY_PREFIX_FIELD: u32 = 3;
/// Wire type 0 (varint) and wire type 2 (length-delimited).
const WIRE_VARINT: u32 = 0;
const WIRE_LEN: u32 = 2;

/// One top-level field, located in the input buffer.
struct FieldSpan {
    number: u32,
    /// Offset of the field's tag varint.
    start: usize,
    /// Offset one past the field's last byte.
    end: usize,
    /// Payload range, meaningful only for wire type 2.
    payload: Option<(usize, usize)>,
}

fn read_varint(buf: &[u8], pos: &mut usize) -> Result<u64> {
    let mut value: u64 = 0;
    let mut shift = 0u32;
    loop {
        let Some(&byte) = buf.get(*pos) else {
            bail!("truncated varint at offset {}", *pos);
        };
        *pos += 1;
        if shift >= 64 {
            bail!("varint wider than 64 bits at offset {}", *pos);
        }
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
        shift += 7;
    }
}

fn write_varint(out: &mut Vec<u8>, mut value: u64) {
    loop {
        let byte = (value & 0x7f) as u8;
        value >>= 7;
        if value == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

/// Walk `buf` as a protobuf message body and return one [`FieldSpan`] per field.
fn scan_fields(buf: &[u8]) -> Result<Vec<FieldSpan>> {
    let mut fields = Vec::new();
    let mut pos = 0usize;
    while pos < buf.len() {
        let start = pos;
        let tag = read_varint(buf, &mut pos)?;
        let number = u32::try_from(tag >> 3).unwrap_or(u32::MAX);
        let wire = (tag & 0x7) as u32;
        if number == 0 {
            bail!("protobuf field number 0 at offset {start}");
        }
        let payload = match wire {
            WIRE_VARINT => {
                read_varint(buf, &mut pos)?;
                None
            }
            1 => {
                pos = pos
                    .checked_add(8)
                    .filter(|end| *end <= buf.len())
                    .ok_or_else(|| anyhow::anyhow!("truncated 64-bit field at offset {start}"))?;
                None
            }
            WIRE_LEN => {
                let len = usize::try_from(read_varint(buf, &mut pos)?)
                    .map_err(|_| anyhow::anyhow!("length-delimited field too large"))?;
                let payload_start = pos;
                pos = pos
                    .checked_add(len)
                    .filter(|end| *end <= buf.len())
                    .ok_or_else(|| {
                        anyhow::anyhow!("truncated length-delimited field at offset {start}")
                    })?;
                Some((payload_start, pos))
            }
            5 => {
                pos = pos
                    .checked_add(4)
                    .filter(|end| *end <= buf.len())
                    .ok_or_else(|| anyhow::anyhow!("truncated 32-bit field at offset {start}"))?;
                None
            }
            other => bail!("unsupported protobuf wire type {other} at offset {start}"),
        };
        fields.push(FieldSpan {
            number,
            start,
            end: pos,
            payload,
        });
    }
    Ok(fields)
}

/// Emit a length-delimited field: tag, length, payload.
fn push_len_field(out: &mut Vec<u8>, number: u32, payload: &[u8]) {
    write_varint(out, u64::from(number) << 3 | u64::from(WIRE_LEN));
    write_varint(out, payload.len() as u64);
    out.extend_from_slice(payload);
}

/// Rebuild a `NormalizerSpec` body with `add_dummy_prefix` forced to `false`.
fn normalizer_spec_without_dummy_prefix(body: &[u8]) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(body.len() + 2);
    for field in scan_fields(body)? {
        if field.number == NORMALIZER_ADD_DUMMY_PREFIX_FIELD {
            continue;
        }
        out.extend_from_slice(&body[field.start..field.end]);
    }
    write_varint(
        &mut out,
        u64::from(NORMALIZER_ADD_DUMMY_PREFIX_FIELD) << 3 | u64::from(WIRE_VARINT),
    );
    out.push(0);
    Ok(out)
}

/// Return `proto` with `normalizer_spec.add_dummy_prefix` set to `false`.
///
/// The rest of the message is copied verbatim, including any field this code
/// does not understand, so an unfamiliar SentencePiece revision round-trips
/// rather than losing data. Returns an error only when `proto` is not a
/// well-formed protobuf message.
pub(crate) fn disable_add_dummy_prefix(proto: &[u8]) -> Result<Vec<u8>> {
    let fields = scan_fields(proto)?;

    let mut out = Vec::with_capacity(proto.len() + 2);
    let mut copied_to = 0usize;
    let mut rewrote_any = false;
    // Every occurrence is rewritten, not just the first. `normalizer_spec` is
    // a singular field, so a well-formed model carries one, but the wire
    // format permits repeats and protobuf merges them, which means a later
    // occurrence carrying `add_dummy_prefix = true` would undo a patch applied
    // only to the first one.
    for field in &fields {
        if field.number != MODEL_NORMALIZER_SPEC_FIELD {
            continue;
        }
        let Some((body_start, body_end)) = field.payload else {
            bail!("ModelProto.normalizer_spec is not a length-delimited field");
        };
        let body = normalizer_spec_without_dummy_prefix(&proto[body_start..body_end])?;
        out.extend_from_slice(&proto[copied_to..field.start]);
        push_len_field(&mut out, MODEL_NORMALIZER_SPEC_FIELD, &body);
        copied_to = field.end;
        rewrote_any = true;
    }
    out.extend_from_slice(&proto[copied_to..]);

    if !rewrote_any {
        // No normalizer_spec at all: the proto2 default `add_dummy_prefix =
        // true` is in force, so one has to be appended to override it.
        let mut body = Vec::with_capacity(2);
        write_varint(
            &mut body,
            u64::from(NORMALIZER_ADD_DUMMY_PREFIX_FIELD) << 3 | u64::from(WIRE_VARINT),
        );
        body.push(0);
        push_len_field(&mut out, MODEL_NORMALIZER_SPEC_FIELD, &body);
    }

    Ok(out)
}

#[cfg(test)]
#[path = "spm_proto_tests.rs"]
mod spm_proto_tests;
