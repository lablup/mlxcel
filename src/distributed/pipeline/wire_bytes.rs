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

//! Compact encoding for the bulk byte fields of the stage protocol (#2656).
//!
//! The coordinator commands, stage responses and activation messages are
//! serde_json documents. A `Vec<u8>` field encodes by default as a JSON array
//! of numbers, about 3.6 characters per byte, so the per-token `RunResult`
//! of a 150k-vocabulary model crossed the wire as a 1 to 2 MB frame for
//! 300 to 600 KB of logits. The `with` modules here encode such fields as one
//! base64 string (1.33 characters per byte) when the format is human
//! readable, and as raw bytes otherwise, so the same annotation stays correct
//! should the protocol ever move to a binary serde format.
//!
//! Wire change: a build with this encoding does not interoperate with one
//! without it. Mixed versions in a cluster were already unsupported and the
//! cluster handshake `PROTOCOL_VERSION` was bumped in the same change.
//!
//! Used by: remote_service (`RemoteStageCommand::RunEntry`,
//! `RemoteStageResponse::RunResult`), activation_transfer
//! (`ActivationMessage`).

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use serde::de::{self, Visitor};
use serde::{Deserializer, Serializer};

/// `#[serde(with = "wire_bytes::base64_bytes")]` for a `Vec<u8>` field.
pub mod base64_bytes {
    use super::*;

    pub fn serialize<S: Serializer>(bytes: &[u8], serializer: S) -> Result<S::Ok, S::Error> {
        if serializer.is_human_readable() {
            serializer.serialize_str(&STANDARD.encode(bytes))
        } else {
            serializer.serialize_bytes(bytes)
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<u8>, D::Error> {
        if deserializer.is_human_readable() {
            deserializer.deserialize_str(Base64Visitor)
        } else {
            deserializer.deserialize_byte_buf(RawVisitor)
        }
    }
}

/// `#[serde(with = "wire_bytes::base64_bytes_opt")]` for an
/// `Option<Vec<u8>>` field (`None` stays JSON `null`).
pub mod base64_bytes_opt {
    use super::*;

    pub fn serialize<S: Serializer>(
        bytes: &Option<Vec<u8>>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        match bytes {
            Some(bytes) => {
                if serializer.is_human_readable() {
                    serializer.serialize_some(&STANDARD.encode(bytes))
                } else {
                    serializer.serialize_some(&RawBytes(bytes))
                }
            }
            None => serializer.serialize_none(),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<Vec<u8>>, D::Error> {
        deserializer.deserialize_option(OptVisitor)
    }

    struct OptVisitor;

    impl<'de> Visitor<'de> for OptVisitor {
        type Value = Option<Vec<u8>>;

        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("null or base64-encoded bytes")
        }

        fn visit_none<E: de::Error>(self) -> Result<Self::Value, E> {
            Ok(None)
        }

        fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
            Ok(None)
        }

        fn visit_some<D: Deserializer<'de>>(
            self,
            deserializer: D,
        ) -> Result<Self::Value, D::Error> {
            base64_bytes::deserialize(deserializer).map(Some)
        }
    }
}

/// Raw bytes for a non-human-readable serializer inside `serialize_some`.
struct RawBytes<'a>(&'a [u8]);

impl serde::Serialize for RawBytes<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_bytes(self.0)
    }
}

struct Base64Visitor;

impl Visitor<'_> for Base64Visitor {
    type Value = Vec<u8>;

    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("a base64 string")
    }

    fn visit_str<E: de::Error>(self, value: &str) -> Result<Self::Value, E> {
        STANDARD
            .decode(value)
            .map_err(|err| E::custom(format!("invalid base64 bytes: {err}")))
    }
}

struct RawVisitor;

impl<'de> Visitor<'de> for RawVisitor {
    type Value = Vec<u8>;

    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("raw bytes")
    }

    fn visit_bytes<E: de::Error>(self, value: &[u8]) -> Result<Self::Value, E> {
        Ok(value.to_vec())
    }

    fn visit_byte_buf<E: de::Error>(self, value: Vec<u8>) -> Result<Self::Value, E> {
        Ok(value)
    }

    fn visit_seq<A: de::SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        let mut out = Vec::with_capacity(seq.size_hint().unwrap_or(0));
        while let Some(byte) = seq.next_element::<u8>()? {
            out.push(byte);
        }
        Ok(out)
    }
}

#[cfg(test)]
#[path = "wire_bytes_tests.rs"]
mod tests;
