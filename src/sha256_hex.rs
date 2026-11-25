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

//! Lowercase hex SHA-256 of a byte slice.
//!
//! `sha2` 0.11 returns a `hybrid_array::Array` that no longer implements
//! `LowerHex`, so `format!("{:x}", Sha256::digest(..))` stopped compiling.
//! This is the one place that formats a digest, with the same output as the
//! old `{:x}`: 64 lowercase hex characters.

use std::fmt::Write as _;

use sha2::{Digest, Sha256};

// Used by: distributed::stage_assignment, server::webui::assets (ETag), and
// the fixture-hash checks in the vision/dense tests.
/// Lowercase hex SHA-256 digest of `data`.
pub(crate) fn sha256_hex(data: impl AsRef<[u8]>) -> String {
    hex_lower(Sha256::digest(data.as_ref()))
}

// Used by: incremental digests (a `Sha256` fed with `update`) that already
// hold a finished digest. Passing `hasher.finalize()` to `sha256_hex` would
// hash the digest a second time (#3365).
/// Lowercase hex encoding of an already computed digest, with no hashing.
pub(crate) fn hex_lower(digest: impl AsRef<[u8]>) -> String {
    let digest = digest.as_ref();
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        // Writing to a `String` cannot fail.
        let _ = write!(out, "{byte:02x}");
    }
    out
}

#[cfg(test)]
mod tests {
    use sha2::{Digest, Sha256};

    use super::{hex_lower, sha256_hex};

    #[test]
    fn hex_lower_encodes_a_finished_digest_without_rehashing() {
        let mut hasher = Sha256::new();
        hasher.update(b"abc");
        assert_eq!(hex_lower(hasher.finalize()), sha256_hex("abc"));
    }

    #[test]
    fn matches_known_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex("abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
