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

//! Automatic Prefix Caching (APC) — block-granularity hash chain.
//!
//! This module implements the block-hash construction described in upstream
//! `mlx-vlm` PR #1114 (built on WIP PR #1103). The token sequence is split
//! into fixed-size blocks (default 16) and each block hash is derived from a
//! triple `(parent_hash, block_tokens, extra_hash)`. The chained construction
//! has two desirable properties:
//!
//! 1. **Locality of effect.** A request that shares the first `K` blocks with
//!    a previously seen request produces the same first `K` block hashes,
//!    enabling KV-cache reuse at block granularity. Whole-prefix hashing (used
//!    by [`super::PromptCacheKey`] for the *bucket* identity) cannot express
//!    this since it produces a single digest for the entire prefix.
//!
//! 2. **Multimodal isolation.** The `extra_hash` parameter feeds the
//!    [`MultimodalDigest`] of the request into every block, so two requests
//!    with identical tokens but different image/audio bytes diverge starting
//!    at the very first block. This prevents accidental KV-cache reuse across
//!    requests that share image-placeholder tokens but differ on the actual
//!    image content.
//!
//! The hash algorithm is configurable. `sha256` is the default for
//! wire-compatibility with the upstream APC implementation; `blake3` is also
//! supported as a faster alternative when wire-format parity is not needed.
//! Neither digest is treated as a security primitive — both serve solely as
//! cache-bucket identifiers.
//!
//! ## Example
//!
//! ```ignore
//! use mlxcel::server::prompt_cache::{
//!     ApcHashAlgo, BlockHashChain, MultimodalDigest,
//! };
//!
//! let tokens = (0..40i32).collect::<Vec<_>>();
//! let chain = BlockHashChain::compute(
//!     &tokens,
//!     16, // block_size
//!     ApcHashAlgo::Sha256,
//!     MultimodalDigest::empty().as_bytes(),
//! );
//! // 40 / 16 == 2 full blocks, last 8 tokens form an incomplete tail block.
//! assert_eq!(chain.len(), 3);
//! assert_eq!(chain.full_blocks(), 2);
//! ```

use std::fmt;
use std::str::FromStr;

use sha2::{Digest, Sha256};

use super::key::MultimodalDigest;

/// Default block size matching the upstream APC implementation.
///
/// 16 tokens balances reuse granularity against per-block hashing overhead.
pub const DEFAULT_APC_BLOCK_SIZE: usize = 16;

/// Hash algorithm used to compute APC block hashes.
///
/// `Sha256` is the default and matches the wire format used by upstream
/// `mlx-vlm` so warm-disk artifacts remain interchangeable. `Blake3` is offered
/// as a faster in-process alternative (~3-5x faster on Apple Silicon) for
/// deployments that do not need cross-runtime artifact compatibility.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ApcHashAlgo {
    /// SHA-256, 32-byte digest. Default for wire-compatibility with upstream.
    #[default]
    Sha256,
    /// BLAKE3, 32-byte digest. Faster but not wire-compatible with upstream.
    Blake3,
}

impl ApcHashAlgo {
    /// Stable string identifier suitable for CLI/env serialization. Mirrors
    /// upstream's `APC_HASH=sha256` parity.
    pub fn as_str(&self) -> &'static str {
        match self {
            ApcHashAlgo::Sha256 => "sha256",
            ApcHashAlgo::Blake3 => "blake3",
        }
    }
}

impl fmt::Display for ApcHashAlgo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Error returned when parsing an [`ApcHashAlgo`] from a string fails.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseApcHashError(String);

impl fmt::Display for ParseApcHashError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "unsupported APC hash algorithm {:?} (expected one of: sha256, blake3)",
            self.0
        )
    }
}

impl std::error::Error for ParseApcHashError {}

impl FromStr for ApcHashAlgo {
    type Err = ParseApcHashError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "sha256" | "sha-256" => Ok(ApcHashAlgo::Sha256),
            "blake3" | "blake-3" => Ok(ApcHashAlgo::Blake3),
            other => Err(ParseApcHashError(other.to_string())),
        }
    }
}

/// A 32-byte block-hash output.
///
/// Both supported algorithms (SHA-256, BLAKE3) emit a 32-byte digest. The
/// uniform width keeps the chained construction independent of the chosen
/// algorithm — only the per-block hash bytes differ.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ApcBlockHash([u8; 32]);

impl ApcBlockHash {
    /// All-zero hash. Used as the seed `parent_hash` for the first block of a
    /// chain (analogous to a Merkle-tree zero-root).
    pub const ZERO: ApcBlockHash = ApcBlockHash([0u8; 32]);

    /// Raw bytes for serialization.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Lowercase hex string, 64 chars.
    pub fn to_hex(self) -> String {
        let mut out = String::with_capacity(64);
        for byte in self.0 {
            out.push_str(&format!("{byte:02x}"));
        }
        out
    }

    /// Short hex prefix, 16 chars (8 bytes), suitable for log lines.
    pub fn short_hex(self) -> String {
        let mut out = String::with_capacity(16);
        for byte in &self.0[..8] {
            out.push_str(&format!("{byte:02x}"));
        }
        out
    }
}

impl fmt::Debug for ApcBlockHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ApcBlockHash({})", self.short_hex())
    }
}

impl fmt::Display for ApcBlockHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_hex())
    }
}

/// A chained sequence of block hashes covering a token sequence.
///
/// The first hash in [`BlockHashChain::hashes`] is derived from the seed
/// parent hash (zero), the first `block_size` tokens, and the extra hash. Each
/// subsequent block uses the previous block's hash as its parent so that any
/// modification to an earlier block invalidates the entire downstream chain
/// (Merkle-DAG semantics).
///
/// When the token count is not a multiple of `block_size`, the trailing block
/// hashes a *partial* slice. The trailing block is conceptually less useful
/// for KV reuse (a future request must either match the partial block exactly
/// or extend it from the parent) — callers may inspect
/// [`BlockHashChain::full_blocks`] to discriminate.
#[derive(Clone, Debug)]
pub struct BlockHashChain {
    /// Per-block 32-byte hashes, one entry per (full or partial) block.
    pub hashes: Vec<ApcBlockHash>,
    /// Block size in tokens used to construct this chain.
    pub block_size: usize,
    /// Total number of tokens covered by the chain.
    pub token_count: usize,
    /// Hash algorithm used.
    pub algo: ApcHashAlgo,
}

impl BlockHashChain {
    /// Compute the block-hash chain for `tokens`.
    ///
    /// Arguments:
    /// - `tokens`: full token sequence to be split.
    /// - `block_size`: tokens per block. Must be `>= 1`. A `block_size` of 0
    ///   is a programmer error; this function panics in debug builds and
    ///   silently substitutes [`DEFAULT_APC_BLOCK_SIZE`] in release builds so
    ///   a misconfigured operator cannot crash the server.
    /// - `algo`: hash algorithm.
    /// - `extra_hash`: 32 bytes folded into every block hash. Pass
    ///   `MultimodalDigest::empty().as_bytes()` for text-only requests, or
    ///   the request's resolved [`MultimodalDigest::as_bytes`] for
    ///   multimodal requests.
    ///
    /// The implementation is intentionally allocation-light: a single `Vec`
    /// is allocated up-front for the output and each block-hash computation
    /// reuses a stack-resident hasher state.
    pub fn compute(
        tokens: &[i32],
        block_size: usize,
        algo: ApcHashAlgo,
        extra_hash: &[u8; 32],
    ) -> Self {
        let block_size = if block_size == 0 {
            debug_assert!(false, "BlockHashChain::compute called with block_size=0");
            DEFAULT_APC_BLOCK_SIZE
        } else {
            block_size
        };

        let block_count = tokens.len().div_ceil(block_size);
        let mut hashes = Vec::with_capacity(block_count);
        let mut parent = ApcBlockHash::ZERO;

        for block_idx in 0..block_count {
            let start = block_idx * block_size;
            let end = ((block_idx + 1) * block_size).min(tokens.len());
            let block_tokens = &tokens[start..end];
            let next = hash_block(algo, parent.as_bytes(), block_tokens, extra_hash);
            hashes.push(next);
            parent = next;
        }

        Self {
            hashes,
            block_size,
            token_count: tokens.len(),
            algo,
        }
    }

    /// Compute the chain using a [`MultimodalDigest`] directly.
    ///
    /// Convenience wrapper around [`BlockHashChain::compute`] that supplies
    /// the digest's raw bytes as `extra_hash`.
    pub fn compute_with_mm(
        tokens: &[i32],
        block_size: usize,
        algo: ApcHashAlgo,
        mm_digest: &MultimodalDigest,
    ) -> Self {
        Self::compute(tokens, block_size, algo, mm_digest.as_bytes())
    }

    /// Number of block hashes in the chain.
    pub fn len(&self) -> usize {
        self.hashes.len()
    }

    /// Whether the chain is empty (no tokens).
    pub fn is_empty(&self) -> bool {
        self.hashes.is_empty()
    }

    /// Number of *full* blocks (i.e. blocks that contain exactly
    /// `block_size` tokens). The last block may be a partial trailing
    /// block when `token_count % block_size != 0`.
    pub fn full_blocks(&self) -> usize {
        self.token_count / self.block_size
    }

    /// Get the hash for block index `i`, or `None` if `i` is out of range.
    pub fn get(&self, i: usize) -> Option<ApcBlockHash> {
        self.hashes.get(i).copied()
    }

    /// Hash of the last block in the chain, useful as a quick fingerprint.
    pub fn tail(&self) -> Option<ApcBlockHash> {
        self.hashes.last().copied()
    }
}

/// Domain separator for APC block hashes. Bumped if the input layout ever
/// changes so old artifacts cannot collide with new ones.
const APC_BLOCK_DOMAIN: &[u8] = b"mlxcel:apc:block:v1";

/// Compute one block hash: `H(domain || parent || tokens || extra)`.
fn hash_block(
    algo: ApcHashAlgo,
    parent: &[u8; 32],
    tokens: &[i32],
    extra: &[u8; 32],
) -> ApcBlockHash {
    match algo {
        ApcHashAlgo::Sha256 => {
            let mut hasher = Sha256::new();
            hasher.update(APC_BLOCK_DOMAIN);
            hasher.update(parent);
            // Length-prefix the token slice so adjacent fields cannot collide.
            hasher.update((tokens.len() as u64).to_le_bytes());
            for tok in tokens {
                hasher.update(tok.to_le_bytes());
            }
            hasher.update(extra);
            let out = hasher.finalize();
            let mut bytes = [0u8; 32];
            bytes.copy_from_slice(&out);
            ApcBlockHash(bytes)
        }
        ApcHashAlgo::Blake3 => {
            let mut hasher = blake3::Hasher::new();
            hasher.update(APC_BLOCK_DOMAIN);
            hasher.update(parent);
            hasher.update(&(tokens.len() as u64).to_le_bytes());
            for tok in tokens {
                hasher.update(&tok.to_le_bytes());
            }
            hasher.update(extra);
            let mut bytes = [0u8; 32];
            hasher.finalize_xof().fill(&mut bytes);
            ApcBlockHash(bytes)
        }
    }
}

#[cfg(test)]
#[path = "block_hash_tests.rs"]
mod tests;
