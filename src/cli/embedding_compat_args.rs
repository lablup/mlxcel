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

//! llama-server b10621 embedding and reranking mode flags (issue #1452).
//!
//! Four options, defined once and flattened into both server binaries:
//! `--embedding` / `--embeddings`, `--rerank` / `--reranking`, `--pooling` and
//! `--embd-normalize`.
//!
//! # What the two mode flags mean here
//!
//! b10621 has one model and one set of weights, so `--embeddings` is a
//! server-wide restriction: it stops generation and turns the embedding
//! endpoints on. mlxcel loads a dedicated embedding or reranking worker, chosen
//! today by `--embedding-model` / `--reranker-model` or by detecting what `-m`
//! is. The flags therefore do two things here, and both halves are needed for
//! the b10621 behavior to be reproduced rather than approximated:
//!
//! 1. They **select**: `--embeddings` requires `-m` (or `--embedding-model`) to
//!    resolve to an embedding worker, and `--rerank` requires a reranker. A
//!    command line that asks for the mode and gives no checkpoint that can
//!    serve it fails at startup, naming the flag, instead of booting a server
//!    whose embedding route answers 501 forever.
//! 2. They **restrict**: generation routes answer the same 501 they answer when
//!    no chat model is loaded, naming the flag that turned generation off. That
//!    is what a client written against `llama-server --embeddings` observes.
//!
//! # `--pooling`
//!
//! b10621's five values do not all name a pooling kernel. `mean`, `cls` and
//! `last` do, and map onto mlxcel's [`PoolingMode`]. `rank` does not: upstream
//! uses it to put the model on its reranking path, which is exactly what
//! `--reranking` does, so it is accepted as a synonym for that flag rather than
//! invented as a pooling kernel that would have nothing to compute. `none`
//! means "return one vector per token", which mlxcel's embedding pipeline
//! cannot produce at all: every family pools to `[B, D]` before the engine sees
//! the output. It is refused at startup with that reason.
//!
//! mlxcel's own `max` pooling has no b10621 spelling and stays reachable
//! through `MLXCEL_EMBEDDING_POOLING` and the checkpoint's own
//! `1_Pooling/config.json`.
//!
//! # `--embd-normalize`
//!
//! The whole b10621 domain, `-1`, `0`, `1`, `2` and any `p > 2`, is served; see
//! [`crate::embeddings::EmbdNormalize`].
//!
//! Upstream reference:
//! <https://github.com/ggml-org/llama.cpp/blob/c1d0e7a004015f23bc0233470b747b596f29b264/common/arg.cpp>
//!
//! Used by: mlxcel serve, mlxcel-server.

use clap::Args;

use crate::embeddings::{EmbdNormalize, PoolingMode};

use super::ggml_compat_args::env_flag;

/// Shared embedding and reranking mode flag group.
#[derive(Args, Debug, Default, Clone)]
#[command(next_help_heading = "Embeddings and Reranking (llama-server compatibility)")]
pub struct EmbeddingCompatArgs {
    /// Serve embeddings only: refuse generation and require an embedding
    /// checkpoint.
    ///
    /// `-m` must be an embedding checkpoint, or `--embedding-model` must name
    /// one; otherwise startup fails rather than answering 501 forever.
    #[arg(
        long = "embedding",
        visible_alias = "embeddings",
        action = clap::ArgAction::SetTrue
    )]
    pub embedding: bool,

    /// Serve reranking only: refuse generation and require a reranker.
    ///
    /// `-m` must be a reranker checkpoint, or `--reranker-model` must name one.
    #[arg(
        long = "rerank",
        visible_alias = "reranking",
        action = clap::ArgAction::SetTrue
    )]
    pub rerank: bool,

    /// Pooling for embeddings: `none`, `mean`, `cls`, `last` or `rank`.
    ///
    /// Unset uses the checkpoint's `1_Pooling/config.json`, then the family
    /// default. `rank` selects reranking, as `--reranking` does. `none` is
    /// refused: mlxcel pools inside the family and cannot return one vector per
    /// token.
    #[arg(
        long = "pooling",
        env = "LLAMA_ARG_POOLING",
        value_name = "{none,mean,cls,last,rank}"
    )]
    pub pooling: Option<String>,

    /// Embedding normalization: `-1` none, `0` max absolute int16, `1`
    /// taxicab, `2` euclidean, above 2 that p-norm.
    #[arg(long = "embd-normalize", value_name = "N")]
    pub embd_normalize: Option<i32>,
}

/// What this group resolves to, once precedence and validation are applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EmbeddingCompatResolution {
    /// `--embedding` / `--embeddings` / `LLAMA_ARG_EMBEDDINGS`, or `--pooling`
    /// having selected an embedding-only server.
    pub embedding_only: bool,
    /// `--rerank` / `--reranking` / `LLAMA_ARG_RERANKING`, or `--pooling rank`.
    pub rerank_only: bool,
    /// The pooling kernel `--pooling` asked for, `None` when it named `rank`
    /// or was not given.
    pub pooling: Option<PoolingMode>,
    /// The `--embd-normalize` value, `None` when unset: the checkpoint's own
    /// `normalize` flag then decides.
    pub embd_normalize: Option<EmbdNormalize>,
}

impl EmbeddingCompatArgs {
    /// Apply precedence and refuse what cannot be served.
    ///
    /// Both mode flags read their `LLAMA_ARG_*` variable through
    /// [`env_flag`] rather than through clap, because b10621 fires a
    /// value-less option from the environment only for its own truthy set and
    /// clap's boolish parser both accepts more and errors outside it.
    ///
    /// Asking for both modes on the *command line* is refused, because it names
    /// two workers for one `-m`. The same pair arriving from different layers
    /// is not: `--reranking` sets the embedding restriction itself, exactly as
    /// b10621's handler does, so an inherited `LLAMA_ARG_EMBEDDINGS` next to an
    /// explicit `--rerank` is coherent and resolves to reranking.
    pub fn resolve(&self) -> Result<EmbeddingCompatResolution, String> {
        let mut embedding_only = self.embedding || env_flag("LLAMA_ARG_EMBEDDINGS");
        let mut rerank_only = self.rerank || env_flag("LLAMA_ARG_RERANKING");
        let mut pooling = None;

        if let Some(raw) = self.pooling.as_deref() {
            match parse_pooling(raw)? {
                PoolingChoice::Mode(mode) => pooling = Some(mode),
                PoolingChoice::Rank => rerank_only = true,
            }
        }

        // b10621's `--reranking` sets its embedding flag too, so reranking
        // implies the same generation restriction.
        if rerank_only {
            embedding_only = true;
        }

        if self.embedding && self.rerank {
            return Err(
                "--embeddings and --reranking select different workers: pass one. A single \
                 mlxcel server loads one embedding checkpoint or one reranker for -m, not both; \
                 to serve both, pass a chat model to -m with --embedding-model and \
                 --reranker-model."
                    .to_string(),
            );
        }

        let embd_normalize = self.embd_normalize.map(EmbdNormalize::new).transpose()?;

        Ok(EmbeddingCompatResolution {
            embedding_only,
            rerank_only,
            pooling,
            embd_normalize,
        })
    }
}

/// What one `--pooling` value selects.
enum PoolingChoice {
    /// A pooling kernel mlxcel implements.
    Mode(PoolingMode),
    /// b10621's `rank`, which is the reranking path rather than a kernel.
    Rank,
}

/// Map one b10621 `--pooling` value onto mlxcel.
fn parse_pooling(raw: &str) -> Result<PoolingChoice, String> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "mean" => Ok(PoolingChoice::Mode(PoolingMode::Mean)),
        "cls" => Ok(PoolingChoice::Mode(PoolingMode::Cls)),
        "last" => Ok(PoolingChoice::Mode(PoolingMode::LastToken)),
        "rank" => Ok(PoolingChoice::Rank),
        "none" => Err(
            "--pooling none asks for one embedding vector per token, which mlxcel cannot \
             produce: every embedding family pools to one vector per input inside its own \
             forward pass, before the engine sees the output, so there is no unpooled hidden \
             state left to return. Pass mean, cls or last, or drop the flag to use the \
             checkpoint's 1_Pooling/config.json. Tracked by #1452."
                .to_string(),
        ),
        other => Err(format!(
            "--pooling {other} is not a b10621 pooling type: pass none, mean, cls, last or rank"
        )),
    }
}

/// Resolve the group from the environment alone.
///
/// Parsing an argv of just the program name makes clap fill `--pooling` from
/// `LLAMA_ARG_POOLING`, and [`Self::resolve`] reads the two value-less
/// variables itself, so a caller with no command line still goes through one
/// set of definitions.
pub fn from_env() -> Result<EmbeddingCompatResolution, String> {
    use clap::Parser;

    #[derive(Parser)]
    struct EnvOnly {
        #[command(flatten)]
        embedding_compat: EmbeddingCompatArgs,
    }

    EnvOnly::try_parse_from(["mlxcel-server"])
        .map_err(|e| e.to_string())?
        .embedding_compat
        .resolve()
}

#[cfg(test)]
#[path = "embedding_compat_args_tests.rs"]
mod embedding_compat_args_tests;
