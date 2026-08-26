# mlxcel documentation layout

This directory is the shared documentation root for public release material.
It may contain:

1. **GitHub-facing Markdown documents** linked directly from the root `README.md`.
2. **MkDocs site content** under MkDocs-specific source trees. None is present
   here; see "The MkDocs manual" below.
3. **Git/GitHub workflow documents** for maintainers and contributors.

The current top-level files are GitHub-facing documents linked from the root
README. They should remain readable as standalone Markdown files even if richer
MkDocs pages are added later.

Current GitHub-facing docs:

1. `installation.md` - platform prerequisites and build flags.
2. `environment-variables.md` - `MLXCEL_*` runtime, build, downloader, cache, and diagnostic knobs.
3. `benchmarks.md` - benchmark methodology and the requirements for future raw result tables.
4. `supported-models.md` - maintained architecture/checkpoint support matrix.
5. `architecture.md` - runtime architecture and major components.
6. `distributed.md` - tensor/pipeline parallel setup and limitations.
7. `turbo-kv-cache.md` - TurboQuant modes, the unified paged KV cache, quality/performance trade-offs, and flags.
8. `CONTINUOUS_BATCHING.md` - continuous-batching scheduler, paged decode, and disaggregated prefill/decode/router serving.
9. `responses-api.md` - implemented `/v1/responses` subset and gaps.
10. `audio-api.md` - implemented `/v1/audio` endpoints: Whisper STT setup, request/response reference, WAV encoding details, and request validation order.
11. `adding-models.md` - contribution guide for new model architectures.
12. `block-diffusion.md` - DiffusionGemma block-diffusion generation: canvas denoising vs autoregressive, CLI flags, throughput, and phase 1 limitations.
13. `python-client.md` - the `mlxcel` Python client over the OpenAI-compatible server, covering managed and connect modes, streaming, chat, structured output, the `openai_client` escape hatch, async usage, and troubleshooting.
14. `audio-preprocessing.md` - shared model-input WAV normalization, loaded family policy, resource limits, cancellation, metrics, and current XLA capability boundary.
15. `speculative-acceptance.md` - speculative-decoding acceptance rules, which rule each code path runs, the distribution-preservation guarantee and its RNG dependency, the kill switch, and how to read the active rule off a log.
16. `mla-absorbed-decode.md` - DeepSeek-family matrix-absorbed MLA decode over a compressed-latent KV cache: the identity, the cache layout, the flags, and what is and is not verified.
17. `cascade-attention.md` - shared-prompt-prefix (cascade) decode: computing a prefix shared by several concurrent sequences once per step instead of once per sequence, the two-level decomposition, the flags, and how to tell which path a launch took.
18. `sparse-paged-decode.md` - sparse attention reduced to page indirection over the fused v2 decode kernel: the addressing argument, the MiniMax-M3 routing, why DeepSeek Sparse Attention is not routed, where the dispatch floor sits, the kill switch, and the benchmark harness.
19. `mtp-policy-api.md` - the supported read interface for the adaptive B=1 MTP verdict (`GET /v1/internal/mtp-policy`): the response body, the four states, the unavailable reasons, and the schema versioning and compatibility policy.
20. `code-guidelines.md` - the file-size and module-split thresholds, including when to extract a `<name>_helpers.rs` and when inline tests move to a sibling `_tests.rs`, the `// Used by:` annotation convention for shared functions, the JIT kernel rule that every varying input dtype must appear in `template_args` because CUDA keys the compiled-module cache on it, and the `HashMap` iteration-order rule covering what counts as an order-sensitive consumer, why a stable sort on a non-total key does not pin the result, the fresh-map-per-iteration testing requirement, and why no static check enforces it.
21. `embeddings.md` - the `/v1/embeddings` and `/v1/rerank` endpoints plus `mlxcel embed` and `mlxcel rerank`: embedding and reranker detection, pooling and scoring, multimodal and late-interaction inputs, request/response schemas, server flags, error codes, and family-registration guidance.

## Architecture Decision Records

`adr/` holds numbered Architecture Decision Records, one significant decision per file, immutable once Accepted. See `adr/README.md` for the index.

## The MkDocs manual

The manual published at <https://mlxcel.lablup.ai/en/manual/> is not built from
this directory. Its sources (`docs/en`, `docs/ko`, `docs/shared`,
`docs/requirements.txt`, `docs/scripts`) are maintained in a separate
documentation tree and are not part of this repository.

The four mkdocs configs at the repository root (`mkdocs.yml`, `mkdocs.ko.yml`,
`mkdocs.pdf.yml`, `mkdocs.ko.pdf.yml`) belong to that tree and are kept here in
sync with it. Read their `docs_dir`, `custom_dir`, and `nav:` entries as paths
into that tree: none of them names a file listed above, and none of the files
listed above appears in a `nav:`. That is deliberate, not drift. The
GitHub-facing documents here are meant to read as plain Markdown on GitHub, and
the manual is a separately authored artifact.

The `docs-*` Makefile targets build the manual from those sources. In this
repository they stop immediately with an explanation rather than failing partway
through a `uv`, symlink, or site build. Read the published manual instead.

Expected future layout examples:

- `docs/github/...` for GitHub issue/PR/release workflow notes.
- `docs/git/...` for branch, commit, tag, and mirroring procedures.

Keep root README links stable unless the corresponding top-level document is
intentionally replaced with a redirect-style index page.
