# mlxcel documentation layout

This directory is the shared documentation root for public release material.
It may contain both:

1. **GitHub-facing Markdown documents** linked directly from the root `README.md`.
2. **MkDocs site content** added under the MkDocs-specific source trees.
3. **Git/GitHub workflow documents** for maintainers and contributors.

The current top-level files are GitHub-facing documents linked from the root
README. They should remain readable as standalone Markdown files even if richer
MkDocs pages are added later.

Current GitHub-facing docs:

1. `installation.md` — platform prerequisites and build flags.
2. `environment-variables.md` — `MLXCEL_*` runtime, build, downloader, cache, and diagnostic knobs.
3. `benchmarks.md` — benchmark methodology and the requirements for future raw result tables.
4. `supported-models.md` — maintained architecture/checkpoint support matrix.
5. `architecture.md` — runtime architecture and major components.
6. `distributed.md` — tensor/pipeline parallel setup and limitations.
7. `turbo-kv-cache.md` — TurboQuant modes, the unified paged KV cache, quality/performance trade-offs, and flags.
8. `CONTINUOUS_BATCHING.md` — continuous-batching scheduler, paged decode, and disaggregated prefill/decode/router serving.
9. `responses-api.md` — implemented `/v1/responses` subset and gaps.
10. `audio-api.md` — implemented `/v1/audio` endpoints: Whisper STT setup, request/response reference, WAV encoding details, and request validation order.
11. `adding-models.md` — contribution guide for new model architectures.
12. `block-diffusion.md` — DiffusionGemma block-diffusion generation: canvas denoising vs autoregressive, CLI flags, throughput, and phase 1 limitations.
13. `python-client.md`: the `mlxcel` Python client over the OpenAI-compatible server, covering managed and connect modes, streaming, chat, structured output, the `openai_client` escape hatch, async usage, and troubleshooting.
14. `audio-preprocessing.md` — shared model-input WAV normalization, loaded family policy, resource limits, cancellation, metrics, and current XLA capability boundary.
15. `speculative-acceptance.md` — speculative-decoding acceptance rules, which rule each code path runs, the distribution-preservation guarantee and its RNG dependency, the kill switch, and how to read the active rule off a log.
16. `mla-absorbed-decode.md` — DeepSeek-family matrix-absorbed MLA decode over a compressed-latent KV cache: the identity, the cache layout, the flags, and what is and is not verified.
17. `cascade-attention.md` — shared-prompt-prefix (cascade) decode: computing a prefix shared by several concurrent sequences once per step instead of once per sequence, the two-level decomposition, the flags, and how to tell which path a launch took.

## Architecture Decision Records

`adr/` holds numbered Architecture Decision Records, one significant decision per file, immutable once Accepted. See `adr/README.md` for the index.

Expected future layout examples:

- `docs/en/...` and `docs/ko/...` for MkDocs/manual pages.
- `docs/github/...` for GitHub issue/PR/release workflow notes.
- `docs/git/...` for branch, commit, tag, and mirroring procedures.

Keep root README links stable unless the corresponding top-level document is
intentionally replaced with a redirect-style index page.
