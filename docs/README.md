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
7. `turbo-kv-cache.md` — TurboQuant modes, quality/performance trade-offs, and flags.
8. `responses-api.md` — implemented `/v1/responses` subset and gaps.
9. `adding-models.md` — contribution guide for new model architectures.

## Architecture Decision Records

`adr/` holds numbered Architecture Decision Records, one significant decision per file, immutable once Accepted. See `adr/README.md` for the index.

Expected future layout examples:

- `docs/en/...` and `docs/ko/...` for MkDocs/manual pages.
- `docs/github/...` for GitHub issue/PR/release workflow notes.
- `docs/git/...` for branch, commit, tag, and mirroring procedures.

Keep root README links stable unless the corresponding top-level document is
intentionally replaced with a redirect-style index page.
