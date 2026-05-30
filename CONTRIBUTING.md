# Contributing to mlxcel

Thank you for your interest in contributing to mlxcel! This document covers the basics for getting started. For the deeper working contract — invariants for the request path, MLX upstream pin synchronization, model-porting checklist — read [`AGENTS.md`](AGENTS.md) after this.

## Quick links

| You want to... | Read |
|----------------|------|
| Report a security vulnerability | [`SECURITY.md`](SECURITY.md) — **do not** open a public issue |
| File a bug or feature request | [GitHub Issues](https://github.com/lablup/mlxcel/issues) (use the templates) |
| Build and test locally | [`docs/installation.md`](docs/installation.md) |
| Understand the architecture | [`docs/architecture.md`](docs/architecture.md) |
| Add a new model family | [`docs/adding-models.md`](docs/adding-models.md) |
| Understand the project conventions | [`AGENTS.md`](AGENTS.md) |

## How to contribute

### Reporting issues

- Search existing issues first.
- Use the bug-report or feature-request template — they prompt for the information we need to act on the issue.
- Include the mlxcel version (`mlxcel --version`), platform (macOS Apple Silicon / Linux CUDA + GPU model), and the checkpoint you were running.
- For inference correctness or performance reports, also include the prompt and seed so the run is reproducible.

### Submitting pull requests

1. Fork the repository and create a feature branch off `main`:
   ```bash
   git checkout -b feat/short-description
   ```
2. Make your changes. Keep one PR scoped to **one logical change** — a model port, an MLX bump, and a CLI rename are three PRs.
3. Build and test for your target:
   ```bash
   # macOS (Apple Silicon)
   cargo build --release --features metal,accelerate
   cargo test --release

   # Linux / CUDA
   cargo build --release --features cuda
   cargo test --release
   ```
4. Run the local quality gates:
   ```bash
   cargo fmt --all -- --check   # enforced by CI; fmt violations block merge
   cargo clippy --all-targets --features metal,accelerate -- -D warnings   # enforced by CI on self-hosted macOS runner
   cargo test --release --features metal,accelerate                         # enforced by CI on self-hosted macOS runner
   cargo deny check             # advisories + licenses + sources
   ```
   CI enforces `clippy` (with `-D warnings`) and `cargo test` on the `self-hosted-macos-26-arm64` runner on every PR that touches Rust files. CUDA verification is not gated at PR time — that stays exclusive to `release.yml`.
5. For inference changes, validate against a real checkpoint — synthetic or build-only validation is not enough (see [`AGENTS.md`](AGENTS.md) for why).
6. Commit with a conventional prefix (see below) and a clear message.
7. Push to your fork and open a Pull Request. The PR template will prompt for a summary, test plan, and linked issues.

### Commit and PR conventions

Write commits, PR titles, and issue comments in **English**. Use Conventional Commits prefixes:

| Prefix | When |
|--------|------|
| `feat:` | New user-visible feature |
| `fix:` | Bug fix |
| `perf:` | Performance improvement with measurable evidence |
| `refactor:` | Internal restructuring without behavior change |
| `chore:` | Build, CI, dependencies, release infrastructure |
| `docs:` | Documentation |
| `test:` | Tests only |

### Code standards

- Follow standard Rust conventions: `rustfmt`, `clippy -D warnings`, idiomatic ownership and error handling.
- Tests live next to the code (`_tests.rs` files) for unit tests, and under `tests/` for end-to-end integration.
- When modifying a function shared by multiple models, update the `// Used by: Model1, Model2, …` comment above it. See [`docs/code-guidelines.md`](docs/code-guidelines.md).
- Do not introduce Python on the inference request path. Python is acceptable only for benchmarks and out-of-band tooling.

### Cross-repository issue references

`#NNN` auto-links to `lablup/mlxcel`, so use a bare `#NNN` **only** for issues and PRs in this repository. Any reference to another repository must be qualified so it resolves correctly and never leaks a private-repo number:

- Upstream references are written `org/repo#NNN` — `ml-explore/mlx-lm#1240`, `Blaizzy/mlx-vlm#1181`, `ml-explore/mlx#3475`, `huggingface/transformers#NNN`.
- `mlxcel-internal` (private) numbers must never appear anywhere — code comments, docs, commit subjects, or PR bodies. Map an internal reference to its public-equivalent PR/issue when one demonstrably exists; otherwise describe the change without a number.

Pre-flight before pushing — review every bare 3+-digit reference you add:

```bash
git diff origin/main...HEAD | grep -nE '#[0-9]{3,}'
# or, scoped and classified (advisory by default; STRICT=1 to gate):
python3 scripts/ci/check_cross_repo_refs.py
```

CI runs the same check on every pull request (advisory).

### Adding a new model family

See [`docs/adding-models.md`](docs/adding-models.md) for the full checklist. The short version: land one working checkpoint plus tests before broadening, mirror the `mlx-lm` / `mlx-vlm` directory shape where it helps, and update [`docs/supported-models.md`](docs/supported-models.md) plus the detection table in `src/models/detection.rs`.

### Bumping the MLX upstream pin

The pinned MLX C++ commit lives in three files that must stay in sync — see the "MLX upstream commit upgrade" section of [`AGENTS.md`](AGENTS.md) for the list and the mandatory post-bump validation.

## Development environment

Detailed setup instructions are in [`docs/installation.md`](docs/installation.md).

Minimum:

- Rust **1.93+** (project uses edition 2024)
- macOS: Apple Silicon Mac on macOS Sonoma+; Xcode Command Line Tools
- Linux: CUDA 13+ toolchain, OpenBLAS, LAPACK (see [`docs/installation.md`](docs/installation.md) for the package list)

Recommended local tooling:

```bash
cargo install cargo-deny --locked
cargo install cargo-audit --locked
```

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to abide by its terms.

## Questions

- General questions, design discussion: open a [GitHub Discussion](https://github.com/lablup/mlxcel/discussions) (when enabled) or a `question` issue.
- Security: see [`SECURITY.md`](SECURITY.md).

## License

By contributing to mlxcel, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).
