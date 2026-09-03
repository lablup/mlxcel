# mlxcel Makefile
# High-performance LLM/VLM/VLA inference on Apple Silicon

# ============================================================================
# Configuration
# ============================================================================

CARGO := cargo
RUSTFLAGS := RUSTFLAGS="-C target-cpu=native"

# ----------------------------------------------------------------------------
# Release accelerator features (platform-aware)
#
# macOS (Apple Silicon) always builds with Metal + Accelerate, matching the CI
# feature set and the canonical `--features metal,accelerate` build in CLAUDE.md.
#
# Linux is intentionally accelerator-neutral: a Linux host may carry any of
# several accelerators, so we never assume CUDA here. Pick an explicit target
# (`make release-cuda`, plus future siblings) for the chip you actually have.
# ----------------------------------------------------------------------------
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
RELEASE_FEATURES := metal,accelerate
endif
# Expands to `--features <list>` only when RELEASE_FEATURES is set; empty on Linux.
RELEASE_FEATURE_FLAG := $(if $(RELEASE_FEATURES),--features $(RELEASE_FEATURES))

# Binary names
BIN_CLI := mlxcel
BIN_SERVER := mlxcel-server

# Default model path for examples (override with MODEL=path)
MODEL ?= ./models/default

# Default prompt for examples
PROMPT ?= "Hello, world!"

# Optional test-name filter forwarded to `cargo test` by the test-fast targets
# below, e.g. `make test-fast-cuda FILTER=server::chat_request` (issue #809).
FILTER ?=

# Dev-profile server template renders can overflow libtest's default 2 MiB
# thread stack; keep the larger stack limited to the plain dev-profile test
# entry points and leave test-fast / release gates unchanged (#1384).
DEV_TEST_RUST_MIN_STACK ?= 16777216

# Server settings
HOST ?= 127.0.0.1
PORT ?= 8080

# Colors for output
CYAN := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
RESET := \033[0m
BOLD := \033[1m

# ============================================================================
# Default Target
# ============================================================================

.PHONY: help
help: ## Show this help message
	@echo ""
	@echo "$(BOLD)mlxcel$(RESET) - High-performance LLM/VLM/VLA inference on Apple Silicon"
	@echo ""
	@echo "$(BOLD)Usage:$(RESET)"
	@echo "  make $(CYAN)<target>$(RESET) [VARIABLE=value]"
	@echo ""
	@echo "$(BOLD)Build Targets:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '(build|release|debug)' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(BOLD)IREE Backend Targets:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '(iree)' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(BOLD)Test Targets:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '(test|check|lint|clippy)' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(BOLD)Run Targets:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '(run|serve|generate)' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(BOLD)Help & Documentation:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '(help|doc|info)' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(BOLD)Benchmark Targets:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '(bench)' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(BOLD)Webpage Targets:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '(webpage)' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(BOLD)Utility Targets:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '(clean|fmt|install)' | grep -v '^bench' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(BOLD)Variables:$(RESET)"
	@echo "  $(CYAN)MODEL$(RESET)    Path to model directory (default: ./models/default)"
	@echo "  $(CYAN)PROMPT$(RESET)   Generation prompt (default: \"Hello, world!\")"
	@echo "  $(CYAN)HOST$(RESET)     Server host (default: 127.0.0.1)"
	@echo "  $(CYAN)PORT$(RESET)     Server port (default: 8080)"
	@echo ""
	@echo "$(BOLD)Examples:$(RESET)"
	@echo "  make build                           # Debug build"
	@echo "  make release                         # Optimized release build"
	@echo "  make run-generate MODEL=./models/llama PROMPT=\"Tell me a joke\""
	@echo "  make serve MODEL=./models/qwen PORT=8000"
	@echo "  make test                            # Run all tests"
	@echo "  make verify                          # macOS merge gate (fmt + clippy + test)"
	@echo "  make verify-test-cuda                # Linux/NVIDIA merge gate (workspace, single threaded)"
	@echo "  make test-fast-cuda FILTER=server::chat_request  # Fast filtered test-fast-profile run"
	@echo ""

# ============================================================================
# Build Targets
# ============================================================================

.PHONY: build
build: ## Build in debug mode
	@echo "$(CYAN)Building in debug mode...$(RESET)"
	$(CARGO) build
	@echo "$(GREEN)Build complete!$(RESET)"

.PHONY: build-cli
build-cli: ## Build only the CLI binary
	@echo "$(CYAN)Building CLI...$(RESET)"
	$(CARGO) build --bin $(BIN_CLI)

.PHONY: build-server
build-server: ## Build only the server binary
	@echo "$(CYAN)Building server...$(RESET)"
	$(CARGO) build --bin $(BIN_SERVER)

.PHONY: release
release: ## Build in release mode (optimized; macOS adds metal,accelerate)
	@echo "$(CYAN)Building in release mode...$(RESET)"
	$(RUSTFLAGS) $(CARGO) build --release $(RELEASE_FEATURE_FLAG)
	@echo "$(GREEN)Release build complete!$(RESET)"
	@echo "Binaries: target/release/$(BIN_CLI), target/release/$(BIN_SERVER)"

.PHONY: release-cli
release-cli: ## Build CLI in release mode
	@echo "$(CYAN)Building CLI in release mode...$(RESET)"
	$(RUSTFLAGS) $(CARGO) build --release $(RELEASE_FEATURE_FLAG) --bin $(BIN_CLI)

.PHONY: recipes-registry
recipes-registry: release-cli ## Rebuild the committed recipes architecture registry snapshot
	@mkdir -p recipes/registry
	@version="$$(./target/release/$(BIN_CLI) --version | awk '{print $$2}')" ; \
	./target/release/$(BIN_CLI) arch --json > "recipes/registry/$${version}.json" ; \
	printf '%s\n' "$${version}" > recipes/registry/CURRENT ; \
	echo "recipes/registry/$${version}.json"

.PHONY: release-server
release-server: ## Build server in release mode
	@echo "$(CYAN)Building server in release mode...$(RESET)"
	$(RUSTFLAGS) $(CARGO) build --release $(RELEASE_FEATURE_FLAG) --bin $(BIN_SERVER)

# ----------------------------------------------------------------------------
# Linux accelerator release targets (explicit, opt-in)
#
# One target per backend. CUDA is the first; add siblings (e.g. release-rocm,
# release-vulkan) here as backends land. Each builds both binaries with the
# matching mlxcel-core feature gate.
# ----------------------------------------------------------------------------
.PHONY: release-cuda
release-cuda: ## Build in release mode with CUDA (Linux/NVIDIA)
	@echo "$(CYAN)Building in release mode with CUDA...$(RESET)"
	$(RUSTFLAGS) $(CARGO) build --release --features cuda
	@echo "$(GREEN)Release (CUDA) build complete!$(RESET)"
	@echo "Binaries: target/release/$(BIN_CLI), target/release/$(BIN_SERVER)"

.PHONY: debug
debug: build ## Alias for build (debug mode)

# ============================================================================
# IREE Toolchain (OpenXLA `xla-iree` backend runtime)
# ============================================================================
#
# Source-build the version-matched IREE runtime + pinned `iree-compile` that the
# `xla-iree` feature links against, wrapping scripts/iree/setup-{cuda,macos}.sh so
# the measurement environment is reproducible from a fresh checkout. The scripts
# are idempotent (they reuse an existing clone / build / venv), so re-running does
# not rebuild when the pinned artifact is already present. `make iree` auto-detects
# the host: CUDA on Linux, Metal on macOS.

IREE_CUDA_SCRIPT := scripts/iree/setup-cuda.sh
IREE_MACOS_SCRIPT := scripts/iree/setup-macos.sh

.PHONY: iree
iree: ## Set up the IREE backend toolchain for this host (CUDA/Linux, Metal/macOS)
ifeq ($(UNAME_S),Darwin)
	@$(MAKE) --no-print-directory iree-metal
else
	@$(MAKE) --no-print-directory iree-cuda
endif

.PHONY: iree-cuda
iree-cuda: ## Set up the IREE CUDA toolchain (Linux/NVIDIA)
	@$(IREE_CUDA_SCRIPT)

.PHONY: iree-metal
iree-metal: ## Set up the IREE Metal toolchain (macOS/Apple Silicon)
	@$(IREE_MACOS_SCRIPT)

.PHONY: iree-env
iree-env: ## Print the resolved IREE toolchain paths + pinned version
ifeq ($(UNAME_S),Darwin)
	@$(IREE_MACOS_SCRIPT) --info
else
	@$(IREE_CUDA_SCRIPT) --info
endif

# ============================================================================
# Test Targets
# ============================================================================

.PHONY: test
test: ## Run all tests
	@echo "$(CYAN)Running tests...$(RESET)"
	RUST_MIN_STACK=$(DEV_TEST_RUST_MIN_STACK) $(CARGO) test -- --test-threads=1
	@echo "$(GREEN)All tests passed!$(RESET)"

.PHONY: test-verbose
test-verbose: ## Run tests with verbose output
	@echo "$(CYAN)Running tests (verbose)...$(RESET)"
	RUST_MIN_STACK=$(DEV_TEST_RUST_MIN_STACK) $(CARGO) test -- --nocapture --test-threads=1

.PHONY: test-lib
test-lib: ## Run library tests only
	RUST_MIN_STACK=$(DEV_TEST_RUST_MIN_STACK) $(CARGO) test --lib -- --test-threads=1

.PHONY: test-doc
test-doc: ## Run documentation tests
	$(CARGO) test --doc

# ----------------------------------------------------------------------------
# Fast dev-iteration test targets (issue #809)
#
# `release*` and a hand-run `cargo test --release` build under
# [profile.release] (fat LTO, codegen-units = 1), which is the right choice for
# shipping validation but measured at 4 to 6 minutes per incremental rebuild on
# the ~390k-line main crate, which is expensive for the edit-test-edit loop of
# local and agent development. These targets build under [profile.test-fast]
# instead (no cross-crate LTO, parallel codegen, incremental compilation); see the
# profile's Cargo.toml comment and docs/installation.md ("Fast iteration
# builds") for the measured speedup. Set FILTER to narrow the run, e.g.
# `make test-fast-cuda FILTER=server::chat_request`.
#
# `verify-test` below also builds under [profile.test-fast] as of #1000, so
# these targets and the CI-faithful gate no longer differ in codegen. What they
# do still differ in is scope and reporting: the gate adds `--workspace` and
# `--no-fail-fast` (#1007) and pins the feature set, while these stay on the
# root package with `--test-threads=1` and a FILTER hook, which is what makes
# them fast enough for an edit-test loop.
#
# Neither of these is a gate, on either platform, and the scope difference is
# the reason. Without `--workspace` a bare `cargo test` here resolves to
# `-p mlxcel`, so `test-fast-cuda` cannot run a single one of mlxcel-core's 1410
# tests, which is the crate holding the MLX bridge, layers.rs, the KV cache and
# the quantization loaders. Use `verify-test` (macOS) or `verify-test-cuda`
# (Linux/NVIDIA) below before you push; these two are for the edit-test loop.
# ----------------------------------------------------------------------------

.PHONY: test-fast
test-fast: ## Run tests under the fast dev-iteration profile (macOS adds metal,accelerate; set FILTER=<path>; issue #809)
	@echo "$(CYAN)Running tests (test-fast profile)...$(RESET)"
	$(CARGO) test --profile test-fast $(RELEASE_FEATURE_FLAG) $(FILTER) -- --test-threads=1
	@echo "$(GREEN)All tests passed!$(RESET)"

.PHONY: test-fast-cuda
test-fast-cuda: ## Run tests under the fast dev-iteration profile with CUDA (Linux/NVIDIA; set FILTER=<path>; issue #809)
	@echo "$(CYAN)Running tests (test-fast profile, CUDA)...$(RESET)"
	$(CARGO) test --profile test-fast --features cuda $(FILTER) -- --test-threads=1
	@echo "$(GREEN)All tests passed!$(RESET)"

.PHONY: check-fast
check-fast: ## Check all targets under the fast dev-iteration profile (issue #809)
	@echo "$(CYAN)Checking code (test-fast profile)...$(RESET)"
	$(CARGO) check --all-targets --profile test-fast $(RELEASE_FEATURE_FLAG)

.PHONY: check
check: ## Check code without building
	@echo "$(CYAN)Checking code...$(RESET)"
	$(CARGO) check

.PHONY: check-all
check-all: ## Check all targets including tests
	$(CARGO) check --all-targets

.PHONY: clippy
clippy: ## Run clippy linter
	@echo "$(CYAN)Running clippy...$(RESET)"
	$(CARGO) clippy -- -W warnings

.PHONY: clippy-fix
clippy-fix: ## Run clippy and apply fixes
	$(CARGO) clippy --fix --allow-dirty

.PHONY: lint
lint: clippy ## Alias for clippy

# ============================================================================
# Run Targets
# ============================================================================

.PHONY: run
run: run-generate ## Alias for run-generate

.PHONY: run-generate
run-generate: build-cli ## Run text generation
	@echo "$(CYAN)Generating text...$(RESET)"
	$(CARGO) run --bin $(BIN_CLI) -- generate -m $(MODEL) -p $(PROMPT)

.PHONY: run-generate-release
run-generate-release: release-cli ## Run text generation (release mode)
	@echo "$(CYAN)Generating text (release)...$(RESET)"
	./target/release/$(BIN_CLI) generate -m $(MODEL) -p $(PROMPT)

.PHONY: run-list
run-list: build-cli ## List supported models
	$(CARGO) run --bin $(BIN_CLI) -- list

.PHONY: serve
serve: build-server ## Start the HTTP server
	@echo "$(CYAN)Starting server on $(HOST):$(PORT)...$(RESET)"
	$(CARGO) run --bin $(BIN_SERVER) -- --model $(MODEL) --host $(HOST) --port $(PORT)

.PHONY: serve-release
serve-release: release-server ## Start the HTTP server (release mode)
	@echo "$(CYAN)Starting server on $(HOST):$(PORT) (release)...$(RESET)"
	./target/release/$(BIN_SERVER) --model $(MODEL) --host $(HOST) --port $(PORT)

# ============================================================================
# Help & Documentation Targets
# ============================================================================

.PHONY: help-cli
help-cli: build-cli ## Show CLI help
	@echo ""
	@echo "$(BOLD)=== mlxcel CLI Help ===$(RESET)"
	@echo ""
	$(CARGO) run --bin $(BIN_CLI) -- --help

.PHONY: help-generate
help-generate: build-cli ## Show generate command help
	@echo ""
	@echo "$(BOLD)=== Generate Command Help ===$(RESET)"
	@echo ""
	$(CARGO) run --bin $(BIN_CLI) -- generate --help

.PHONY: help-server
help-server: build-server ## Show server help
	@echo ""
	@echo "$(BOLD)=== mlxcel-server Help ===$(RESET)"
	@echo ""
	$(CARGO) run --bin $(BIN_SERVER) -- --help

.PHONY: help-all
help-all: help-cli help-generate help-server ## Show all help messages

.PHONY: doc
doc: ## Generate documentation
	@echo "$(CYAN)Generating documentation...$(RESET)"
	$(CARGO) doc --no-deps
	@echo "$(GREEN)Documentation generated at target/doc/mlxcel/index.html$(RESET)"

.PHONY: doc-open
doc-open: doc ## Generate and open documentation
	$(CARGO) doc --no-deps --open

.PHONY: info
info: ## Show project information
	@echo ""
	@echo "$(BOLD)mlxcel Project Info$(RESET)"
	@echo "======================"
	@echo ""
	@echo "$(CYAN)Binaries:$(RESET)"
	@echo "  - mlxcel        : CLI for text generation"
	@echo "  - mlxcel-server : OpenAI-compatible HTTP server"
	@echo ""
	@echo "$(CYAN)Supported Models (57+):$(RESET)"
	@echo "  Transformer : Llama, Qwen, Gemma, Phi, Mistral, DeepSeek, etc."
	@echo "  MoE         : Mixtral, DeepSeek V2/V3, Qwen MoE, GLM4 MoE"
	@echo "  SSM/RNN     : Mamba 1/2, RWKV7, RecurrentGemma"
	@echo "  Hybrid      : Jamba, Qwen3 Next, Nemotron-H"
	@echo ""
	@echo "$(CYAN)Key Features:$(RESET)"
	@echo "  - Sampling: temperature, top-p, top-k, min-p, XTC"
	@echo "  - Repetition: penalty, DRY (Don't Repeat Yourself)"
	@echo "  - Acceleration: LoRA adapters, speculative decoding"
	@echo "  - Server: OpenAI-compatible API, streaming support"
	@echo ""
	@echo "$(CYAN)Documentation:$(RESET)"
	@echo "  - docs/model_implementations.md : Supported models"
	@echo "  - docs/ARCHITECTURE.md          : System architecture"
	@echo ""

.PHONY: models
models: run-list ## Alias for run-list (show supported models)

# ============================================================================
# Utility Targets
# ============================================================================

.PHONY: clean
clean: ## Clean build artifacts
	@echo "$(CYAN)Cleaning build artifacts...$(RESET)"
	$(CARGO) clean
	rm -rf webpage/site/.next webpage/site/out
	@echo "$(GREEN)Clean complete!$(RESET)"

.PHONY: clean-release
clean-release: ## Clean only release artifacts
	rm -rf target/release

.PHONY: fmt
fmt: ## Format code
	@echo "$(CYAN)Formatting code...$(RESET)"
	$(CARGO) fmt

.PHONY: fmt-check
fmt-check: ## Check code formatting
	$(CARGO) fmt -- --check

.PHONY: install
install: release ## Install binaries to ~/.cargo/bin
	@echo "$(CYAN)Installing binaries...$(RESET)"
	$(CARGO) install --path .
	@echo "$(GREEN)Installed: $(BIN_CLI), $(BIN_SERVER)$(RESET)"

.PHONY: uninstall
uninstall: ## Uninstall binaries
	$(CARGO) uninstall mlxcel

.PHONY: update
update: ## Update dependencies
	@echo "$(CYAN)Updating dependencies...$(RESET)"
	$(CARGO) update

.PHONY: tree
tree: ## Show dependency tree
	$(CARGO) tree

.PHONY: outdated
outdated: ## Check for outdated dependencies
	$(CARGO) outdated 2>/dev/null || echo "Install cargo-outdated: cargo install cargo-outdated"

.PHONY: bloat
bloat: release ## Analyze binary size
	$(CARGO) bloat --release 2>/dev/null || echo "Install cargo-bloat: cargo install cargo-bloat"

# ============================================================================
# Development Workflow
# ============================================================================

.PHONY: dev
dev: fmt check test ## Development workflow: format, check, test

.PHONY: ci
ci: fmt-check check clippy test ## CI workflow: format check, check, lint, test

.PHONY: pre-commit
pre-commit: fmt clippy test ## Pre-commit checks

# ----------------------------------------------------------------------------
# CI-faithful local gate (matches .github/workflows/nightly-verify.yml)
#
# The `verify*` targets ARE what the nightly workflow runs: nightly-verify.yml
# invokes these same Makefile targets so the local gate and the scheduled
# backstop cannot drift apart. Note that ci.yml gates only fmt and cargo-deny
# at PR time, so running this before you push is the real gate, not a
# formality; the nightly is only a net that catches a red `main` within a day.
# They differ from the looser `clippy` / `test` targets above in three ways
# that have repeatedly bitten us:
#
#   1. `--features metal,accelerate` — the CI feature set. Without it, large
#      gated regions of mlxcel-core (parts of cache/turbo/quant, the
#      attention-dispatch helpers, etc.) are never type-checked locally, so
#      lint or build errors land first on CI.
#   2. `-D warnings` — promotes every clippy warning to an error, matching
#      `-- -D warnings` in CI. The default `clippy` target uses `-W warnings`
#      and silently hides regressions.
#   3. `--profile test-fast`: CI runs tests optimised, not in debug, for
#      realistic MLX/Metal codegen. Debug-mode tests can pass while optimised
#      tests hit different paths. `test-fast` keeps `opt-level = 3`, which is
#      what that argument actually rests on, and drops cross-crate LTO plus
#      `codegen-units = 1` of `[profile.release]`, which exist to tune a
#      shipped binary and were costing the nightly its whole budget in
#      codegen and linking (#1000/#1406). release.yml still builds and links
#      what ships under `[profile.release]`. The residual gap is a defect that
#      reproduces only under release LTO or single-unit codegen; use
#      `cargo test --release --features metal,accelerate` by hand when you
#      are chasing one.
#   4. `--workspace`: all five members, not just the root package. The
#      workspace root IS the `mlxcel` package, so a bare `cargo test` or
#      `cargo clippy` resolves to `-p mlxcel` and never builds mlxcel-core,
#      mlxcel-mlx-pin, mlxcel-surgery or mlxcel-xla, let alone their test
#      targets. 1754 tests
#      were invisible on that basis, mlxcel-core's 1354 of them, and so was
#      test-only lint debt: six clippy errors sat in mlxcel-xla's lib-test
#      target while the root gate passed clean, and during #973 five
#      `Debug`-bound compile errors in new mlxcel-core tests passed both of the
#      commands above, only `cargo test -p mlxcel-core` finding them (#1007).
#      `verify-fmt` never had the hole because `cargo fmt --all` was already
#      workspace-wide.
#
#      This is a flag here rather than `default-members` in Cargo.toml on
#      purpose. `default-members` would silently re-scope every bare cargo
#      invocation in the repository, release.yml's
#      `cargo build --release --target aarch64-apple-darwin --locked` included,
#      which would then compile the default-off `mlxcel-xla` into every release
#      build. It would also move the gate's scope into a manifest field that
#      the gate's own command line does not mention, and a scope you cannot
#      read off the command is the defect this item exists to fix.
#
#      What `--workspace` covers is each member at the feature set the root
#      selects: mlxcel-core resolves to metal + accelerate through the root's
#      forwarding, while mlxcel-mlx-pin, mlxcel-surgery and mlxcel-xla resolve
#      to their (empty) defaults. mlxcel-mlx-pin is a leaf with no production
#      role: it hosts the unit tests for the MLX-pin logic in
#      mlxcel-core/build_support/mlx_pin.rs, and it costs the gate almost
#      nothing because it does not depend on mlxcel-core and so never triggers
#      an MLX build. mlxcel-xla's `iree` feature stays off, so its build script
#      skips the C shim and the gate needs no IREE distribution; the code
#      behind `iree`, `diagnostics` and `micro-oracle` is still ungated here.
#
#      Cargo builds all the test binaries and then runs them one at a time, so
#      the mlxcel-core suite never overlaps the root suite on the Metal device.
#      Measured on cargo 1.97.1 against a three-crate probe workspace: the
#      build completes in full before the first test binary starts, and each
#      binary finishes before the next one begins. That is what keeps
#      `--workspace` clear of #1008, where two concurrent mlxcel-core suites
#      aborted 7 of 12 runs. Anything that starts running test binaries in
#      parallel here (cargo-nextest, say) has to re-establish that; the
#      `no_other_mlxcel_core_test_binary_is_sharing_the_gpu` guard only sees a
#      second mlxcel-core binary, not a root-suite one.
#
#      What that argument covers is concurrency *between* binaries, and only
#      that. Item 6 covers the other axis, which is the one that took `main`
#      down: concurrency inside a single binary.
#   5. `--no-fail-fast` on the test target. Without it the first failing test
#      binary ends the run, which was harmless when the run was one package and
#      is not now: a single red root-suite test would hide all of mlxcel-core,
#      mlxcel-surgery and mlxcel-xla behind it. A nightly that costs the better
#      part of an hour should report everything that is red in one pass. It
#      does not weaken the gate, since cargo still exits non-zero, and it does
#      not skip compile errors, which stop the run either way.
#   6. `--test-threads=1`, for the reason `verify-test-cuda` already carries it
#      (#1092). Cargo's sequencing above bounds concurrency between binaries
#      and says nothing about concurrency inside one, and libtest defaults to
#      one test thread per logical CPU. The 2026-08-16 nightly died with
#      `signal: 11, SIGSEGV` in the mlxcel-core binary, with no panic and no
#      `test result` line, and the macOS crash report from the local repro on
#      an 18-core M5 Max names the shape without ambiguity: 18 libtest worker
#      threads live at the fault, every one of them in an MLX-backed cache
#      test, two inside `iokit_user_client_trap` and two inside the allocator,
#      faulting on an address in no mapped region. That is #1048's CUDA abort
#      on the other backend, and `--test-threads` is the flag that bounds it.
#      `--jobs 1` is not: it bounds the build, which has already finished.
#
#      It costs almost nothing, because the work serializes on the one Metal
#      device whether or not the host threads do. Measured on M5 Max at
#      `5dfcb390`, warm, whole workspace, 101 binaries and 8128 tests: 69.17s
#      parallel against 76.39s serialized. That is +7.2s on a step the nightly
#      budgets 180 minutes for and whose time goes to the build. The two big
#      members move in opposite directions and nearly cancel: mlxcel-core
#      costs +23s serialized (10.2s to 33.2s) while the root suite gains 12s
#      (23.5s to 11.6s), thread contention across 5695 tests being worse than
#      running them in a row.
#
#      `make test-fast` has passed `--test-threads=1` on macOS since #809, so
#      this makes the gate agree with the edit-test loop instead of diverging
#      from it. Narrowed hand-runs stay parallel and that is fine; it is
#      whole-suite runs that crash.
#
# `verify-test-cuda` is the Linux/NVIDIA counterpart of `verify-test`: same
# scope, same profile, same `--no-fail-fast`, `--features cuda` instead of
# `metal,accelerate`, plus `--test-threads=1`. Two things about it:
#
#   * Until #1048 there was no CUDA target that ran mlxcel-core's tests at all.
#     `verify-test` is the macOS feature set and `test-fast-cuda` is root-package
#     only, so the crate with the MLX bridge and the KV cache had no gate on this
#     platform. That is #1007's blindness a second time, on the other backend.
#   * `--test-threads=1` is load-bearing, not tidiness. Measured on GB10 at MLX
#     pin 2c46b953: the 20-thread run dies with SIGABRT from
#     `cudaStreamEndCapture ... previous error during capture`, at a different
#     test each time; the same binary serialized finishes 1410 tests in 88s.
#     Turning graph capture off does not rescue the parallel run, it only
#     re-reports the abort as `cuLaunchKernelEx ... invalid argument`, so
#     MLX_USE_CUDA_GRAPHS=0 is not the workaround it looks like. Capture stays
#     fully on under this target. mlxcel-core carries a
#     `the_cuda_test_suite_must_run_single_threaded` guard so a hand-run
#     `cargo test --workspace --features cuda` names the problem instead of
#     aborting anonymously.
#   * Neither gate sets `MLX_ENABLE_TF32`, and that is deliberate rather than an
#     oversight: the pin lives in the test binaries instead. MLX defaults the
#     variable to 1 in `mlx/utils.h`, which routes f32 GEMMs through a
#     reduced-precision kernel (TF32 on CUDA, NAX on Apple GPU generation 17)
#     and breaks the suite's algorithm-equivalence tests, the ones that compute
#     the same quantity two ways and expect full-f32 agreement. #1088 hit this
#     on the CUDA gate and #1259 hit the same mechanism on Metal. The fix
#     (#1260) is a `ctor` in `src/lib.rs` and in `src/lib/mlxcel-core/src/lib.rs`
#     that sets `MLX_ENABLE_TF32=0` before MLX latches the value into its
#     process-wide static, so it covers every gate running those two binaries
#     without the two gates' command lines diverging from each other. An
#     explicit operator setting still wins over the pin.
#
#     The policy this encodes: unit tests verify algorithm equivalence at full
#     f32, while shipped numerics stay on MLX defaults and are covered by the
#     runtime exactness probes (#1188, #1189). So do not answer a future red
#     here by widening a parity tolerance before checking whether the run had
#     TF32 on. Measured on GB10 (sm_121) at `670512c2`: this gate is 8167 passed
#     and 0 failed, while re-running with `MLX_ENABLE_TF32=1` reds every one of
#     the four tests #1088 listed. All four now fail on every run, each at a
#     bit-identical magnitude: klear's is 1.0485351e-3 against its 1e-3 bound,
#     versus 5.364418e-7 under the pin, both reproducing on 10 of 10 runs.
#
#     klear used to be the exception here, and #1265 found out why. It failed
#     only 8 runs in 10 with its delta wandering over 5.9e-4 to 2.0e-3, which
#     earlier notes at this spot read as a property of the backend. It was not.
#     `filled_weights` in `src/models/klear_tests.rs` walked a `WeightMap`, which
#     is a `HashMap`, and advanced one seed per key; `RandomState` randomizes that
#     iteration order per process, so every run built a DIFFERENT random model.
#     Sorting the keys fixed it, and the same fixture bug was in afmoe, phixtral
#     and bailing_moe_linear. Two claims that stood here before are therefore
#     withdrawn: the variance was not klear's own numerics, and the fixture is not
#     a quantized MoE model at all (it ships no `.scales`, so the expert stacks
#     load dense and neither the fused nor the `gather_qmm` quantized path runs).
#     Measured on GB10 alongside the fix: both arms are bit-identical to
#     themselves across repeats in one process, at both precision settings.
#
#     What this all means here is that a single green run under
#     `MLX_ENABLE_TF32=1` is not evidence that a test has stopped depending on
#     the pin. The pin is load-bearing, not decorative.
#
# There is deliberately no `verify-clippy-cuda` here yet: the CUDA lint half is
# a separate hole from the CUDA test half, and #1048 is about the test gate.
#
# `verify-test-video` is the third gate of the same shape, for a capability that
# lives outside the binary entirely: video decoding shells out to the system
# ffmpeg. It is NOT part of `verify`, because a contributor without ffmpeg
# should still get a green `make verify`. That exemption is exactly what went
# wrong in #1172: the ffmpeg-backed tests skipped when the binary was absent and
# libtest counted the skip as a pass, so the suite stayed green for as long as
# it took ffmpeg 8 to remove `-vsync` and break every video path in the runtime.
# The tests are `#[ignore]` now, so a plain run reports them as ignored instead
# of passed, and this target is the one that runs them for real:
# `--include-ignored` to select them and `MLXCEL_TEST_VIDEO=1` so a missing
# ffmpeg is a hard failure rather than a skip. nightly-verify.yml installs
# ffmpeg and runs it.
#
# `--skip bench_single_pass_768_frames` because that one is `#[ignore]` for an
# unrelated reason: it is a wall-clock benchmark asserting a 500 ms bound, and
# `--include-ignored` would otherwise sweep it into a correctness gate where a
# loaded runner turns a timing bound into a red build. Run it by hand when you
# want the number.
#
# The filter list carries a second entry that is not under `multimodal::video`:
# `vision::processors::gemma4::tests::process_videos_pixel_values_match_input_color`.
# That test calls `load_video` on the same path and swallowed a decode failure
# the same way, so #1172 hid there too, and a module-scoped filter would leave
# the one ffmpeg-backed test outside `multimodal::video` ungated. libtest ORs
# positional filters, but `cargo test` itself accepts only one TESTNAME, so both
# filters have to sit after the `--`.
#
# Run `make verify` before opening or updating a PR. Run `make verify-clean`
# (which prepends `cargo clean`) when you suspect clippy's per-crate result
# cache is masking a regression — most often after editing shared code in
# mlxcel-core that other crates inherit lints from.
# ----------------------------------------------------------------------------

.PHONY: verify-fmt
verify-fmt: ## CI-faithful: cargo fmt --all -- --check
	@echo "$(CYAN)[verify] fmt check...$(RESET)"
	$(CARGO) fmt --all -- --check

.PHONY: verify-clippy
verify-clippy: ## CI-faithful: clippy --workspace --all-targets --features metal,accelerate -- -D warnings
	@echo "$(CYAN)[verify] clippy (workspace, features=metal,accelerate, -D warnings)...$(RESET)"
	$(CARGO) clippy --workspace --all-targets --features metal,accelerate -- -D warnings

.PHONY: verify-test
verify-test: ## CI-faithful: cargo test --workspace --profile test-fast --features metal,accelerate --no-fail-fast -- --test-threads=1 (issue #1092)
	@echo "$(CYAN)[verify] test (workspace, test-fast profile, features=metal,accelerate, single threaded)...$(RESET)"
	$(CARGO) test --workspace --profile test-fast --features metal,accelerate --no-fail-fast -- --test-threads=1

.PHONY: verify-test-video
verify-test-video: ## Video gate: run the ffmpeg-backed video tests for real (needs ffmpeg 5.0+ on PATH, issue #1172)
	@echo "$(CYAN)[verify] video (multimodal::video + gemma4 pixel content, ffmpeg required)...$(RESET)"
	MLXCEL_TEST_VIDEO=1 $(CARGO) test --profile test-fast --features metal,accelerate \
		-p mlxcel --lib -- --include-ignored \
		--skip bench_single_pass_768_frames \
		multimodal::video \
		vision::processors::gemma4::tests::process_videos_pixel_values_match_input_color

.PHONY: verify-test-cuda
verify-test-cuda: ## CUDA gate: cargo test --workspace --profile test-fast --features cuda --no-fail-fast -- --test-threads=1 (issue #1048)
	@echo "$(CYAN)[verify] test (workspace, test-fast profile, features=cuda, single threaded)...$(RESET)"
	$(CARGO) test --workspace --profile test-fast --features cuda --no-fail-fast -- --test-threads=1

.PHONY: verify-versions
verify-versions: ## Assert every version-tracking workspace crate carries the root `mlxcel` version
	@echo "$(CYAN)[verify] workspace crate versions...$(RESET)"
	@python3 scripts/ci/check_crate_versions.py

.PHONY: verify-kernel-dtype-keys
verify-kernel-dtype-keys: ## Assert every CUDA JIT kernel launch keys its cache on the input dtypes (issues #1053, #1054)
	@echo "$(CYAN)[verify] kernel dtype cache keys...$(RESET)"
	@python3 scripts/ci/check_kernel_dtype_keys.py

# Offline structural half of the llama-server b10621 compatibility gate
# (issue #1443, epic #1431): validates the checked-in manifest under
# compat/llama-server/b10621/ without network or the b10621 archive. The
# `llama-compat manifest` CI job runs the same script with
# --check-issues-open added (it has a GH token; a local run may not). The
# binary-facing half runs inside `verify-test` as
# tests/llama_compat_manifest.rs and src/server/llama_compat_tests.rs.
# check_llama_compat_manifest_test.sh is the validator's own negative
# coverage: it mutates a throwaway copy of the manifest and asserts the gate
# rejects it, so the rules keep failing on a bad manifest rather than only
# passing on a good one.
.PHONY: verify-llama-compat
verify-llama-compat: ## Assert the llama-server b10621 compatibility manifest is structurally valid (issue #1443)
	@echo "$(CYAN)[verify] llama-server b10621 compatibility manifest...$(RESET)"
	@python3 scripts/ci/check_llama_compat_manifest.py
	@bash scripts/ci/check_llama_compat_manifest_test.sh

.PHONY: bump-version
bump-version: ## Release: set every version-tracking crate to VERSION and sync Cargo.lock (make bump-version VERSION=0.5.0)
	@test -n "$(VERSION)" || { echo "$(RED)usage: make bump-version VERSION=0.5.0$(RESET)"; exit 1; }
	@python3 scripts/ci/check_crate_versions.py --set "$(VERSION)"
	@$(CARGO) update $$(python3 scripts/ci/check_crate_versions.py --print-update-args)
	@$(MAKE) --no-print-directory verify-versions

.PHONY: verify
verify: verify-versions verify-kernel-dtype-keys verify-llama-compat verify-fmt verify-clippy verify-test ## Run the full CI-faithful gate locally (recommended before push)
	@echo "$(GREEN)[verify] OK: matches the nightly-verify GitHub Actions job$(RESET)"

.PHONY: verify-clean
verify-clean: ## Run `verify` after a `cargo clean` (use when clippy's cache may be hiding a regression)
	@echo "$(YELLOW)[verify-clean] dropping target/ to force a fresh clippy/test pass...$(RESET)"
	$(CARGO) clean
	@$(MAKE) --no-print-directory verify

# ============================================================================
# Quick Examples
# ============================================================================

.PHONY: example-greedy
example-greedy: build-cli ## Example: greedy decoding
	@echo "$(BOLD)Example: Greedy Decoding (temp=0)$(RESET)"
	$(CARGO) run --bin $(BIN_CLI) -- generate -m $(MODEL) -p $(PROMPT) --temp 0

.PHONY: example-creative
example-creative: build-cli ## Example: creative sampling
	@echo "$(BOLD)Example: Creative Sampling$(RESET)"
	$(CARGO) run --bin $(BIN_CLI) -- generate -m $(MODEL) -p $(PROMPT) \
		--temp 0.8 --top-p 0.95 --top-k 40

.PHONY: example-dry
example-dry: build-cli ## Example: DRY penalty
	@echo "$(BOLD)Example: DRY Penalty (prevents repetition)$(RESET)"
	$(CARGO) run --bin $(BIN_CLI) -- generate -m $(MODEL) -p $(PROMPT) \
		--temp 0.7 --dry-multiplier 1.0 --dry-base 1.75

.PHONY: example-speculative
example-speculative: build-cli ## Example: speculative decoding
	@echo "$(BOLD)Example: Speculative Decoding$(RESET)"
	@echo "Requires DRAFT_MODEL variable"
	$(CARGO) run --bin $(BIN_CLI) -- generate -m $(MODEL) -p $(PROMPT) \
		--draft-model $(DRAFT_MODEL) --num-draft-tokens 4

# ============================================================================
# Aliases
# ============================================================================

.PHONY: b r t c l
b: build      ## Alias for build
r: release    ## Alias for release
t: test       ## Alias for test
c: check      ## Alias for check
l: lint       ## Alias for lint

# ============================================================================
# Model Benchmark Targets
# ============================================================================

# Benchmark configuration
MODELS_DIR := ./models
TEST_PROMPT := "Hello, how are you today?"
TEST_IMAGE := /tmp/test_cat.jpg
MAX_TOKENS := 100
BENCH_LOG := benchmark_results.log
BENCH_BIN := ./target/release/mlxcel
BENCH_TIMEOUT := 120

# VLM models (require --image flag)
VLM_MODELS := \
	aya-vision-8b \
	bunny-llama3-8b \
	gemma3-4b \
	gemma3n-e2b \
	gemma3n-e4b \
	gemma3n-e4b-4bit \
	llama4-scout \
	llava-1.5-7b \
	llava-next-7b \
	llava-qwen-0.5b \
	mimo-7b \
	paligemma2-3b \
	phi3.5-vision \
	pixtral-12b \
	qwen2-vl-2b \
	qwen2.5-vl-3b \
	qwen3-vl-2b \
	qwen3-vl-moe-30b

# Text models (all models in MODELS_DIR minus VLMs, computed dynamically)
ALL_MODELS := $(sort $(notdir $(wildcard $(MODELS_DIR)/*)))
TEXT_MODELS := $(filter-out $(VLM_MODELS),$(ALL_MODELS))

# Inline shell helper: run_bench <model_name> [extra_flags]
# Defined as a Make variable so it can be embedded in recipes via $(BENCH_FN_INLINE)
# Each recipe must include this followed by calls to run_bench
BENCH_FN_INLINE = run_bench() { \
	local m=$$1; local ef="$$2"; \
	local md="$(MODELS_DIR)/$$m"; \
	if [ ! -d "$$md" ]; then \
		printf "\033[33m[SKIP]\033[0m %-35s Model not found\n" "$$m"; \
		echo "[SKIP] $$m  Model not found" >> $(BENCH_LOG); \
		return; \
	fi; \
	local out; out=$$($(BENCH_BIN) generate -m "$$md" $$ef -p $(TEST_PROMPT) -n $(MAX_TOKENS) 2>&1); \
	local ec=$$?; \
	if [ $$ec -eq 0 ]; then \
		local met; met=$$(echo "$$out" | grep -oE 'Generated [0-9]+ tokens in [0-9.]+s = [0-9.]+ tok/s' | tail -1); \
		if [ -n "$$met" ]; then \
			local toks; toks=$$(echo "$$met" | grep -oE '[0-9.]+ tok/s'); \
			local nt; nt=$$(echo "$$met" | grep -oE 'Generated [0-9]+' | grep -oE '[0-9]+'); \
			local sc; sc=$$(echo "$$met" | grep -oE 'in [0-9.]+s' | grep -oE '[0-9.]+'); \
			printf "\033[32m[PASS]\033[0m %-35s %s (%s tokens in %ss)\n" "$$m" "$$toks" "$$nt" "$$sc"; \
			echo "[PASS] $$m  $$toks ($$nt tokens in $${sc}s)" >> $(BENCH_LOG); \
		else \
			printf "\033[32m[PASS]\033[0m %-35s (no metrics found)\n" "$$m"; \
			echo "[PASS] $$m  (no metrics)" >> $(BENCH_LOG); \
		fi; \
	else \
		local err; err=$$(echo "$$out" | grep -iE 'error|panic|fatal' | head -1); \
		if [ -z "$$err" ]; then err=$$(echo "$$out" | tail -1); fi; \
		printf "\033[31m[FAIL]\033[0m %-35s Error: %s\n" "$$m" "$$err"; \
		echo "[FAIL] $$m  Error: $$err" >> $(BENCH_LOG); \
	fi; \
}

.PHONY: bench
bench: bench-text bench-vlm ## Run all model benchmarks (text + VLM)
	@echo ""
	@echo "\033[1mBenchmark complete. Results saved to $(BENCH_LOG)\033[0m"

.PHONY: bench-text
bench-text: ## Run all text model benchmarks
	@echo ""
	@echo "\033[1m=== Text Model Benchmarks ===\033[0m"
	@echo "---"
	@echo "[`date '+%Y-%m-%d %H:%M:%S'`] Text model benchmarks" >> $(BENCH_LOG)
	@$(BENCH_FN_INLINE); \
	for model in $(TEXT_MODELS); do \
		run_bench "$$model" ""; \
	done

.PHONY: bench-vlm
bench-vlm: ## Run all VLM model benchmarks
	@echo ""
	@echo "\033[1m=== VLM Model Benchmarks ===\033[0m"
	@echo "---"
	@if [ ! -f "$(TEST_IMAGE)" ]; then \
		echo "\033[33mCreating test image at $(TEST_IMAGE)...\033[0m"; \
		python3 -c "from PIL import Image; Image.new('RGB', (224, 224), (128, 128, 200)).save('$(TEST_IMAGE)')" 2>/dev/null \
		|| python3 -c "import struct,zlib;raw=b''.join(b'\x00'+bytes([128,128,200]*224) for _ in range(224));c=zlib.compress(raw);import struct as S;ck=lambda t,d:S.pack('>I',len(d))+t+d+S.pack('>I',zlib.crc32(t+d)&0xffffffff);open('$(TEST_IMAGE)','wb').write(b'\x89PNG\r\n\x1a\n'+ck(b'IHDR',S.pack('>IIBBBBB',224,224,8,2,0,0,0))+ck(b'IDAT',c)+ck(b'IEND',b''))" 2>/dev/null \
		|| echo "\033[31mFailed to create test image. VLM tests may fail.\033[0m"; \
	fi
	@echo "[`date '+%Y-%m-%d %H:%M:%S'`] VLM model benchmarks" >> $(BENCH_LOG)
	@$(BENCH_FN_INLINE); \
	for model in $(VLM_MODELS); do \
		run_bench "$$model" "--image $(TEST_IMAGE)"; \
	done

.PHONY: bench-model
bench-model: ## Run single model benchmark (MODEL=models/name)
	@model_name=$$(basename $(MODEL)); \
	is_vlm=0; \
	for v in $(VLM_MODELS); do \
		if [ "$$v" = "$$model_name" ]; then is_vlm=1; break; fi; \
	done; \
	$(BENCH_FN_INLINE); \
	if [ $$is_vlm -eq 1 ]; then \
		if [ ! -f "$(TEST_IMAGE)" ]; then \
			echo "\033[33mCreating test image at $(TEST_IMAGE)...\033[0m"; \
			python3 -c "from PIL import Image; Image.new('RGB', (224, 224), (128, 128, 200)).save('$(TEST_IMAGE)')" 2>/dev/null || true; \
		fi; \
		run_bench "$$model_name" "--image $(TEST_IMAGE)"; \
	else \
		run_bench "$$model_name" ""; \
	fi

.PHONY: bench-report
bench-report: ## Show last benchmark results summary
	@if [ ! -f "$(BENCH_LOG)" ]; then \
		echo "No benchmark results found. Run 'make bench' first."; \
		exit 1; \
	fi
	@echo ""
	@echo "\033[1m=== Benchmark Results Summary ===\033[0m"
	@echo ""
	@pass=$$(grep -c '^\[PASS\]' $(BENCH_LOG) 2>/dev/null; true); \
	fail=$$(grep -c '^\[FAIL\]' $(BENCH_LOG) 2>/dev/null; true); \
	skip=$$(grep -c '^\[SKIP\]' $(BENCH_LOG) 2>/dev/null; true); \
	pass=$${pass:-0}; fail=$${fail:-0}; skip=$${skip:-0}; \
	total=$$((pass + fail + skip)); \
	echo "Total: $$total  \033[32mPASS: $$pass\033[0m  \033[31mFAIL: $$fail\033[0m  \033[33mSKIP: $$skip\033[0m"
	@echo ""
	@echo "\033[1mPassed:\033[0m"
	@grep '^\[PASS\]' $(BENCH_LOG) 2>/dev/null | sort || echo "  (none)"
	@echo ""
	@if grep -q '^\[FAIL\]' $(BENCH_LOG) 2>/dev/null; then \
		echo "\033[1mFailed:\033[0m"; \
		grep '^\[FAIL\]' $(BENCH_LOG); \
		echo ""; \
	fi
	@if grep -q '^\[SKIP\]' $(BENCH_LOG) 2>/dev/null; then \
		echo "\033[1mSkipped:\033[0m"; \
		grep '^\[SKIP\]' $(BENCH_LOG); \
		echo ""; \
	fi

.PHONY: bench-clean
bench-clean: ## Remove benchmark log
	@rm -f $(BENCH_LOG)
	@echo "Benchmark log removed."

# ============================================================================
# Webpage (Next.js static site)
# ============================================================================

.PHONY: webpage-dev
webpage-dev: ## Run download webpage dev server
	@echo "$(CYAN)Starting webpage development server...$(RESET)"
	cd webpage/site && pnpm install && pnpm dev

.PHONY: webpage-build
webpage-build: docs-guard ## Build download webpage (static export) (manual sources not in this checkout)
	@echo "$(CYAN)Building documentation for webpage...$(RESET)"
	rm -rf webpage/site/public/en/manual webpage/site/public/ko/manual
	uv run zensical build -f mkdocs.yml -d webpage/site/public/en/manual
	uv run zensical build -f mkdocs.ko.yml -d webpage/site/public/ko/manual
	@echo "$(CYAN)Building webpage...$(RESET)"
	cd webpage/site && pnpm install && pnpm build
	@echo "$(GREEN)Build complete. Output in webpage/site/out/$(RESET)"

# webpage-deploy is intentionally not gated behind docs-guard. deploy_webpage.sh
# never invokes zensical or reads docs_dir, so it cannot fail the way
# webpage-build does, and docs-guard checks for docs/en (a build input) rather
# than for the manual output deploy actually reads. What deploy needs is the
# manual already built into webpage/site/public/{en,ko}/manual by a prior
# 'make webpage-build' (or copied in from elsewhere); a missing docs/en does
# not imply that output is missing. So this checks the actual precondition and
# warns without blocking, since a maintainer may be re-deploying an unchanged
# site from a previous build.
.PHONY: webpage-deploy
webpage-deploy: ## Deploy download webpage to GitHub Pages
	@echo "$(CYAN)Deploying webpage...$(RESET)"
	@if [ ! -d webpage/site/public/en/manual ] || [ ! -d webpage/site/public/ko/manual ]; then \
		echo "$(YELLOW)Warning: webpage/site/public/en/manual or .../ko/manual is missing.$(RESET)"; \
		echo "  Run 'make webpage-build' first, or the deployed site will be missing (or serving a stale copy of) the manual pages."; \
	fi
	./scripts/deploy_webpage.sh

# ============================================================================
# Documentation (Zensical / MkDocs-compatible)
# ============================================================================
#
# The docs-* targets below build the MkDocs manual. Its sources (docs/en,
# docs/ko, docs/shared, docs/requirements.txt, docs/scripts) are maintained in a
# separate documentation tree and are not part of this repository, so every one
# of these targets depends on docs-guard. The guard is a presence check, not an
# unconditional refusal: where the sources exist the targets run exactly as
# before, and where they do not the build stops with an explanation instead of
# an opaque uv, ln, or zensical failure. See docs/README.md for the split.

DOCS_MANUAL_DIR := docs/en
DOCS_MANUAL_URL := https://mlxcel.lablup.ai/en/manual/

.PHONY: docs-guard
docs-guard:
	@test -d "$(DOCS_MANUAL_DIR)" || { \
		echo "The MkDocs manual sources are not present in this checkout."; \
		echo ""; \
		echo "  '$(DOCS_MANUAL_DIR)' is missing, and so are docs/ko, docs/shared,"; \
		echo "  docs/requirements.txt and docs/scripts. They are maintained in a"; \
		echo "  separate documentation tree, along with its own copies of mkdocs.yml,"; \
		echo "  mkdocs.ko.yml and the two PDF configs. The docs_dir, custom_dir and"; \
		echo "  nav: entries in the configs kept here name paths in that tree, not the"; \
		echo "  docs/*.md files in this repository, so no docs-* target can build"; \
		echo "  anything from this checkout."; \
		echo ""; \
		echo "  Read the published manual instead: $(DOCS_MANUAL_URL)"; \
		echo "  The documents that do live here are indexed in docs/README.md."; \
		exit 1; \
	}

.PHONY: docs-install
docs-install: docs-guard ## Install documentation dependencies and create shared symlinks (manual sources not in this checkout)
	@command -v uv >/dev/null 2>&1 || { \
		echo "Error: uv is not installed. Install it from https://docs.astral.sh/uv/"; \
		exit 1; \
	}
	uv pip install -r docs/requirements.txt
	rm -rf docs/en/shared docs/ko/shared
	ln -s ../shared docs/en/shared
	ln -s ../shared docs/ko/shared
	@echo "Documentation dependencies installed and symlinks created. Run 'make docs-serve' to start the server."

.PHONY: docs-serve
docs-serve: docs-guard ## Serve all docs locally, builds KO first then serves EN (manual sources not in this checkout)
	@echo "Building Korean docs..."
	uv run zensical build -f mkdocs.ko.yml
	@echo "Serving English docs..."
	uv run zensical serve -f mkdocs.yml

.PHONY: docs-serve-en
docs-serve-en: docs-guard ## Serve English docs with live reload (manual sources not in this checkout)
	uv run zensical serve -f mkdocs.yml

.PHONY: docs-serve-ko
docs-serve-ko: docs-guard ## Serve Korean docs with live reload (manual sources not in this checkout)
	uv run zensical serve -f mkdocs.ko.yml

.PHONY: docs-build
docs-build: docs-guard ## Build English docs (manual sources not in this checkout)
	uv run zensical build -f mkdocs.yml

.PHONY: docs-build-ko
docs-build-ko: docs-guard ## Build Korean docs (manual sources not in this checkout)
	uv run zensical build -f mkdocs.ko.yml

.PHONY: docs-build-all
docs-build-all: docs-guard ## Build all docs, EN and KO (manual sources not in this checkout)
	@echo "Building English docs..."
	uv run zensical build -f mkdocs.yml
	@echo "Building Korean docs..."
	uv run zensical build -f mkdocs.ko.yml
	@echo "All docs built in site/"

.PHONY: docs-build-strict
docs-build-strict: docs-guard ## Build all docs in strict mode for CI (manual sources not in this checkout)
	uv run zensical build -f mkdocs.yml
	uv run zensical build -f mkdocs.ko.yml

.PHONY: docs-pdf-setup
docs-pdf-setup: docs-guard ## Install Playwright browser for PDF export, one-time (manual sources not in this checkout)
	uv venv --python 3.13
	uv pip install -r docs/requirements.txt
	uv run python -m playwright install chromium
	@echo "PDF export dependencies ready."

.PHONY: docs-pdf-en
docs-pdf-en: docs-guard ## Export English documentation as PDF (manual sources not in this checkout)
	@echo "Building English documentation as PDF..."
	uv run mkdocs build --config-file mkdocs.pdf.yml -d site/en/manual
	@echo "Fixing PDF internal links..."
	uv run python docs/scripts/fix_pdf_links.py mkdocs.pdf.yml site/en/manual/mlxcel-Manual-en.pdf
	@echo "PDF generated: site/en/manual/mlxcel-Manual-en.pdf"

.PHONY: docs-pdf-ko
docs-pdf-ko: docs-guard ## Export Korean documentation as PDF (manual sources not in this checkout)
	@echo "Building Korean documentation as PDF..."
	uv run mkdocs build --config-file mkdocs.ko.pdf.yml -d site/ko/manual
	@echo "Fixing PDF internal links..."
	uv run python docs/scripts/fix_pdf_links.py mkdocs.ko.pdf.yml site/ko/manual/mlxcel-Manual-ko.pdf
	@echo "PDF generated: site/ko/manual/mlxcel-Manual-ko.pdf"

.PHONY: docs-pdf
docs-pdf: docs-guard docs-pdf-en docs-pdf-ko ## Export all documentation as PDF (manual sources not in this checkout)
	@echo "All PDFs generated:"
	@echo "  - site/en/manual/mlxcel-Manual-en.pdf"
	@echo "  - site/ko/manual/mlxcel-Manual-ko.pdf"

.PHONY: docs-clean
docs-clean: docs-guard ## Remove built docs (manual sources not in this checkout)
	rm -rf site/
	@echo "Built docs removed."
