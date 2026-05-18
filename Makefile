# mlxcel Makefile
# High-performance LLM/VLM/VLA inference on Apple Silicon

# ============================================================================
# Configuration
# ============================================================================

CARGO := cargo
RUSTFLAGS := RUSTFLAGS="-C target-cpu=native"

# Binary names
BIN_CLI := mlxcel
BIN_SERVER := mlxcel-server

# Default model path for examples (override with MODEL=path)
MODEL ?= ./models/default

# Default prompt for examples
PROMPT ?= "Hello, world!"

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
release: ## Build in release mode (optimized)
	@echo "$(CYAN)Building in release mode...$(RESET)"
	$(RUSTFLAGS) $(CARGO) build --release
	@echo "$(GREEN)Release build complete!$(RESET)"
	@echo "Binaries: target/release/$(BIN_CLI), target/release/$(BIN_SERVER)"

.PHONY: release-cli
release-cli: ## Build CLI in release mode
	@echo "$(CYAN)Building CLI in release mode...$(RESET)"
	$(RUSTFLAGS) $(CARGO) build --release --bin $(BIN_CLI)

.PHONY: release-server
release-server: ## Build server in release mode
	@echo "$(CYAN)Building server in release mode...$(RESET)"
	$(RUSTFLAGS) $(CARGO) build --release --bin $(BIN_SERVER)

.PHONY: debug
debug: build ## Alias for build (debug mode)

# ============================================================================
# Test Targets
# ============================================================================

.PHONY: test
test: ## Run all tests
	@echo "$(CYAN)Running tests...$(RESET)"
	$(CARGO) test -- --test-threads=1
	@echo "$(GREEN)All tests passed!$(RESET)"

.PHONY: test-verbose
test-verbose: ## Run tests with verbose output
	@echo "$(CYAN)Running tests (verbose)...$(RESET)"
	$(CARGO) test -- --nocapture --test-threads=1

.PHONY: test-lib
test-lib: ## Run library tests only
	$(CARGO) test --lib -- --test-threads=1

.PHONY: test-doc
test-doc: ## Run documentation tests
	$(CARGO) test --doc

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
# CI-faithful local gate (matches .github/workflows/ci.yml exactly)
#
# The `verify*` targets reproduce the GitHub Actions `clippy + test (macOS
# ARM64)` job step-for-step. They differ from the looser `clippy` / `test`
# targets above in three ways that have repeatedly bitten us:
#
#   1. `--features metal,accelerate` — the CI feature set. Without it, large
#      gated regions of mlxcel-core (parts of cache/turbo/quant, the
#      attention-dispatch helpers, etc.) are never type-checked locally, so
#      lint or build errors land first on CI.
#   2. `-D warnings` — promotes every clippy warning to an error, matching
#      `-- -D warnings` in CI. The default `clippy` target uses `-W warnings`
#      and silently hides regressions.
#   3. `cargo test --release` — CI runs tests in release mode for realistic
#      MLX/Metal codegen. Debug-mode tests can pass while release-mode tests
#      hit different optimisation paths.
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
verify-clippy: ## CI-faithful: clippy --all-targets --features metal,accelerate -- -D warnings
	@echo "$(CYAN)[verify] clippy (features=metal,accelerate, -D warnings)...$(RESET)"
	$(CARGO) clippy --all-targets --features metal,accelerate -- -D warnings

.PHONY: verify-test
verify-test: ## CI-faithful: cargo test --release --features metal,accelerate
	@echo "$(CYAN)[verify] test (release, features=metal,accelerate)...$(RESET)"
	$(CARGO) test --release --features metal,accelerate

.PHONY: verify
verify: verify-fmt verify-clippy verify-test ## Run the full CI-faithful gate locally (recommended before push)
	@echo "$(GREEN)[verify] OK — matches GitHub Actions clippy+test job$(RESET)"

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
webpage-build: ## Build download webpage (static export)
	@echo "$(CYAN)Building documentation for webpage...$(RESET)"
	rm -rf webpage/site/public/en/manual webpage/site/public/ko/manual
	uv run zensical build -f mkdocs.yml -d webpage/site/public/en/manual
	uv run zensical build -f mkdocs.ko.yml -d webpage/site/public/ko/manual
	@echo "$(CYAN)Building webpage...$(RESET)"
	cd webpage/site && pnpm install && pnpm build
	@echo "$(GREEN)Build complete. Output in webpage/site/out/$(RESET)"

.PHONY: webpage-deploy
webpage-deploy: ## Deploy download webpage to GitHub Pages
	@echo "$(CYAN)Deploying webpage...$(RESET)"
	./scripts/deploy_webpage.sh

# ============================================================================
# Documentation (Zensical / MkDocs-compatible)
# ============================================================================

.PHONY: docs-install
docs-install: ## Install documentation dependencies and create shared symlinks
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
docs-serve: ## Serve all docs locally (builds KO first, then serves EN)
	@echo "Building Korean docs..."
	uv run zensical build -f mkdocs.ko.yml
	@echo "Serving English docs..."
	uv run zensical serve -f mkdocs.yml

.PHONY: docs-serve-en
docs-serve-en: ## Serve English docs with live reload
	uv run zensical serve -f mkdocs.yml

.PHONY: docs-serve-ko
docs-serve-ko: ## Serve Korean docs with live reload
	uv run zensical serve -f mkdocs.ko.yml

.PHONY: docs-build
docs-build: ## Build English docs
	uv run zensical build -f mkdocs.yml

.PHONY: docs-build-ko
docs-build-ko: ## Build Korean docs
	uv run zensical build -f mkdocs.ko.yml

.PHONY: docs-build-all
docs-build-all: ## Build all docs (EN + KO)
	@echo "Building English docs..."
	uv run zensical build -f mkdocs.yml
	@echo "Building Korean docs..."
	uv run zensical build -f mkdocs.ko.yml
	@echo "All docs built in site/"

.PHONY: docs-build-strict
docs-build-strict: ## Build all docs with strict mode (for CI)
	uv run zensical build -f mkdocs.yml
	uv run zensical build -f mkdocs.ko.yml

.PHONY: docs-pdf-setup
docs-pdf-setup: ## Install Playwright browser for PDF export (one-time setup)
	uv venv --python 3.13
	uv pip install -r docs/requirements.txt
	uv run python -m playwright install chromium
	@echo "PDF export dependencies ready."

.PHONY: docs-pdf-en
docs-pdf-en: ## Export English documentation as PDF
	@echo "Building English documentation as PDF..."
	uv run mkdocs build --config-file mkdocs.pdf.yml -d site/en/manual
	@echo "Fixing PDF internal links..."
	uv run python docs/scripts/fix_pdf_links.py mkdocs.pdf.yml site/en/manual/mlxcel-Manual-en.pdf
	@echo "PDF generated: site/en/manual/mlxcel-Manual-en.pdf"

.PHONY: docs-pdf-ko
docs-pdf-ko: ## Export Korean documentation as PDF
	@echo "Building Korean documentation as PDF..."
	uv run mkdocs build --config-file mkdocs.ko.pdf.yml -d site/ko/manual
	@echo "Fixing PDF internal links..."
	uv run python docs/scripts/fix_pdf_links.py mkdocs.ko.pdf.yml site/ko/manual/mlxcel-Manual-ko.pdf
	@echo "PDF generated: site/ko/manual/mlxcel-Manual-ko.pdf"

.PHONY: docs-pdf
docs-pdf: docs-pdf-en docs-pdf-ko ## Export all documentation as PDF
	@echo "All PDFs generated:"
	@echo "  - site/en/manual/mlxcel-Manual-en.pdf"
	@echo "  - site/ko/manual/mlxcel-Manual-ko.pdf"

.PHONY: docs-clean
docs-clean: ## Remove built docs
	rm -rf site/
	@echo "Built docs removed."
