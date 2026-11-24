# Nemotron-H validation handoff (#3344)

The implementation targets the official `nvidia/Nemotron-H-4B-Base-8K` checkpoint at revision `faba3b731ad7ea5781b9518ae75fb610a94affcf`. Download its original HF bf16 safetensors, tokenizer, and config to `models/hf/nemotron-h-4b`. Transformers 5.12.1 has native Nemotron-H support; use its CPU torch fallback with `trust_remote_code=False`. The config contains bare `Infinity`, an attention head width of 128 (4096 projection width, hidden width 3072), and an SSM width of 7168; do not rewrite these dimensions.

Record `git rev-parse HEAD`, checkpoint revision, compiler/runtime versions, device identity, command exit codes, and full token IDs with every result. Local CPU structural checks do not establish ROCm acceptance. Keep #3344 open until the required hardware gates have passed.

The independent CPU HF reference is committed in `nemotron_h_reference/hf-f32-oracle.json`, with package versions, checkpoint metadata hashes, and a portable reproduction command in `nemotron_h_reference/provenance.json`. It contains six prompt tokens and forty continuation tokens. This is an HF reference artifact, not XLA or hardware acceptance; it can be passed directly to `--oracle`.

## CUDA GB10 result at `93ddebad`

The implementation was validated on NVIDIA GB10 with driver 580.178.04, f32 recurrence, and the original checkpoint revision above. The compiled binaries came from `93ddebad382478540fdfc987fbca655744be941d`; the CLI/server evidence was captured from the later documentation-only checkout `d7e8a0409d4daa9d6f3f4a784ccfc263e38a3cdb`. Every GPU command ran under `gpu-lock`.

- The real-model oracle gate matched all 40 HF fp32 continuation tokens exactly at context capacity 256.
- The B=4, eight-request gate matched all 170 generated tokens against per-request single-sequence references.
- The CLI printed the same 40 token IDs as the HF oracle.
- Four overlapping HTTP completions each matched a separately captured CLI reference in rendered text and token counts, with 24 completion tokens per request. The HTTP API does not expose token IDs, so this is text and count parity rather than ID parity.
- The server started with `--ctx-size 2048`, B=4, an explicit XLA capacity of 256, and prompt cache/APC disabled. The earlier `--ctx-size 1024` attempt was correctly rejected because 256 tokens per slot is below the server's 512-token minimum.
- Six Mamba2 state-carry tests, five hybrid state-carry tests, and the Nemotron-H prompt-cache exclusion unit test passed on the tested source.

The raw oracle, batch, CLI/server, concurrency, provenance, and focused regression outputs are in `nemotron_h_reference/`. The test-fast throughput values are informational and are not release-build performance claims. No ROCm acceptance result is recorded yet.

Reproduction uses the same explicit capacity and f32 environment:

```bash
eval "$(scripts/iree/setup-cuda.sh --env)"
gpu-lock run --tag issue-3344-cuda-real-oracle -- env MLXCEL_XLA_DEVICE=cuda MLXCEL_XLA_PRECISION=f32 MLXCEL_CACHE_DIR=/tmp/epic564-cache-cuda target/test-fast/examples/xla_oracle_check --model models/hf/nemotron-h-4b --oracle spike/openxla/nemotron_h_reference/hf-f32-oracle.json --device cuda
gpu-lock run --tag issue-3344-cuda-real-batch -- env MLXCEL_XLA_DEVICE=cuda MLXCEL_XLA_PRECISION=f32 MLXCEL_CACHE_DIR=/tmp/epic564-cache-cuda target/test-fast/examples/xla_batch_bench --model models/hf/nemotron-h-4b --prompts spike/openxla/nemotron_h_reference/hf-f32-oracle.json --device cuda --batch 4 --requests 8 --maxcap 24
gpu-lock run --tag issue3344-cli-server -- env MLXCEL_BACKEND=xla MLXCEL_XLA_DEVICE=cuda MLXCEL_XLA_PRECISION=f32 MLXCEL_XLA_CONTEXT_CAPACITY=256 MLXCEL_CACHE_DIR=/tmp/epic564-cache-cuda target/test-fast/mlxcel generate -m models/hf/nemotron-h-4b -p 'The capital of France is' -n 40 --temp 0 --no-chat-template
gpu-lock run --tag issue3344-cli-server -- env MLXCEL_BACKEND=xla MLXCEL_XLA_DEVICE=cuda MLXCEL_XLA_PRECISION=f32 MLXCEL_XLA_CONTEXT_CAPACITY=256 MLXCEL_CACHE_DIR=/tmp/epic564-cache-cuda target/test-fast/mlxcel-server -m models/hf/nemotron-h-4b --host 127.0.0.1 --port 59511 --max-batch-size 4 --ctx-size 2048
gpu-lock run --tag issue-3344-cuda-mamba-regressions -- env MLXCEL_XLA_DEVICE=cuda MLXCEL_XLA_PRECISION=f32 cargo test --profile test-fast --features xla-iree,test-utils --test xla_ssm_state_carry -- --ignored --test-threads=1
gpu-lock run --tag issue-3344-cuda-mamba-regressions -- env MLXCEL_XLA_DEVICE=cuda MLXCEL_XLA_PRECISION=f32 cargo test --profile test-fast --features xla-iree,test-utils --test xla_hybrid_nemotron_h -- --ignored --test-threads=1
cargo test --profile test-fast -p mlxcel --lib server::xla_prompt_cache_guard::tests::an_emitter_loadable_nemotron_h_gets_no_prompt_cache_or_apc -- --exact
```

## ROCm gfx1151

Every build and device operation on the 31 GiB Strix Halo host runs under its guard. Run the HF fp32 oracle (~16 GB weights) and XLA f32 engine sequentially. F32 is the recurrent default; an explicit f16 comparison cannot satisfy token-exact acceptance.

```bash
git rev-parse HEAD
eval "$(scripts/iree/setup-rocm.sh --env)"
export MLXCEL_XLA_DEVICE=hip MLXCEL_XLA_ROCM_TARGET=gfx1151
export MLXCEL_XLA_PRECISION=f32
export OPENSSL_INCLUDE_DIR=/usr/include OPENSSL_LIB_DIR=/usr/lib/x86_64-linux-gnu CARGO_BUILD_JOBS=6
export MLXCEL_ORACLE_PYTHON=spike/openxla/.venv/bin/python
# Verify torch==2.12.1+cpu and transformers==5.12.1 before generating the oracle.
scripts/rocm_gpu_guard.sh --hold -- "$MLXCEL_ORACLE_PYTHON" spike/openxla/oracle_continuation.py --model models/hf/nemotron-h-4b --out /tmp/nemotron-h-hf-f32.json --prompt 'The capital of France is' --max-new 40
scripts/rocm_gpu_guard.sh --hold -- cargo test --release --features xla-iree,test-utils --test xla_hybrid_nemotron_h -- --ignored --test-threads=1
scripts/rocm_gpu_guard.sh --hold -- scripts/xla/validate_arch.sh --model models/hf/nemotron-h-4b --oracle /tmp/nemotron-h-hf-f32.json --device hip --batch 4 --requests 8 --maxcap 24
scripts/rocm_gpu_guard.sh --hold -- cargo build --release --features xla-iree --bin mlxcel --bin mlxcel-server
scripts/rocm_gpu_guard.sh --hold -- env MLXCEL_BACKEND=xla MLXCEL_PRINT_TOKEN_IDS=1 target/release/mlxcel generate -m models/hf/nemotron-h-4b -p 'The capital of France is' -n 40 --temp 0 --no-chat-template
```

Require the CLI continuation IDs to equal the independent HF oracle exactly. Never accept `validate_arch.sh`'s XLA local-task fallback as an HF oracle; provide the explicit oracle file above. Run four concurrent `/v1/completions` requests at temperature zero with different prompts and compare each stream with a separately captured CLI continuation using identical tokenizer/prompt settings. Capture the server launch command and payloads with the evidence. Prompt caching and APC must stay disabled for this recurrent model.

Rerun the #3340 Mamba2 gates after the hybrid check:

```bash
scripts/rocm_gpu_guard.sh --hold -- cargo test --release --features xla-iree,test-utils --test xla_ssm_single_seq -- --ignored --test-threads=1
scripts/rocm_gpu_guard.sh --hold -- cargo test --release --features xla-iree,test-utils --test xla_ssm_state_carry -- --ignored --test-threads=1
```

On GB10, GPU commands use `gpu-lock run --tag issue-3344-validation -- ...`; CUDA evidence does not substitute for ROCm evidence. The tiny synthetic fixture can also run with `MLXCEL_XLA_DEVICE=local-task` and the installed IREE runtime. No recurrent embeddings/deepstack/diagnostics graph support is claimed.
