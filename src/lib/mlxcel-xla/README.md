# mlxcel-xla

OpenXLA / StableHLO compiler-family inference backend for mlxcel (issue #449,
ADR 0004 Track B). **Default-off.** The root crate compiles this only under the
`xla-backend` feature, so Apple-Silicon and CUDA shipping builds never touch it.

It hosts [`XlaInferenceSession`], which fills in the engine-neutral
`InferenceSession` contract from `mlxcel-core` (token-level `prefill` /
`decode_step` with on-device sampling). A model is authored once as a StableHLO
graph (the Rust emitter, issue #451), `iree-compile` lowers `prefill` and a
single-token `decode_step` to a vmfb, and the IREE runtime executes them with the
weights resident on the device and the next-token argmax computed on-device.

## Two feature gates

| Build | What compiles | Needs IREE dist? |
|-------|---------------|------------------|
| default / `--features xla-backend` (root) | Pure Rust: the crate + seam. `prefill` / `decode_step` return a clear "built without `iree`" error. | No (so CI builds it) |
| `--features xla-iree` (root) = `xla-backend` + `mlxcel-xla/iree` | Real execution: the C shim (`csrc/xla_iree.c`) is compiled against the prebuilt IREE runtime and the session drives the bundled graphs. | Yes (`IREE_DIST`) |

Why the split: `--features xla-backend` must stay buildable in CI, which has no
IREE distribution. The native execution path is behind the extra `iree` feature.

## Running it (Phase 3 M2)

```bash
# 1. Get the prebuilt IREE dist (runtime static libs + headers + iree-compile).
#    Pin the version used to author the bundled vmfbs (see spike/iree-ffi).
export IREE_DIST=/path/to/extracted/iree-dist-<ver>-linux-<arch>

# 2. Build with real execution on.
cargo build --release --features xla-iree

# 3. Select the backend at runtime and generate.
MLXCEL_BACKEND=xla ./target/release/mlxcel generate \
  -m <Llama-3.2-1B-Instruct dir> -p "..." -n 48
```

CPU (`local-task`) is the proven path, token-exact (48/48) vs the HF temp-0
reference. `MLXCEL_XLA_DEVICE` selects the HAL device (default `local-task`).

### CUDA (GPU) build

The prebuilt dist is CPU/Vulkan only (no CUDA driver, and its `iree-compile` has
no CUDA codegen). The CUDA path therefore uses a **source-built cuda-enabled IREE
runtime** plus a **cuda-capable `iree-compile`**, version-matched to each other.
It is a separate, mutually-exclusive build mode (set `IREE_CUDA_HOME` instead of
`IREE_DIST`). `make iree-cuda` (wrapping `scripts/iree/setup-cuda.sh`) automates
it: it installs the pinned cuda `iree-compile` into a private venv and
source-builds the version-matched runtime (`local-task`/`local-sync`/`cuda`
drivers), then prints the env. Idempotent: re-running reuses the clone/build/venv.

```bash
# 1. Source-build the cuda IREE runtime + pinned iree-compile (idempotent).
make iree-cuda                                 # or `make iree` (auto-detects the host)
eval "$(scripts/iree/setup-cuda.sh --env)"     # export IREE_CUDA_HOME / _COMPILE / MLXCEL_XLA_IREE_COMPILE
# (inspect the resolved paths + pinned version any time: `make iree-env`)

# 2. Build with real execution on.
cargo build --release --features xla-iree

# 3. Run on the GPU.
MLXCEL_BACKEND=xla MLXCEL_XLA_DEVICE=cuda ./target/release/mlxcel generate \
  -m <Llama-3.2-1B-Instruct dir> -p "..." -n 48
```

Validated on a GB10 (Grace-Blackwell, sm_121): token-exact 48/48, ~5 tok/s
(~2.6x the CPU path). Vulkan via the prebuilt dist does **not** work on the GB10
(IREE's Vulkan allocator vs NVIDIA's unified memory), so CUDA is the GPU path.

### macOS (Apple Silicon, Metal) build

On Apple Silicon, **MLX is the default and primary backend**; this OpenXLA path
is a development / parity path, opt-in exactly like CUDA. IREE publishes no macOS
`iree-dist` (only linux dists + python wheels), so the runtime is **source-built**
like the CUDA path. `make iree-metal` (wrapping `scripts/iree/setup-macos.sh`)
automates it: it installs the pinned macOS `iree-compile` (metal-spirv codegen)
from the universal2 wheel into a
private venv, source-builds the IREE runtime (`local-task`/`local-sync`/`metal`
drivers), and prints the env.

```bash
# 1. One-time setup: source-build the IREE runtime (idempotent).
make iree-metal                              # or `make iree` (auto-detects the host)
eval "$(scripts/iree/setup-macos.sh --env)"  # export IREE_MACOS_HOME, MLXCEL_XLA_IREE_COMPILE
# (inspect the resolved paths + pinned version: `make iree-env`)

# 2. Build with real execution on (alongside the usual MLX features).
cargo build --release --features metal,accelerate,xla-iree

# 3. Opt into XLA at runtime; MLX stays default when MLXCEL_BACKEND is unset.
#    On Apple Silicon the XLA device defaults to `metal`, so just:
MLXCEL_BACKEND=xla ./target/release/mlxcel generate \
  -m <Llama-3.2-1B-Instruct dir> -p "The capital of France is" -n 48
#    Force the CPU path (token-exact vs the HF temp-0 reference) if you need it:
MLXCEL_BACKEND=xla MLXCEL_XLA_DEVICE=local-task ./target/release/mlxcel generate \
  -m <Llama-3.2-1B-Instruct dir> -p "The capital of France is" -n 48
```

On Apple Silicon `MLXCEL_XLA_DEVICE` defaults to `metal` (the GPU); set it to
`local-task` to force the CPU path. Validated on an M1 Ultra: `metal` is
token-exact with the `local-task` path (which is itself token-exact vs the HF
temp-0 reference) and ~1.9x its tok/s.

This is a correctness / parity path, **not** a performance path: on the same
M1 Ultra and model (Llama-3.2-1B-Instruct, greedy, 64 tokens), MLX runs at
~186 tok/s versus ~1.6 tok/s for XLA-on-Metal (~117x), because the bundled XLA
graphs are f32 and use generic metal-spirv codegen with a per-step host logits
readback, while MLX uses bf16 hand-tuned Metal kernels. MLX remains the
production Apple-Silicon backend; XLA-on-Metal exists to run the same StableHLO
graphs that target CUDA/Linux on a Mac for development and parity.

The root `build.rs` macOS arm uses Apple `ld` (`-force_load`, no
`--whole-archive` / `--start-group` / `-lgcc`) and links the
Metal/Foundation/QuartzCore frameworks; the C shim is unchanged (the `metal`
driver registers via `use_all_available_drivers`).

### Precision (low-precision matmul)

The contraction (matmul) input precision is authored in the emitted StableHLO
graph, so it applies on every IREE target (CPU, CUDA, Metal, and future NPUs),
not just one backend:

- `f16` / `bf16`: demote the f32 inputs of every `dot_general` to the narrow
  type while keeping the f32 accumulate and output, so only the matmuls change
  and the sensitive elementwise ops (norm, softmax, RoPE) stay f32. A blanket
  program-wide f32 to f16 is deliberately not done (it regressed accuracy and was
  slower).
- `f32`: no demotion.

**Default is per device:** `f16` on the GPU devices (`metal`, `cuda`), `f32` on
the CPU (`local-task` / `local-sync`). `MLXCEL_XLA_PRECISION` (`f16` | `bf16` |
`f32`) overrides the default on any device. So on Apple Silicon the Metal path
runs f16 by default:

```bash
# f16 by default on metal (the device default); no env needed:
MLXCEL_BACKEND=xla ./target/release/mlxcel generate \
  -m <Llama-3.2-1B-Instruct dir> -p "..." -n 48
# Force f32 for a token-exact reference:
MLXCEL_XLA_PRECISION=f32 MLXCEL_BACKEND=xla ./target/release/mlxcel generate \
  -m <Llama-3.2-1B-Instruct dir> -p "..." -n 48
```

On an M1 Ultra (Llama-3.2-1B-Instruct, greedy) `f16` is ~1.9x the `f32` tok/s and
token-exact with it. The same graph change also speeds up the CPU path. Weights
are uploaded f32 and demoted in the graph. Keeping the resident weights small is
the separate weight-quantization axis (`MLXCEL_XLA_QUANT=packed`, issue #516): an
MLX 4/8-bit checkpoint uploads the packed `ui32` weight + f16 scales/biases and
dequantizes in the graph. It is token-exact but off by default, because IREE does
not fuse the in-graph dequant into the matmul on CUDA, so it regresses throughput
(see "int8 packed dequant on GB10 / CUDA" below). This is a transferable,
correctness-first lever; it does not close the gap to MLX (see the perf note above).

### Prefill from embeddings

The emitter exposes a second, non-overloaded StableHLO module named
`prefill_embeddings.main`. It begins at the shared transformer stack and returns
the same logits/argmax and per-layer K/V tensors as `prefill.main`; the existing
token module remains byte-identical at the compatibility capacity. `Lp` is the
configured static context capacity for both modules, not a separate embeddings
limit. Runtime and C-ABI wiring is intentionally a separate integration step.

After the model weights, the embeddings entry has one deterministic argument
order: `embeddings`, `positions`, `real_len`, `attention_bias`.

- `embeddings` is `[Lp, hidden_size]` f32 and contains post-token-embedding-scale hidden states. The graph performs no token lookup and applies neither Gemma's `sqrt(hidden_size)` scale nor an architecture embedding multiplier again.
- `positions` is `[Lp]` i32 and retains the existing one-dimensional text position contract. M-RoPE uses a later, distinct schema rather than changing this argument's meaning.
- `real_len` is an i32 scalar in `1..=Lp` and selects the output row at `real_len - 1`.
- `attention_bias` is `[Lp, Lp]` f32. `0.0` means allowed and the finite value `-1e30` means masked; NaN, infinity, and intermediate additive values are rejected. The caller supplies the complete static bucket, including padding rows. Token-parity padding retains the ordinary causal pattern on padded rows while real rows mask future and padded keys.

`PrefillEmbeddingsInputMetadata` and the additive-bias validator provide the
pre-compilation/invocation checks for hidden size, sequence length, dtypes,
position/mask shapes, `real_len`, and mask values. The embedding table remains a
weight argument because tied checkpoints use it as the LM head; untied
checkpoints continue to consume their dedicated `lm_head` in the same weight
order.

The pure-Rust structural/golden checks run with `cargo test -p mlxcel-xla --lib`.
The real execution fixture gathers embeddings from a deterministic tied and
untied checkpoint, compiles both modules for IREE llvm-cpu, and compares logits
plus every K/V element across one-token, nonzero-padding, and near-capacity cases:

```bash
spike/openxla/.venv/bin/python spike/openxla/prefill_embeddings_check.py
```

### LLaVA independent reference gate

`scripts/xla/validate_llava_reference.sh` is the reproducible multimodal
correctness gate. It compares the production host preprocessor and IREE
prefill/decode bundle with Hugging Face Transformers, rather than treating an
MLX or IREE result as its own oracle. The reference is pinned to:

- source `llava-hf/llava-interleave-qwen-0.5b-hf` at
  `1090956dd1c79bc93ae98dcf395590369435ec91`;
- converted checkpoint `mlx-community/llava-interleave-qwen-0.5b-bf16` at
  `ba7385935f69c5417bfbe29c3809858a98afc22f`;
- the model and tokenizer SHA-256 values recorded in
  `spike/openxla/llava_reference_oracle.py`.

The source uses the Tongyi Qianwen Research License. Obtain and use it only
under those research/evaluation terms. Model files and generated binary
captures must remain outside Git. One way to obtain immutable local snapshots
is:

```bash
hf download llava-hf/llava-interleave-qwen-0.5b-hf \
  --revision 1090956dd1c79bc93ae98dcf395590369435ec91 \
  --local-dir /tmp/llava-interleave-qwen-0.5b-hf
hf download mlx-community/llava-interleave-qwen-0.5b-bf16 \
  --revision ba7385935f69c5417bfbe29c3809858a98afc22f \
  --local-dir /path/outside/repo/llava-interleave-qwen-0.5b-bf16
```

After configuring a real IREE CPU, CUDA, or Metal runtime, run:

```bash
scripts/xla/validate_llava_reference.sh \
  --source-model /tmp/llava-interleave-qwen-0.5b-hf \
  --model /path/outside/repo/llava-interleave-qwen-0.5b-bf16 \
  --image tests/fixtures/test_image.png \
  --out /tmp/mlxcel-llava-reference \
  --device cuda
```

The companion command sets `MLX_ENABLE_TF32=0`: the declared reference policy
is true F32, while MLX CUDA's default FAST_TF32 mode is an intentional,
lower-precision throughput policy and is not labeled as F32 evidence.

The gate verifies exact processor tokens, positions, and masks; every vision
hidden state plus first-block internal stages; selected/projected image
features; merged embeddings; first prefill logits; compact all-layer K/V
samples; and the greedy token trajectory. Cases cover one image plus text,
mandatory two-image ordering, and text-only input. Negative checks cover
malformed placeholder counts and effective context overflow before device
execution. Floating-point comparisons use the per-stage dtype policy recorded
in the reference manifest and report the first divergent stage.

The same command also checks non-streaming CLI output and the streaming OpenAI
chat-completions contract, including role/content/finish/usage/`[DONE]` order
and batch metrics. Manifests record compile/load, host preprocessing, prefill,
decode, host peak RSS, and runtime device-memory evidence. Add
`--text-model /path/to/text/checkpoint` to run `validate_arch.sh` as the
ordinary text and batch regression companion. Use `--skip-surfaces` only when
isolating a stage-level diagnostic failure.

On the qualified GB10 host, IREE `local-task` could not create its worker pool
under either 20-core or 4-core affinity (`thread creation failed with 22 (code
13)`). `local-sync` is the CPU execution fallback on such constrained hosts; it
ran the identical three-case reference comparison successfully. This is a mixed
runtime check: MLX CUDA owns host vision preprocessing and IREE `local-sync`
owns the text decoder. Use `--device local-sync` with the CUDA-enabled IREE
runtime, or `local-task` where worker creation is permitted.

The initial non-regression baseline was recorded on 2026-07-24 on an NVIDIA GB10
(driver 580.159.03, CUDA 13.0), IREE 3.12.0rc20260721 CUDA, context 1536,
four-token greedy decode, and `MLX_ENABLE_TF32=0`:

| Case | Host preprocess | IREE prefill | Decode |
|------|----------------:|-------------:|-------:|
| one image + text (743 effective tokens) | 1.351 s | 0.985 s | 8.37 tok/s |
| two images + text (1473 effective tokens) | 1.278 s | 0.670 s | 8.36 tok/s |
| text only (14 effective tokens) | 0.0004 s | 0.675 s | 8.27 tok/s |

Host component load was 0.220 s and IREE compile/load was 1.316 s. Peak process
RSS was 2,551,368 KiB and MLX reported 2,401,427,528 peak device bytes. Dedicated
IREE device bytes are unavailable on the GB10 unified-memory runtime, so the
manifest also records Linux available-memory before/after. The host loader's
allowlist retains processor, vision tower, projector, and the one text embedding
table; decoder layers and LM head are rejected by its structural test. Prefill,
selected-KV capture, and decode share one resident IREE text decoder bundle.

The `local-sync` CPU fallback on the same host passed all stages and exact
greedy tokens. It recorded 1.375 s compile/load, 1.035 s one-image host
preprocessing, 19.158 s prefill, 1.30 decode tok/s, and 6,374,948 KiB peak RSS.
The higher RSS and latency are expected from the static 1536-token CPU graph and
serve as the CPU non-regression baseline.

### Scope / limits

- The bundled graphs are authored for **Llama-3.2-1B-Instruct** specifically;
  `load` verifies `config.json` matches and errors otherwise.
- The context capacity is a static graph shape selected with `MLXCEL_XLA_CONTEXT_CAPACITY` (compatibility default `256`).
- Greedy sampling only; LLaVA prepared-prefill validation is limited to the
  pinned family above, and draft-model decoding remains out of scope.
- `XlaInferenceSession` is single-sequence; `XlaBatchEngine` (below) adds
  multi-sequence throughput.
- **Metal precision (issue #575):** `f16` is the transferable lever on Metal (the
  `metal` default, 2.3x over f32 and token-exact for 64 steps on an M1 Ultra).
  `bf16` cannot lower on `metal-spirv` (the Metal GPU target has no bf16 compute),
  so `MLXCEL_XLA_PRECISION=bf16` on a `metal` device is **rejected at load** with a
  message pointing at `f16` (issue #612), rather than dumping an opaque
  `iree-compile` legalization error mid-run. The packed int8 path
  (`MLXCEL_XLA_QUANT=packed`) compiles for Metal but the prefill invoke faults at
  runtime (a metal HAL `Metal command buffer failed` fault, surfaced as IREE
  `INTERNAL`, issue #613); it has only run on the CUDA runtime, and its bandwidth
  win is un-demonstrable on the compute-bound Metal decode regardless. So
  `MLXCEL_XLA_QUANT=packed` on a `metal` device is **rejected at load** with a
  message pointing at the CUDA / CPU targets, rather than faulting mid-run.

## Batched continuous batching (Stage 2b)

### Static context capacity

Set `MLXCEL_XLA_CONTEXT_CAPACITY` before loading a model to select the sequence dimension shared by the prefill graph, decode graph, RoPE tables, masks, and single/ragged KV caches. For example, `MLXCEL_XLA_CONTEXT_CAPACITY=1024` provides enough static space for a 729-token image expansion plus text and generation headroom. The selected value is part of the compiled-artifact identity; modules and runtime buffers with a different capacity are rejected instead of being paired.

Admission uses `effective_prompt_len + max_new_tokens <= context_capacity`. Text requests use their token count. A multimodal preprocessor must supply the length after placeholder expansion to the same validation helper before native execution. The error reports the effective prompt length, requested generation budget, and configured capacity, and batch rejection occurs before a request id or slot is consumed.

Larger capacities trade flexibility for resources: KV memory grows linearly with capacity, while the static prefill attention mask and score tensors grow quadratically and compilation can take longer. This backend emits a separate static StableHLO graph for each capacity; it does not claim or rely on dynamic StableHLO sequence shapes.

`XlaBatchEngine` runs many sequences at once: `B_max` slots share one rank-5 KV
cache and serve a request stream, so the device stays full. Requests of different
lengths join and leave the batch at different times; a freed slot is recycled by
a new request whose prompt KV is written **device-side** into just that slot
(no host round-trip), so admitting one sequence does not disturb the others. The
ragged decode graph advances every active slot one token per step, each at its
own position. Greedy, fixed `B_max ∈ {4, 8}` (the bundled ragged graphs),
contiguous per-slot KV.

The engine is backend-neutral at the request level (`submit` a prompt + budget +
[`SampleParams`](src/sampler.rs), `pump` a step, read per-request `EngineEvent`s,
`cancel`); it holds no server types, so the server adapter wraps it unchanged.

### Sampling (Stage 2d)

The engine reads the per-row logits back to the host (the logits graph variant)
and samples there, so it honors **temperature, top-k, top-p, min-p, and a seed**.
Greedy (`temperature 0`) is host argmax of the logits, token-exact with the
single-sequence argmax path. Reading the full `[B, vocab]` logits per step is a
small fraction of the decode matmuls; on-device sampling is a later optimization.

### Serving (Stage 2c/2d)

On an `xla-iree` build, `mlxcel-server` serves through this engine when
`MLXCEL_BACKEND=xla` is selected: a server-side `XlaServeWorker` adapts the engine
to the server's backend-neutral `BatchEngine` contract (the MLX scheduler
implements the same contract), so the HTTP path is unchanged.

```bash
MLXCEL_BACKEND=xla MLXCEL_XLA_DEVICE=cuda ./target/release/mlxcel-server \
  -m <Llama-3.2-1B-Instruct dir> --port 8080
# then POST /v1/completions (temperature / top_p / top_k / seed are honored).
```

The serve path accepts text and qualified LLaVA image requests. LLaVA image
decoding and the production MLX vision/projector path run in a bounded host
preprocessing stage, then the resulting prepared embeddings enter the same IREE
batch engine as text requests. The `xla-diagnostics` feature supplies the CUDA
host runtime used by the reproducible mixed-runtime gate above. Sampling
(temperature / top-k / top-p / min-p / seed), history-based penalties
(repetition / frequency / presence / DRY), stop strings, `max_tokens`, and the
model's EOS ids are honored. Requests the engine cannot serve faithfully are
rejected with a clear error: logprobs, structured / JSON-schema output, and
unsupported audio/video inputs. `--max-batch-size` maps to the engine's bundled
`B_max` (`>= 8` -> 8, else 4).

Prove the engine without the server with the reference-equivalence + throughput
example (every request's batched stream must equal its independent single-seq
reference):

```bash
# CPU (prebuilt dist):
IREE_DIST=/path/to/iree-dist cargo run --release --features xla-iree \
  --example xla_batch_bench -- --batch 4 --requests 8 --maxcap 24
# CUDA (GB10): source-built runtime + cuda iree-compile (as above), then:
MLXCEL_XLA_DEVICE=cuda cargo run --release --features xla-iree \
  --example xla_batch_bench -- --device cuda --batch 8 --requests 16 --maxcap 48
```

## Per-architecture validation harness (#496)

Validating an architecture has two tiers, split so the cheap one can gate every
change while the expensive one stays opt-in. Both are reusable, so adding a family
is turnkey.

**Structural (byte-exact, pure Rust, no GPU).** The emitter must reproduce a
frozen StableHLO golden for each registered architecture, byte for byte. This is
the fast regression gate that catches a graph which drifted from its validated
form. It is the `src/validation.rs` module, run as a normal test:

```bash
cargo test -p mlxcel-xla --lib validation
```

`validation::REGISTERED` lists the fixtures (currently `llama-3.2-1b`, whose
goldens live in `assets/llama-3.2-1b/`). The check honors `MLXCEL_XLA_PRECISION`:
the goldens are the default `f32` graphs, so a byte-exact run under `f16` / `bf16`
is rejected with a clear message instead of a confusing diff.

**Execution (token-exact + reference-exact).** One command produces the HF fp32
oracle and runs both run gates (needs a real IREE build and a checkpoint):

```bash
export IREE_DIST=/path/to/iree-dist          # or IREE_CUDA_HOME=... for CUDA
scripts/xla/validate_arch.sh --model <checkpoint dir> --device local-task
```

It (0) runs the structural pre-gate, (1) produces the oracle with
`spike/openxla/oracle_continuation.py` (loads the checkpoint in fp32,
dequantizing an MLX 4-bit / 8-bit checkpoint offline first with the same affine
formula as `src/weights.rs`), (2) runs `xla_oracle_check` (single-seq greedy ==
HF oracle), and (3) runs `xla_batch_bench` (every batched request == its
single-seq reference). It exits non-zero if any run gate fails.
`--structural-only` runs just the pure-Rust pre-gate (no IREE, no GPU).

On a host without the HF oracle venv (e.g. macOS / Metal), the `xla_traj_dump`
example dumps the engine's greedy trajectory at the ambient `MLXCEL_XLA_PRECISION`
as the same oracle JSON, so the token-exact gate can compare one precision against
another (issue #575 checks `f16` against an XLA-`f32` reference this way):

```bash
MLXCEL_XLA_PRECISION=f32 MLXCEL_XLA_DEVICE=metal cargo run --release \
  --features xla-iree --example xla_traj_dump -- \
  --model models/llama-3.2-1b-4bit --max-new 64 --out /tmp/f32.json
MLXCEL_XLA_PRECISION=f16 MLXCEL_XLA_DEVICE=metal cargo run --release \
  --features xla-iree --example xla_oracle_check -- \
  --model models/llama-3.2-1b-4bit --oracle /tmp/f32.json --device metal
```

### Adding a family is turnkey

1. Emit correctly: extend `Config::from_json` and the emitter for the
   architecture (the structural invariant tests in `emitter/mod.rs` cover the new
   switches: RoPE kind, q/k/v bias, tied / untied head, soft-caps).
2. Prove it: `scripts/xla/validate_arch.sh --model <checkpoint>` must report a
   clean token-exact + reference-exact pass on a real checkpoint.
3. Freeze goldens (optional, for a byte-exact CI guard): emit each graph with
   `validation::emit_graphs(config_json, kinds)` and write them to
   `assets/<arch>/*.mlir`; an already-registered family re-freezes in place with
   `MLXCEL_FREEZE_GOLDENS=1 cargo test -p mlxcel-xla --lib validation::tests::freeze_goldens`.
4. Register: add an `ArchFixture` to `validation::REGISTERED`; the byte-exact test
   then guards it forever.

Not every family bundles goldens: Qwen2.5 (`assets/qwen2.5-0.5b/`) is emitted at
load and covered by the emitter's structural tests plus the execution tier, with
no committed `.mlir`. The harness still drives its emit through `emit_graphs`.

## Performance and the low-precision decision

This backend is a portability / parity path, not a performance path. On Apple
Silicon, MLX is and remains the production backend; XLA runs the same StableHLO
graphs that target CUDA/Linux on a Mac, for development and cross-checking.

### Measured (M1 Ultra, Llama-3.2-1B-Instruct, greedy)

| | Metal | CPU (`local-task`) |
|---|---|---|
| one decode step, f32 | ~600 ms | ~233 ms |
| one decode step, f16 | ~291 ms (~2.1x) | ~187 ms (~1.25x) |
| end-to-end, f32 | ~1.5 tok/s | ~0.75 tok/s |
| end-to-end, f16 | ~3.0 tok/s | - |
| MLX (reference) | ~186 tok/s | - |

The decode-step figures are `iree-benchmark-module` (pure runtime, no host glue).

### Where the time goes

The Metal decode step is ~600 ms with the GPU busy the whole time (its host-side
`process_time` is ~50 ms), so the time is in the GPU kernels, not invoke overhead
or host round-trips. On the **same** StableHLO graph a 13-thread CPU (~233 ms)
beats the Metal GPU (~600 ms), which is the tell: the bottleneck is IREE's
`metal-spirv` kernel codegen (generic, unfused MSL via SPIRV-Cross), not
bandwidth. ~600 ms is ~110x the ~5 ms/token bandwidth floor MLX runs at.

### What is and is not worth optimizing

- **Graph-level (precision, quantization, op selection): in scope, transferable.**
  Authored once in the portable graph, it helps every IREE target (CPU, CUDA,
  Metal, future NPUs). f16 (landed) is ~1.9x and token-exact, and speeds up the
  CPU path too. For NPUs, low precision / quantization is not a 2x optimization
  but the entry ticket (they are int8 / fp16 native). This is where investment
  pays off.
- **Per-backend kernel codegen (the remaining ~50x to MLX): out of scope.** That
  is upstream IREE's job, is Metal-specific (does not transfer to non-SPIR-V
  NPUs), and MLX already owns Apple-Silicon performance.

So Metal's absolute tok/s is a *pessimistic* proxy for an NPU, which brings its
own optimized kernels; what transfers is the graph, not the Metal tuning.

### Decision (2026-07)

- **In scope:** graph-level low precision. f16 / bf16 is landed. int8 / int4
  weight quantization (the packed in-graph dequant, `MLXCEL_XLA_QUANT=packed`,
  issue #516) is landed too, **token-exact** but **opt-in and off by default**
  because on the one available int-capable target it does not yet pay off (see
  below).
- **Out of scope:** hand-writing Metal kernels or tuning IREE's `metal-spirv`
  codegen to chase MLX.

### int8 packed dequant on GB10 / CUDA: the fusion gate (2026-07-01)

The int8 lever was picked up on a GB10 (Grace-Blackwell, CUDA via the source-built
IREE runtime), the one int-native target available. `MLXCEL_XLA_QUANT=packed` keeps
the MLX 4/8-bit weights resident **packed** (`ui32` + f16 scales/biases) and
reconstructs each weight in the StableHLO graph (`Builder::dequant_affine`: bit
unpack -> `q*scale + bias`), instead of dequantizing to f32 at load.

Measured on the GB10 (Llama-3.2-1B, 4-bit `group_size` 64, greedy, warm vmfb):

| decode path | tok/s | GPU util | note |
|-------------|-------|----------|------|
| f16, dequant-at-load (default) | ~6.7 | 84% | resident f32 weights, f16 matmul inputs |
| packed, dequant-in-graph | ~1.6 | 96% | **token-exact** with the f16 path, ~4.3x slower |

The packed path is **correct** (bit-identical reconstruction, so the greedy token
stream matches the f32/f16 path) but **slower**, because IREE's CUDA codegen does
**not fuse** the unpack+dequant into the matmul: the decode step is ~678 dispatches
and the reconstructed f32 weight is materialized to DRAM every step, so the graph
pays *more* bandwidth + compute, not less (GPU util rises 84% -> 96%). Fusion flags
(`--iree-dispatch-creation-enable-aggressive-fusion`, `--iree-opt-generalize-matmul`,
`--iree-dispatch-creation-enable-early-trunc-fusion`) leave the dispatch count
unchanged (678 -> 677).

So the memory-bandwidth lever is **not** realized by authoring the dequant in the
portable graph alone; it needs the target to fuse dequant->matmul (a
quantized-matmul op / int8 `dot_general` lowering to the hardware's int8 path). That
is the same split the f16 note reaches, now confirmed for int8: the graph-level
change is necessary but the fused kernel is upstream IREE's job. The packed path
therefore lands **behind the `MLXCEL_XLA_QUANT=packed` gate, off by default**, as the
correctness-verified foundation for a fused quantized-matmul follow-up. It also
carries the whole packed ABI (emitter dequant, per-dtype weight upload, the
`ui32`/f16 device buffers) that a fused path reuses. The v1 packed path covers the
standard Llama layout ([`Config::supports_packed_quant`]: non-fused-qkv,
non-fused-gate-up, non-dense-MLP, non-MoE); embed / lm_head stay f32-resident.

### int8 fusion spike (#573): in-tree vs upstream on GB10

Following the fusion gate above, #573 measured whether any in-tree lever gets IREE's
CUDA codegen to fuse the packed dequant into the matmul (so the reconstructed weight
is not materialized to DRAM every decode step). The signal is the static decode
dispatch count (`iree-compile --compile-to=flow`, count `flow.dispatch`); tok/s is
end-to-end through the mlxcel binary. Reproduce with `scripts/xla/fusion_spike.sh`.

| decode path | flow.dispatch | tok/s | note |
|---|---|---|---|
| packed, in-graph dequant (#568) | 678 | ~1.5 | weight reconstructed + materialized every step |
| f16-resident (#572) | 454 | ~7.6 | no dequant; f16 weight stored directly |

Findings (GB10, IREE 3.11.0rc @ e4a3b04):

- **No iree-compile flag fuses it.** An 11-config flag sweep (aggressive-fusion,
  generalize-matmul, early-trunc-fusion, horizontal-contractions, fuse-multi-use,
  elementwise-multi-reduction, encoding-fusion, data-tiling x2, and all combined)
  leaves the count at 677-678. Data-tiling is a no-op on CUDA: the encoding layout
  resolvers exist only for the `hip` / `rocm` targets.
- **The dequant is a separate dispatch.** A one-matmul microtest (int4 unpack +
  scale then `dot_general`) is 2 dispatches (elementwise dequant producer + matmul)
  and stays 2 under every fusion flag: IREE does not fuse an elementwise producer
  into a matmul operand load on CUDA.
- **`linalg.quantized_matmul` / ukernels are unavailable for CUDA.** The two IREE
  facilities that would carry a fused quantized matmul (data-tiling encodings and
  ukernels) are gated to `hip` / `rocm` / `llvm-cpu`, not the `cuda` target.
- **int8 `dot_general` lowers, but without int8 tensor cores.** A bare `i8 x i8 ->
  i32` matmul compiles to a single CUDA kernel (like the f16 one), but the codegen
  upcasts `i8 -> i32` (17 `sext`, zero `dp4a` / `mma.*.s8`), so an int8 path would
  not beat f16 here.

**Split.** The graph-level dequant is necessary but the fused kernel is upstream
IREE's job (a CUDA quantized-matmul codegen path / encoding resolver / ukernel that
does not exist in 3.11.0). No in-tree representation or flag realizes it on CUDA. The
realized low-precision bandwidth win today is **f16-resident weights (#572)**; the
packed int8 path is a correctness-verified foundation but is dominated on CUDA until
upstream fusion lands. #574 therefore ships the minimal in-tree packed representation
(folding in the bf16 scales / biases limitation) and tracks the upstream fused
quantized-matmul, rather than forcing an int8-beats-f16 win in-tree.

## File map

| Path | Purpose |
|------|---------|
| `src/lib.rs` | `XlaInferenceSession`: the single-sequence `InferenceSession` impl + greedy drive loop. |
| `src/iree.rs` | (feature `iree`) FFI to the shim; `IreeLlama` (single-seq) and `IreeRaggedLlama` (batched) load weights, compile + run the graphs. |
| `src/batch.rs` | (feature `iree`) `XlaBatchEngine`: the continuous-batching engine (slots + queue + admit/decode/evict) and `XlaReferenceEngine` (single-seq reference for validation). The backend-neutral `Scheduler` bookkeeping is unit-tested without IREE. |
| `src/validation.rs` | (issue #496) Reusable per-architecture structural harness: the `ArchFixture` registry + `check_arch` byte-exact golden gate + `emit_graphs` freeze primitive. Pure Rust; runs under `cargo test`. The execution tier lives in `scripts/xla/validate_arch.sh` + `spike/openxla/oracle_continuation.py`. |
| `csrc/xla_iree.c` | C shim over the IREE runtime C API (one session, resident weights, threaded KV; single-seq `prefill`/`decode` plus the ragged `prefill_slot`/`decode_ragged` with a device-side per-slot KV write). |
| `build.rs` | (feature `iree`) compiles the shim against `IREE_DIST` headers. The runtime link recipe lives in the **root** `mlxcel/build.rs` (a dependency's link-args do not propagate to the binary). |
| `assets/llama-3.2-1b/` | The #451-emitted `prefill` / `decode_step` StableHLO graphs (on-device-argmax variant) plus the ragged `decode_ragged_b{4,8}` graphs, compiled to vmfbs at load. |
