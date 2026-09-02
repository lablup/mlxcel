# Technical Report: PR #1576 - fix(lora): refuse adapters that do not map onto the model

**Date**: 2026-09-02
**Author**: mlxcel maintainers
**Reviewer**: n/a
**Status**: Completed
**Languages**: Rust
**Risk Level**: Medium

---

## Executive Summary

LoRA adapter loading silently discarded every tensor it could not apply, so an adapter trained for a different architecture, a renamed projection, or a DoRA checkpoint produced a running server that answered from unmodified base weights while its startup log reported the adapter as fused. This PR validates every adapter tensor against the base weight map before the first write, reports every offender in one error, replaces the fabricated "fused into N layers" count with the number of pairs actually applied, and refuses DoRA by name. The fused path, the pipeline-parallel stage path, and the unfused runtime serving path now share one validator.

---

## 1. Problem Statement

### 1.1 Background

`mlxcel-server --adapter <dir>` and `mlxcel generate --adapter <dir>` fuse a LoRA adapter into the base weight map before model construction. The fusion loop grouped adapter tensors by their `.lora_a` / `.lora_b` stem, resolved each stem to a base weight key, and added `scale * (B @ A)` into that key. Every step of that resolution was best-effort.

### 1.2 Existing Issues

- **Missing base weight was a warning.** `find_base_weight_name` never failed: when none of its three candidates existed in the checkpoint it returned `"{name}.weight"` anyway, "the most likely candidate". The fusion loop then looked that key up, missed, logged `tracing::warn!("Base weight not found for LoRA layer ...")`, and continued. An adapter with 84 pairs against a model that has none of them fused nothing and still loaded.
- **Unrecognised tensors were discarded.** Anything that did not end in `.lora_a` or `.lora_b` was skipped by a comment that read `// Ignore other weights (like scales for DoRA)`. A DoRA magnitude vector, a stray `.weight`, and the HuggingFace PEFT `.lora_A.weight` spelling this build does not read all vanished at that line.
- **The reported count was fabricated.** `apply_lora_adapters` and `apply_stage_lora_adapter` both computed `modified_count` by counting `.lora_a` keys in the adapter file and logged it as `Fused LoRA adapters into N layers`. The number was the adapter's own size, entirely independent of how many pairs survived to reach `mlxcel_core::add`.
- **DoRA was accepted as LoRA.** `AdapterConfig::is_lora()` returned true for `FineTuneType::DoRA`, so a DoRA adapter passed the type gate and had its low-rank pair applied without the per-output-row magnitude vectors its own tooling folds in.
- **The unfused runtime path had the same three defects.** `stage_runtime_adapters`, the default channel for the b10621 `--lora` spellings since #1439, carried an identical `warn!`-and-continue block with a source comment stating that #1328 owned making both paths strict.

Measured on the pre-change `target/release/mlxcel` against local checkpoints: `models/mlx/qwen2.5-0.5b-bf16` with a copy of `models/lora-dense-test` whose `model.layers.0.self_attn.q_proj.lora_a` had been renamed to `...lora_a.bogus` loaded, fused 71 of 72 pairs, and generated normally. The same base with the same adapter declared `"fine_tune_type": "dora"` also loaded and generated.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| Serving base weights under an adapter's name, with no error anywhere | High | High for any adapter/checkpoint mismatch |
| Partially applied adapter (some layers fused, some skipped) | High | High for a layer-count mismatch |
| DoRA applied as LoRA, producing weights that match neither base nor fine-tune | Medium | Medium, gated on DoRA adapter availability |
| Operator trusting a fused-layer count that never described reality | Medium | Certain whenever any tensor was skipped |

---

## 2. Technical Review

### 2.1 Security

Not a security change. The failure class is a silent correctness fault, not an exposure. One adjacent property does improve: an adapter directory is now fully described by the error it produces, so an operator can no longer be misled into believing a fine-tune is in effect when it is not.

**Checklist:**
- [x] Input validation (adapter tensor names and base-weight resolution are now checked, not assumed)
- [ ] Authentication/Authorization (not applicable)
- [ ] Data encryption (not applicable)
- [x] Logging (the fused count is now a measured value; no sensitive data is logged)

### 2.2 Performance

No measurable effect. Validation walks the adapter's tensor names once, which is bounded by the adapter file size (hundreds of keys), and the work it does was already being done inline in the fusion loop. Nothing changes in the MLX call sequence for a valid adapter, so decode throughput and load time are unaffected.

### 2.3 Compatibility & Dependencies

- **Breaking Changes**: yes, deliberately. Adapters that used to load while applying nothing, or applying only part of themselves, now fail the load. This is the point of the change. Three public signatures changed: `fuse_lora_weights_into` returns `Result<usize>` instead of `Result<()>`, `apply_stage_lora_adapter` returns `Result<usize>` instead of `Result<()>`, and `AdapterConfig::is_lora()` is replaced by `AdapterConfig::is_fusable_lora()`. `fuse_lora_weights` and `apply_lora_adapters` keep their signatures.
- **New Dependencies**: none.
- **Compatibility**: mlx-lm adapters (`<layer>.lora_a` / `<layer>.lora_b`) and HuggingFace PEFT `<layer>.base_layer` naming both still resolve. The PEFT `lora_A.weight` / `lora_B.weight` spelling was never read by this code and now fails loudly instead of loading as a no-op.

### 2.4 Code Quality

- **Test Coverage**: the `lora` module gained 14 tests (11 in `loader_tests.rs`, 4 in the new `runtime_tests.rs`, 1 in `config.rs`, offset by the two stage-path tests added in `partial_loading_adapter_tests.rs`). The `--lib lora` selector goes from 47 to 61 passing tests.
- **Code Complexity**: `fuse_lora_weights_into` shrank. Its pairing, resolution, and skip handling moved into `validate_adapter_tensors`, leaving a loop that only computes and adds deltas.
- **Technical Debt**: decreased. The runtime path's `// #1328 owns making both strict` marker is discharged, and the two paths now share one definition of what a valid adapter tensor is instead of two drifting copies.

---

## 3. Technical Decisions

### 3.1 Collect every violation instead of failing on the first

**Context:**
An adapter built for the wrong architecture does not have one bad tensor; it has one bad tensor per layer per target module. A 28-layer adapter against a 24-layer model has 12 unmapped pairs, and a genuinely foreign adapter has all of them.

**Alternatives Considered:**

| Option | Pros | Cons |
|--------|------|------|
| Fail on the first offender | Simplest code, shortest error | Needs one load attempt per offending layer to learn the extent of the mismatch; each attempt reloads the whole checkpoint |
| Warn on all, fail if none applied | Keeps partial adapters working | Preserves the exact defect: a partially applied adapter is the dangerous case, not the fully unmapped one |
| **Chosen: collect all, fail once** | One load attempt gives the complete picture; the offender list distinguishes "wrong architecture" from "one renamed tensor" at a glance | Longer error text for a fully foreign adapter |

**Rationale:**
The diagnostic value is in the shape of the offender list. Twelve consecutive layer indices at the tail says "the adapter has more layers than the model"; one line says "somebody renamed a tensor"; every layer says "wrong architecture entirely". A first-failure error cannot convey any of that.

**Trade-offs:**
A foreign adapter produces a long error. Accepted: the alternative is a short error that requires repeated whole-checkpoint loads to expand.

### 3.2 Remove the `find_base_weight_name` fallback rather than check the result at the call site

**Context:**
`find_base_weight_name` returned `Result<String>` but could not fail. When no candidate matched it returned the first candidate anyway, and both callers followed it with a `contains_key` check and a warning.

**Alternatives Considered:**

| Option | Pros | Cons |
|--------|------|------|
| Keep the fallback, make the callers bail instead of warn | Smallest diff | The function still reports a key it knows does not exist, and the next caller repeats the mistake |
| **Chosen: return `Option<String>`, drop the fallback** | "Resolved" and "exists" become the same answer; the type makes the check unskippable | Touches every caller, including the runtime path |

**Rationale:**
The fallback is the root cause, not the symptom. A function that answers "here is the key, it may or may not be there" invites exactly the `warn!`-and-continue the issue is about, and it invited it twice independently. Making the absence a `None` moves the decision to the type system.

**Trade-offs:**
The runtime path had to change in the same PR because it shared the function. That turned out to be the right scope anyway (see 3.4).

### 3.3 Zero applied pairs is an error on the whole-model path only

**Context:**
An adapter that applies nothing is exactly the failure this PR exists to report. But a pipeline-parallel stage that owns layers 16 to 31 legitimately applies nothing when the adapter targets layers 0 to 15.

**Alternatives Considered:**

| Option | Pros | Cons |
|--------|------|------|
| Error on zero everywhere | Uniform rule | Breaks every valid partial-layer adapter under pipeline parallelism |
| Never error on zero | No false positives | An empty adapter file loads silently, which is the defect again |
| **Chosen: error on the whole-model path, return the count on the stage path** | Both cases are correct | The rule has to be documented where a future reader would otherwise "fix" the asymmetry |

**Rationale:**
The whole-model path sees the entire adapter and the entire checkpoint, so zero is unambiguous. A stage sees a filtered slice of both and cannot distinguish "this adapter is wrong" from "this adapter is not mine". The stage returns the count so its caller has the number, and `src/distributed/pipeline/stage_executor/llama.rs` carries a comment stating why the check is absent there.

**Trade-offs:**
A pipeline-parallel run where *no* stage applies anything is still not detected. Catching that needs cross-stage coordination and is left as future work (section 8).

### 3.4 Extend the same validator to the unfused runtime path

**Context:**
The issue's "Out of scope" section names runtime LoRA, but it was written before #1439 landed `stage_runtime_adapters`, and that function carries the comment `Unmatched tensors warn with the same posture as the fused path (#1328 owns making both strict)`.

**Alternatives Considered:**

| Option | Pros | Cons |
|--------|------|------|
| Fused path only, minimal edit to the runtime path for the `Option` signature | Smallest scope | Leaves the identical defect on the channel that `--lora` selects by default; ships a fix whose acceptance criterion "DoRA adapters are refused" is only half true |
| **Chosen: share `validate_adapter_tensors` and `reject_unsupported_fine_tune_type`** | One definition of a valid adapter; discharges the marker the tree already left | Larger diff than the issue body implies |

**Rationale:**
The runtime path had to be edited regardless because it called `find_base_weight_name`. Given that, leaving it lenient would ship a fix that does not hold on the default server channel, and would leave two copies of the pairing logic to drift.

**Trade-offs:**
`stage_runtime_adapters` gained a zero-staged check that `RuntimeLoraSet` route tests do not exercise; it is covered by the new `runtime_tests.rs` instead.

---

## 4. Implementation Details

### 4.1 Architecture Changes

```
[Before]
apply_lora_adapters ─┐
                     ├─> fuse_lora_weights_into ─> pair inline ─> warn+skip ─> add
apply_stage_lora_adapter ┘                                  ^
                                                            └── find_base_weight_name (never fails)
stage_runtime_adapters ──> pair inline ──> warn+skip ──> stage term

[After]
apply_lora_adapters ─┐
                     ├─> fuse_lora_weights_into ─┐
apply_stage_lora_adapter ┘                       ├─> validate_adapter_tensors ─> Vec<FusablePair> | one error
                                                 │        ^
stage_runtime_adapters ──────────────────────────┘        └── find_base_weight_name -> Option<String>
```

### 4.2 Key Code Changes

**File: `src/lora/loader.rs`**

```rust
// Before
let base_weight_name = find_base_weight_name(&base_name, base_weights)?;
let Some(base_weight) = base_weights.get(&base_weight_name) else {
    tracing::warn!("Base weight not found for LoRA layer {}: tried {}", base_name, base_weight_name);
    continue;
};

// After
let pairs = validate_adapter_tensors(base_weights, adapter_weights)?;
```

**Reason for change:** the resolution and the existence check are one decision, taken for the whole adapter before any write, so a failure names every offender and leaves `base_weights` untouched.

**File: `src/lora/loader.rs`**

```rust
// After
if !violations.is_empty() {
    violations.sort();
    let count = violations.len();
    let noun = if count == 1 { "tensor" } else { "tensors" };
    anyhow::bail!(
        "{count} adapter {noun} cannot be applied to this model:\n  {}\n\
         Every tensor in a fusable adapter has to be one half of a <layer>.lora_a / \
         <layer>.lora_b pair whose base weight this checkpoint holds; skipping the rest \
         would serve weights that match neither the base model nor the fine-tune.",
        violations.join("\n  "),
    );
}
```

**Reason for change:** `WeightMap` is a `HashMap`, so an unsorted report would reorder between runs and make the error untestable and hard to diff. The returned `Vec<FusablePair>` is sorted by base weight key for the same reason.

**File: `src/lora/config.rs`**

```rust
// Before
pub fn is_lora(&self) -> bool {
    matches!(self.fine_tune_type, FineTuneType::LoRA | FineTuneType::DoRA)
}

// After
pub fn is_fusable_lora(&self) -> bool {
    self.fine_tune_type == FineTuneType::LoRA
}
```

**Reason for change:** the old name asserted something the body did not deliver, and every one of its three call sites used it as a "can I apply this?" gate. Renaming it removes the trap rather than patching the callers.

### 4.3 Data Model Changes

None. No on-disk format, config schema, or wire format changes.

---

## 5. Learning Points

### 5.1 A resolver that cannot fail pushes the failure to every caller

**Concept:**
When a lookup helper returns a best guess instead of an absence, its return type stops carrying the information the caller needs. Every caller then has to re-derive "did this actually resolve?" and each one can get it wrong independently.

**Application in this PR:**
`find_base_weight_name` returned `Result<String>` and ended with `Ok(format!("{}.weight", lora_name))` when nothing matched. Both of its callers followed it with a `contains_key` check and a `tracing::warn!`, and both chose to continue. Changing the signature to `Option<String>` made the absence unrepresentable as a success and collapsed two divergent handlings into one.

**Common Use Cases:**
- Name or path resolvers that fall back to a "default" candidate
- Config lookups that substitute a default instead of reporting the key as unset
- Any `Result` whose body has no `Err` arm reachable from the failure it is supposed to describe

**Example Code:**
```rust
// The shape to avoid: the Ok value may or may not exist.
fn resolve(name: &str, map: &Map) -> Result<String> {
    for c in candidates(name) { if map.contains_key(&c) { return Ok(c); } }
    Ok(format!("{name}.weight")) // "most likely candidate"
}

// The shape that forces the caller to decide once.
fn resolve(name: &str, map: &Map) -> Option<String> {
    candidates(name).into_iter().find(|c| map.contains_key(c))
}
```

### 5.2 A count derived from the input is not a report about the output

**Concept:**
Logging "processed N items" where N came from measuring the input, not from counting completed work, produces a log line that is correct exactly when nothing went wrong and misleading exactly when something did.

**Application in this PR:**
`modified_count` counted `.lora_a` keys in the adapter file. It matched reality only when every pair fused, which is the case where nobody reads the log. On every skip it overstated, which is the case where somebody does.

**Common Use Cases:**
- Batch processors reporting the batch size instead of the success count
- Migration tools reporting the number of files found instead of the number applied
- Any "N items" log written before the loop that produces them

### 5.3 Adapter fusion and quantized checkpoints

**Concept:**
A 4-bit MLX checkpoint stores a projection's `.weight` as packed `uint32` with a separate `.scales` / `.biases` plane. A LoRA delta is a dense float matrix in `[out_features, in_features]`.

**Application in this PR:**
Establishing an honest smoke plan surfaced that the fused `--adapter` path cannot be exercised on a quantized checkpoint at all: `models/mlx/qwen3-0.6b-4bit` stores `q_proj.weight` as `[2048, 128]` `U32` while the delta is `[2048, 1024]`, so the pre-existing shape guard fires before this PR's validation is even reached. The unfused runtime path handles this correctly, because `validate_pair_shapes` pins `in_features` through the scales plane's group count rather than the packed width. This is not changed by this PR; it is recorded so the next person planning an adapter smoke test picks a bf16 checkpoint for the fused path and a quantized one for `--lora`.

---

## 6. Further Learning

### Key Terms

| Keyword | Description | Relevance |
|---------|-------------|-----------|
| `LoRA` | Low-rank adaptation: a frozen base weight plus `scale * B @ A` | The delta this code applies |
| `DoRA` | Weight-decomposed LoRA: the low-rank pair plus a per-output-row magnitude vector | Accepted as LoRA before this PR, refused after |
| `fusion` | Adding the delta into the base weight at load, before model construction | The path `--adapter` and `--lora-fuse` take |
| `runtime LoRA` | Keeping the pair unfused behind a live scale handle the layers read per forward | The default `--lora` channel since #1439 |
| `base_layer` | HuggingFace PEFT's name for the wrapped frozen projection | One of the three base-weight candidates |
| `LayerFilter` | A pipeline stage's owned layer range plus embedding/lm_head flags | Why zero applied pairs is valid on the stage path |

### Related Technologies/Frameworks

- **mlx-lm**: reference implementation for adapter naming and scaling
  - https://github.com/ml-explore/mlx-lm
- **HuggingFace PEFT**: the `base_layer` / `lora_A` / `lora_B` naming convention
  - https://github.com/huggingface/peft

### Related PRs/Issues

- Issue #1328: the defect this PR closes
- Issue #1439: added the unfused runtime path, which left the `#1328 owns making both strict` marker this PR discharges

---

## 7. Change Summary

### Statistics

| Item | Value |
|------|-------|
| Files changed | 11 |
| Lines added | +941 |
| Lines deleted | -180 |
| Tests added | 17 |

### Changes by Category

| Category | Count | Summary |
|----------|-------|---------|
| Correctness | 5 | Tensor validation, base-weight resolution, applied-pair count, DoRA refusal, runtime-path parity |
| Code Quality | 2 | Shared validator across three call paths, shared on-disk adapter fixture for tests |
| Documentation | 1 | `docs/server-features.md` records the acceptance rule and the stage exception |

### Related Commits

| Hash | Type | Message |
|------|------|---------|
| `fc6467a` | fix | fix(lora): refuse adapters whose tensors do not map onto the model |

---

## 8. Follow-up Actions

### Required

- [ ] Real-checkpoint smoke against `models/mlx/qwen2.5-0.5b-bf16` with `models/lora-dense-test` (72 applied pairs), `models/lora-runtime-test` (12 unmapped pairs reported), a renamed-tensor copy, and a DoRA-declared copy. The exact commands and expectations are in the PR body.

### Monitoring Required

- Startup failures naming `adapter tensors cannot be applied` on deployments that previously started successfully. Such a failure is the defect being reported, not a regression, but the adapter in question was serving base weights and the operator should be told.

### Future Improvements

- A pipeline-parallel run in which no stage applies any pair is still undetected. Detecting it needs a cross-stage tally after all stages report.
- DoRA fusion (`W' = m * (W + scale * B A) / ||W + scale * B A||` per output row) remains unimplemented; it needs a DoRA checkpoint to validate against.
- Fused adapter application on a quantized base is unsupported (dequantize, add, requantize). Today it fails on the shape guard. The unfused `--lora` path is the working answer for quantized checkpoints.
- The HuggingFace PEFT `lora_A.weight` / `lora_B.weight` spelling is now reported clearly rather than silently ignored, but is still not read.

---

## Appendix

### A. Test Results

| Command | Result |
|---------|--------|
| `cargo test --profile test-fast --features metal,accelerate --lib lora` | 61 passed, 0 failed |
| `cargo test --profile test-fast --features metal,accelerate --lib distributed::pipeline` | 323 passed, 0 failed |
| `cargo test --profile test-fast --features metal,accelerate --lib loading::` | 312 passed, 0 failed |
| `cargo clippy --profile test-fast --lib --tests --features metal,accelerate -- -D warnings` | clean |
| `cargo fmt --all -- --check` | clean |

### B. Performance Benchmarks

Not applicable. Validation touches load time only, bounded by the adapter's key count, and the MLX call sequence for a valid adapter is unchanged.

### C. References

- `src/lora/loader.rs`: `validate_adapter_tensors`, `find_base_weight_name`, `reject_unsupported_fine_tune_type`
- `src/lora/runtime.rs`: `stage_runtime_adapters`
- `docs/server-features.md`: "LoRA adapters"
