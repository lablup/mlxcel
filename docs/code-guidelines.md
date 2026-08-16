# Code Modification Guidelines

## File Size and Module Structure

Guidance for when a file should stay as one unit, when to extract a helpers file, and when to split into a directory module. This section is the authoritative source for the numeric thresholds. Anything else that cites them, including agent-facing tooling under `.claude/skills/`, should link here rather than restate them, so the numbers cannot drift apart.

A feature that is self-contained and small is fine as a single file, for example `src/models/dbrx.rs` (a complete model in about 600 lines) or `src/distributed/heartbeat.rs` (a complete feature in about 310 lines). Beyond that, apply the following thresholds.

| Size | Action |
|------|--------|
| Under 800 lines | Fine as a single file |
| 800 to 1,200 lines | Acceptable for complex models; consider extracting a helpers file |
| 1,200+ lines | Extract helpers into `<name>_helpers.rs` |
| 1,500+ lines | Strongly consider splitting into a directory module |
| 2,000+ lines with no helpers file | Anti-pattern; extract `<name>_helpers.rs` |

These are guidelines, not gates. The reasons a file is allowed past them matter more than the numbers:

- `src/models/nemotron_h.rs` (about 2,900 lines, no helpers file) is a documented exception. It is a hybrid Mamba plus Transformer model, and splitting it would break the layer-interleaving logic that makes it one thing.
- `src/models/llama4.rs` (about 1,860 lines) shows the 1,200+ row applied: MoE, iGQA and ChunkedKV are tightly coupled, so the model itself stayed one file, and the mask construction and weight loading that were separable moved out to `src/models/llama4_helpers.rs`. Under `src/models/` the only other such extractions are `gemma3n_helpers.rs` and `qwen3_next_helpers.rs`.

Several files sit well past these numbers with no recorded justification: `src/models/gemma4.rs` (about 6,700 lines) and `src/models/qwen3_5.rs` (about 3,500 lines) have no helpers file, and `src/models/gemma3n.rs` is about 5,600 lines. Read those as debt rather than as precedent. The largest file in the tree is not evidence of what the project permits, and it is not a number you can cite in a review.

Line counts drift as models are edited, so treat every figure above as approximate and check the file rather than trusting a number quoted anywhere, this section included.

### Inline tests versus a sibling `_tests.rs` file

Keep tests inline in `#[cfg(test)] mod tests { ... }` while they stay short and tightly coupled to the module, roughly under 100 lines. Once a module's tests grow past that, and by 200 lines at the latest, move them to a sibling file (`config.rs` gets `config_tests.rs`), wired in with `#[cfg(test)] mod config_tests;` or a `#[path = "config_tests.rs"] mod config_tests;` attribute.

### Naming

| Pattern | Example | When |
|---------|---------|------|
| `<name>_helpers.rs` | `gemma3n_helpers.rs` | Extracted helpers for a large model |
| `<name>_tests.rs` | `config_tests.rs` | Sibling test file, once inline tests outgrow the module |

## Shared Function Comments

When modifying shared/common functions that multiple models depend on (e.g., attention implementations, normalization, KV cache, activation functions), **always add or update comments** indicating which models use that function.

**Format:**
```rust
// Used by: Llama, Qwen, Gemma2, Gemma3
fn repeat_kv(keys: &MlxArray, values: &MlxArray, n_rep: i32) -> ... {
    ...
}
```

**Why this matters:**
- Prevents regression when fixing one model from breaking others
- Makes it clear which models need retesting after changes
- Helps future developers understand the impact of modifications

**When to update:**
1. When adding a new model that uses an existing shared function
2. When modifying a shared function's behavior
3. When discovering that a function is used by a model not listed

**When the caller list is too long to enumerate:**

Past roughly a dozen callers, a hand-written roster is wrong by the next release and nobody repairs it. Do not enumerate in that case. Write a rule that says *why* a caller is on the list, name a few representatives per group, name the families that are deliberately absent, and close with the `grep` one-liner that regenerates the exact set:

```rust
/// Used by: decoders that materialize an explicit prefill mask instead of
/// leaving `mask: None` for fused SDPA to apply causality itself. At this
/// commit that is 44 non-test files under `src/models`, in four groups.
/// (groups and representatives ...)
///
/// Not used by the mainstream dense decoders (Llama3, Mixtral, Gemma2 and
/// similar): they pass `mask: None` for `seq_len > 1`, so they are unaffected
/// by changes here.
///
/// The caller set is too large to enumerate by name without going stale, so
/// the groups above are a summary. Regenerate the exact list with
/// `grep -rln '\bcreate_causal_mask(' src --include='*.rs'`.
pub fn create_causal_mask(size: i32, offset: i32) -> UniquePtr<MlxArray> {
```

`create_causal_mask` in `utils.rs` is the worked example. The "not used by" half carries as much weight as the list: it is what stops a contributor from assuming a shared helper still covers a family that moved off it several releases ago. Derive every list by running the grep at the current commit, never from memory, and prefer `///` doc comments over `//` on public items so the annotation survives into rustdoc.

**Key shared components to track:**
- `src/lib/mlxcel-core/src/layers.rs` - KVCache, Attention, Normalization
- `src/lib/mlxcel-core/src/utils.rs` - create_causal_mask, softcap, repeat_kv
- Model-specific attention variants in `src/models/*.rs`

## JIT Kernel Cache Keys

Every `template_args` list passed to a `cuda_kernel` launch must name the dtype of each input whose dtype can vary:

```cpp
std::vector<std::pair<std::string, TemplateArg>> template_args = {
    {"Dim", dim},
    {"KVType", k_pool.dtype()},   // keys the cache, not read by the kernel body
};
```

**Why this matters:**

MLX generates a custom kernel's buffer parameter types from the runtime dtypes of its inputs, but only Metal folds those dtypes into the kernel name. CUDA names a kernel `"custom_kernel_" + name + template_arguments_hash(template_args)`, and `cu::get_jit_module` memoises the compiled module under exactly that name in a process-global map, invoking the source builder only on a cache miss.

With int-only template args, the first dtype to compile is then served to every later dtype at the same geometry. That call reads its buffers through the wrong pointer type and returns numbers unrelated to its inputs. Nothing throws, and macOS never sees it. Issues #1053 and #1054 are two symptoms; the sampler was affected in production, because `gumbel_max_sample_accepts` admits f32, f16 and bf16 at one `NumSplits`.

The entry may stay unreferenced by the kernel body. Its job is the cache key, so do not remove it as dead code.

**Enforcement:**

`make verify-kernel-dtype-keys` (part of `make verify`, and the `kernel dtype keys` CI job) runs [`scripts/ci/check_kernel_dtype_keys.py`](../scripts/ci/check_kernel_dtype_keys.py). The rule is scoped by the presence of `cuda_kernel(` in the file rather than by a hand-maintained list, so a Metal-only launcher is out of scope until someone adds a CUDA port to it, at which point the check starts applying on its own. Read the script rather than trusting this section.
