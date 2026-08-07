# Code Modification Guidelines

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
