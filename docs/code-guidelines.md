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

## HashMap Iteration Order

When a `HashMap` or `HashSet` iteration result becomes ordered state, or feeds a consumer that is sensitive to order, the ordering has to be established explicitly at the point of iteration:

```rust
// Wrong. `entries` comes out of a HashMap, so its order is arbitrary, and
// `sort_by_key` is stable, so any two entries tied on `last_accessed` keep
// that arbitrary order.
let mut entries: Vec<_> = self.allocations.values().collect();
entries.sort_by_key(|a| a.last_accessed);

// Right. The id component makes the key a TOTAL order, so the stable sort is
// never asked to break a tie and hash order cannot reach the result.
let mut entries: Vec<_> = self.allocations.values().collect();
entries.sort_by_key(|a| (a.last_accessed, a.sequence_id));
```

**Why this matters:**

`RandomState` seeds each map *instance*, not the process. The common mental model, "HashMap iteration order is randomized per run", is wrong in the direction that matters: it implies a fixed binary on a fixed machine sees a fixed order, and it does not. Two maps built from identical input, in the same function, microseconds apart, iterate differently.

Measured on this repository's Linux box with `rustc -O` (1.97.1): build ten `HashMap<&str, u32>` instances from the same five keys inside one process, record each map's `.keys()` order, and count the distinct orders. Over 200 processes, 127 produced 10 distinct orders out of 10, 62 produced 9, and 11 produced 8. Not one process out of 200 had the ten maps agree. The exact split moves between measurement runs, because the experiment is itself random. What does not move is that the ten never agree.

The result is a probabilistic defect, so a green test run is not evidence. `cargo check` is clean, clippy is clean, and the tests pass most of the time.

**Order-sensitive consumers.** The iteration is only a problem once something downstream cares about position. These all care:

- Positional indexing, including round-robin selection by index.
- `min_by_key` and `max_by_key`, which tie-break on input order in opposite directions. `min_by_key` and `min_by` return the **first** minimum; `max_by_key` and `max_by` return the **last** maximum. A fix therefore has to face the tie component the right way for the call it is fixing, and two arms of one policy can need opposite directions.
- A prefix: `take(n)`, or an early `break`, or an early `return` out of the loop.
- A loop-carried accumulator, such as a seed advanced once per element.
- Anything documented as a priority order, whether or not the current consumer reads it that way.

The consumer is frequently in another module from the iteration, which is why this survives review: the accessor that returns the unordered list looks unremarkable on its own.

### A sort is not automatically a fix

This is the least obvious part of the rule and the part that costs the most to rediscover. `slice::sort_by_key` and `slice::sort_by` are **stable**: elements that compare equal keep their input order. When the input came out of a `HashMap`, that retained order is hash order. So a sort does not make a `HashMap`-derived list deterministic unless the sort key is a **total order** over the elements, meaning no two distinct elements can compare equal.

Three of the eight instances below sorted their candidate list and were still nondeterministic for exactly this reason. A reader who saw the sort reasonably concluded the result was pinned. It was not.

`sort_unstable_by` is not the remedy either. It does not preserve input order, but it does not define one; it substitutes a different arbitrary tie order, and the outcome is still not reproducible.

The remedy is to make the key total by appending something unique to each element. All three forms below are correct and the choice between them is stylistic:

```rust
// Tuple key. Cheapest when the unique component is Copy.
entries.sort_by_key(|a| (a.last_accessed, a.sequence_id));

// Comparator with `then_with`. Preferred when the unique component is a
// String, since `sort_by_key` would force a clone into the tuple.
completed.sort_by(|(key_a, t_a), (key_b, t_b)| t_a.cmp(t_b).then_with(|| key_a.cmp(key_b)));

// Sort on the unique component alone, when the policy key is not needed.
nodes.sort_by(|a, b| a.config.id.cmp(&b.config.id));
```

Add a comment at the call site naming the unique component as the thing that makes the key total. Without it the component reads as redundant and the next reader simplifies it away.

### Testing the fix

Every fix in this family needed a test shaped a particular way, and a test that misses any of these three points passes against the unfixed code:

1. **Build a fresh map inside each iteration, and loop at least 32 times.** Because `RandomState` seeds per instance, probing one map repeatedly measures nothing; the map has to be rebuilt. The in-tree loop counts are `ORDER_RESOLVE_ITERATIONS = 32` (`src/lang_bias.rs:755`), `ITERATIONS = 64` (`src/vision/detection/rt_detr_v2/sanitize.rs:388`), `ORDER_ITERATIONS = 64` (`src/distributed/registry_tests.rs:257`) and `SELECTION_ITERATIONS = 32` (`src/distributed/disaggregated/request_router_tests.rs:865`).
2. **Construct an actual tie.** Distinct keys already give a total order, so a test built from distinct keys passes without the fix and proves nothing. The tie is the thing under test.
3. **Write equal timestamps in deliberately.** `Instant::now()` resolves to nanoseconds on the platforms this project targets, so two entries stamped by two separate calls effectively never collide. A test that stamps entries with `Instant::now()` and expects them to tie will never construct the case it is trying to cover.

The reason for the loop is that the pre-fix failure rate is well under 100 percent. PR #1281's new test failed on **27 of 64** freshly built maps against the unfixed function. A single-shot version of that test would have passed on the majority of runs and shipped nothing. PR #1269's three ordering tests first failed at iteration 0, iteration 0 and iteration 2.

Validate the test by running it against the unfixed code first. If it does not fail there, it is not testing the defect.

### Choosing BTreeMap instead

`BTreeMap` and `BTreeSet` have a defined iteration order, so switching the container is the structural fix and removes the whole class at that site. It costs O(1) `get` for O(log n), so it is the right move where lookups are not hot and the wrong one where they are. #1277 is the worked example of the other direction: the registry's node map is read on the primary request path, so it kept its `HashMap` and the four accessors sort on the node id instead.

**The eight instances.** These were found in one review sweep between 2026-08-20 and 2026-08-22, across six modules, with no part of the toolchain flagging any of them:

- **#1265** (PR #1266). Four `filled_weights` test fixtures walked a `WeightMap` and advanced one LCG seed per key, so every process built a different synthetic model. A loop-carried accumulator, with no sort and no index anywhere in sight. Before the cause was found it had already put two wrong conclusions into a tracked file, both since withdrawn in place at `Makefile:604-616`.
- **#1267** (PR #1269). `LangBiasYamlConfig::bias` was `Option<HashMap<String, BiasValueStr>>` feeding `LangBiasSet.ordered`, which is documented as the priority order and consumed first-language-wins. A multi-CJK YAML config therefore assigned different biases to shared Han tokens on every run, and the schema example in the doc comment is itself a three-CJK config, so copying it was enough to trigger this.
- **#1276** (PR #1281). `needs_sanitize` returned from inside a `weights.keys()` walk on the first marker it saw, so a checkpoint carrying both marker families got a coin-flip layout verdict. The wrong direction is the expensive one: re-running the sanitize pipeline over already-MLX weights double-transposes conv weights into a shape-valid tensor that nothing downstream can flag.
- **#1277** (PR #1284). Four registry accessors (`all_nodes`, `nodes_with_role`, `peer_addresses`, `topology_summary`) returned `HashMap` values unordered into consumers that index positionally or tie-break on input order, reaching node selection on the primary request path.
- **#1286** (PR #1288). Three eviction paths, in `src/distributed/tensor_parallel/cache_manager.rs`, `src/distributed/pipeline/cache_manager.rs` and `src/distributed/request_tracker.rs`. All three sorted their candidate list and were still nondeterministic, which is the case the stable-sort subsection above exists for.
- **#1293** (PR #1301). `BatchScheduler::select_eviction_victim` chose the preemption victim with `max_by_key` on `generated_tokens.len()` under `LongestFirst` and `min_by` on priority-then-length under `LowestPriority`, both over `ActiveBatch::iter_sequences`, which is `HashMap::values`. The only one of the eight whose tie is reachable through ordinary state rather than through a coarsening of the key: sequences admitted together decode in lockstep and therefore share a token count, and both `PreemptionPolicy::LongestFirst` and `RequestPriority::Normal` are the shipped defaults. Nothing was corrupted, since any tied-longest sequence does satisfy the policy, but two identically configured workers under identical load preempted different requests and an operator could not reproduce the choice from the batch state. Fixed by appending `seq_id`, facing opposite ways in the two arms because `max_by_key` takes the last maximum and `min_by` the first minimum.

**Enforcement:**

There is no `make verify-*` target, no CI job and no script for this rule. That is a measured decision rather than an omission, taken in #1287, and the numbers behind it are recorded here so the next person does not re-derive them. Two candidate checks in the shape of [`scripts/ci/check_kernel_dtype_keys.py`](../scripts/ci/check_kernel_dtype_keys.py) were prototyped and run over all 1,248 `.rs` files under `src/`, against the current tree and against the reconstructed pre-fix tree of each of the first five fixes above (#1293 was found after the measurement ran):

- **`min_by_key` / `max_by_key` on an unordered-map receiver.** Four flags on the current tree and zero false positives: the LRU eviction sites at `src/server/prompt_cache/store.rs:315` and `:336`, `src/server/responses_store.rs:243`, and `src/server/conversation_store.rs:151`. It catches **0 of the 8** instances above. Every reconstructed pre-fix tree produced those same four flags and nothing else, because none of the seven then known used `min_by_key` or `max_by_key`. #1293 does use `max_by_key` and was missed anyway, for the reason recorded at the end of this section. The four are also not live defects: all four keys are `std::time::Instant`, and each writer computes `Instant::now()` once per call and stamps a single entry with it, so at nanosecond resolution the tie is not reachable today. They are true positives for the lint and zero live bugs. The distinction matters: saying the check found four bugs would be false, and the rule would lose its credibility the first time somebody checked. Gating on this would mean suppressing four of four findings on the day it landed, and a suppression does not re-arm on the change that would make the pattern real, which is the key becoming coarser. The pattern is genuinely fragile and worth knowing about; it is not worth a gate.
- **A `Vec` built from an unordered-map view and consumed order-sensitively.** Twenty-two flags on the current tree. It flags all three #1286 sites *after* PR #1288 fixed them, because the fix keeps the shape and changes only whether the sort key is total, a distinction no regex can draw. Its 3-of-8 catch rate held only against pre-fix trees; on any tree where the class has been fixed, those three become permanent false positives. It reaches none of the four separately filed instances, each for a different structural reason: #1265 is a loop-carried seed with no `Vec`, no sort and no index; #1267 iterates the map directly in a `for` and never calls `.keys()` or `.values()` at all; #1276 returns early out of a `for` with no `Vec` binding to bind a consumer to; and #1277's producer and its order-sensitive consumers are in different modules. A tree-wide scan for `type X = HashMap<...>` does remove the type-alias blind spot that `WeightMap` creates, so that one limit is fixable; the other three are not, at this level of analysis.

The general form is undecidable from the source alone, because whether an iteration order matters depends on a consumer that is often in another module. So this rule is enforced by review and by the testing rule above. Re-open the question if a future instance takes a shape either candidate would have caught.

That blind spot has already cost a finding, recorded here so the decision above stays honest rather than only favourable. #1293 is an eighth instance, in `BatchScheduler::select_eviction_victim`, and unlike the four latent ones its tie is reachable: it breaks on `generated_tokens.len()`, a small integer that sequences decoding in lockstep share routinely. Neither candidate check flagged it, because the receiver is a cross-module accessor method (`ActiveBatch::iter_sequences`) rather than a literal `.values()` on a map declared in the same file, which is exactly the limitation described above for #1277. A human sweep found it. That is the cost of the decline, and it is the number to weigh if the question is re-opened.
