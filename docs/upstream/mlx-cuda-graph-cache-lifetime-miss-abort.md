# MLX CUDA: LRUCache's thrashing check aborts long-lived processes on a lifetime miss count

Upstream bug report draft for ml-explore/mlx. Observed in mlxcel (lablup/mlxcel#818) and mitigated there by raising the CUDA graph-cache default (commit `bac0fca`); tracked for upstream in lablup/mlxcel#821. Every file and line reference below was read at MLX commit `b7c3dd6d27f45b5365b08a840310187dc503f1db`, which is mlxcel's current pin.

## Summary

`LRUCache::emplace` (`mlx/backend/cuda/lru_cache.h:91`) throws `std::runtime_error("Cache thrashing is happening, ...")` once `cache_misses_` passes `2 * capacity_`. `cache_misses_` (line 131) is a lifetime counter. It is incremented on every insert-miss and is never reset: not on a hit, not in `trim()`, not in `clear()`. It measures how many distinct keys the process has ever inserted, not the rate at which the cache is missing.

Four consequences:

1. Any process whose distinct-key working set grows past `2 * capacity` over its lifetime aborts, at any hit rate. A cache running at a 99% hit rate for hours still dies when its 801st distinct key arrives, at the CUDA graph cache's default capacity of 400.
2. The abort is a `throw` from inside a normal eval, and MLX does not catch it. A host application that does not wrap every eval in a handler gets `std::terminate`. For an inference server that is process death with every in-flight request dropped.
3. The advice in the message (raise the environment variable) only moves the threshold, because the threshold is derived from the capacity. No value makes an unbounded-lifetime process safe. Raising `MLX_CUDA_GRAPH_CACHE_SIZE` from 400 to 2000 buys 4000 lifetime misses instead of 800, at proportional memory cost, and postpones rather than removes the abort.
4. The state is unrecoverable even if the caller does catch. `++cache_misses_` is evaluated inside the throwing expression and the throw precedes the insert, so the offending key is never cached and every subsequent miss re-throws.

Upstream's own test suite works around this: `python/tests/mlx_tests.py:9` sets `MLX_ENABLE_CACHE_THRASHING_CHECK=0` with the comment "Do not abort on cache thrashing", added by `787c0d90` ("Detect cache thrashing in LRUCache", #2600), the same commit that introduced the check.

## Evidence

### The counter is lifetime-scoped

| Location | Fact |
|---|---|
| `lru_cache.h:91` | `if (env_name_ && ++cache_misses_ > 2 * capacity_) throw ...` |
| `lru_cache.h:131` | `size_t cache_misses_{0};`, the only write outside line 91 is the initializer |
| `lru_cache.h:86-89` | hit path in `emplace` splices and returns; `cache_misses_` untouched |
| `lru_cache.h:75-81` | `find` splices on hit; `cache_misses_` untouched |
| `lru_cache.h:122-128` | `trim()` evicts to `capacity_`; `cache_misses_` untouched |
| `lru_cache.h:70-73` | `clear()` empties `map_` and `vlist_`; `cache_misses_` untouched |
| `lru_cache.h:52-55` | `resize()` changes `capacity_`, which does move the threshold since it is recomputed per check, but nothing in the CUDA backend calls it |
| `lru_cache.h:35-40` | the check is armed only by the env-name constructor, and only while `MLX_ENABLE_CACHE_THRASHING_CHECK` (default 1) is truthy |
| `device.cpp:210` | `graph_cache_("MLX_CUDA_GRAPH_CACHE_SIZE", /* default_capacity */ 400)`, so the throw fires at 800 lifetime misses |

### The CUDA graph cache key is topology, not tensor shape

The graph cache is keyed by `graph_nodes_key_ + ":" + graph_deps_key_` (`device.cpp:490`), consulted through `operator[]` (`device.cpp:491`), so a miss goes straight into the `emplace` path above.

`graph_nodes_key_` is a concatenation of per-node type codes (`device.cpp:142-143`): `"K"` for a kernel node added directly (`device.cpp:349`, `device.cpp:357`), `"E"` for the empty node used to join concurrent work (`device.cpp:105`), and for a child graph the parenthesized string built by `subgraph_to_key` (`device.cpp:361-422`), which emits `K<clusterDim.x>` per kernel, `M` per memset, `H` per host node, `W`/`R` per event node, and recurses into nested child graphs. `graph_deps_key_` is a concatenation of `<from-ordinal>-<to-ordinal>` pairs (`device.cpp:172-175`) where the ordinals are `node_count_` at insertion time (`device.cpp:130`).

A key therefore identifies the structure of one committed graph: which kinds of nodes it holds, in what order, joined by which dependency edges. It encodes neither tensor shapes nor kernel identities. Distinct keys come from distinct op-stream structures per commit, and commits are cut by `needs_commit()` at `max_ops_per_graph_` nodes or `max_mb_per_graph_` megabytes (`device.cpp:458-461`). Those budgets are small on some devices: `get_graph_limits` (`device.cpp:181-203`) gives DGX Spark (compute capability 12.1) 20 ops and 25 MB, against 100 ops and 1000 MB on H100 and B200.

A server that interleaves requests does not accumulate one key per model shape, then stop. The cut points move as the op stream changes, so the number of distinct node-type sequences grows with the diversity of concurrent work rather than with the number of tensor shapes the model uses. That is the mechanism by which batched speculative decoding reaches 800 distinct keys quickly: draft and verify phases, varying batch composition, and varying sequence-length buckets all reshape the op stream between commits, and a 20-node commit budget slices it finely.

(mlxcel's original report, lablup/mlxcel#818, described the key as "one entry per distinct captured graph shape". That wording is imprecise; the key is topological as described above. The failure mode is unchanged.)

### Field observation

A `mlxcel-server` on GB10 (DGX Spark, sm_121, CUDA 13, Linux 6.17) running batched speculative decoding with a dflash drafter died with:

```
terminate called after throwing an instance of 'std::runtime_error'
  what():  Cache thrashing is happening, please set the environment variable MLX_CUDA_GRAPH_CACHE_SIZE to a larger value than 400 to fix degraded performance.
```

Whole-process abort, no request-level error, all in-flight requests dropped. The scenario was six concurrent chat completions with varying prompts, repeated in bursts. It reproduced on two different mlxcel builds of the same scenario, one dying during the first burst plus a follow-up request and the other on the second burst. With `MLX_CUDA_GRAPH_CACHE_SIZE=2000` the same scenario completed 13/13 requests across multiple bursts with speculative decoding engaged.

That 13/13 figure is the only quantitative datum available. The throughput and memory effect of the larger cache was not measured, and no attempt was made to find the smallest sufficient capacity, because any fixed capacity only postpones the abort.

## Root cause

File: `mlx/backend/cuda/lru_cache.h` at `b7c3dd6d`.

```cpp
  template <typename U>
  std::pair<iterator, bool> emplace(const K& key, U&& value) {
    auto it = map_.find(key);
    if (it != map_.end()) {
      vlist_.splice(vlist_.begin(), vlist_, it->second);
      return {it->second, false};              // hit: cache_misses_ untouched
    }

    if (env_name_ && ++cache_misses_ > 2 * capacity_) {
      throw std::runtime_error(
          fmt::format(
              "Cache thrashing is happening, please set the environment variable "
              "{} to a larger value than {} to fix degraded performance.",
              env_name_,
              capacity_));
    }

    vlist_.emplace_front(key, std::forward<U>(value));
    map_[key] = vlist_.begin();

    trim();

    return {vlist_.begin(), true};
  }
```

```cpp
  const char* env_name_{nullptr};
  size_t cache_misses_{0};
```

Thrashing means the working set does not fit and the cache keeps missing on keys it just evicted. That is a property of the recent miss rate. A lifetime count cannot tell it apart from a working set that legitimately grew, and the two have opposite correct responses: the first is a real performance problem worth reporting, the second is a cache doing its job.

The escape hatch is also all-or-nothing per process. The constructor at lines 35-40 latches `env_name_` only when `MLX_ENABLE_CACHE_THRASHING_CHECK` is truthy, so setting it to 0 disables the check for every `LRUCache` built with the env-name constructor at once: the CUDA graph cache, the cuDNN conv cache, both SDPA caches, and the cuFFT plan cache. That variable appears nowhere under `docs/`.

## Minimal fix

Recommended: measure the miss rate over a bounded window, and downgrade the response from a `throw` to a one-shot warning. Hits reach the cache through `find()` (`operator[]` and every in-tree caller do `find()` first and `emplace()` only on miss), so the hit counter belongs there rather than in `emplace` alone.

```diff
--- a/mlx/backend/cuda/lru_cache.h
+++ b/mlx/backend/cuda/lru_cache.h
@@
 #include <cstring>
+#include <iostream>
 #include <list>
 #include <unordered_map>
 #include <utility>
@@
   iterator find(const K& key) {
     auto it = map_.find(key);
     if (it == map_.end())
       return end();
+    ++window_hits_;
     vlist_.splice(vlist_.begin(), vlist_, it->second);
     return it->second;
   }
@@
     auto it = map_.find(key);
     if (it != map_.end()) {
       vlist_.splice(vlist_.begin(), vlist_, it->second);
+      ++window_hits_;
       return {it->second, false};
     }
 
-    if (env_name_ && ++cache_misses_ > 2 * capacity_) {
-      throw std::runtime_error(
-          fmt::format(
-              "Cache thrashing is happening, please set the environment variable "
-              "{} to a larger value than {} to fix degraded performance.",
-              env_name_,
-              capacity_));
-    }
+    ++window_misses_;
+    report_thrashing_once();
 
     vlist_.emplace_front(key, std::forward<U>(value));
     map_[key] = vlist_.begin();
@@
  private:
+  // Thrashing is a high miss *rate* against a working set that does not fit,
+  // not a large number of misses accumulated over a long process lifetime.
+  // Evaluate the rate over a window of lookups so a process whose working set
+  // simply grew is never reported, and warn instead of throwing: a cache that
+  // is too small is a performance problem, and the caller cannot recover from
+  // an exception here anyway.
+  void report_thrashing_once() {
+    if (!env_name_ || warned_) {
+      return;
+    }
+    size_t lookups = window_hits_ + window_misses_;
+    if (lookups < 4 * capacity_) {
+      return;
+    }
+    if (4 * window_misses_ > 3 * lookups) { // more than 75% misses
+      warned_ = true;
+      std::cerr << fmt::format(
+          "[mlx] Cache thrashing detected: {} of the last {} lookups missed "
+          "against a capacity of {}. Set the environment variable {} to a "
+          "larger value to recover the lost performance.\n",
+          window_misses_,
+          lookups,
+          capacity_,
+          env_name_);
+    }
+    window_hits_ = 0;
+    window_misses_ = 0;
+  }
+
   void trim() {
@@
   const char* env_name_{nullptr};
-  size_t cache_misses_{0};
+  size_t window_hits_{0};
+  size_t window_misses_{0};
+  bool warned_{false};
```

The two halves are independent and both are worth taking:

- The windowed rate fixes the diagnosis. A workload whose working set grows once and then hits steadily has a low miss rate in every window and is never reported. A workload whose working set does not fit misses on nearly every lookup and is reported inside the first window, earlier than the current check fires.
- The warning fixes the response. If only one half is acceptable, this is the half that removes the outage: nothing about a too-small cache justifies killing the host process, and there is no recovery path for a caller that catches, since the failing insert is skipped and the counter keeps climbing.

If the window bookkeeping is unwanted, a one-line alternative is to reset the counter on any hit, which turns `cache_misses_` into a count of consecutive misses:

```diff
   iterator find(const K& key) {
     auto it = map_.find(key);
     if (it == map_.end())
       return end();
+    cache_misses_ = 0;
     vlist_.splice(vlist_.begin(), vlist_, it->second);
     return it->second;
   }
@@
     auto it = map_.find(key);
     if (it != map_.end()) {
       vlist_.splice(vlist_.begin(), vlist_, it->second);
+      cache_misses_ = 0;
       return {it->second, false};
     }
```

That keeps the throw but changes its meaning to "`2 * capacity` lookups in a row without a single hit", which a healthy workload does not produce. It is weaker than the windowed version, since a thrashing subset interleaved with well-behaved traffic is never reported, but it errs toward not aborting.

Two smaller points worth folding in either way: document `MLX_ENABLE_CACHE_THRASHING_CHECK`, and consider whether the check should be off by default for the CUDA graph cache specifically. That cache's key space grows with workload diversity, while the conv, SDPA, and FFT caches are keyed by a comparatively fixed set of layer configurations.

## Minimal standalone repro

Observed repro (the one that produced the abort quoted above), on GB10 with MLX at the default `MLX_CUDA_GRAPH_CACHE_SIZE`:

1. Start a long-lived inference server that keeps one MLX process alive across requests, configured for batched speculative decoding (a draft model plus a target model, scheduler batch size 4).
2. Fire six concurrent completions with different prompts and short `max_tokens`, so the batch composition changes as requests finish at different steps.
3. Repeat the burst.

The process aborts within the first two bursts. The essential ingredients are process longevity and op-stream diversity, not any particular model: draft and verify phases, changing batch composition, and sequence-length bucketing keep producing committed graphs with node-type sequences that have not been seen before, and the lifetime miss counter never comes back down.

A pure-MLX reproducer follows the same shape. It is a sketch, not run on this system (no MLX Python build with CUDA was available here), and the observed repro above is the one this report is based on:

```python
import mlx.core as mx

# Run with a small capacity so the threshold is reached quickly:
#   MLX_CUDA_GRAPH_CACHE_SIZE=32 python repro.py   -> throws at 65 lifetime misses

x = mx.ones((256, 256))

def chain(n):
    y = x
    for _ in range(n):
        y = mx.sin(y)
    mx.eval(y)

for n in range(1, 200):
    chain(n)              # a graph structure not seen before
    for _ in range(20):
        chain(1)          # a hot structure that always hits
    print("distinct structures so far:", n, flush=True)
```

The point the loop makes is that the hit rate is above 95%, every eviction is of a structure that is never requested again (so no key is ever missed twice, which is what thrashing actually means), and the process still aborts. Vary the graph structure, not the tensor shapes: shapes do not enter the key.

## Affected surface

Every `LRUCache` constructed with the env-name constructor carries the fatal check. At `b7c3dd6d` that is:

| Cache | Env var | Default capacity | Lifetime misses before abort | Location |
|---|---|---|---|---|
| CUDA graph exec cache | `MLX_CUDA_GRAPH_CACHE_SIZE` | 400 | 800 | `device.cpp:210`, member at `device.h:156` |
| cuDNN SDPA forward | `MLX_CUDA_SDPA_CACHE_SIZE` | 256 | 512 | `scaled_dot_product_attention.cpp:178-182` |
| cuDNN conv | `MLX_CUDA_CONV_CACHE_SIZE` | 128 | 256 | `conv.cpp:41-47` |
| cuFFT plans | `MLX_CUDA_FFT_CACHE_SIZE` | 128 | 256 | `fft.cu:89-94` |
| cuDNN SDPA backward | `MLX_CUDA_SDPA_BACKWARD_CACHE_SIZE` | 64 | 128 | `scaled_dot_product_attention.cpp:184-188` |

The graph cache is the one that bites first in practice, because it is the only one whose key space grows with the shape of the host's op stream rather than with a fixed set of layer configurations, and because its per-key working set is the largest.

Scope notes:

- The graph cache is a `CommandEncoder` member, so there is one counter per stream. The conv and SDPA caches are `thread_local`, and the FFT cache is a function-local static.
- `LRUCache(size_t capacity)` leaves `env_name_` null, so caches built that way never throw.
- `LRUCache` lives only in the CUDA backend, so Metal and CPU are unaffected.
