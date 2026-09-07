# ADR 0006: SafeTensors shard discovery falls back to a directory glob past a stale index

**Status:** Accepted (2026-09-07). Records the rationale for the tolerance already implemented in `collect_shard_paths`, backed by an upstream audit of the affected repositories.

## Context

A HuggingFace checkpoint that ships more than one weight file also ships `model.safetensors.index.json`, which maps every tensor name to the shard holding it. Reading that index is the documented way to know which files to open, and it is what a strict loader does.

A share of the mlx-community VLM conversions publish the *source* model's index verbatim beside the quantized weights. The weights are rewritten and re-sharded by quantization; the index is not regenerated. The result names shards that were never uploaded and declares a `total_size` that is the full-precision size.

The failure is total rather than partial. In every affected repository audited, **none** of the shards the index names exist, so a loader that trusts the index does not load a subset, it cannot open the checkpoint at all.

This is not a local packaging accident. Two mlxcel benchmark hosts, an M5 Max and an M1 Ultra, downloaded these checkpoints independently and produced byte-identical fingerprints for the same twelve of them (`benchmarks/fingerprints_m5max_2026-09-07.json`, `benchmarks/fingerprints_m1ultra_2026-09-07.json`), and the published repositories reproduce the same state.

## Decision

Keep shard discovery index-first with a glob fallback, and keep the warning that fires when the fallback is taken.

`collect_shard_paths` (`src/lib/mlxcel-core/src/weights.rs`) parses the index when one is present and uses `validate_index_shards` to resolve it. When that validation fails because named shards are not on disk, and the directory still holds usable `*.safetensors` files, it globs the directory, prints a warning naming the likely cause, and proceeds. The actionable missing-shard error survives for the case it was written for: a directory with nothing loadable in it.

The index stays load-bearing wherever it carries information a glob cannot. `unindexed_safetensors` uses it to find an auxiliary head such as an MTP `mtp.safetensors` that sits beside a target-only index, and the pipeline-parallel partial loader (`src/distributed/pipeline/partial_loading.rs`) uses it to open only the shards a rank needs. This ADR does not weaken either: it decides what happens when the index is provably wrong, not whether to read it.

## Measurement

Audited with `scripts/audit_hf_index.py`, which compares each repository's file tree against the shard names its index declares. Every repository backing a stale-index checkpoint on either benchmark host:

| Repository | Ships | Index declares | Declared shards absent |
|---|---|---|---|
| `mlx-community/gemma-3-4b-it-4bit` | 1 shard, 3.4 GB | 2 shards, 8.6 GB | 2 of 2 |
| `mlx-community/gemma-3n-E2B-it-4bit` | 1 shard, 4.5 GB | 3 shards, 10.9 GB | 3 of 3 |
| `mlx-community/gemma-3n-E4B-it-4bit` | 2 shards, 5.8 GB | 4 shards, 15.7 GB | 4 of 4 |
| `mlx-community/GLM-4.1V-9B-Thinking-4bit` | 2 shards, 7.1 GB | 4 shards, 20.6 GB | 4 of 4 |
| `mlx-community/GLM-4.5V-4bit` | 12 shards, 61.9 GB | 46 shards, 215.4 GB | 46 of 46 |
| `mlx-community/Kimi-VL-A3B-Thinking-4bit` | 2 shards, 9.8 GB | 7 shards, 32.8 GB | 7 of 7 |
| `mlx-community/Mistral-Small-3.1-24B-Instruct-2503-4bit` | 3 shards, 14.1 GB | 10 shards, 48.0 GB | 10 of 10 |
| `Aliyovic/molmo2-4b-mlx-8bit` | 2 shards, 7.8 GB | 4 shards, 19.4 GB | 4 of 4 |
| `mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit` | 4 shards, 18.3 GB | 13 shards, 62.1 GB | 13 of 13 |
| `mlx-community/Qwen3-VL-32B-Instruct-4bit` | 4 shards, 19.6 GB | 14 shards, 66.7 GB | 14 of 14 |
| `mlx-community/Qwen3-VL-4B-Instruct-4bit` | 1 shard, 3.1 GB | 2 shards, 8.9 GB | 2 of 2 |
| `mlx-community/Qwen3-VL-8B-Instruct-4bit` | 2 shards, 5.8 GB | 4 shards, 17.5 GB | 4 of 4 |

The declared figures are the source model's, to the byte. `mlx-community/gemma-3-4b-it-4bit` declares 2 shards and 8.60 GB against `google/gemma-3-4b-it` at 2 shards and 8.60 GB; `mlx-community/Qwen3-VL-32B-Instruct-4bit` declares 14 shards and 66.71 GB against `Qwen/Qwen3-VL-32B-Instruct` at 14 shards and 66.71 GB; `mlx-community/GLM-4.5V-4bit` declares 46 shards and 215.42 GB against `zai-org/GLM-4.5V` at 46 shards and 215.42 GB. The index is the pre-quantization one, carried through the conversion unchanged.

Scope, from the same audit:

- Text conversions are unaffected. `mlx-community/Qwen3-8B-4bit`, `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` and `mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit` are all consistent.
- Among the 60 most-downloaded `mlx-community` image-text-to-text repositories, 53 are consistent and 7 are stale. Four of those seven are not on either benchmark host: `gemma-3-12b-it-4bit`, `gemma-3-12b-it-qat-4bit`, `gemma-3-27b-it-qat-4bit`, `gemma-3-4b-it-qat-4bit`.
- The conversion version does not predict it. `mlx-community/GLM-4.5V-4bit` (mlx-vlm 0.3.2) is stale while `mlx-community/LFM2-VL-1.6B-4bit` (also 0.3.2) is consistent; 0.0.13 and 0.1.11 conversions are consistent while 0.1.18, 0.1.19, 0.1.23 and 0.3.4 conversions are stale. A version allowlist would be wrong in both directions.

Reproduce:

```bash
# Every locally held checkpoint whose shards came from the glob fallback,
# resolved to its source repository through each README.
python3 scripts/checkpoint_fingerprint.py benchmarks/metal_m5max_2026-09-06.csv \
  | python3 -c 'import json,sys; print("\n".join(k for k,v in json.load(sys.stdin).items() if v.get("shard_source")=="glob-stale-index"))' \
  | sed 's|^|models/|' | xargs python3 scripts/audit_hf_index.py --local

# The org slice, and a text control.
python3 scripts/audit_hf_index.py --search image-text-to-text --limit 60
python3 scripts/audit_hf_index.py mlx-community/Qwen3-8B-4bit
```

## Consequences

Twelve checkpoints in the benchmark set load and measure normally, including `qwen3-vl-32b-4bit` and `glm-4.5v-4bit`, which a strict loader would refuse outright. Removing the fallback would take them out of the supported set without any change on their side.

The warning is the cost. It fires on every load of an affected checkpoint and reads as a defect to anyone who has not seen this ADR. That is the intended trade: silence would make a genuinely corrupt directory indistinguishable from a repackaging artifact.

Detection of the state is available outside the loader. `scripts/checkpoint_fingerprint.py` records a `shard_source` field per checkpoint (`index`, `glob`, or `glob-stale-index`), which is how the twelve were enumerated across two hosts, and how a cross-host size comparison avoids reading a directory-level byte total as a model property.

The tolerance is not uniform across load paths. `identify_required_shards` in the pipeline-parallel partial loader builds its path list from the index without checking that those files exist, so an affected checkpoint is not usable under pipeline parallel. Nothing in the benchmark set exercises that combination today, which is why it is recorded here rather than fixed here.

## What reopens this

- A shard-selective load path that needs the index for a checkpoint whose index is stale. The glob returns the whole directory, so it cannot answer "which shard holds this tensor". The fix is to rebuild the mapping by reading each shard's SafeTensors header rather than to reinstate a hard failure.
- A directory holding two exports at once, where the glob loads both and a tensor name present in each is silently overwritten. Observed once, on a benchmark host that kept a superseded export beside a current one. A shape-mismatch check at merge time addresses it; failing the load on a stale index does not.
- Upstream regenerating these indexes. The audit script's exit status is 1 while any audited repository is still stale, so this is checkable rather than assumed.

## References

- `src/lib/mlxcel-core/src/weights.rs`, `collect_shard_paths` and `validate_index_shards`.
- `src/distributed/pipeline/partial_loading.rs`, `identify_required_shards`.
- `scripts/audit_hf_index.py`, the upstream audit.
- `scripts/checkpoint_fingerprint.py` and `benchmarks/fingerprints_m5max_2026-09-07.json`, `benchmarks/fingerprints_m1ultra_2026-09-07.json`, the two-host enumeration.
