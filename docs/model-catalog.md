# Model Catalog

`docs/model-catalog.tsv` records what each directory under the benchmark store actually holds, on which machine, and on what evidence. It exists because a benchmark row is read by its name, so a name that misstates the model misstates the measurement, and because the same checkpoint has carried different names on different machines and at different dates.

The store is `models/` and nothing else, but `models/` is not the same shape on every machine. M5 Max holds its checkpoints directly under it; M1 Ultra makes it a symlink and keeps two roots below, `models/mlx/` and `models/mlx-big/`, the second for checkpoints above 120 GB. Two consequences follow and both have already caused a wrong document. Scan every root: reading `models/mlx/` alone reported nine checkpoints as absent from M1 Ultra that were present, eight of them with measured rows already committed under `benchmarks/`. And match on basename rather than on a path literal: a check written as `ls -d models/<name>` passes on one host and fails on the other for the same checkpoint, so it tests the layout instead of the store. `~/.cache/mlxcel/models` is out of scope: its contents differ per machine, so a table assembled from it cannot be read beside another host's. A checkpoint reachable only through a symlink into that cache is not in the store; move the real directory in. Note that `mlxcel list` reports that cache and not the store, so it can print "No models downloaded" while the store holds two hundred checkpoints, or print a short list that looks complete and is not.

## Naming rule

A directory name should say what the checkpoint is: family, size, tuning, precision. Four steps decide it, in order.

1. **Read the README heading.** An mlx-community conversion carries a `# mlx-community/<repo>` line naming the artifact itself, quantization included. This is the authority when present.
2. **Fall back to `base_model:` only as a flag, never as an answer.** That frontmatter field names the checkpoint the artifact was *converted from*, so following it renames a 4-bit conversion after its bf16 original. A checkpoint published by the original author rather than converted has no heading, and the fallback firing is itself the signal that the resulting name may describe something else.
3. **Keep a precision token the upstream name lacks.** `mlx-community/Llama-3.1-8B-Instruct` carries no precision, but the local directory did, and it sits beside a 4-bit sibling. Dropping the suffix makes the pair read as base against quantized when it is bf16 against quantized. Restore a token that was lost; do not invent one that was never there.
4. **Check the name against the config, and believe the config.** `hunyuan-1.8b-4bit`'s README points at `Hunyuan-4B-Instruct-1.8bit`, but the checkpoint measures 1.85B parameters at `bits: 4`. The directory name is right and the README is a copy error, so the name stays.

With no heading and no `base_model`, leave the name alone. A name assembled from config reads as verified when it is a guess, and `id_source: none` says so plainly instead.

## Columns

| Column | Meaning |
|---|---|
| `local_name` | Directory name under `models/`. |
| `aliases` | Names this checkpoint was measured under before a rename, `;`-separated. A past-dated CSV row joins to the current one through this. An alias records that one set of weights had another name, not that a directory was once misnamed after a different model: `jamba-v0.1-4bit` held AI21-Jamba-Reasoning-3B and no Jamba v0.1 was ever in the store, so renaming that directory was right and listing the old name as an alias was not. |
| `upstream_repo_id` | The HuggingFace repo, empty when neither source yielded one. |
| `id_source` | `heading`, `base_model` or `none`. Reliability is readable from this column alone. |
| `precision` | From `config.json`, or from the shard headers when `torch_dtype` is absent. |
| `size_gb` | On-disk footprint, which is what a transfer between machines has to be planned against. |
| `present_on_m1ultra`, `present_on_m5max` | Each host owns its own column. |
| `verified_by` | `config` when an upstream id was cross-checked against the config, `none` otherwise. |
| `store` | `repo` for the standard store. |
| `shards_ok_m1ultra`, `shards_ok_m5max` | Per-host shard verdict, `ok(N)` or a fault. |
| `csv_rows`, `doc_mentions` | How many benchmark rows and doc references the name and its aliases carry. This prices a rename before it is made; the cost is not uniform, and the largest entries span over a hundred rows. |
| `m5_absence` | Why a checkpoint present on one host is absent from the other, when the absence is a decision rather than a fault. |

## Shard integrity

`shards_ok_*` compares each shard's declared length, `8 + header_len + max(data_offsets[1])`, against the file size. It catches a truncated download and a corrupt header without reading the weights.

Directory presence alone is not enough, which is the reason this column exists. Two entries once held a config, a README and no `.safetensors` at all, with tens of gigabytes sitting unmaterialised in their `.cache/` blobs. Both were named correctly, both were listed for transfer to the other host, and the only signal was a benchmark row with every measurement column empty.

## Regenerating

Walk `models/*/`, read the README heading then the `base_model` line, read `config.json` for precision and quantization, run the shard check, and count occurrences of the name and its aliases across `benchmarks/*.csv` and `docs/`. Count doc mentions with a trailing boundary (`(?![A-Za-z0-9._-])`) or a short name absorbs every longer one that starts with it.

Each host fills its own presence and shard columns and leaves the other host's alone, so the two can be merged without either overwriting what it cannot observe.
