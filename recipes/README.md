# recipes/

This directory does not hold recipe files. Recipes themselves live at [mlxcel.ai/recipes](https://mlxcel.ai/recipes); if you were routed here by the recipe request issue template, that site is where the recipe content is authored, not this repository.

`recipes/registry/` holds committed snapshots of the machine-readable architecture registry, one JSON file per released `mlxcel` version plus a `CURRENT` pointer, so the recipes site and other downstream automation can pin against a specific runtime's supported-family data without building the binary.

## What a snapshot is

Each `recipes/registry/<version>.json` file is the exact output of `mlxcel arch --json` for that version, captured as `{"mlxcel_version": "<version>", "families": [...]}`. Every entry in `families` corresponds to a loadable family in `ALL_MODEL_TYPES` (plus a few standalone runtime families, such as detector-only checkpoints) and carries its detection keys, supported runtimes and modalities, per-backend status (Metal and CUDA in the 0.7.0 and earlier snapshots; snapshots generated after ROCm support landed also carry a `rocm` entry), tensor/pipeline parallel flags, speculative drafter support, and supported KV modes. See [`docs/supported-models.md`](../docs/supported-models.md) for the field-by-field description; the registry is a family-level contract, not a per-checkpoint qualification.

A snapshot describes the CLI as it existed when it was generated. A run only writes the file for the CLI's current version (rerunning at the same version overwrites that one file); snapshots for other versions are never deleted or rewritten, so the directory accumulates one file per version that was ever regenerated here.

## Regenerating the snapshot

```bash
make recipes-registry
```

This target builds the release CLI, reads its version from `mlxcel --version`, runs `mlxcel arch --json` into `recipes/registry/<version>.json`, and writes that same version string into `recipes/registry/CURRENT`. Both files are written via a temp file and an atomic `mv`, so a failed run never leaves a partial snapshot in place. Run it after bumping the crate version so the snapshot is filed under the released version name rather than the previous one; `0.7.0-beta.1.json` is a pre-release snapshot generated before the 0.7.0 version bump landed.

## What `CURRENT` means

`CURRENT` is a one-line text file holding the version string of the most recently generated snapshot. Downstream consumers that want "the latest registry" read `recipes/registry/$(cat recipes/registry/CURRENT).json` rather than guessing at the newest filename.

## Relation to `mlxcel arch`

`mlxcel arch` prints the same architecture catalog as a human-readable table; `mlxcel arch --json` emits it as data. The registry snapshot is that JSON output pinned at a release, so a consumer that only needs "what does mlxcel version X support" can read a snapshot instead of installing and running that version of the binary. See the "Supported models" section of the top-level [`README.md`](../README.md) for the CLI commands themselves.
