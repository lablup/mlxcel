# VDA Relative CPU Artifact Recovery — Independent Review

Reviewed on 2026-10-23 without rerunning a model or modifying a running job.

## Decision

The relative CPU artifact is acceptable as a **recovered oracle** through a dedicated, fail-closed recovery gate. It must not be described as a clean wrapper run.

## Independently verified artifact

- Output: `tests/fixtures/video_depth_anything/small_relative_40.safetensors`
  - SHA-256: `980442c3e1ebfc2291be035a80a4b8d6647dcfb31dcd9ad21e376d51fe532af8`
  - Size: 1,966,288 bytes
  - Exact tensor set: `depth`
  - Shape/dtype: `[40, 96, 128]`, float32
  - All values finite; minimum `0.0`, maximum `12.007184982299805`
  - Safetensors metadata: upstream revision `abd3b2d7b728cb5372d4e092260da325e5ded209`, device `cpu`, checkpoint dtype `float32`
- Provenance: `tests/fixtures/video_depth_anything/small_relative_40.provenance.json`
  - SHA-256: `4cb484a350a0c049b7e019331c93d320cdbad999ebc4752c0e631b1149c116bf`
  - Exact 17-key legacy schema, with no invented current-schema fields
  - Recorded output hash matches the artifact
  - Recorded duration: `25175.408284508158` seconds
  - Recorded peak RSS: `721000` KiB
  - Checkpoint config and all 351 parameter names, shapes, and float32 dtypes exactly match the pinned trusted GPU provenance (`37c9da1f2e47c374e03435178abb402fa8c77a5ce54cdd3d37288aa85b36bf02`)

## Trust-chain checks

- Archived relative generator SHA-256: `b5de2cb196409a31a1e85c4716a599375633aa178822e1a81c3c64d4db23833c`
- Relative checkpoint config/model/preprocessor hashes match their pinned values.
- Input clip, decoded 40-frame fixture, reference contract, upstream source manifest, and native runtime provenance match their pinned hashes.
- All 82 cached upstream-manifest entries match their recorded sizes and hashes.
- All 34 applicable native runtime and MLX Python files match the pinned runtime provenance.
- Full recovered CPU output versus the pinned upstream GPU output:
  - Relative RMS: `3.4021821538643326e-7`
  - Maximum absolute difference: `5.793571472167969e-5`
  - CPU and GPU output hashes are distinct.

## Preserved failure and uncertainty

- Wrapper exit: `127`
- Python exit: unknown; it was not independently captured.
- The original completion log was overwritten after the computation and must not be reconstructed or claimed.
- Before the overwrite, root reviewers witnessed the CPU runtime and processor identity, completion of window 1, and the start of window 2. They did not preserve a terminal `PASS` line.
- The generator wrote the tensor at `18:32:25.574`, then the complete provenance at `18:32:25.766`; the wrapper failure log followed at `18:32:26.866`.
- The wrapper file changed while the process was running. A shell offset reread explains the later `video_depth_anything: command not found` fragment, but this remains a hypothesis.
- Recovery packet SHA-256: `34fac1b79c33ac0fe830ec5c908adb0228a9f916ebda284129144aaca267044f`
- Preserved failure log SHA-256: `7489c3ef37f6a4f9005b446becd0e57f97da808267f3250f9c1cd4ec1412717f`
- Preserved failure status SHA-256: `743c7850cccfba5e53a9002663ec1ddd1079315a98bdbfdde10e6044f56abefe`

## Required recovery gate

For `relative` only, the comparator must pin the exact output and provenance hashes, authenticate the preserved recovery packet and wrapper failure evidence, require wrapper exit 127 plus unknown Python exit and a lost completion log, run the existing complete tensor/provenance checks, and require full CPU-to-pinned-GPU relative RMS at or below `1e-4`. It must record the recovery state and limitations in `cpu-parity.json`.

The metric oracle must retain the ordinary completion-log gate. The recovered oracle does not replace the separate requirement for a clean Rust CPU capture and Rust-to-oracle parity result.
