# Video Depth Anything CPU CLI acceptance evidence

This directory archives the compact evidence for the genuine 40-frame CPU CLI
parity checks. Both saved float32 depth tensors have shape `[40, 96, 128]`,
contain only finite values, and match the authenticated CPU reference fixtures
within the `1e-4` relative-RMS threshold.

| Checkpoint | Relative RMS | Threshold | Result |
| --- | ---: | ---: | --- |
| Small relative | `5.142513694620207e-6` | `1e-4` | pass |
| Small metric | `7.445840126949833e-7` | `1e-4` | pass |

`*-cpu-parity.json` is the independently reviewed result produced under
optimized Python (`python -O`) by comparator SHA-256
`37c3616b4d6ce4087ba14020839fa7fa6f8db2255203d4d84e68f40e21a30386`.
The comparator itself is intentionally kept out of this fixture directory
because it is a review tool rather than a runtime or regeneration dependency.
`*-comparator.log` and `*-comparator.status` preserve its output and zero exit
status.

The original capture wrappers exited 1 after the underlying CLI had already
exited 0. The saved-output validator then authenticated the complete output,
metadata, previews, CLI log, and zero CLI status without rerunning inference.
The wrapper failure came from a display-case assertion after inference. The
exact pre-edit capture script bytes executed by the already-running processes
were not preserved. `archived-post-edit-capture.py` is the authenticated
post-edit snapshot only; it changed during the runs and is not treated as proof
of the executed script. Accordingly, both acceptance results record
`executed_capture_script_sha256: null` and the provenance limitation explicitly.
No `finished_unix` value was fabricated.

The acceptance results bind the capture binary SHA-256
`dcc303d1e3a3c62ec3aa004ce71bd68f2f1c2160983620c8175cfa31ee41809b`
and the reviewed implementation commit
`3f01fb6a3d2d20c366a4673c0da3cb3d130961a7`. These historical identities remain
unchanged when the development branch is rebased. The large predicted tensors
and preview files are omitted here because their exact hashes are authenticated
by the acceptance JSON; the committed CPU reference tensors remain the durable
numerical oracle.

For each checkpoint:

- `*-capture-receipt.json` is the original saved-output receipt.
- `*-cli.log` and `*-cli-exit.status` prove the underlying CLI exit.
- `*-meta.json` records the depth kind, units, shape, frame count, and dtype.
- `*-saved-output-validation.status` records the independent validator exit.
- `*-wrapper.log` and `*-wrapper.status` preserve the post-CLI wrapper failure.
- `*-cpu-parity.json`, `*-comparator.log`, and `*-comparator.status` preserve the
  authenticated numerical comparison and its result.

`capture-wrapper.sh` and `saved-output-validator.py` are the exact archived
validation helpers. `SHA256SUMS` authenticates every non-document evidence file
in this directory.
