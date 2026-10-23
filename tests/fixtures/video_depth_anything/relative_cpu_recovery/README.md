# Relative CPU recovered-oracle evidence

These files preserve the independently reviewed recovery packet for
`small_relative_40.safetensors`. They document an accepted recovered oracle,
not a clean generator-wrapper run:

- the shell wrapper exited with status 127;
- the Python process exit was not independently captured;
- the original detailed completion log was lost; and
- no terminal `PASS` line was reconstructed.

The output tensor and its original legacy provenance remain in the parent
fixture directory without modification. The comparator's dedicated recovery
gate authenticates every file listed in `SHA256SUMS`, validates the full tensor
and provenance chain, and recomputes parity against the pinned GPU reference.
Metric reference validation continues to use its ordinary completion log.

File mapping:

- `epic-2382-vda-relative-cpu-recovery-inspection.json`: preserved inspection
  packet made after the wrapper failure.
- `epic-2382-vda-relative-cpu-wrapper-failure.log`: preserved shell failure.
- `epic-2382-vda-relative-cpu-wrapper-failure.status`: preserved exit status.
- `epic-2382-vda-relative-cpu-independent-recovery-review.md`: independent
  audit and acceptance conditions.
- `epic-2382-vda-relative-cpu-recovery-review.json`: machine-readable reviewed
  decision, limitations, and CPU-to-GPU corroboration.
