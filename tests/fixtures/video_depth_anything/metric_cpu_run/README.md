# Metric CPU reference run evidence

This directory preserves the original clean completion evidence for
`small_metric_40.safetensors`. The status is zero, and the log records the
authenticated CPU runtime, exact processor output, both 32-frame temporal
windows, both backbone and temporal-head stages, and the terminal fixture
`PASS` line.

The generated tensor and provenance remain in the parent fixture directory:

- tensor SHA-256:
  `c0f5e28da23645fa9cd146c07db270667aac490217f870984f1eb927e699a2cc`
- provenance SHA-256:
  `ccdee4a9abddaefbe1b39af4944dcbc2c06e4cb5633f1429786ff16f1d28e881`

`SHA256SUMS` authenticates the two original run records byte for byte.
