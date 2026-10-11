# Trajectory-backed final acceptance

The coordinator runs the unchanged full acceptance matrix here after validating the consecutive-low-yield correction. The failed first full attempt remains in the parent directory. The second attempt failed all five short-copy comparisons and was stopped early; its partial raw data remain in `attempt-2`. The `trajectory-diagnostic` and `trajectory-diagnostic-after` directories explain the observed governor failure and the corrected transitions.

This directory uses the same workloads and thresholds. Final results are pending until the full matrix, provenance checks, and acceptance analyzer complete on a fresh production release binary under gpu-lock after a quiet-host gate.
