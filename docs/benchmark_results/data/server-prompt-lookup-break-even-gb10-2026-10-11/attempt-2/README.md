# Corrected implementation acceptance

This directory is reserved for the full unchanged acceptance matrix after fixing fully accepted short proposal rounds. The first complete failed attempt remains in the parent directory, with `attempt-1.md` and `ATTEMPT1-SHA256SUMS`. Its observations are neither overwritten nor excluded from the report.

The copied measurement clients, arm runners, and analyzer retain the same workloads and thresholds. The coordinator runs `run_all.sh` here under exclusive gpu-lock after release build, runtime tests, clippy, formatting, contract checks, and the quiet-host gate. The environment records the exact corrected binary source and hash.
