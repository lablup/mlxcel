# First integrated acceptance attempt: failed

The complete quiet GB10 matrix ran on 2026-10-11 from source `0960e32414377c2931838f8f425e160b5321f707`, after the integrated release build, full CUDA tests (8,936 root tests and 2,086 core tests), clippy, formatting, and three contract checks passed. `environment.json`, `environment.txt`, `binary-after.sha256`, and `hostgate.log` record the exact executable, native pin, and quiet-host provenance. `ATTEMPT1-SHA256SUMS` preserves all 140 generated raw artifacts from this attempt.

The analyzer exited 1. General chat passed at a median paired -0.917% (range -4.553% to +1.567%). C=2 passed the non-disjoint-regression rule. The four short-copy gains were +41.968%, +41.364%, +41.644%, and +42.567%, below their unchanged 90%-of-historical-gain floors. Paged 1024 and 2048 ratios were 2.4569 and 2.5562, below floors 2.521 and 2.611. Paged 4096 (1.4274, floor 1.378) and dense 8192 (2.0378, floor 1.99) passed. The paged-8192 gate-closed control passed its proposal-counter checks.

No raw observations were removed or overwritten. This is a failed acceptance attempt, not final performance evidence. Any corrected implementation must use a separate directory and preserve these results.
