# Third complete acceptance: chat criterion remains unmet

Source `20c9ea8e73c0ad19252ebd1ce3088835d4610099` includes the trajectory-backed consecutive-low-yield correction `719ea08b`. The production release, full CUDA suite (8,936 root and 2,088 core tests, all integration tests), clippy, formatting, and three contract checks passed before this unchanged full matrix ran under gpu-lock after the quiet-host gate.

All measurement arms completed successfully. The nested copy of the analyzer initially failed because it assumed the original directory depth. `analyzer-initial-error.log` and the original empty `summary.txt` preserve that failure. The parent canonical analyzer was corrected only to locate the repository and historical evidence through ancestors; all 17 synthetic checks, including a new nested-directory case, pass. No measurement, formula, threshold, or selection changed. Running the corrected parent analyzer against this directory produced `summary-corrected.txt` and exit 1.

The only failed criterion is (a): chat median -1.419%, range -4.350% to +0.813%, versus a -1% floor. All copy gains (+142.132%, +151.215%, +145.537%, +146.083%), concurrency criteria, long-copy gain floors (paged 1024/2048/4096: 2.6686/2.7573/1.4565; dense 8192: 2.2823), and paged-8192 gate-closed checks pass. This remains a failed overall acceptance run.

Replay the analysis from the repository root using the parent `analyze_break_even.py` with this directory as its argument. The nested runner/analyzer copies remain unchanged as the exact measurement protocol. `SHA256SUMS` preserves all generated artifacts and the recorded analysis failure/recovery.

The additional `analyze-corrected.py` is the fixed parent analyzer copied into this nested directory. Running that copy against this directory also exits 1 and produces byte-identical `summary-corrected.txt`, directly checking both script-depth and data-depth resolution.
