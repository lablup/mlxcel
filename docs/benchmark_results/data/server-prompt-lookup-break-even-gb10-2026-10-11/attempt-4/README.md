# Fourth complete acceptance: all criteria pass

Production release source is `b274abf0`; its exact hash and release binary SHA-256 are recorded in the environment and source files. Release build, clippy, formatting, three contract checks, and the full CUDA workspace suite passed before this unchanged measurement matrix ran under gpu-lock on a quiet GB10. The workspace suite passed 8,936 root tests and 2,093 core tests, with no integration failures. Original local full test output remains at `/tmp/mlxcel-auto-plan/2264-fourth-full-cuda.log`; the complete suite summaries are archived here.

The analyzer exits 0: chat median -0.275% (range -2.174% to +1.944%); short-copy gains +159.283%, +157.037%, +151.289%, +148.602%; concurrency-two ranges overlap; long-copy ratios paged 1024/2048/4096 are 2.7349/2.8189/1.5221 and dense 8192 is 2.3475. Paged 8192 correctly reports no proposals. All original acceptance thresholds and workloads are unchanged.

The selective grace condition is a trace-backed heuristic measured on these workloads, not a general guarantee for arbitrary prose or models. All prior failed attempts remain preserved. Replay `python3 -I -B analyze_break_even.py .` from this directory, or use the parent analyzer with this directory as its argument.
