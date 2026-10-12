# Issue 2264 checkpoint

Implementation is not complete and no PR has been opened. Five sibling issues in the approved auto-impl run are merged; issue 2264 remains open. The user approved three additional attempts on 2026-10-12 after the initial three-attempt limit. Exact chat trajectory diagnosis and selective-policy validation are complete; attempt 4 is pending.

- Attempt 1, parent directory: complete failed matrix. Chat passed, but short-copy and paged 1024/2048 gain preservation failed.
- Attempt 2: all five short-copy comparisons still failed after correcting fully accepted short proposals; stopped intentionally before the remaining matrix completed. Its partial evidence is preserved.
- Before/after trajectory diagnostics: isolated misses after profitable copy caused immediate pause and no later resume. Requiring two consecutive incomplete low-yield rounds restored the exact copy workload from 137 proposed / 112 accepted / 21 rounds to 239 / 214 / 35. Diagnostic timings are not performance evidence.
- Attempt 3: complete matrix. All criteria pass except chat, whose median paired delta is -1.419% against the -1% floor. See `attempt-3/summary-corrected.txt`.

The added grace is too costly on ordinary chat: compared with attempts 1/2, the exact eight-question workload adds 60 proposals, only 8 accepted tokens, and 12 verify rounds, emitting 20 tokens total (1.67 per added round). The next evidence step is to trace those exact chat requests and derive a condition that grants isolated-miss grace only after established profitable-copy evidence. The selective correction now grants grace only to the first complete miss whose preceding accepted-token EMA meets the floor. Actual diagnostic runs preserve copy 239/214/35 and reduce chat from 226/83/43 to 173/75/32; 54 governor/drafter and 20 scheduler tests pass. Attempt 4 performance remains pending.

The selective policy has passed focused runtime validation; its release build, full CUDA suite, and unchanged quiet performance matrix are pending. Floor 4, shadow confirmation 7, immediate server re-prime, and context cap 4608 remain unchanged. CLI default, Graded, and nonadaptive behavior remain unchanged. No temporary runtime tracing remains in the source.
