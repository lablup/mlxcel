# Issue 2264 checkpoint

Attempt 4 passes every unchanged acceptance criterion. Release source `b274abf0` passed release build, clippy, formatting, three contract checks, and the full CUDA workspace suite (8,936 root and 2,093 core tests, no integration failures). Publication and CI are pending.

- Attempt 1: complete failed matrix; chat passed, copy retention failed.
- Attempt 2: short-copy failure persisted after the full-short-acceptance correction; intentionally stopped with partial evidence preserved.
- Attempt 3: blanket consecutive-low grace restored copy but chat failed at -1.419%.
- Chat and selective trajectory diagnostics: restrict grace to an isolated complete miss after accepted-token EMA meets the floor. Copy stays 239/214/35; chat drops from 226/83/43 to 173/75/32. All 54 governor/drafter and 20 scheduler tests pass. Temporary instrumentation was removed.
- Attempt 4: complete PASS, chat -0.275%, short-copy +148.602% to +159.283%, C2 pass, all long-copy floors and paged-8192 guard pass. See `attempt-4/summary.txt`.

The user approved three additional attempts on 2026-10-12; the first additional attempt passed. Floor 4, shadow confirmation 7, immediate server re-prime, and context cap 4608 remain unchanged. CLI defaults, Graded, and nonadaptive behavior remain unchanged.
