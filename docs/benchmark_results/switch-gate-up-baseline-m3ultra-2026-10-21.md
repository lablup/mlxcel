# Switch gate/up kernel wire-in baseline on M3 Ultra, 2026-10-21

PR #2514 lands the switch gate/up Metal kernel in the shared `SwitchGLU`
and wires it into `forward_activated` (inkling's decode) first. The follow-up
commit extends the wire-in to `SwitchGLU::forward`, the path the other MoE
families take at multi-token decode (batch 2 to 8, `total < 64`) and wherever
the #268 fused-expert kernel declines. Before that extension, the affected
state was recorded so its effect is measured against numbers on file, not
against recall.

The coverage finding first: no locally available model reaches
`SwitchGLU::forward`. `deepseek-v4-flash-4bit` runs `LimitedSwitchGlu`, a
separate struct with its own forward; inkling-small takes
`forward_activated` plus the down-combine kernel (the Part 1 path);
inkling-0.6b is unquantized and the kernel gate declines; `qwen3.8-27b` and
the `gemma-4` variants are dense hybrids with no experts. The `SwitchGLU`
families in the tree (qwen3_moe, glm4_moe, hunyuan_moe, solar_open,
bailing_moe, ernie4_5_moe, nemotron_h, olmoe, dbrx, mixtral, llama4, ...) have
no local checkpoints. The functional gate for the extension is therefore the
full test suite's per-family parity tests on synthetic fixtures, and no local
decode bench exercises the extended path. The dense qwen3.8 bench is a
no-regression reference only, and inkling is unaffected by construction.

## Functional baseline

Full workspace test suite at `4984c07a`, before the wire-in extension:

```
test result: FAILED. 9629 passed; 6 failed; 168 ignored; 0 measured; 0 filtered out; finished in 11.16s
```

The 6 failures are all `models::switch_layers::sorted_gather_*` numeric tests,
the known Metal grouped-`gather_mm` 2.6 ulp divergence documented since the
243bdf09/bff10220 pins (both pins show the same deviation; CI has no GPU and
skips them). The previously documented suite baseline was 9590 passed / 7
failed; the 7th was a florence2 flake that passes standalone and did not
appear here. The test count grew by 39 between the two baselines.

## Performance baseline

`models/qwen3.8-27b-4bit`, dense, extension has no effect on it (reference
only), 4 runs cool under `caffeinate -i`:

```
Prefill: 116.15-117.54 tok/s
Decode:   36.83-36.90 tok/s
```

`models/inkling-small-mlx-4bit`, the Part 1 path, also untouched by the
`forward` wire-in: 45.29-45.45 tok/s decode after `4984c07a`, 92.8% of the
48.94 tok/s mlx-lm baseline.

## Environment

| Field | Value |
|---|---|
| Host | Mac Studio, Apple M3 Ultra, 512 GB unified memory, macOS 26.6.1 (25G76) |
| Apple GPU generation | 15 (`applegpu_g15d`) |
| Build | `cargo test --release` (workspace, rust LTO) |
| Branch | `perf/moe-switch-gate-up` at `4984c07a` (PR #2514, unmodified for the baseline) |
| Harness | `cargo test --release` (functional); `mlxcel-bench-decode` (performance) |
| Wrapper | `caffeinate -i` |
| Target | `models/qwen3.8-27b-4bit` (dense reference), `models/inkling-small-mlx-4bit` (Part 1 path) |
| Sampling | decode `-n 100 --warmup-tokens 20`, 4 runs, fastest trusted |
