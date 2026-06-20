# Architecture Decision Records

This directory holds Architecture Decision Records (ADRs) for mlxcel. Each ADR captures one significant decision, the context that forced it, and the consequences that follow. ADRs are numbered sequentially and are immutable once their status is Accepted: a later decision that changes course gets a new ADR that supersedes the old one rather than editing it in place.

## Index

- [ADR 0001: Paged-attention gather strategy and KV pool tensor layout](0001-paged-attention-gather-vs-fused-kernel.md). Accepted. Adopts gather-then-SDPA for the unified paged KV cache (epic #116) and records the pool tensor layout decision, backed by `examples/page_gather_microbench.rs`.
- [ADR 0002: Turbo KV decode is split dequant plus native SDPA](0002-turbo-kv-split-dequant-vs-fused.md). Accepted. Records why mlxcel's split Turbo KV path does not reproduce the upstream turboquant_plus sparse-V `+22.8%` speedup (the V step is a negligible decode fraction in a split design), keeps dequant-SDPA (#369) as the default, and names fused V dequant (#370) as the one lever, backed by the A/B in `benchmarks/turbo_kv/`.
