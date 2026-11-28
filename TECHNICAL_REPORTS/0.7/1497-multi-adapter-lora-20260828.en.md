# Technical Report: PR #1497 - Multi-adapter LoRA to the fused-weights boundary

**Date**: 2026-08-28

**Status**: Completed

**Languages**: Rust

**Risk Level**: Medium

## Executive Summary

Implements the b10621 multi-adapter LoRA surface (#1439) up to the boundary mlxcel's design sets: adapters fuse into the base weights at model load, so `--lora a,b` and `--lora-scaled a:0.5,b:2` load any number of adapters with deterministic order and per-adapter scaling, `GET /lora-adapters` reports upstream's inventory shape, and the runtime-swap surfaces (`POST /lora-adapters`, the per-request `lora` field, `--lora-init-without-apply`'s later activation) accept exactly the inert values and refuse the rest with diagnostics. Three manifest entries flip to `supported`; three stay `deferred` on the fused-weights divergence, so issue #1439 remains open.

## 1. Problem Statement

b10621 keeps LoRA adapters as runtime-swappable layers: any number load at startup, scales change per request or through `POST /lora-adapters`, and unlisted adapters drop to scale 0.0. mlxcel had a single `--lora` adapter fused permanently into the weights and no inventory or update surface, so a llama-server deployment using several adapters or runtime swaps had no honest migration target.

## 2. Technical Decisions

### 2.1 Fusion stays; the boundary is stated rather than hidden

Runtime-swappable adapters need unfused A/B matrices applied in every family's forward pass, a core-crate change with codebase-wide ripple. Fusion already exists and is correct for the static case, so the static surface (multi-adapter, per-adapter scale, order-deterministic load, inventory) is implemented fully, and every dynamic surface answers upstream's own resolution rule (`construct_lora_list`: listed ids set scales, unlisted drop to 0.0, unknown ids ignored) and then either acknowledges the configuration already in force or refuses with a diagnostic naming `--lora-scaled` and a restart. A request is never silently served on weights it did not ask for, which is the epic's core rule.

### 2.2 Scale domain enforced at startup

A user scale multiplies into the adapter's own `alpha / r` at fusion, exactly upstream's arithmetic. NaN, infinite, and non-numeric scales fail the command line: a non-finite delta would poison every weight the adapter touches and surface only as garbage output. Missing adapter directories fail startup too, and combining multi/scaled adapters with tensor or pipeline parallelism is refused rather than partially applied.

### 2.3 Compatibility seams kept narrow

The trivial single-adapter case reduces onto the pre-existing `adapter_path` channel, keeping that path byte-identical. Router-mode pool children deliberately do not inherit adapters, because an adapter is base-model-specific and the lenient legacy loader would otherwise skip mismatched tensors silently, the #1328 failure mode. #1328's strict validation itself is untouched.

## 3. Change Summary

| Item | Value |
|------|-------|
| Files changed | 24 |
| Manifest entries | supported: --lora, --lora-scaled, GET /lora-adapters; deferred: --lora-init-without-apply, POST /lora-adapters, field:lora |

Validation: unit and route tests (flag parsing with NaN/inf refusals, legacy reduction, inventory shape, POST inert/refuse/non-array, per-request inert/refuse) plus a real checkpoint (`qwen2.5-0.5b-bf16`, greedy) with two synthesized real-format adapters: `--lora` changes the output versus base, `:0` is identical to base, `:3` diverges further, both adapters load together, and every route behavior above was exercised live. Two chain-surfaced test regressions were fixed on the way (the shadowed duplicate `--no-webui` stubs; the #1438 migration guard now firing before model resolution).

## 4. Follow-up Actions

- The unfused runtime-LoRA forward path (or a #1438 model-pool reload flow) unlocks the three deferred entries; it stays on #1439.
- #1328 (strict adapter-tensor validation) remains open and unaffected.
