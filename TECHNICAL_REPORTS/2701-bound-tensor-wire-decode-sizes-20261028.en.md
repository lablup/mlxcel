# PR #2701: Bound tensor wire decode sizes and fix int16 decode

**Date**: 2026-10-28
**Status**: Validation complete; PR open as a draft (report written on the PR branch before merge)
**Languages**: Rust, C++
**Risk Level**: Low (peer-facing decode hardening; wire format and cluster `PROTOCOL_VERSION` unchanged; three crate-public Rust signatures change)

## Executive Summary

Issue #2691 collected three defects in how a stage decodes a tensor received from a cluster peer, found during the security review of #2671. A quantized or LZ4-compressed tensor header let the sender choose the size of the buffer the receiver allocated (4 GiB from an 8-byte or 13-byte payload, or a process abort from a larger shape), a malformed quantized payload decoded to zeros instead of failing, and the C++ `from_bytes` bridge decoded INT16 and UINT16 one byte per element. This PR makes every size a peer controls into a checked value that is validated before allocation: `TensorHeader::num_elements` fails on overflow, `dequantize_int8` / `dequantize_int4` require the exact payload layout, `decompress` takes a caller limit and checks two prefixes plus a 256:1 expansion bound before allocating, and `deserialize_tensor` checks every decoded tensor against its shape. The bridge gains typed 16-bit integer branches, and `validate_raw_tensor` refuses FLOAT64 and COMPLEX64, which the bridge still cannot decode. No honest traffic changes shape on the wire; the only behavior change is that malformed tensors now return an error.

## 1. Problem Statement

### 1.1 Background

Cluster frames are authenticated by the handshake from #2650 / #2674 and, since #2684, sealed per frame with AES-256-GCM. Those layers decide who may send a frame, not what the frame may ask the receiver to do. A compromised member or a buggy peer still reaches the tensor decoder with arbitrary header fields.

The decoder honors the header's flags rather than what an honest sender would produce. Pipeline activations are serialized with `SerializeOptions::default()` (no quantization, no compression), so the compressed and quantized branches of `deserialize_tensor` are never taken by honest stage traffic, yet a peer that sets those flag bits reaches them on the activation path: `deserialize_wire_tensor` calls `ActivationMessage::deserialize_activation`, which calls `deserialize_tensor`. That is why the hostile surface sat in code that normal runs never exercise.

### 1.2 Existing Issues

- **Quantized output sized from the shape** (MEDIUM): `num_elements` folded the shape with `checked_mul` and then `unwrap_or(u64::MAX)`. `deserialize_tensor` cast the result with `as usize` and passed it to `dequantize_int8` / `dequantize_int4`, which returned `vec![0u8; num_elements * 2]` whenever the payload was short, had a zero group count or size, or was shorter than the shape implied. A QUANTIZED int8 header with shape `[1 << 31]` and an 8-byte payload returned `Ok` with 4 GiB of zeros; shape `[1 << 40]` aborted the process on allocation; a saturated `u64::MAX` overflowed `num_elements * 2`. A malformed payload became silently wrong data.
- **Unquantized payload never checked against its shape** inside `deserialize_tensor`. `serialize_tensor` enforced the rule on the way out; nothing enforced it on the way in.
- **LZ4 prefix trusted** (MEDIUM): `decompress` read the outer u64 length and discarded it, then called `lz4_flex::decompress_size_prepended`, which in lz4_flex 0.11.6 runs `vec![0; n]` on the peer-supplied inner u32 before decoding a byte. A 13-byte payload declaring `0xFFFF_FFFF` cost 4 GiB per frame.
- **16-bit integers misdecoded** (LOW): `from_bytes` had typed cases for 32/64-bit integers and 16-bit floats and sent everything else through `case 1: default:`, which builds the array from `uint8_t*` and casts each byte. INT16 (code 6) and UINT16 (code 2) produced wrong values from the first half of the buffer, and FLOAT64 (11) and COMPLEX64 (13) were misdecoded the same way. `validate_raw_tensor` checked the byte count first, so nothing over-read, but the tensor was wrong. An int16 MLX array crossing a stage boundary was enough to trigger it.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| A member forces multi-GiB allocations per frame (memory pressure, OOM kill) | High | Low (requires an authenticated peer) |
| A crafted shape aborts the receiving process | High | Low |
| A malformed quantized payload is accepted as zeros and propagates as wrong activations or KV entries | Medium | Low |
| int16 / uint16 tensors silently corrupted at a stage boundary | Medium | Low (stage activations are normally float16, bfloat16 or float32) |

## 2. Technical Decisions

### 2.1 Validate the exact layout, then allocate

**Context:** the old dequantizers checked a lower bound (`payload.len() >= required_len`) with saturating arithmetic and fell back to a zero buffer on every failure.

**Alternatives considered:**

| Option | Pros | Cons |
|--------|------|------|
| Keep the lower-bound check, return `Err` instead of zeros | Small diff | Trailing bytes still accepted; output still bounded only by the header |
| Cap the output at a fixed size | One constant | A legitimate large tensor can still fail; a small frame can still claim the cap |
| **Chosen: exact layout match** (`group_size >= 1`, `num_groups == ceil(n / group_size)`, `len == 8 + 2 * num_groups + data_len`, all checked) | Output is tied to bytes actually received: at most 2x the payload for int8, 4x for int4 | Rejects any payload with trailing bytes, which the encoder never produces |

**Rationale:** once the payload length equals the layout implied by `num_elements`, the allocation `num_elements * 2` is a fixed multiple of what the peer actually sent, so no separate size cap is needed for dequantization. A shared `validate_layout` returns a `QuantizedLayout` (`num_groups`, `group_size`, `data_start`) that both dequantizers use, and slice indices in the loops stay inside the validated region. The int4 path also stops materializing an intermediate `nibbles` vector of `num_elements` bytes; it reads each nibble from the packed byte directly.

**Trade-off:** `dequantize_*` changed from `Vec<u8>` to `Result<Vec<u8>>`, so every caller had to propagate the error. The caller audit in the PR review found three callers (`tensor_serialize.rs`, `kv_cache_transfer/quantized.rs`, `kv_cache_transfer/streamed.rs`), all updated.

### 2.2 LZ4: checks before the buffer, `decompress_into` after

`decompress(data, max_len)` now:

1. requires 12 bytes of prefixes (`split_first_chunk`, no `unwrap`);
2. requires the outer u64 and the inner lz4_flex u32 to agree (`LZ4 size prefix mismatch`);
3. requires `declared <= max_len` (`decompressed size {declared} exceeds limit {max_len}`);
4. requires `declared <= body.len() * 256` (saturating);
5. allocates exactly `declared` bytes and decodes with `lz4_flex::block::decompress_into`, rejecting a block that writes any other length.

Calling `decompress_into` on the block body, rather than `decompress_size_prepended`, is what moves the allocation under our control: the library function allocates from the inner prefix before any of our checks could run. The 256:1 bound comes from the LZ4 block format, which spends at least one byte per 255 bytes of match length; it ties the allocation to the received block without rejecting valid data. The `decompress_round_trips_64_mib_of_zeros` test pins that claim with the most compressible input and also checks that `max_len - 1` is rejected.

Requiring prefix agreement has a side effect worth knowing: `compress` writes the inner prefix through `compress_prepend_size`, which stores a u32. An input of 4 GiB or more would get a truncated inner prefix; before this PR the receiver silently decoded a truncated tensor, now it reports a prefix mismatch. Honest tensors that large are not compressed today, so this was not exercised.

### 2.3 Header-derived limit, with a 4 GiB constant on top

`deserialize_tensor` passes `min(max_decoded_payload_len(header), MAX_DECODED_TENSOR_BYTES)`. The header bound is the exact data size for an unquantized tensor (`div_ceil(2)` for Int4) and the largest valid layout for a quantized one: `int8_payload_max_len(n) = 8 + 3n` and `int4_payload_max_len(n) = 8 + 2n + ceil(n / 2)`, which cover every `group_size >= 1`. The same two helpers bound `reassemble_layer_payload` in the streamed KV path, so the bounds are written once.

`MAX_DECODED_TENSOR_BYTES = 1 << 32` equals the u32 limit of the inner prefix. Because step 2 above already forces `declared <= u32::MAX`, the constant never binds on its own today. It is kept as an explicit, named ceiling next to `MAX_METADATA_LEN` so a future change of the prefix format cannot quietly lift the limit, and so callers have one value to cite. The effective guard is the combination of the header bound and the 256:1 rule: with a consistent header, a stage still allocates up to about 4 GiB from a frame of about 16 MiB, which the security review recorded as accepted under the issue's caps.

### 2.4 Reuse `validate_data_size` at the end of `deserialize_tensor`

`serialize_tensor` already rejected a buffer that did not match its shape and dtype. Calling the same `validate_data_size` after decompression and dequantization makes decode symmetric with encode: every tensor that leaves `deserialize_tensor`, quantized or not, matches its shape exactly. Reusing the helper instead of writing a second check keeps one definition of "the right size" (including the packed Int4 rule). Since the encoder enforces the identical rule, every tensor it produces still round-trips.

### 2.5 Fix int16 in C++, refuse FLOAT64 / COMPLEX64 in Rust

`from_bytes` returns `UniquePtr<MlxArray>` through cxx and has no error channel. Two choices were available for the dtypes it misdecodes:

| Option | Pros | Cons |
|--------|------|------|
| Change the bridge signature to return `Result` | One place to reject | Touches every `from_bytes` caller, including checkpoint loaders that never see peer data |
| **Chosen: typed branches for codes 2 and 6; reject 11 and 13 in `validate_raw_tensor`** | int16 / uint16 now decode bit-exactly through the existing `make_array_typed` helper; FLOAT64 / COMPLEX64 are refused on the only path peer data takes | FLOAT64 / COMPLEX64 remain wrong if a future caller reaches `from_bytes` without `validate_raw_tensor` |

The fallthrough is now explicit (`case 0: case 1: case 5: default:`) with a comment that only 1-byte dtypes may reach it. Every peer-reachable `from_bytes` call goes through `reconstruct_mlx_array`, which runs `validate_raw_tensor` first (KV cache layers, paged blocks, `deserialize_wire_tensor`); the remaining callers load local checkpoints with fixed or file-derived dtypes. `mlx_dtype_element_size` still knows the width of codes 11 and 13, so size computation elsewhere is unchanged.

### 2.6 No wire or protocol change

All checks run on the receiver over the existing layout `[u64 LE len][u32 LE len][LZ4 block]` and `[u32 num_groups][u32 group_size][f16 scales][values]`. `PROTOCOL_VERSION` stays as #2684 left it, and a peer on main interoperates with a peer on this branch for every tensor the encoder produces.

## 3. Decode Path After the Change

```
frame (AEAD-opened, #2684)
  -> base64 tensor field (wire_bytes.rs, bounded by the 256 MiB frame cap; unchanged)
  -> deserialize_tensor
       header.num_elements()?            overflow -> Err
       usize::try_from                   no `as` casts
       [compressed]  decompress(payload, min(header bound, 4 GiB))
                       prefix agreement, <= max_len, <= 256 x block, exact decode length
       [quantized]   dequantize_int8/int4  exact layout before allocation
       validate_data_size(dtype, shape, payload)
  -> reconstruct_mlx_array
       validate_raw_tensor                dtype 11/13 -> Err, exact byte count
  -> ffi::from_bytes                      typed u16/i16 branches
```

The streamed KV path (`reassemble_layer_payload`) takes the same decompress and dequantize steps with a limit derived from `header.original_num_elements`: `n * 2` at level None, the int8 / int4 payload bounds otherwise.

## 4. Change Summary

### Statistics

| Item | Value |
|------|-------|
| Files changed | 16 (8 source, 8 test) |
| Lines added | +850 |
| Lines deleted | -147 |
| Tests added | 29 (28 in the `mlxcel` lib, 1 in `mlxcel-core`) |

### Changes by module

| Module | Change |
|--------|--------|
| `tensor_protocol.rs` | `num_elements() -> Result<u64>` (`shape product overflow for ...`); `pub const MAX_DECODED_TENSOR_BYTES = 1 << 32` |
| `tensor_quantize.rs` | `validate_layout` / `QuantizedLayout`; `dequantize_int8` / `dequantize_int4` return `Result` with no zero-fill paths; `int8_payload_max_len` / `int4_payload_max_len`; int4 decode without the intermediate nibble vector |
| `tensor_compress.rs` | `decompress(data, max_len)`; `LZ4_MAX_EXPANSION = 256`; `decompress_into` with exact-length check |
| `tensor_serialize.rs` | checked element count; `max_decoded_payload_len`; `validate_data_size` on every decoded tensor |
| `kv_cache_transfer/streamed.rs` | header-derived LZ4 limit; dequantize errors propagated with context |
| `kv_cache_transfer/quantized.rs` | `?` on the new `Result` |
| `kv_cache_serde/types.rs` | `validate_raw_tensor` rejects dtype 11 and 13 |
| `mlxcel-core/cpp/mlx_cxx_bridge.cpp` | typed `uint16_t` / `int16_t` branches; explicit 1-byte fallthrough |

### Related Commits

| Hash | Type | Message |
|------|------|---------|
| `a056d30f` | fix | fix(distributed): bound tensor wire decode sizes and fix int16 decode |
| `bedb4be9` | merge | origin/main into the feature branch |

## 5. Compatibility

- **Wire**: unchanged. No `PROTOCOL_VERSION` bump; mixed main / branch peers interoperate for every tensor the encoder emits.
- **Rust API (crate-public)**: `tensor_compress::decompress` gains a `max_len` argument; `TensorHeader::num_elements` and `tensor_quantize::dequantize_int8` / `dequantize_int4` return `Result`. All in-workspace callers (src, tests, benches, examples) were updated.
- **Behavior**: tensors that previously decoded to zeros, decoded with trailing bytes, or decoded to the wrong size now return an error. FLOAT64 / COMPLEX64 raw tensors are refused for wire reconstruction; no serialize path emits them for activations or KV caches.

## 6. Validation

**Failing-first on main 54d63b0a.** The new Rust signatures do not compile against main, so the Rust tests ran against a signature-only shim that kept the old bodies (`decompress` ignored `max_len`, `dequantize_*` wrapped the old result in `Ok`, `num_elements` wrapped the saturated value). 20 of 24 new tests failed. `deserialize_rejects_quantized_shape_larger_than_payload` returned `Ok` with 4294967296 bytes of zeros (allocated for real on the 128 GB host); `deserialize_rejects_quantized_shape_overflow` panicked with `capacity overflow`; the int16 tests saw bytes `[0, 0, 128, 0, 255, 0, ...]`. The 4 that passed (`dequantize_accepts_empty_tensor`, `payload_max_len_bounds_every_quantized_payload`, `test_header_num_elements_empty_shape_is_zero`, `deserialize_round_trips_compressed_int4_quantized_tensor`) are kept as regression guards. In `mlxcel-core`, `from_bytes_round_trips_16bit_integers_bit_exactly` failed against the unmodified bridge. `dequantize_rejects_huge_element_count_without_allocating` (`1 << 40` elements, 2 TiB) was not run against old code, and the three `reassemble_*` tests were added with the fix.

**Branch head a056d30f** (`--profile test-fast --features cuda`, `--test-threads=1` under `gpu-lock`): fmt and clippy `--lib --tests -D warnings` clean; `distributed::tensor_` 445 passed, `kv_cache_transfer::` 62, `kv_cache_serde::` 43, `pipeline::wire_tensor` 9; all 28 new lib tests pass; GB10 remote real-model parity (`MLXCEL_REQUIRE_MODELS=1 MLXCEL_SDPA_DETERMINISTIC=1`) 6 passed: llama, gemma3, gemma4, qwen3, qwen3.5 parity and the llama drain/shutdown transition.

**Pre-merge gate at bedb4be9** (after merging main):

| Target | Result |
|--------|--------|
| verify scripts (fmt, versions, env-var docs, kernel dtype keys, kernel port dispatch, binary assets, Windows paths) | all exit 0 |
| contract tests | 27 + 4 passed |
| clippy, build | clean |
| `distributed::` lib module | 1461 passed, 0 failed, 4 ignored |
| `pipeline_remote_runtime_*` real-model parity | 6 passed |
| `pipeline_server_remote_coordinator_*` vs dense baseline | 5 passed |
| `pipeline_multi_stage_two_host_logical_smoke` | 1 passed |
| `disaggregated_e2e` | 51 passed |
| `mlxcel-core` `from_bytes` tests | 6 passed |

No performance measurement was taken. The uncompressed, unquantized activation path gains O(1) checks and no copies; LZ4 and dequantize allocate the same amounts as before for valid input, and int4 decode drops one temporary buffer.

## 7. Residual Risks and Follow-up

- **4 GiB from about 16 MiB remains possible** when the header shape is consistent with it (256:1 bound times the u32 prefix limit). A tighter cap derived from the model configuration (hidden size, max sequence length) would need a follow-up; the issue's caps were met as specified.
- **Streamed KV level None bound** is `original_num_elements * 2`, while `prepare_layer_payload` counts `keys.len() / 2 + values.len() / 2`. For 1-byte dtypes with odd key and value byte counts the bound is short by up to 2 bytes and a valid layer would be rejected. `reassemble_layer_payload` has no production caller today (only `prepare_layer_payload` is used, from `parallel.rs` and `benchmark.rs`), so nothing breaks; revisit if the streamed receiver is wired in.
- **Uncompressed level None in `reassemble_layer_payload`** does not compare the payload length with `original_num_elements`. Pre-existing and out of scope; it amplifies nothing because the output is a copy of the wire bytes.
- **Empty-shape semantics differ**: `validate_data_size` treats `[]` as 0 elements, `validate_raw_tensor` as a 1-element scalar. The encoder rejects scalars, so no regression, but a future scalar wire tensor would hit the mismatch.
- **FLOAT64 / COMPLEX64 still misdecode inside `from_bytes`** if a future caller skips `validate_raw_tensor`. The C++ comment records the invariant.
- **Pre-existing copy**: `deserialize_tensor` copies the compressed payload before decompressing it.
- **`MAX_DECODED_TENSOR_BYTES = 1 << 32`** does not compile on a 32-bit `usize` target. Every target mlxcel ships for is 64-bit, so this is a note, not a defect.

## 8. Learning Points

### 8.1 Validate before allocate, and make the allocation a function of received bytes

A header field that sizes a buffer is input, not metadata. The robust pattern is to derive an exact expected length from the header, require the payload to equal it, and only then allocate an output that is a fixed multiple of the payload. A lower-bound check plus a fallback value (`unwrap_or(u64::MAX)`, `vec![0; n]` on failure) turns a parse error into an amplified allocation and wrong data at once.

```rust
let expected_len = num_groups
    .checked_mul(2)
    .and_then(|s| s.checked_add(8))
    .and_then(|d| d.checked_add(data_len))
    .ok_or_else(|| anyhow!("... overflows ..."))?;
if payload.len() != expected_len { bail!("... length mismatch ..."); }
let out_len = num_elements.checked_mul(2).ok_or_else(|| anyhow!("... overflows ..."))?;
let mut out = Vec::with_capacity(out_len); // at most 2x the payload
```

### 8.2 Library convenience functions can allocate before you can check

`lz4_flex::decompress_size_prepended` reads its own prefix and allocates from it. Wrapping it with checks on a different prefix does not help, because the allocation happens inside the call. Splitting the frame yourself and calling the lower-level `block::decompress_into` with a buffer you sized puts the allocation behind your checks. The same question applies to any "size-prepended" API on untrusted input.

### 8.3 Decoders honor flags that honest senders never set

Honest stages never compress or quantize activations, so the vulnerable branches had no coverage from real runs. Failing-first tests built hand-made wire buffers (`hand_built_wire`, `framed`) to reach those branches directly. When auditing a decoder, enumerate every flag combination the header can express, not only the ones the encoder emits.

### 8.4 An FFI function without an error channel needs a gatekeeper

`from_bytes` cannot fail through cxx, so the dtypes it cannot decode are rejected one layer up, on the only path that carries peer data. The invariant lives in two comments (C++ fallthrough and `validate_raw_tensor`) so that a future caller sees it.

### Key Terms

| Keyword | Description | Relevance |
|---------|-------------|-----------|
| absmax group quantization | per-group f16 scale plus int8 / int4 values | the layout `dequantize_*` now validates exactly |
| LZ4 block format | literal runs plus (offset, match length) sequences | bounds expansion to about 255:1 |
| `decompress_into` | lz4_flex block decoder into a caller buffer | lets the receiver size the buffer |
| `cxx` bridge | Rust / C++ FFI generator | `from_bytes` has no `Result` return |

### Related PRs and Issues

- Issue #2691: the three defects and their acceptance criteria (this PR closes it).
- Issue #2654 / PR #2671: the security review that found them.
- PR #2684: per-frame AEAD and base64 tensor bytes; the base64 decoder was checked and is not affected.
- PR #2650, PR #2674: cluster authentication that limits these paths to authenticated peers.
- PR #2695 (open): fixes the disaggregated serving and router e2e failures on main, which the PR body cites as the reason the streamed KV path had no usable integration target; `disaggregated_e2e` passed 51/51 at bedb4be9.
