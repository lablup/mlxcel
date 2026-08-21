# Technical Report: PR #1275 - Link libc after the IREE archives

## Executive Summary

No integration test could be linked with `--features cuda,xla-iree`, the natural production feature set for the OpenXLA backend on a CUDA host. The link failed on an undefined `__stack_chk_guard` reference from IREE's runtime archive, naming the dynamic linker as a "DSO missing from command line". The fix is one entry: repeat `-lc` after the IREE archives in the `IREE_CUDA_HOME` recipe in `build.rs`.

The value of this report is mostly in what did not work. Three successive diagnoses were wrong, and the shipped change is smaller than the first attempt because each half of that attempt was ablated separately rather than accepted together once the link went green.

## 1. Problem Statement

```text
/usr/bin/ld: libiree_runtime_unified.a(call.c.o): undefined reference to symbol '__stack_chk_guard@@GLIBC_2.17'
/usr/bin/ld: /lib/ld-linux-aarch64.so.1: error adding symbols: DSO missing from command line
```

`cargo check` never links, so `cargo check --features cuda,xla-iree --all-targets` passed on the same tree. The failure only appeared when something actually produced a test binary, which is why it survived in `main`.

It was also not uniform. `chat_template_kwargs` linked under the same features while `molmo2_xla_vision_parity` did not, and `molmo2_xla_vision_parity` linked under `xla-diagnostics`. That non-uniformity is what made the first two diagnoses look plausible.

## 2. Technical Decisions

### 2.1 The error message points away from the fix

The message names `ld-linux-aarch64.so.1`, which suggests linking the dynamic linker. `readelf` settles where the symbol actually lives: `libc.so.6` carries `__stack_chk_guard` as `UND`, and `ld-linux-aarch64.so.1` defines it. So libc does not supply the symbol, and the initial conclusion drawn from that ("adding `-lc` cannot work") was wrong for a subtle reason: the fix does not need libc to define the symbol, only to be positioned where ld can follow its `DT_NEEDED` chain.

### 2.2 Ordering, not a missing library

`rustc-link-arg` can only append. The IREE archives therefore land after rustc's own `-lc`, and `call.c.o` is compiled with the stack protector, so its reference to `__stack_chk_guard` appears at a point where no libc follows it. Repeating `-lc` after the archives restores the ordering an ordinary C program gets for free.

### 2.3 The policy flag was ablated out

The first attempt added `-Wl,--copy-dt-needed-entries`, on the theory that ld was refusing the transitive resolution. It failed on its own: rustc appends our arguments after its `-lc`, so the flag landed at argument 80 while `-lc` sat at argument 29, and it only governs inputs that follow it.

Both changes together linked. Rather than ship that, each half was tested alone:

| configuration | result |
| --- | --- |
| trailing `-lc` only | links |
| `-Wl,--copy-dt-needed-entries` only | fails, same error |
| both | links |

The flag is redundant, and it is a global relaxation that can hide a genuinely missing `-l` elsewhere in the same link. Shipping the pair would have added that risk for no benefit. This is the main reason the report exists: a green link is not evidence that every part of the change earned its place.

### 2.4 The sibling recipes are left alone

`IREE_DIST` emits a nearly identical group and is likely affected the same way. It is not changed, because this host has neither an `IREE_DIST` tree nor macOS, and a link recipe should not carry changes nobody has linked. Issue #1274 explicitly allows those recipes to be either unchanged or verified.

## 3. Change Summary

| File | Change |
| --- | --- |
| `build.rs` | `-lc` appended to the `IREE_CUDA_HOME` group, with the ordering rationale and the ablation result recorded inline |
| `build.rs` | `link_args.insert(5, ...)` replaced by a positional `push`, so adding an entry ahead of it can no longer move the vendored printf archive out of `--start-group` silently |
| `build.rs` | Each library in the group now carries the reason it is present |

## 4. Review Findings

Self-review during implementation caught the redundant policy flag before the commit. No external review findings.

The magic-index removal was not requested by the issue. It was made because this change added an entry to the same vector, and the existing `insert(5, ...)` would have silently relocated the conditional printf archive if the entry had been prepended instead of appended. That is a defect waiting for the next edit, not a style preference.

## 5. Validation

Four targets linked, deliberately including cases that already passed:

| target | features | before | after |
| --- | --- | --- | --- |
| `molmo2_xla_vision_parity` | `cuda,xla-iree` | fails | links |
| `cli_help_consistency` | `cuda,xla-iree` | fails | links |
| `molmo2_xla_vision_parity` | `xla-diagnostics` | links | links |
| `chat_template_kwargs` | `cuda,xla-iree` | links | links |

Plus `cargo check --features cuda --all-targets` clean and `cargo fmt --all -- --check`. The previously-passing rows are not padding: the failure was target-dependent, so a fix verified only against failing targets could have regressed the rest without anyone noticing.

## 6. Related Work

Issue #1274 filed the bug and carries a correction section recording which of its original claims held up. Issue #1270 tracks the CI coverage gap that let this reach `main` in the first place, since no workflow compiles any XLA feature combination.

One question is left open: why the failure was target-dependent and feature-dependent. Both working binaries carry `ld-linux-aarch64.so.1` as an explicit `DT_NEEDED` and the failing links do not, so something in those link lines already pulled it in. The `surgery` default feature was suspected and is neither confirmed nor ruled out. This fix removes the ordering dependency for every target, so the asymmetry no longer has practical consequences, but it is recorded rather than left as folklore.
