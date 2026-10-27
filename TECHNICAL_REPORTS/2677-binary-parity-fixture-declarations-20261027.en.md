# Technical Report: PR #2677 — binary parity fixture declarations

**Date**: 2026-10-27
**Status**: Completed
**Risk Level**: Low

## Executive Summary

Restore the binary asset gate after Bonsai parity fixtures landed during the earlier FLUX.2/Z-Image repair in #2675. Only the allow-list changes; existing binary contents and test behavior remain unchanged.

## Problem Statement

Three Bonsai fixtures were consumed by the image-generation parity tests but lacked exact size declarations. Their PNG exceeded the generic synthetic-image allowance and their tensors were undeclared. The failure existed on main and blocked otherwise unrelated changes in #2574.

## Change Summary

Declare the exact three paths with their measured ceilings and add their combined 1,452,561 bytes to the existing total allowance. Earlier #2675 similarly declared eight FLUX.2/Z-Image fixtures totaling 2,834,247 bytes. Neither change relaxes generic image limits or admits arbitrary new binary paths.

`python3 scripts/ci/check_binary_assets.py` passes with 103 declared binaries: 40,272.0 KiB within a 40,472 KiB ceiling. `git diff --check` passes. Future fixture additions must still add explicit declarations and justify their sizes.
