# Technical Report: PR #3086 - Update the create_causal_mask caller roster after #3083

**Date**: 2026-11-15
**Status**: Completed
**Risk Level**: Low

## Executive Summary

#3083 moved DeepSeek V3's multi-token MLA attention to causal-mode SDPA, so `deepseek_v3.rs` stopped calling `create_causal_mask`. `utils::tests::create_causal_mask_roster_count_matches_repo` then failed on main, because the doc comment stated 51 caller files and the grep found 50. The count is now 50, and the MLA group names DeepSeekV2 in place of DeepSeekV3. No PR job runs the `mlxcel-core` lib suite. This test is fast, so the orchestrated runs now include it in their merge-time contract checks.
