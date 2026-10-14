#!/usr/bin/env bash
# The tests issue #2192 names, each run under its own narrow selector with
# `--features rocm` on gfx1151, single-threaded. The build must already exist
# (`cargo test --workspace --profile test-fast --features rocm --no-run`) so
# no compile overlaps the runs. Every run's output lands in OUT_DIR, and the
# summary lists per selector: the `test result:` line, the number of
# `skipping` lines (must be 0 for a run to count as executed), and whether
# the selector matched zero tests.
#
# Usage: named_tests.sh OUT_DIR
set -uo pipefail
OUT="$1"; mkdir -p "$OUT"
export OPENSSL_INCLUDE_DIR=/usr/include OPENSSL_LIB_DIR=/usr/lib/x86_64-linux-gnu CARGO_BUILD_JOBS=8
SUMMARY="$OUT/summary.tsv"
echo -e "selector\tresult\tskipping_lines\tstatus" > "$SUMMARY"

run() {
  local name="$1"; shift
  local log="$OUT/$name.log"
  >&2 echo ">>> $name ($(date -u +%H:%M:%S))"
  cargo test --profile test-fast --features rocm "$@" > "$log" 2>&1
  local rc=$?
  local result skips status=ok
  result=$(grep -h "^test result:" "$log" | sed 's/; finished.*//' | paste -sd ';' -)
  skips=$(grep -ci "skipping" "$log")
  [[ $rc -eq 0 ]] || status="FAIL(rc=$rc)"
  if grep -q "^test result: ok. 0 passed; 0 failed; 0 ignored" "$log" && ! grep -q "^test result: ok. [1-9]" "$log"; then
    status="NO_MATCH"
  fi
  echo -e "$name\t$result\t$skips\t$status" >> "$SUMMARY"
}

# Root-crate integration tests.
for t in sampling_gumbel_kill_switch sampling_rejection_kill_switch \
         rocm_slice_update_source rocm_slice_update_reduce \
         rocm_wave64_port_refusal rocm_wave_size \
         rocm_gather_qmm_expert_batched rocm_inflight_bound; do
  run "test_$t" --test "$t" -- --test-threads=1
done

# mlxcel-core lib tests.
for f in sampling_fixed_key_tests sampling_rejection_tests sampling_gumbel_tests \
         lang_bias_counters::tests::override_is_scoped_nested_and_thread_local \
         config_supports_fused_batch_false_for_token_bias \
         rotating_inplace_warmup_and_wrap_keep_shared_snapshots_intact \
         rotating_steady_state_snapshot_survives_the_next_wrap_write \
         paged_pool_offset_tests \
         validate_rejects_a_merge_input_past_the_u32_index_range \
         kernel_port_tests \
         fused_905_defaults_are_on_for_rocm_builds_only \
         fused_norm_parity_tests \
         ssm_update_parity_tests fused_moe_parity_tests fused_moe_relu2_parity_tests; do
  run "core_${f//::/_}" -p mlxcel-core --lib "$f" -- --test-threads=1
done

# mlxcel lib tests (single-threaded, as the lookahead tests require).
for f in pooling_cache_remainder_survives_an_overlapping_tail_write \
         the_rope_append_bypass_notice_needs_an_explicit_truthy_value \
         switch_layers::mxfp_tests \
         runtime::tests::cache_limit_ \
         scheduler_model_owned_lookahead_tests \
         model_owned_sequence_primes_the_decode_lookahead_only_with_a_rewind \
         sampling_observability_tests; do
  run "lib_${f//::/_}" -p mlxcel --lib "$f" -- --test-threads=1
done

cat "$SUMMARY"
