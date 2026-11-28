# Technical Report: PR #3079 - Make the side-model join test fail after the worker starts

**Date**: 2026-11-15
**Status**: Completed
**Risk Level**: Low

## Executive Summary

`server_exit_joins_worker::embedding_worker_is_joined_when_a_later_side_model_fails_to_load` failed on main. Its `--image-model` was an absent path, which model resolution now rejects before any side-model worker starts, so the exit path whose worker join the test asserts never ran. The test now points `--image-model` at a directory with an invalid `config.json`. That directory resolves and then fails to load after the embedding worker has started.

## 1. Problem Statement

The test from #2827 checks that a side-model load failure after an earlier worker has started still joins that worker before the process exits ("side model workers exited before process exit"). Resolution of the image-model argument (`src/downloader/resolver.rs`) now fails fast on a path that is not a model, and that is the intended behavior. So the server exited with the resolver error, and the test reported the join line as missing (#3076). No PR job runs the root integration tests, so CI did not catch it.

## 3. Technical Decisions

The server's behavior is correct, so only the test input changed. A temp directory holding `{ not valid json` as `config.json` passes resolution and fails in the loader, which reproduces the late failure the test was written for. The assertions are unchanged: the join log line, exit code 1, no C++ abort text, and the "failed to load --image-model" reason.

## 7. Change Summary

| Item | Value |
|------|-------|
| Files changed | 1 (`tests/server_exit_joins_worker.rs`) |
| Tests | `MLXCEL_REQUIRE_MODELS=1 cargo test --test server_exit_joins_worker`: 5 passed with real checkpoints |

Related: #2827 (the test), #3076 (issue).
