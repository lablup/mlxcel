# 기술 보고서: PR #1096 - 결정론적 Autotune Profiling 테스트

**작성일**: 2026-08-10
**상태**: 구현 완료, platform qualification 대기
**언어**: Rust
**위험도**: Medium

## 요약

PR #1096은 autotune candidate 순서를 검증하는 테스트에서 host scheduler timing jitter를 제거한다. Production profiling은 계속 `Instant`로 실제 `TunableOp::run` 및 `sync` latency를 측정하고, 테스트는 internal manual microsecond clock을 통해 동일한 warmup, interleaving, selection, persistence, memoization logic을 실행한다.

리뷰 과정에서 production resource bound 문제도 함께 해결했다. 이제 caller가 지정한 warmup repetition은 정규화된 `max_reps`로 제한되며, CLI와 API 문서도 이 동작과 일치한다.

## 1. 문제 정의

CPU-only autotune 테스트 두 개는 `FakeOp` cost를 microsecond `thread::sleep`으로 모델링했기 때문에 host load에서 간헐적으로 잘못된 tactic을 선택했다. Scheduler overshoot가 비교 대상인 400-720 microsecond gap보다 커질 수 있어, 결정론적 selection을 검증해야 할 테스트가 operating-system wakeup 동작을 측정했다.

같은 test double이 다른 profile 및 resolve assertion에도 사용되므로 관측된 두 테스트만 수정하면 동일한 잠재 flake가 module에 남는다.

## 2. 기술적 선택과 그 이유

### 2.1 Profiler 경계에 시간 주입

Internal `ProfileTimer` abstraction이 기존 profiling algorithm에 mark와 elapsed microsecond를 제공한다. `profile()`은 `RealTimer`를 사용해 production 경로를 보존하고, test-only helper는 `ManualTimer`를 사용해 선택한 tactic의 선언된 cost만큼 시간을 전진시킨다. 이 seam은 internal로 유지되며 public `TunableOp` contract는 바뀌지 않는다.

### 2.2 Single-candidate budget 테스트에만 real sleep 유지

`timed_repetitions_scale_inversely_with_launch_cost`는 adaptive wall-clock repetition budgeting 자체를 검증하므로 sleep을 유지한다. 이 테스트의 각 profile에는 candidate가 하나뿐이어서 scheduler jitter가 문서화된 bound 안에서 repetition count를 바꿀 수는 있지만 candidate ordering을 결정할 수는 없다.

### 2.3 Selection 민감도 증명 및 caller work 제한

Regression suite는 양쪽 cost 방향을 모두 검증한다. Tuned candidate가 더 빠르면 default를 대체해야 하고, cost를 뒤집으면 더 빠른 default를 유지해야 한다. 임시로 잘못된 expectation을 넣었을 때 실제 red를 확인한 뒤 복원했다. 또한 `max_reps`를 timed repetition floor 이상으로 정규화한 다음 sanitized `warmup`을 그 값으로 제한하여 비정상 CLI/API input이 과도한 warmup loop를 강제하지 못하게 했다.

## 3. 변경 요약

| 항목 | 값 |
|------|----|
| 변경 파일 | 4 |
| 추가 라인 | 210 |
| 삭제 라인 | 56 |
| Focused test | 57 passed |

- Internal real/manual timer 구현을 추가하고 profiler의 warmup 및 timed phase를 이 경로로 연결했다.
- Candidate ordering에 의존하는 모든 fake-op profile 및 resolver 테스트를 deterministic synthetic time으로 옮겼다.
- Inverted-cost 및 warmup-ceiling 회귀 테스트를 추가했다.
- Public API 변경 없이 test-only resolver 실행을 위한 profiling closure seam을 추가했다.
- `ProfileConfig` 및 `mlxcel tune` help/output에 warmup이 `max_reps`로 제한됨을 명확히 기록했다.

## 4. 리뷰 발견 사항

| 발견 사항 | 심각도 | 해결 |
|-----------|--------|------|
| Caller-supplied `warmup`이 문서화된 `max_reps` ceiling을 넘을 수 있음 | Medium | `max_reps` 정규화 후 warmup을 제한하고 7-call 회귀 테스트 추가 |
| Cap 추가 후에도 public 및 CLI 문구가 warmup을 무조건적인 minimum으로 설명함 | Low | Profiler 문서, CLI help, tune summary output 갱신 |

남은 Critical 또는 High finding은 없다. Timer abstraction은 internal이며 production은 계속 `Instant`를 사용하고 deterministic resolver entry point는 test-only다.

## 5. 검증

- `cargo fmt --check -p mlxcel-core`: 통과.
- `cargo fmt --check -p mlxcel`: 통과.
- `cargo test --profile test-fast -p mlxcel-core --lib autotune:: -- --test-threads=1`: 57 passed, 0 failed, 1,355 filtered out.
- Inverted synthetic-cost selection test: 통과. 복원 전 임시로 잘못된 expectation이 요구대로 실패함을 확인했다.
- Warmup-ceiling regression: 통과. 제한된 warmup 5회와 timed repetition 2회를 증명했다.
- Hosted change detection, crate-version, kernel-dtype-key, cross-repository-reference, cargo-fmt, cargo-deny, CLA check 통과. MLX pin을 변경하지 않아 extraction은 skip되었다.
- macOS `metal,accelerate` 및 loaded-host full-lib gate는 사용할 수 없어 통과로 주장하지 않는다. 현재 Linux full-lib 실행은 1,411개 테스트를 시작했지만 CUDA backend가 없어 관련 없는 CUDA-backed cache test에서 abort되었다.

## 6. 관련 작업

- Issue #1079: sleep accuracy flake 및 deterministic timing 요구사항.
- Issue #997: 다른 mechanism과 platform gate를 가진 별도 concurrent-load flake.
