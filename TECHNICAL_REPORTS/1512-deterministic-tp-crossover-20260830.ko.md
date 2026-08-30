# 기술 보고서: PR #1512 - 결정적 TP 크로스오버 검증

**날짜**: 2026-08-30
**상태**: 열린 PR, 중앙 병합 준비 완료
**언어**: Rust
**위험도**: 낮음

## 요약

PR #1512는 `e2e_crossover_larger_models_benefit_more`에서 호스트 스케줄러 타이밍 잡음을 제거한다. 일반 `run_tp_benchmark` 호출은 기존처럼 실제 경과 시간을 측정하지만, 크로스오버 분석은 주입된 명목 타이머로 hidden-size 모델을 평가한다. 대상 테스트는 정확한 모델 효율과 benefit 플래그를 검증하므로 마이크로초 단위 wall-clock 잡음이 아니라 테스트 이름이 뜻하는 속성을 확인한다.

## 1. 문제

기존 테스트는 `run_crossover_analysis`가 `Instant`로 마이크로초 `spin_wait` 루프를 측정한 뒤 `large.scaling_efficiency >= small.scaling_efficiency`를 비교했다. 부하가 있는 전체 스위트 실행에서는 스케줄러 지연이 2048과 8192 synthetic timing 차이를 압도해 간헐적으로 assertion을 뒤집을 수 있었다. 이슈 스레드는 실제 남은 범위를 이 TP 테스트 하나로 좁혔다. 같은 계열로 처음 언급된 autotune 테스트들은 PR #1096에서 manual timer로 이미 해결되었다.

## 2. 기술 결정

### 2.1 벤치마크 harness 경계에 타이밍 주입

private `BenchmarkTimer` seam이 `wait`와 `measure`를 제공한다. `run_tp_benchmark`는 `RealBenchmarkTimer`를 생성하므로 기존 `Instant`와 `spin_wait` 경로가 유지된다. `run_crossover_analysis`는 `NominalBenchmarkTimer`를 생성하며, 이 타이머는 요청된 duration만큼 synthetic elapsed time을 정확히 전진시킨다.

### 2.2 크로스오버 분석을 결정적으로 유지

크로스오버 분석은 wall-clock 벤치마크가 아니라 model-size 비교다. baseline 및 TP 실행에 명목 simulated duration을 사용하므로 throughput, scaling efficiency, all-reduce overhead, benefit 판정이 호스트 부하와 무관하게 안정적이다.

### 2.3 정확한 모델 출력 검증

대상 e2e 테스트는 이제 2048 hidden-size TP=2 entry의 scaling efficiency가 0.5이고, 8192 hidden-size TP=2 entry의 scaling efficiency가 0.8인지 확인한다. 또한 작은 모델은 break-even이고 큰 모델은 beneficial임을 검증한다. 필요한 entry가 없으면 assertion을 건너뛰지 않고 명시적으로 실패한다.

## 3. 변경 요약

| 항목 | 값 |
|------|----|
| 변경 파일 | 2 |
| Rust 변경량 | 133 insertions, 30 deletions |
| public API 추가 | 없음 |
| production timing path | `run_tp_benchmark`에서 유지 |
| crossover timing path | 결정적 명목 타이머 |

- private real/nominal benchmark timer를 추가했다.
- `run_tp_benchmark`를 signature 변경 없이 real timer로 경유시켰다.
- `run_crossover_analysis`를 nominal timer로 경유시키고 해당 동작을 문서화했다.
- flaky crossover assertion을 정확한 명목 모델 기대값과 benefit 검증으로 교체했다.

## 4. 검증

- `cargo fmt --check`: 통과.
- `git diff --check -- src/distributed/tensor_parallel/benchmark.rs tests/tp_e2e.rs`: 통과.
- `cargo test --profile test-fast --test tp_e2e e2e_crossover_larger_models_benefit_more -- --exact --nocapture`: 통과, 1 passed, 35 filtered out.
- `cargo test --profile test-fast -p mlxcel distributed::tensor_parallel::benchmark::tests::crossover_analysis_basic -- --exact --nocapture`: 통과, matching unit test 1개 통과, 나머지 filtered test는 0 run으로 보고됨.
- Mutation check: crossover compute scaling을 quadratic에서 linear로 임시 변경했다. exact 대상 테스트가 `small-model scaling efficiency 0.6666666666666666 should match the injected nominal model 0.5`로 실패했고, 복원 후 다시 통과했다.
- `cargo clippy --profile test-fast --test tp_e2e -- -D warnings`: 통과.

focused local validation은 `/proc/loadavg`가 `12.14 9.86 6.28`을 보고하던 상태에서 실행되었다. 이 auto-implementation unit이 broad cargo, workspace, serial all-test command를 명시적으로 금지했기 때문에 전체 workspace 또는 full-suite loaded-host 실행은 수행하지 않았다.

## 5. 리뷰 결과

| 항목 | 심각도 | 처리 |
|------|--------|------|
| 대상 assertion이 마이크로초 호스트 스케줄링에 의존함 | Medium | 크로스오버 분석 timing source를 결정적 주입 명목 시간으로 교체 |
| 기대 entry가 없으면 테스트가 조용히 통과할 수 있음 | Low | optional `if let` 대신 명시적 `expect` 검사로 교체 |
| 기대 상수가 caller-configured timing 값으로 오해될 수 있음 | Low | 상수가 hidden-size timing model에서 온다는 주석 추가 |

새로운 보안 문제는 없다. 이 변경은 외부 입력 파싱, I/O, public API, 또는 기존 configured benchmark loop를 넘어서는 unbounded resource use를 추가하지 않는다.

## 6. 남은 제한

- 중앙 병합 전 hosted CI 결과는 GitHub에서 확인해야 한다.
- hardware/model validation은 주장하지 않는다. 이 경로는 simulated tensor-parallel benchmark이며 실제 모델을 로드하지 않는다.
