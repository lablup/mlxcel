# 기술 보고서: PR #1514 - KV Binary Byte Units

**작성일**: 2026-08-30
**상태**: 열린 PR, 중앙 병합 준비 완료
**언어**: Rust
**위험도**: Low

## 요약

PR #1514는 KV-transfer benchmark에서 1024 단위로 나눈 byte-size 값이 binary unit 이름으로 표시되도록 수정한다. 변경 범위는 display string과 정확한 formatter assertion으로 제한되며, benchmark measurement arithmetic, transfer strategy behavior, throughput unit, model execution path는 변경하지 않았다.

## 1. 문제 정의

KV-transfer benchmark는 byte count를 각 tier에서 `1024`로 나누면서도 결과를 `GB`, `MB`, `KB`로 표시했다. 이 label은 decimal unit을 뜻하지만 계산은 binary 기준이었다. 이 불일치는 benchmark report string에 영향을 주며, KV cache size와 transfer summary를 읽는 사람이 단위를 잘못 해석할 수 있다.

이슈 스레드에는 comment가 없었으므로 issue body가 전체 specification이었다. 요청된 수정은 label을 `GiB`, `MiB`, `KiB`로 바꾸고 formatter test를 업데이트하는 것이었다.

## 2. 기술적 선택과 그 이유

### 2.1 Measurement가 아니라 display label만 수정

변경은 label에만 적용했다. Byte count, benchmark input, compression ratio, timing, throughput calculation은 그대로 유지했다. 이렇게 하면 measurement behavior는 보존하면서 rendering unit만 기존 arithmetic과 맞출 수 있다.

### 2.2 Benchmark module 안의 두 byte-size formatter를 함께 정리

private `format_bytes` helper가 issue의 직접 대상이었다. 같은 benchmark module의 `TransferBenchConfig::total_size_str`도 동일하게 1024 기반 formatting pattern을 사용했으므로 같은 report-string defect로 수정했다. `MB/s` throughput label은 byte-size formatter가 아니며 이 이슈 범위 밖이므로 유지했다.

### 2.3 Substring check를 exact assertion으로 교체

기존 테스트는 `KB`, `MB`, `GB`가 포함되어 있으면 통과했기 때문에 잘못된 label을 놓칠 수 있었다. 이제 테스트는 `2.0KiB`, `5.0MiB`, `2.0GiB`, `256.0 MiB` 같은 정확한 output을 assertion한다.

## 3. 변경 요약

| 항목 | 값 |
|------|----|
| 변경 파일 | implementation/test 2개 및 reports |
| source/test delta | 11 insertions, 11 deletions |
| Public API 변경 | 없음 |
| Measurement logic 변경 | 없음 |
| Hardware/model requirement | 없음 |

- `TransferBenchConfig::total_size_str` label을 `GB`/`MB`/`KB`에서 `GiB`/`MiB`/`KiB`로 변경했다.
- private KV-transfer `format_bytes` helper label을 `GB`/`MB`/`KB`에서 `GiB`/`MiB`/`KiB`로 변경했다.
- `bench_config_total_size_str`가 정확한 `256.0 MiB` output을 assertion하도록 수정했다.
- `format_bytes_ranges`가 byte, KiB, MiB, GiB output을 정확히 assertion하도록 수정했다.

## 4. 검증

- `cargo fmt --check`: 통과.
- `git diff --check -- src/distributed/kv_cache_transfer/benchmark.rs src/distributed/kv_cache_transfer/benchmark_tests.rs`: 통과.
- `cargo test --profile test-fast -p mlxcel distributed::kv_cache_transfer::benchmark::tests::format_bytes_ranges -- --exact --nocapture`: 통과, matching test 1개 통과, 나머지 test는 filtered out.
- `cargo test --profile test-fast -p mlxcel distributed::kv_cache_transfer::benchmark::tests::bench_config_total_size_str -- --exact --nocapture`: 통과, matching test 1개 통과, 나머지 test는 filtered out.
- `cargo clippy --profile test-fast -p mlxcel --lib -- -D warnings`: 통과.

이 wave-runner unit이 broad cargo와 workspace command를 금지했기 때문에 broad workspace test는 실행하지 않았다. 이 PR은 benchmark display string과 formatter assertion만 변경하므로 hardware/model validation은 해당하지 않는다.

## 5. 리뷰 결과

| 항목 | 심각도 | 처리 |
|------|--------|------|
| 1024 기반 byte-size 값이 decimal unit label로 표시됨 | Low | binary label `KiB`, `MiB`, `GiB` 사용 |
| Formatter test가 substring check라 label drift를 놓칠 수 있음 | Low | 관련 formatter output에 exact assertion 사용 |

새로운 security issue는 없다. 이 변경은 input parsing, I/O, unsafe code, public API surface, concurrency, caching, resource-management behavior를 추가하지 않는다.

## 6. 남은 제한

- 중앙 병합 전 hosted CI 상태는 GitHub에서 확인해야 한다.
- 이 PR은 `MB/s` 같은 unrelated throughput label을 표준화하지 않는다. 해당 unit은 byte-size formatter 범위 밖이다.
