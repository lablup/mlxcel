# PR #2976: ragged load timing 분리와 decode-only 벤치 처리량 복원

**작성일**: 2026-11-11
**상태**: 완료 (draft PR, #2961 close)

## 요약

보고된 f16 `ragged-context-and-upload` 회귀는 회귀가 아니었고, `0.7 s` 기준값은 재현되지 않는다. `92c07eb8^`(`fd2a65ea`)에서 `Cargo.lock`과 IREE ROCm 배포판을 동일하게 맞춰 빌드한 pre-#2933 control은 같은 구간을 B=4에서 `4.89 s`, B=8에서 `4.93 s`에 로드한다. HEAD는 `5.62 s`와 `5.03 s`로, 둘 다 copy 시간의 실행 간 편차 안에 있다. 비용은 f16 host 확장 자체다. dequantize `1.8`에서 `1.9 s`, f16 pack `1.9 s`, device copy `0.9`에서 `1.9 s`로 146개 tensor의 `2,997,100,544` 바이트를 옮긴다. capacity에 대해 평평하다.

이슈를 촉발한 `20.6`에서 `22.5 tok/s` 수치는 원인이 다르다. #2936이 `xla_batch_bench`의 타이머 안쪽으로 engine 생성을 옮겨, 일회성 weight upload가 분모에 들어갔다. decode만 보면 같은 실행이 `64.0 tok/s`다. 벤치마크는 이제 timed region 밖에서 로드하고 로드 시간을 따로 출력한다.

이를 바로잡으면 #2933 절대 목표는 대부분 충족된다. native B=4 `61.7`(목표 60.0-62.2, 통과), native B=8 `75.8`(목표 74.5-76.7, 통과), f16 B=8 `60.4`(목표 59.8, 통과), f16 B=4 `64.0` 대 `65.6`은 2.4퍼센트 부족이며 그대로 보고한다.

## 1. 문제 정의

#2961은 f16 ragged 로드가 약 `4.8 s`인데 pre-#2933 binary는 약 `0.7 s`로 보였고, f16 arm이 B=4에서 `20.6`에서 `22.5 tok/s`로 #2933의 `65.6`에 못 미쳤기 때문에 열렸다. 두 로드 함수(`upload_weights_with_sink`, `create_ctx_with_diagnostics`)는 `92c07eb8^`와 바이트 동일하므로, 이슈는 추측이 아니라 측정을 요구했다. phase를 나누고, shape 스케일링을 시험하고, 업로드 횟수를 세고, control을 정직하게 재현할 것.

## 2. 측정

### 2.1 phase 분리

`report_load_timing`이 이제 `language-module-load`(instance, device, session, vmfb)와 `language-weight-upload`를 따로 보고하고, `language-weight-upload-parts`가 `tensors`, `uploaded_bytes`, `dequantize_ms`, `pack_ms`, `copy_ms`를 덧붙인다. capacity bucket은 base context의 weight를 공유하므로 parts 줄을 0 카운터로 출력하며, 이는 device test로도 고정했다.

3회 실행의 중앙값, `gfx1151`, 표기한 경우를 제외하면 capacity 256:

| Arm | Cap | B | Load s | Module s | Upload s | Dequant s | Pack s | Copy s | Tensors | Bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HEAD f16 | 256 | 4 | 5.62 | 0.20 | 5.37 | 1.88 | 1.91 | 1.56 | 146 | 2,997,100,544 |
| HEAD f16 | 256 | 8 | 5.03 | 0.20 | 4.83 | 1.88 | 1.94 | 0.99 | 146 | 2,997,100,544 |
| HEAD f16 | 1024 | 4 | 4.92 | 0.20 | 4.71 | 1.80 | 1.91 | 0.98 | 146 | 2,997,100,544 |
| HEAD f16 | 1024 | 8 | 4.94 | 0.20 | 4.74 | 1.83 | 1.91 | 0.97 | 146 | 2,997,100,544 |
| HEAD native | 256 | 4 | 1.17 | 0.21 | 0.95 | 0.45 | 0 | 0.46 | 370 | 1,598,300,160 |
| HEAD native | 256 | 8 | 1.15 | 0.21 | 0.93 | 0.41 | 0 | 0.43 | 370 | 1,598,300,160 |
| Control f16 | 256 | 4 | 4.89 | - | - | - | - | - | - | - |
| Control f16 | 256 | 8 | 4.93 | - | - | - | - | - | - | - |

control은 새 라벨을 출력할 수 없고, 같은 구간을 감싸는 내부 `language-vmfb-and-device-upload`를 쓴다. harness는 ragged engine을 두 번 로드하므로 phase를 구분했다. phase 1은 단일 시퀀스 reference engine(147 tensor, `3,522,437,120` 바이트, upload 약 `12.6 s`), phase 2는 ragged batch engine이며 `ragged-context-and-upload`를 출력하는 유일한 로드다.

### 2.2 수치가 말하는 것

- module load는 모든 행에서 `0.20 s`이므로 vmfb 인스턴스화가 비용이 아니다.
- upload는 capacity와 bucket에 대해 평평하므로 shape 의존 재업로드가 아니다.
- 업로드 횟수는 선언된 spec당 한 번이고, bucket context는 호출 0회와 바이트 0을 보고한다.
- f16 upload는 weight당 2바이트로 `3.0 GB`이고 native packed word는 `1.6 GB`다. host 작업(dequantize와 pack)이 `5.4 s` upload 구간 중 `3.8 s`를 차지한다. 이것이 비용이며 #2933 이후의 변화가 아니다.

## 3. 처리량 수치

#2936과 이 PR 사이의 `xla_batch_bench`는 `let t_run = Instant::now();` 안쪽에서 batch engine을 생성했다. 로드가 창에 들어가면 f16 B=4가 이슈의 수치를 재현한다. 중앙값 `20.8 tok/s`(`19.5`, `20.8`, `22.4`)다. decode만 보면 같은 실행이 `64.0`이다. 벤치마크는 먼저 로드하고 `phase 2 load: ... <n> s`를 따로 출력하므로, `tok/s`와 `BENCH` 줄이 이제 decode를 서술한다.

## 4. #2933 목표 상태

decode-only, 3회 실행, 중앙값:

| Arm | B | #2933 목표 | 실행 | 중앙값 | 상태 |
|---|---:|---|---:|---:|---|
| f16 | 4 | 65.6 | 66.6, 61.2, 64.0 | 64.0 | 2.4퍼센트 부족, 깨끗한 통과 아님 |
| f16 | 8 | 59.8 | 59.9, 60.4, 65.0 | 60.4 | 통과 |
| native | 4 | 60.0-62.2 | 59.8, 61.7, 70.7 | 61.7 | 통과 |
| native | 8 | 74.5-76.7 | 69.6, 75.8, 80.9 | 75.8 | 통과 |

같은 프로토콜의 control 기준점: f16 B=4 중앙값 `63.4`(`60.0`, `63.4`, `63.7`), B=8 중앙값 `63.5`(`56.3`, `63.5`, `71.2`). host 편차는 5에서 15퍼센트이고 완만한 상승 경향이 있어, 중앙값 하나가 아니라 모든 실행을 보고한다.

모든 native 실행과 모든 f16 B=8 실행이 `REFERENCE-EXACT PASS`를 출력한다. f16 B=4 arm은 합성 random-id prompt artifact에서 request 2, token 3에서 발산하며, HEAD의 세 실행과 control의 세 실행 모두 그렇다. 같은 shape의 자연어 artifact로 반복하면 `REFERENCE-EXACT PASS`이며 `63.1 tok/s`다. 따라서 이 발산은 prompt 특유의 근접 동률이지 경로 회귀가 아니며, 숨기지 않고 기록한다.

## 5. 결정

f16 ragged engine은 f16-resident weight를 유지한다. host dequantize와 pack의 `3.8 s`를 없애려면 packed U32를 올리고 device에서 dequantize해야 하며, 이는 자체 정확성 표면을 가진 별도 ragged 그래프라 이 이슈 범위 밖이다. 측정은 로드가 decode 목표 미달의 원인이 아니고 벤치마크 창이 원인이었음을 보여준다.

## 6. 검증

- `make verify-fmt` 통과.
- `cargo clippy -p mlxcel-xla --features iree --lib --tests -- -D warnings` 통과.
- `cargo test -p mlxcel-xla --lib` 407 passed, 19 ignored.
- `cargo test --release -p mlxcel-xla --lib --features iree load_timing_accounting_tests` 3 passed.
- `cargo test --test llama_compat_manifest` 4 passed.
- `make verify-env-var-docs` 15 checks 통과.
- `cargo test --release --features xla-iree,test-utils --test xla_ragged_hardening base_load_uploads_each_weight_once_and_a_bucket_uploads_none -- --ignored` 1 passed. 선언된 spec당 업로드 1회, bucket은 호출과 바이트 0을 확인한다.
- 구성: `MLXCEL_XLA_CONTEXT_CAPACITY` 256과 1024, B=4와 B=8, 양쪽 arm의 모든 실행을 `scripts/rocm_gpu_guard.sh --hold` 아래에서 수행했다.

## 7. 변경 요약

| 항목 | 값 |
|------|-----|
| 변경 파일 | 7 |
| 추가 줄 | +352 |
| 삭제 줄 | -44 |

관련: #2961(이 이슈), #2933(목표), #2936(벤치마크 창 변경), #2955(불일치를 발견한 보고).
