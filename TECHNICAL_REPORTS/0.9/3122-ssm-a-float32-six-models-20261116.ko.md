# 기술 리포트: PR #3122 - 여섯 모델의 SSM A를 A_log에서 float32로 계산

**작성일**: 2026-11-16
**상태**: PR 열림 (#3057 해결)
**리뷰어**: 독립 리뷰, CRITICAL/HIGH 없음. MEDIUM 테스트 공백 하나는 PR 안에서 수정, MEDIUM CUDA 성능 지적 하나는 후속으로 넘김
**언어**: Rust
**위험도**: Low

---

## 요약

selective state space 모델은 `A = -exp(A_log)`를 계산한다. float16에서는 `A_log = 11.09`를 넘으면 `exp`가 넘쳐 `-inf`가 되고, time step이 0이면 `0 * -inf = NaN`이 되어 logits가 NaN이 되고 greedy decoding이 토큰 0만 낸다. #3053이 Mamba에서 이를 고쳤고, 이 PR은 같은 규칙을 Jamba, PLaMo2, Falcon-H1, GraniteMoeHybrid, Mamba2, Nemotron-H에 적용한다. `exp`는 항상 `A_log`의 float32 사본에서 돈다. float32 로드는 비트 단위로 동일하고, bf16 로드는 `exp`의 bf16 반올림만큼만 달라진다.

---

## 1. 문제 정의

### 1.1 배경

체크포인트의 `A_log`는 최대 19.5 정도다. 로드 시 dtype 정책이 bf16을 f16으로 바꾼 경우(또는 f16 체크포인트) `exp(A_log)`가 65504를 넘으면 표현할 수 없다. `mamba2.rs`와 `ssd_scan.rs`처럼 `exp` 뒤에 float32로 캐스트하면 이미 넘친 뒤라 늦다.

### 1.2 기존 문제점

- **여섯 모델이 로드된 dtype으로 `exp`를 계산했다.** Jamba는 직접, 나머지 다섯은 공유 SSD 스캔(`chunked_ssd_scan`)이나 Mamba2의 `ssm_attn`을 통해서다.
- **실패가 조용하다.** 에러 없이 EOS 토큰만 이어진다.

---

## 2. 기술적 선택과 그 이유

### 2.1 세 모델에 `A`를 저장하지 않고 공유 스캔을 고침

**컨텍스트:** 이슈는 `a_log` 필드를 가진 네 모델에서 로드 시 `A`를 저장하라고 했다. 이슈 작성 이후 PLaMo2, Falcon-H1, GraniteMoeHybrid의 prefill은 `chunked_ssd_scan`으로 옮겨졌고, decode는 `A_log`를 `ssm_update_kernel`에 넘기는데 이 커널은 이미 float32로 `exp`를 계산한다(`mlx_cxx_kernels.cpp`).

| 선택지 | 장점 | 단점 |
|---|---|---|
| 네 모델 모두 `A` 저장 | 이슈 문구와 일치 | `A_log`를 받는 decode 커널이 깨짐 |
| **선택: `chunked_ssd_scan`과 `ssm_attn`에서 float32 `exp`, `A` 저장은 Jamba만** | 변경 하나로 다섯 모델과 커널 미지원 시 decode fallback까지 해결 | forward마다 head별 캐스트와 `exp` 하나 |

`ssm_a_from_log`를 둘로 나눴다. `ssm_a_f32`는 지연 그래프(캐스트, `exp`, 부호 반전)를 만들고, `ssm_a_from_log`는 그것을 호출한 뒤 eval한다. forward 경로는 `ssm_a_f32`를 써서 sync가 늘지 않고, 로드 시점 코드는 `ssm_a_from_log`를 쓴다. 로직은 여전히 한 벌이다.

### 2.2 Jamba는 f16을 빼고 activation dtype으로 `A`를 읽음

Jamba는 `MambaBlock::scan_a`를 따른다. `A`를 activation dtype으로 캐스트하되, f16 activation이면 float32로 둔다. graph scan은 (0, 1] 범위인 decay `exp(delta * A)`만 f16으로 되돌리므로(`decay_in_activation_dtype`) recurrent state는 activation dtype을 유지한다.

**트레이드오프:** float32 `A_log`와 bf16 가중치가 섞인 체크포인트는 예전엔 `A`와 decay를 float32로 계산했지만 이제 bf16으로 반올림한다. #3053 이후 Mamba와 같은 동작이다.

---

## 3. 리뷰 결과

- **MEDIUM, PR에서 수정:** f16 activation의 Jamba를 거치는 테스트가 없어서 `A`를 다시 f16으로 캐스트하는 회귀가 통과할 수 있었다. `jamba_f16_model_with_an_overflowing_a_log_keeps_finite_logits`가 fused scan과 강제 graph scan을 모두 덮는다.
- **MEDIUM, 후속:** CUDA에서 f16 activation이면 graph-exact Mamba1 scan 포트가 float32 `A`를 거부해, f16 Jamba(그리고 #3053 이후 Mamba)가 스텝별 graph scan으로 떨어진다. Metal과 ROCm은 float-state 포트를 써서 영향이 없다. CUDA 장비가 없어 측정하지 못했다.
- **LOW, 후속:** `ssd_scan.rs`가 `mamba.rs`에서 helper를 가져온다. 두 helper를 공유 SSM 모듈로 옮기는 편이 깔끔하다.

---

## 4. 변경 요약

| 항목 | 값 |
|---|---|
| 변경 파일 | 13 (소스 5, 테스트 8) |
| 추가 테스트 | 8 |

- `mamba.rs`: `ssm_a_from_log`에서 `ssm_a_f32` 분리.
- `ssd_scan.rs`, `mamba2.rs`: `A`를 `ssm_a_f32`로 계산.
- `jamba.rs`: 로드 시 float32 `A` 저장, `scan_a`와 `decay_in_activation_dtype`.
- 테스트: 모델마다 19.5까지 오르는 f16 `A_log`와 0 step에서 logits가 유한한지(해당 모델은 f32와 동일한지) 확인, `ssd_scan_tests.rs`의 허용오차 테스트, Jamba 전체 f16 테스트, 공유 fixture `ssm_test_support.rs`(`#[cfg(test)]`).

### 검증

- Mutation: `ssm_a_f32`의 float32 캐스트를 빼면 `ssm_a_` 테스트 8개가 모두 실패하고, Jamba f16 분기의 `A`를 f16으로 캐스트하거나 decay를 되돌리지 않으면 Jamba f16 테스트가 실패한다.
- `cargo test --lib models::`: 2840 통과, 0 실패, 96 ignored. clippy, fmt 통과.

---

## 5. 후속 조치

- CUDA graph-exact Mamba1 포트가 f16 모델의 float32 `A`를 받는 방식을 정하고, pre-Ampere GPU에서 fallback 비용을 측정한다.
- `ssm_a_f32`와 `ssm_a_from_log`를 공유 SSM 모듈로 옮긴다.
