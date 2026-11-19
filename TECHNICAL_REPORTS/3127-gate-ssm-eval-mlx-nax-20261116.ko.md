# 기술 리포트: PR #3127 - SSM eval 경계를 MLX의 NAX 플래그로 판단

**작성일**: 2026-11-16
**상태**: PR 열림 (#3123 해결), M3 Ultra와 M5 Max에서 검증
**언어**: Rust
**위험도**: Low

---

## 요약

Nemotron-H와 PLaMo 2는 M5 Max의 NAx GEMM이 float32/float16 지연 그래프를 NaN으로 융합하는 것을 막으려고 SSM mixer마다 `eval`을 강제한다. 이 회피는 mlxcel이 칩을 판단하는 방식(`is_m5_neural_accelerator()`, sysctl brand string)으로 켜졌지만, MLX는 `metal::is_nax_available()`로 NAx 커널 사용 여부를 정한다. 이 PR은 회피와, 예전 경계를 대신하는 finite-decode 테스트를 캐시된 helper `hardware::mlx_nax_available()`을 통해 MLX의 플래그로 판단하게 한다.

---

## 1. 문제 정의

#2806에서 두 판단이 어긋나는 것이 드러났다. 러너 두 대에서 window SDPA dispatch 테스트가 nightly마다 실패했는데, 테스트는 brand string을, MLX는 NAx 플래그를 봤기 때문이다. PR #3110은 그 테스트만 고쳤다. 프로덕션 코드는 여전히 brand string으로 NaN 회피를 켜고 있어서, MLX가 NAx를 쓰는데 brand string이 M5로 인식되지 않는 호스트에서는 회피가 꺼진 채 NaN logits가 나올 수 있었다. 반대로 `MLX_METAL_GPU_ARCH`나 `MLX_METAL_NO_NAX` 빌드로 NAx를 안 쓰는 M5에서는 레이어마다 불필요한 sync를 냈다.

---

## 2. 기술적 선택과 그 이유

### 2.1 MLX가 실제로 고르는 커널 기준

NaN은 MLX가 고른 커널에서 생기므로 판단도 MLX의 플래그를 따른다. `is_m5_neural_accelerator()`는 칩 수준 판단으로 남기고, doc comment에 두 값이 달라지는 경우(`MLX_METAL_GPU_ARCH`, `MLX_METAL_NO_NAX`, Metal 외 백엔드, 인식되지 않는 brand string)를 적었다.

### 2.2 모델 코드의 FFI 대신 캐시 helper

`mlx_nax_available()`은 #3110의 bridge를 `OnceLock<bool>`로 감싼다. MLX가 답을 캐시하므로 프로세스 안에서 값이 바뀌지 않고, forward당 비용은 atomic load 하나다. Metal이 아니면 false를 돌려준다. default device가 아니라 GPU 능력을 보고하므로 NAx 호스트에서 CPU device로 돌려도 `eval`이 켜지는데, sync 비용만 들고 틀리지는 않는다.

### 2.3 "두 판단 일치" 테스트는 넣지 않음

이슈는 일치를 단언하는 테스트와 차이를 문서화하는 방법 중 하나를 허용했다. 테스트를 넣으면 #2806 러너에서 코드 결함 없이 nightly가 빨개지므로 doc comment를 택했다.

---

## 3. 변경 요약

| 항목 | 값 |
|---|---|
| 변경 파일 | 4 |
| 추가 테스트 | 2 (`hardware::tests`) |

- `hardware.rs`: `mlx_nax_available()` 추가, `is_m5_neural_accelerator()` doc 보강.
- `nemotron_h.rs`, `plamo2.rs`: `eval` 경계를 `mlx_nax_available()`로 판단.
- `tests/mamba2_hybrid_decode_finite.rs`: MLX가 NAx를 안 쓰는 곳에서만 NOT APPLICABLE.

### 검증

- M3 Ultra: `hardware::` 45 통과, `MLX NAX false; sysctl view M3` 출력. finite-decode 테스트는 예상대로 NOT APPLICABLE. clippy, fmt, contract check 통과.
- M5 Max: `hardware::` 45 통과, `MLX NAX true; sysctl view M5, Neural Accelerator true, macOS NA support true` 출력. M5 Max에서는 이전과 같이 회피가 켜진다. `mamba2_hybrid_decode_finite`는 NOT APPLICABLE 없이 실행되어 2/2 통과했다. Falcon-H1 tiny와 Granite 4.0-H tiny가 각각 64개 토큰을 유한하고 정상적인 문장으로 생성했다.

---

## 4. 후속 조치

- #2806의 러너 두 대 nightly를 지켜본다. MLX가 NAx를 쓰면 이제 거기서도 회피가 켜진다.
