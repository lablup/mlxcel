# 기술 보고서: PR #1574 - fix(mamba): apply Falcon-Mamba B/C/dt RMS norm once, no per-call ones

**작성일**: 2026-09-02
**작성자**: mlxcel maintainers
**리뷰어**: 지정 전
**상태**: 완료
**언어**: Rust
**위험도**: Low

---

## 요약

`MambaBlock::ssm_step`은 `delta`, `B`, `C`를 호출마다 두 번씩 정규화했고, 이 여섯 번의 호출마다 매번 새 `ones` 가중치 배열을 할당했다. prefill 중에는 레이어당 토큰마다 이 호출이 발생한다. 이 PR은 브리지에 이미 존재하는 `fast_rms_norm_no_weight`를 통해 텐서당 정확히 한 번만 weight 없는 RMS norm을 적용하도록 고쳤으며, 이는 업스트림 mlx-lm 레퍼런스와 일치한다. 더 이상 쓰이지 않는 `ones` 할당 헬퍼도 삭제했다.

---

## 1. 문제 정의

### 1.1 배경

`falcon_mamba` 체크포인트는 `use_bcdt_rms = true`를 설정하는데, 이는 `x_proj(x)`에서 분리한 세 텐서 `delta`, `B`, `C` 각각에 weight 없는 RMS norm을 한 번씩 적용해야 한다는 뜻이다. SSM 스캔은 토큰을 하나씩 처리하므로, 이 호출은 프롬프트 길이가 `T`일 때 레이어당 `T`번 실행되며, 7B 체크포인트 기준 64개 레이어에 걸쳐 반복된다.

### 1.2 기존 문제점

- **이중 정규화**: `ssm_step`이 `delta`, `B`, `C` 각각에 대해 `self.mixer_norm(&self.mixer_norm(&x))`를 호출해, 아키텍처가 요구하는 것보다 norm 실행 횟수가 두 배였다. 이 결과는 단순히 "같은 답을 두 배 비용으로" 얻는 것이 아니다. 이미 거의 unit RMS에 가까운 텐서를 다시 정규화하면 대략 `1/sqrt(1+eps)` 배만큼 다시 스케일이 줄어들기 때문에, 두 번째 적용은 올바른 값 위에 얹힌 작은 추가적 수치 왜곡이지 진짜 no-op이 아니다.
- **호출마다 할당**: 기존 `rms_norm_no_scale` 헬퍼는 weight가 있는 `fast_rms_norm` 커널을 재사용하기 위한 목적만으로, 호출마다 텐서 마지막 차원 크기의 `ones` 배열을 새로 만들었다. 브리지에는 이런 할당이 필요 없는 weight 없는 버전(`fast_rms_norm_no_weight`)이 이미 있고, 다른 모델 계열(`gemma3n.rs`, `gemma4.rs`, `falcon_ocr.rs`)은 이미 이를 사용 중이다.

### 1.3 위험성

| 위험 | 영향도 | 발생 가능성 |
|-----|-------|-----------|
| 레이어당 토큰마다 추가되는 norm 커널 실행이 긴 프롬프트의 디코드 지연을 늘림 | Medium | 확실 (falcon_mamba의 모든 forward pass에서 발생) |
| 호출마다 반복되는 `ones` 할당이 프롬프트 길이에 비례해 lazy graph 노드 수를 늘림 | Low-Medium | 확실 |
| 불필요한 두 번째 norm 적용으로 인한 작은 수치적 편차 | Low | 존재하지만 범위가 제한됨 (`eps = 1e-6`에서 `~1/sqrt(1+eps)` 배) |

---

## 2. 기술적 검토 사항

### 2.2 성능 관점

**검토 항목:**
- [x] 알고리즘 복잡도: `ssm_step` 호출당 norm 실행 6회 중 3회, `ones` 할당 6회 전부 제거
- [ ] 쿼리 최적화: 해당 없음
- [ ] 캐싱 전략: 해당 없음
- [x] 메모리 사용: 호출당 임시 배열 할당 제거

**성능 영향:**

| 영역 | Before | After | 개선율 |
|-----|--------|-------|-------|
| `ssm_step` 호출당 norm 커널 실행 | 6회 (텐서당 2회 x 3개 텐서) | 3회 (텐서당 1회) | norm 실행 50% 감소 |
| `ssm_step` 호출당 `ones` 할당 | 6회 | 0회 | 100% 제거 |
| falcon_mamba(`use_bcdt_rms=true`)에서 레이어당 토큰마다 추가되는 lazy graph 노드 | `ones` 6개 + norm 노드 6개 | norm 노드 3개 | 토큰/레이어당 노드 9개 감소 |

이 PR 범위 안에서 end-to-end 처리량 벤치마크는 실행하지 않았다. `models/falcon-mamba-7b-4bit`에서의 greedy 출력 일치 및 peak 메모리 비교를 통한 실제 체크포인트 검증은 이 PR의 단위 테스트 범위 밖의 후속 조치다.

### 2.3 호환성/의존성 관점

- **Breaking Changes**: 없음. `mixer_norm`의 시그니처와 호출부는 그대로이며, 내부 구현과 `ssm_step`이 이를 몇 번 호출하는지만 바뀌었다.
- **새로운 의존성**: 없음. `fast_rms_norm_no_weight`는 `mlxcel-core` 브리지(`src/lib/mlxcel-core/src/lib.rs`)에 이미 존재했고, 다른 세 모델 계열에서 이미 사용 중이다.
- **호환성**: `src/models/mamba.rs`로 범위가 한정되며, 이 파일은 Falcon-Mamba와 Mamba v1만 다룬다. `mamba2.rs`, `falcon_h1.rs`, `jamba.rs`, `granitemoehybrid.rs`, `plamo2.rs`는 `use_bcdt_rms` 필드가 없어 영향받지 않는다.

### 2.4 코드 품질 관점

- **테스트 커버리지**: `src/models/mamba_tests.rs`에 단위 테스트 2개 추가, 둘 다 수정된 코드 경로를 검증하며 통과.
- **코드 복잡도**: 감소. 10줄짜리 `rms_norm_no_scale` 헬퍼를 삭제했고, `ssm_step`의 정규화 세 줄이 중첩된 이중 호출에서 단일 호출과 갱신된 한 줄 주석으로 바뀌었다.
- **기술 부채**: 감소. (버그였던) 이중 적용 동작을 설명하던 주석도 그 동작과 함께 제거했다.

---

## 3. 기술적 선택과 그 이유

### 3.1 회귀 테스트 판정 기준으로 허용 오차가 아닌 완전 일치 사용

**컨텍스트:**

이슈 자체의 진단에 따르면 weight 없는 RMS norm을 두 번 적용하는 것은 "수치적으로 거의 no-op"에 가깝다. 이미 거의 unit RMS인 텐서를 다시 정규화하면 `1/sqrt(1+eps)` 배만큼 바뀌는데, `eps = 1e-6`일 때 그 크기는 `5e-7` 수준이다. "한 번"과 "두 번" 적용 결과를 부동소수점 허용 오차(예: `atol = 1e-5`인 `allclose`)로 비교하는 안이한 회귀 테스트는, 이중 적용 버그가 다시 들어와도 그 차이가 대부분의 합리적인 허용 오차보다 작기 때문에 십중팔구 통과해버린다.

**고려한 대안:**

| 옵션 | 장점 | 단점 |
|-----|-----|-----|
| Option A: "한 번" vs "두 번" 결과를 `allclose` 기반 허용 오차로 비교 | 단순하고 코드베이스의 다른 곳에 있는 기존 테스트 스타일과 일치 | 실제 회귀를 잡기에는 정밀도가 부족함. `1e-5` 수준의 허용 오차는 `~5e-7` 크기의 차이를 대부분 흡수해버림 |
| Option B: 이슈에 스케치된 대로, 손으로 계산한 기대값을 사용하는 `ssm_step` 수준의 전체 수치 테스트 | 실제 호출 경로를 처음부터 끝까지 검증함 | 실제 `x_proj`/`dt_proj` 가중치를 가진 완전한 `MambaBlock`을 구성하고 softplus/state-update 계산을 손으로 따라가야 함. 같은 회귀 커버리지를 위해 테스트 코드가 훨씬 많이 필요함 |
| **선택: Option C: `mixer_norm` 경계에서 `array_equal`(완전 일치) 사용** | 두 결과가 수치적으로 아무리 가깝더라도, 두 번째 커널 실행이 최소 ULP 수준의 차이를 만들어내므로 "한 번 적용"과 "두 번 적용"을 확실하게 구별함. 테스트 준비 코드도 최소화됨 | `ssm_step`의 전체 수치 파이프라인(softplus, state update)은 검증하지 않음. 서로 다른 두 번의 커널 실행 결과가 비트 단위로 동일하지 않다는 가정에 의존하며, 로컬 검증에서는 이 가정이 성립했음 |

**선택 이유:**

`array_equal`은 허용 오차가 아니라 정확한(완전 일치) 비교를 수행한다. "이중 적용" 버그가 "단일 적용" 결과와 수치적으로는 가깝지만 비트 단위로는 동일하지 않은 결과를 만들어내기 때문에, 완전 일치 비교야말로 두 경우를 실제로 구별해내는 방법이다. `mixer_norm(x)`(수정된 코드)는 `fast_rms_norm_no_weight(x, eps)` 한 번 호출 결과와 `array_equal`임을 단언하고, 그 위에 `fast_rms_norm_no_weight`를 한 번 더 적용한 결과와는 `array_equal`이 **아님**을 명시적으로 단언한다. 만약 누군가 `self.mixer_norm(&self.mixer_norm(&x))`를 다시 들여온다면, 첫 번째 단언이 바로 이를 잡아낸다. 그때 블록의 `mixer_norm` 출력은 "단일 적용" 참조값이 아니라 "이중 적용" 참조값과 일치하게 되기 때문이다.

**트레이드오프:**

이 테스트는 전체 `ssm_step` 파이프라인이 아니라 `mixer_norm` 메서드 경계에서 동작하므로, norm 이후의 softplus/state-update 계산을 독립적으로 검증하지는 않는다. 이 부분의 커버리지는 이번 수정 범위 밖이며(해당 하류 계산은 변경하지 않았음), 별도의 실제 체크포인트 검증 단계에서 다뤄진다.

### 3.2 새 `pub(crate)` API 노출 대신 최소한의 `MambaBlock` 테스트 픽스처 사용

**컨텍스트:**

이슈는 테스트에서 내부에 접근할 수 없다면 `MambaBlock::normalize_bcdt(...)`를 `pub(crate)`로 노출하라고 제안했다. `mamba_tests.rs`는 `mamba.rs` 안에서 `#[path = "mamba_tests.rs"] mod tests;`로 연결되어 있어 `mamba` 모듈의 자식 모듈이 된다. Rust의 모듈 기반 가시성 규칙에 따라 `mixer_norm`을 포함한 `MambaBlock`의 private 필드와 메서드는 이미 그곳에서 보인다.

**선택 이유:**

이런 가시성 덕분에 새로운 `pub(crate)` 노출은 필요하지 않았다. `tiny_mamba_block(use_bcdt_rms, mixer_rms_eps)` 헬퍼는 private struct literal을 통해 `MambaBlock`을 구성하며, `mixer_norm`이 읽지 않는 필드들(`conv_weight`, `in_proj`, `x_proj`, `dt_proj`, `out_proj`, `a_log`, `d_param`)에는 0으로 채운 플레이스홀더 텐서를 사용한다. 이렇게 하면 `mixer_norm`의 구현을 이슈에 명시된 그대로(`&self`를 받는 두 갈래 분기 메서드) 유지하면서도, 보일러플레이트를 최소화한 직접적인 단위 테스트가 가능해진다.

---

## 4. 구현 상세

### 4.2 주요 코드 변경

**파일: `src/models/mamba.rs`**
```rust
// 변경 전
fn rms_norm_no_scale(x: &MlxArray, eps: f32) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    let last_dim = shape[shape.len() - 1];
    let ones = mlxcel_core::ones(&[last_dim], mlxcel_core::array_dtype(x));
    mlxcel_core::fast_rms_norm(x, &ones, eps)
}

impl MambaBlock {
    fn mixer_norm(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        if self.use_bcdt_rms {
            rms_norm_no_scale(x, self.mixer_rms_eps)
        } else {
            mlxcel_core::copy(x)
        }
    }
    // ...
    fn ssm_step(&self, ...) -> ... {
        // ...
        let delta_normed = self.mixer_norm(&self.mixer_norm(&delta_raw));
        let b_normed = self.mixer_norm(&self.mixer_norm(&b_raw));
        let c_normed = self.mixer_norm(&self.mixer_norm(&c_raw));
        // ...
    }
}

// 변경 후
impl MambaBlock {
    fn mixer_norm(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        if self.use_bcdt_rms {
            mlxcel_core::fast_rms_norm_no_weight(x, self.mixer_rms_eps)
        } else {
            mlxcel_core::copy(x)
        }
    }
    // ...
    fn ssm_step(&self, ...) -> ... {
        // ...
        let delta_normed = self.mixer_norm(&delta_raw);
        let b_normed = self.mixer_norm(&b_raw);
        let c_normed = self.mixer_norm(&c_raw);
        // ...
    }
}
```

**변경 이유:** 자유 함수 `rms_norm_no_scale`은 weight가 있는 `fast_rms_norm` 커널을 weight 없는 호출로 바꾸기 위해 `ones` 배열을 만드는 용도로만 존재했다. `fast_rms_norm_no_weight`는 이미 브리지에서 이 동작을 네이티브로 제공한다. 중첩된 `mixer_norm(&mixer_norm(&x))` 호출은 아키텍처(그리고 업스트림 mlx-lm 레퍼런스)가 정확히 한 번만 적용하라고 요구하는 곳에서 norm을 두 번 적용하고 있었다.

---

## 5. 학습 포인트

### 5.1 Weight 없는 fast-norm 브리지 호출

**개념:**

`mlxcel-core`의 FFI 브리지는 weight가 있는 `fast_rms_norm(x, weight, eps)`와 나란히 `fast_rms_norm_no_weight(x, eps)`를 제공하며, C++ 레이어에서는 `mlx::core::fast::rms_norm(x, std::nullopt, eps)`로 구현되어 있다. 모델 아키텍처가 학습된 스케일 없는 norm을 필요로 할 때, 전용 브리지 함수를 호출하면 weight가 있는 커널의 시그니처를 맞추기 위해 `ones` weight 배열을 만들 필요가 없다.

**이 PR에서의 적용:**

`MambaBlock::mixer_norm`은 이제 `ones` 배열을 합성해서 `fast_rms_norm`을 호출하는 대신 `fast_rms_norm_no_weight`를 직접 호출한다.

**일반적인 사용 사례:**
- 학습된 스케일이 없는 토큰별/스텝별 정규화에서, 호출마다 `ones` 배열을 할당하면 피할 수 있는 lazy graph 노드가 늘어나는 경우 (이미 `gemma3n.rs`, `gemma4.rs`, `falcon_ocr.rs`에서 이렇게 하고 있음).

**예시 코드:**
```rust
fn mixer_norm(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
    if self.use_bcdt_rms {
        mlxcel_core::fast_rms_norm_no_weight(x, self.mixer_rms_eps)
    } else {
        mlxcel_core::copy(x)
    }
}
```

---

## 7. 변경 요약

### 통계

| 항목 | 값 |
|-----|---|
| 변경된 파일 수 | 2 |
| 추가된 라인 | +130 |
| 삭제된 라인 | -20 |
| 테스트 추가 | 2 |

### 카테고리별 변경

| 카테고리 | 변경 수 | 주요 내용 |
|---------|--------|----------|
| Performance | 1 | `MambaBlock::ssm_step`에서 중복 norm 실행과 호출당 `ones` 할당 제거 |
| Code Quality | 1 | 더 이상 쓰이지 않는 `rms_norm_no_scale` 헬퍼와 낡은 "applies TWICE" 주석 삭제 |
| Testing | 2 | `mamba_mixer_norm_applies_the_bridge_norm_exactly_once`, `mamba_mixer_norm_is_identity_when_bcdt_rms_disabled` 추가 |

### 관련 커밋

| Hash | Type | Message |
|------|------|---------|
| `3b05da6` | fix | fix(mamba): apply Falcon-Mamba B/C/dt RMS norm once, no per-call ones |

---

## 8. 후속 조치

### 완료 필요
- [ ] 이슈 #1333의 실제 체크포인트 검증 실행: `models/falcon-mamba-7b-4bit`에서 변경 전/후 greedy 출력 일치(또는 첫 토큰 logit이 `1e-2` max-abs 이내로 일치) 확인, 긴 프롬프트에서 peak 메모리가 늘지 않았는지 확인.

### 모니터링 필요
- 표준 nightly `cargo test --workspace --profile test-fast --features metal,accelerate` 게이트 외에 추가로 필요한 모니터링은 없음.

### 향후 개선 사항
- `MambaBlock::forward`의 토큰별 prefill 스캔을 벡터화하는 작업은 이 PR의 범위에서 명시적으로 제외됨 (이슈 #1333에 별도 성능 항목으로 기록됨).

---

## 부록

### A. 테스트 결과

```
cargo test --profile test-fast --features metal,accelerate --lib models::mamba
test result: ok. 18 passed; 0 failed; 2 ignored; 0 measured; 7643 filtered out

cargo clippy --profile test-fast --lib --tests --features metal,accelerate -- -D warnings
Finished `test-fast` profile [optimized] target(s)  (경고 없음)

cargo fmt --all -- --check
(출력 없음; 정상)
```

### C. 참고 자료

- 이슈: #1333
- 업스트림 레퍼런스: [`mlx_lm/models/mamba.py`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/mamba.py) (`ssm_step`, `use_bcdt_rms` 분기)
