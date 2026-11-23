# 기술 리포트: PR #3335 - 패밀리별 MoE expert stacking에서 shape가 다른 expert 거부

**작성일**: 2026-11-23
**상태**: PR 열림 (#3320 종료). 단위 테스트 통과, 로컬 체크포인트 4개가 기존과 동일하게 로드 및 생성
**언어**: Rust
**위험도**: 낮음

---

## 요약

여섯 MoE 패밀리(DeepSeek-V3, Mistral4, PhiMoE, OLMoE, MiMo-V2, Mellum)는 각자의 sanitize 코드에서 expert별 텐서를 stack한다. 한 expert의 plane shape가 다른 체크포인트는 `mlxcel_core::stack`에 그대로 도달했고, MLX의 `[stack] All arrays must have the same shape` 예외가 cxx 브리지를 넘어가 프로세스를 중단시켰다. 이제 각 지점이 stack 전에 #3304의 공용 헬퍼 `refuse_mismatched_expert_parts`로 검사하므로, 같은 입력은 문제의 키와 두 shape를 담은 로드 에러로 실패한다.

---

## 1. 문제 정의

### 1.1 배경

#3304(이슈 #3297)는 `src/models/switch_layers.rs`에 `refuse_mismatched_expert_parts`를 추가하고 공용 stacking 경로(`drain_numbered_expert_planes`, `stack_individual_experts_with_count`)에 적용했다. 자체 루프로 expert를 stack하는 패밀리는 빠져 있었다.

### 1.2 기존 문제점

- **로드 에러 대신 프로세스 중단**: `mlxcel_core::stack`과 `stack_owned`는 FFI에서 실패를 반환할 수 없다. C++ `std::invalid_argument`가 잡히지 않아 `libc++abi`가 프로세스를 종료한다(SIGABRT). 이런 체크포인트를 로드하는 서버는 잘못된 파일을 알리지 못하고 죽는다.
- **실패를 반환할 수 없는 시그니처**: Mistral4 `sanitize_weights`, OLMoE `sanitize_weights`, Mellum `sanitize` / `stack_experts`는 `()` 또는 `WeightMap`을 반환해 문제를 보고할 방법이 없었다.

### 1.3 위험성

| 위험 | 영향 | 가능성 |
|------|------|--------|
| 손상되거나 잘못 변환된 체크포인트가 서빙 프로세스를 종료 | 높음 | 낮음 |
| 어떤 텐서가 잘못됐는지 에러에 단서가 없음 | 중간 | 낮음 |

---

## 2. 기술적 검토 사항

### 2.3 호환성/의존성 관점

- **Breaking change**: crate 내부 시그니처 세 개가 fallible이 됐다(`mistral4::sanitize_weights`는 `pub`이지만 crate 안에서만 사용). 모든 호출부가 에러를 전파한다: `Mistral4Model::load`, `load_from_text_config`, `src/loading/mod.rs`, `src/loading/vlm_pixtral.rs`, `OlmoeModel::load`, `MellumModel::load`.
- **정상 체크포인트**: 변화 없음. 검사는 shape와 dtype 메타데이터만 읽고 평가나 복사를 하지 않는다.

### 2.4 코드 품질 관점

- 패밀리마다 테스트 하나씩 총 6개를 추가했다. 검사를 끄면 각각 SIGABRT로 실패하고, 켜면 통과한다.
- 헬퍼의 `Used by` 목록에 새 호출부를 적었고 `scripts/ci/check_used_by.py`가 통과한다.

---

## 3. 기술적 선택과 그 이유

### 3.1 검사용 stacking 래퍼 대신 각 지점에서 공용 헬퍼 호출

| 옵션 | 장점 | 단점 |
|------|------|------|
| A: `switch_layers.rs`에 `checked_stack_expert_leaf` 래퍼 추가 | 지점마다 호출 한 번 | `stack_expert_leaf`를 쓰는 세 곳에만 맞음. MiMo-V2와 Mellum은 `stack_owned`, OLMoE는 `stack` 사용 |
| **선택: B: `refuse_mismatched_expert_parts` 직접 호출** | 이슈가 지정한 대로 모든 패밀리가 같은 헬퍼와 메시지 사용 | 지점마다 `(expert, &array)` 수집 코드가 몇 줄 추가 |

### 3.2 `global_scale` 제외

`stack_expert_leaf`는 ModelOpt NVFP4 sidecar를 stack하지 않고 평탄화해 concatenate한다. #3304와 같이 DeepSeek-V3, Mistral4, PhiMoE는 이 leaf의 검사를 건너뛰고, 나머지 세 패밀리는 이 leaf를 stack하지 않는다.

### 3.3 PhiMoE 테스트 위치

이슈는 PhiMoE 테스트를 `phimoe_tests.rs`에 두라고 했다. `PhiMoeModel::sanitize_weights`는 private이고 `phimoe_tests.rs`는 형제 모듈이라 접근할 수 없으므로, 함수 가시성을 넓히지 않고 `phimoe.rs`의 inline `sanitize_tests` 모듈에 기존 stacking 테스트와 나란히 두었다.

---

## 4. 구현 상세

각 루프는 이제 expert 인덱스와 함께 part를 모으고, 검사한 다음에야 FFI용 포인터 목록을 만든다. DeepSeek-V3 예:

```rust
let expert_key = |e: usize| format!("{}.mlp.experts.{}.{}.{}", prefix, e, m, k);
let parts: Vec<(usize, &MlxArray)> = (0..n_routed)
    .filter_map(|e| weights.get(&expert_key(e)).map(|w| (e, w.as_ref().expect("..."))))
    .collect();
if !parts.is_empty() {
    if k != "global_scale" {
        refuse_mismatched_expert_parts(&parts, expert_key)?;
    }
    let expert_ptrs: Vec<*const MlxArray> = parts.iter().map(|(_, w)| *w as *const _).collect();
    let (leaf, stacked) = stack_expert_leaf(k, &expert_ptrs);
    ...
}
```

에러는 #3304의 형식대로 expert 1의 전체 키, shape `[4, 6]`과 dtype, expert 0의 키, shape `[4, 8]`과 dtype을 담는다.

에러가 나면 거부된 leaf에는 아무것도 삽입하지 않는다. OLMoE와 Mellum은 해당 leaf의 expert 키를 이미 맵에서 제거했지만, 모든 호출부가 `Err`에서 맵을 버린다.

---

## 5. 학습 포인트

### 5.1 cxx를 넘는 C++ 예외

`Result` 반환 없이 `cxx`로 바인딩된 C++ 함수는 예외를 Rust에 알릴 수 없고, 잡히지 않은 예외는 프로세스를 종료한다. 입력 전제 조건이 있는 FFI 호출(여기서는 `stack`의 동일 shape)은 Rust 쪽에서 먼저 전제 조건을 검사하거나, 바인딩을 fallible로 만들어야 한다.

---

## 7. 변경 요약

### 통계

| 항목 | 값 |
|------|-----|
| 변경 파일 | 소스 파일 12개 |
| 추가 테스트 | 6 |

### 관련 커밋

| Hash | 유형 | 메시지 |
|------|------|--------|
| `05cf5444` | fix | refuse mismatched experts in family-local MoE stacking |

### 관련 PR/이슈

- #3304 / #3297: 이 PR이 재사용하는 공용 expert shape 검사
- #3331 / #3319: 이 PR의 기반인 DeepSeek-V3 비양자화 MoE 로딩

---

## 8. 후속 조치

### 향후 개선 사항

- `global_scale` sidecar는 여전히 검사하지 않는다. 두 expert의 sidecar rank가 다르면 `stack_expert_leaf` 내부의 concatenate가 여전히 중단될 수 있다. #3304의 범위와 같다.
- Mellum은 expert가 빠졌을 때 여전히 에러 없이 일찍 반환한다(#3320에서 명시적으로 범위 밖).

---

## 부록

### A. 테스트 결과

GPU lock 아래에서 `cargo test --profile test-fast --features metal,accelerate --lib <selector>` 실행:

| Selector | 결과 |
|----------|------|
| `mismatched_expert` | 9 통과 (신규 6) |
| `models::deepseek_v3` | 73 통과 |
| `models::mistral4` | 4 통과 |
| `models::phimoe` | 15 통과 |
| `models::olmoe` | 2 통과, 2 ignored |
| `models::mimo_v2` | 20 통과 |
| `models::mellum` | 7 통과 |
| `models::switch_layers` | 66 통과, 1 ignored |
| `coverage_tests` | 18 통과 |

Failing first: 헬퍼를 우회하자 신규 테스트 6개가 모두 `[stack] All arrays must have the same shape`로 중단됐다(SIGABRT).

### B. 체크포인트 회귀 검사

`mlxcel generate -p "The capital of France is" -n 24 --temp 0` (release 빌드):

| 체크포인트 | 패밀리 | Expert 레이아웃 | 출력 |
|------------|--------|-----------------|------|
| `gigachat3-10b-a1.8b-bf16` | DeepSeek-V3 | expert별 (이 코드가 stack) | "Paris." |
| `olmoe-1b-7b-0125-4bit` | OLMoE | expert별 (이 코드가 stack) | "Paris." |
| `gigachat3-10b-a1.8b-mlx-8bit` | DeepSeek-V3 | 양자화 | "Paris." |
| `moonlight-16b-a3b-instruct-4bit` | DeepSeek-V3 | 양자화 | "Paris." |

로컬에 Mistral4, PhiMoE, Mellum 체크포인트는 없다. `MiMo-V2-Flash-4bit`(162 GB)는 100 GB 제한으로 건너뛰었다.
