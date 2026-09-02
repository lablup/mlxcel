# 기술 보고서: PR #1576 - fix(lora): refuse adapters that do not map onto the model

**작성일**: 2026-09-02
**작성자**: mlxcel maintainers
**리뷰어**: 없음
**상태**: 완료
**언어**: Rust
**위험도**: Medium

---

## 요약

LoRA 어댑터 로딩은 적용할 수 없는 텐서를 조용히 버렸다. 그래서 다른 아키텍처용으로 학습된 어댑터, 이름이 바뀐 프로젝션, DoRA 체크포인트를 물려도 서버는 정상 기동했고, 시작 로그는 어댑터가 융합됐다고 보고하는데 실제로는 손대지 않은 베이스 가중치로 응답했다. 이 PR은 첫 쓰기 이전에 모든 어댑터 텐서를 베이스 가중치 맵에 대조해 검증하고, 위반 항목을 하나의 에러에 전부 담아 보고하며, 조작된 "N개 레이어 융합" 카운트를 실제로 적용된 쌍의 개수로 바꾸고, DoRA를 이름을 명시해 거부한다. 융합 경로, 파이프라인 병렬 스테이지 경로, 비융합 런타임 서빙 경로가 이제 하나의 검증기를 공유한다.

---

## 1. 문제 정의

### 1.1 배경

`mlxcel-server --adapter <dir>`와 `mlxcel generate --adapter <dir>`는 모델 생성 전에 LoRA 어댑터를 베이스 가중치 맵에 융합한다. 융합 루프는 어댑터 텐서를 `.lora_a` / `.lora_b` 어간으로 묶고, 각 어간을 베이스 가중치 키로 해석한 뒤, 그 키에 `scale * (B @ A)`를 더했다. 이 해석 과정의 모든 단계가 best-effort였다.

### 1.2 기존 문제

- **베이스 가중치가 없으면 경고로 끝났다.** `find_base_weight_name`은 실패할 수 없는 함수였다. 후보 세 개 중 어느 것도 체크포인트에 없으면 "가장 그럴듯한 후보"라며 `"{name}.weight"`를 그대로 돌려줬다. 융합 루프는 그 키를 찾다 실패하고 `tracing::warn!("Base weight not found for LoRA layer ...")`를 남긴 뒤 계속 진행했다. 대응되는 레이어가 하나도 없는 84쌍짜리 어댑터가 아무것도 융합하지 않은 채로 로딩에 성공했다.
- **인식되지 않는 텐서는 버려졌다.** `.lora_a`나 `.lora_b`로 끝나지 않는 것은 `// Ignore other weights (like scales for DoRA)`라는 주석과 함께 건너뛰었다. DoRA 크기(magnitude) 벡터, 떠도는 `.weight`, 이 빌드가 읽지 않는 HuggingFace PEFT의 `.lora_A.weight` 표기가 모두 그 줄에서 사라졌다.
- **보고된 개수가 조작된 값이었다.** `apply_lora_adapters`와 `apply_stage_lora_adapter`는 어댑터 파일의 `.lora_a` 키를 세어 `modified_count`를 만들고 `Fused LoRA adapters into N layers`로 기록했다. 이 값은 어댑터 자체의 크기일 뿐, 실제로 `mlxcel_core::add`까지 도달한 쌍의 수와 전혀 무관했다.
- **DoRA가 LoRA로 통과했다.** `AdapterConfig::is_lora()`가 `FineTuneType::DoRA`에 대해 true를 반환했다. 그래서 DoRA 어댑터가 타입 게이트를 통과했고, 자체 툴링이 접어 넣는 출력 행별 크기 벡터 없이 저랭크 쌍만 적용됐다.
- **비융합 런타임 경로에 같은 결함 세 가지가 있었다.** #1439 이후 b10621 `--lora` 표기의 기본 채널인 `stage_runtime_adapters`에도 동일한 `warn!` 후 continue 블록이 있었고, #1328이 두 경로를 모두 엄격하게 만드는 일을 담당한다는 주석이 소스에 남아 있었다.

변경 전 `target/release/mlxcel`로 로컬 체크포인트에 대해 측정한 결과: `models/mlx/qwen2.5-0.5b-bf16`에 `models/lora-dense-test`의 복사본(단, `model.layers.0.self_attn.q_proj.lora_a`를 `...lora_a.bogus`로 개명)을 물리면 로딩에 성공해 72쌍 중 71쌍을 융합하고 정상 생성했다. 같은 베이스에 같은 어댑터를 `"fine_tune_type": "dora"`로 선언해도 마찬가지로 로딩되고 생성됐다.

### 1.3 위험 평가

| 위험 | 영향 | 발생 가능성 |
|------|------|------------|
| 어댑터 이름을 달고 베이스 가중치를 서빙, 어디에도 에러 없음 | High | 어댑터/체크포인트 불일치 시 High |
| 부분 적용된 어댑터(일부 레이어만 융합) | High | 레이어 수 불일치 시 High |
| DoRA가 LoRA로 적용되어 베이스도 파인튜닝도 아닌 가중치 생성 | Medium | DoRA 어댑터 확보 여부에 좌우, Medium |
| 운영자가 실제를 반영한 적 없는 융합 레이어 수를 신뢰 | Medium | 텐서가 하나라도 건너뛰어지면 확실 |

---

## 2. 기술 검토

### 2.1 보안

보안 변경은 아니다. 이 결함군은 노출이 아니라 조용한 정합성 오류다. 다만 인접한 속성 하나가 개선된다. 이제 어댑터 디렉터리는 그것이 만들어내는 에러로 완전히 설명되므로, 파인튜닝이 적용되고 있다고 운영자가 오인할 여지가 없어진다.

**체크리스트:**
- [x] 입력 검증(어댑터 텐서 이름과 베이스 가중치 해석을 가정하지 않고 검사)
- [ ] 인증/인가(해당 없음)
- [ ] 데이터 암호화(해당 없음)
- [x] 로깅(융합 개수가 실측값으로 바뀜, 민감 정보 로깅 없음)

### 2.2 성능

측정 가능한 영향 없음. 검증은 어댑터의 텐서 이름을 한 번 순회하며, 그 비용은 어댑터 파일 크기(키 수백 개)로 제한된다. 게다가 같은 작업이 이미 융합 루프 안에서 인라인으로 수행되고 있었다. 유효한 어댑터에 대해 MLX 호출 순서는 그대로이므로 디코드 처리량과 로딩 시간은 변하지 않는다.

### 2.3 호환성 및 의존성

- **파괴적 변경**: 있음, 의도한 것이다. 아무것도 적용하지 않으면서 로딩되던 어댑터, 자신의 일부만 적용하던 어댑터가 이제 로딩에 실패한다. 그것이 이 변경의 목적이다. 공개 시그니처 세 개가 바뀌었다. `fuse_lora_weights_into`가 `Result<()>` 대신 `Result<usize>`를, `apply_stage_lora_adapter`가 `Result<()>` 대신 `Result<usize>`를 반환하고, `AdapterConfig::is_lora()`가 `AdapterConfig::is_fusable_lora()`로 대체됐다. `fuse_lora_weights`와 `apply_lora_adapters`의 시그니처는 유지된다.
- **신규 의존성**: 없음.
- **호환성**: mlx-lm 어댑터(`<layer>.lora_a` / `<layer>.lora_b`)와 HuggingFace PEFT의 `<layer>.base_layer` 명명 모두 그대로 해석된다. PEFT의 `lora_A.weight` / `lora_B.weight` 표기는 이 코드가 원래 읽은 적이 없으며, 이제 무동작으로 로딩되는 대신 명시적으로 실패한다.

### 2.4 코드 품질

- **테스트 커버리지**: `lora` 모듈에 테스트 14개가 늘었다(`loader_tests.rs` 11개, 신규 `runtime_tests.rs` 4개, `config.rs` 1개, 여기에 `partial_loading_adapter_tests.rs`의 스테이지 경로 테스트 2개). `--lib lora` 셀렉터 기준 47개에서 61개로 늘었다.
- **코드 복잡도**: `fuse_lora_weights_into`가 짧아졌다. 쌍 구성, 해석, 건너뛰기 처리가 `validate_adapter_tensors`로 옮겨가고, 델타를 계산해 더하는 루프만 남았다.
- **기술 부채**: 감소. 런타임 경로의 `// #1328 owns making both strict` 표식이 해소됐고, 두 경로가 서로 어긋날 수 있는 복사본 두 벌 대신 유효한 어댑터 텐서의 정의 하나를 공유한다.

---

## 3. 기술적 선택과 그 이유

### 3.1 첫 위반에서 멈추지 않고 전부 모아 보고

**맥락:**
잘못된 아키텍처용 어댑터에는 나쁜 텐서가 하나 있는 게 아니다. 레이어마다, 대상 모듈마다 하나씩 있다. 24레이어 모델에 물린 28레이어 어댑터에는 대응되지 않는 쌍이 12개 있고, 완전히 다른 어댑터라면 전부다.

**검토한 대안:**

| 선택지 | 장점 | 단점 |
|--------|------|------|
| 첫 위반에서 실패 | 코드가 가장 단순, 에러가 짧음 | 불일치 범위를 알려면 위반 레이어 수만큼 로딩을 반복해야 하고, 매번 체크포인트 전체를 다시 읽음 |
| 전부 경고하고 하나도 적용 안 됐을 때만 실패 | 부분 어댑터가 계속 동작 | 결함을 그대로 보존, 위험한 쪽은 완전 불일치가 아니라 부분 적용임 |
| **선택: 전부 모아 한 번에 실패** | 로딩 한 번으로 전체 그림 확보, 위반 목록의 형태만 봐도 "아키텍처가 다름"과 "텐서 하나 개명"이 구분됨 | 완전히 다른 어댑터에서는 에러 텍스트가 길어짐 |

**근거:**
진단 가치는 위반 목록의 형태에 있다. 꼬리 쪽 레이어 인덱스 12개가 연속으로 나오면 "어댑터의 레이어가 모델보다 많다"는 뜻이고, 한 줄이면 "누군가 텐서 이름을 바꿨다"는 뜻이며, 모든 레이어가 나오면 "아키텍처 자체가 다르다"는 뜻이다. 첫 실패만 보고하는 에러는 이 중 어느 것도 전달하지 못한다.

**트레이드오프:**
낯선 어댑터에서는 에러가 길어진다. 감수한다. 대안은 짧은 에러 대신 체크포인트 전체 로딩을 반복해야 범위를 알 수 있는 구조다.

### 3.2 호출부에서 결과를 검사하는 대신 `find_base_weight_name`의 폴백을 제거

**맥락:**
`find_base_weight_name`은 `Result<String>`을 반환했지만 실패할 수 없었다. 어느 후보도 맞지 않으면 첫 후보를 그대로 반환했고, 두 호출부 모두 그 뒤에 `contains_key` 검사와 경고를 붙였다.

**검토한 대안:**

| 선택지 | 장점 | 단점 |
|--------|------|------|
| 폴백은 두고 호출부가 경고 대신 실패하게 | 변경 폭이 가장 작음 | 함수가 여전히 존재하지 않는다고 스스로 아는 키를 보고하고, 다음 호출부가 같은 실수를 반복 |
| **선택: `Option<String>` 반환, 폴백 제거** | "해석됨"과 "존재함"이 같은 답이 되고, 타입이 검사를 생략할 수 없게 만듦 | 런타임 경로를 포함한 모든 호출부를 건드림 |

**근거:**
폴백이 증상이 아니라 근본 원인이다. "키를 줄 텐데, 있을 수도 없을 수도 있다"고 답하는 함수는 이 이슈가 다루는 `warn!` 후 continue를 정확히 유도하고, 실제로 독립적으로 두 번 유도했다. 부재를 `None`으로 만들면 판단이 타입 시스템으로 넘어간다.

**트레이드오프:**
함수를 공유하던 런타임 경로를 같은 PR에서 고쳐야 했다. 결과적으로는 그게 옳은 범위였다(3.4 참조).

### 3.3 적용 쌍 0개는 전체 모델 경로에서만 에러

**맥락:**
아무것도 적용하지 않는 어댑터는 이 PR이 보고하려는 실패 그 자체다. 그러나 레이어 16~31을 소유한 파이프라인 병렬 스테이지는 어댑터가 레이어 0~15를 겨냥할 때 정당하게 아무것도 적용하지 않는다.

**검토한 대안:**

| 선택지 | 장점 | 단점 |
|--------|------|------|
| 모든 곳에서 0을 에러로 | 규칙이 일관됨 | 파이프라인 병렬에서 유효한 부분 레이어 어댑터가 전부 깨짐 |
| 0을 절대 에러로 보지 않음 | 오탐 없음 | 빈 어댑터 파일이 조용히 로딩됨, 결함이 그대로 |
| **선택: 전체 모델 경로만 에러, 스테이지 경로는 개수 반환** | 두 경우 모두 옳음 | 나중에 누군가 이 비대칭을 "고치려" 할 수 있으므로 규칙을 코드에 기록해야 함 |

**근거:**
전체 모델 경로는 어댑터 전체와 체크포인트 전체를 본다. 그래서 0은 모호하지 않다. 스테이지는 양쪽의 걸러진 조각만 보므로 "이 어댑터가 잘못됐다"와 "이 어댑터는 내 것이 아니다"를 구분할 수 없다. 스테이지는 호출자가 그 수를 갖도록 개수를 반환하고, `src/distributed/pipeline/stage_executor/llama.rs`에 그곳에는 왜 검사가 없는지 설명하는 주석을 남겼다.

**트레이드오프:**
어떤 스테이지도 아무것도 적용하지 않은 파이프라인 병렬 실행은 여전히 감지되지 않는다. 이를 잡으려면 스테이지 간 조율이 필요하며 후속 과제로 남긴다(8절).

### 3.4 같은 검증기를 비융합 런타임 경로까지 확장

**맥락:**
이슈의 "Out of scope" 절이 런타임 LoRA를 언급하지만, 그 문장은 #1439가 `stage_runtime_adapters`를 넣기 전에 쓰인 것이고, 해당 함수에는 `Unmatched tensors warn with the same posture as the fused path (#1328 owns making both strict)`라는 주석이 달려 있다.

**검토한 대안:**

| 선택지 | 장점 | 단점 |
|--------|------|------|
| 융합 경로만 고치고 런타임 경로는 `Option` 시그니처 대응만 | 범위가 가장 작음 | `--lora`가 기본으로 선택하는 채널에 동일 결함이 남고, "DoRA 어댑터를 거부한다"는 인수 조건이 절반만 참인 수정이 나감 |
| **선택: `validate_adapter_tensors`와 `reject_unsupported_fine_tune_type` 공유** | 유효한 어댑터의 정의가 하나, 트리에 이미 남아 있던 표식을 해소 | 이슈 본문이 암시하는 것보다 diff가 큼 |

**근거:**
런타임 경로는 `find_base_weight_name`을 호출하므로 어차피 수정해야 했다. 그 상황에서 관대하게 남겨두면 기본 서버 채널에서는 성립하지 않는 수정을 내보내는 셈이고, 쌍 구성 로직 복사본 두 벌이 어긋날 여지를 남긴다.

**트레이드오프:**
`stage_runtime_adapters`에 스테이징 0개 검사가 추가됐는데 `RuntimeLoraSet` 라우트 테스트는 이를 실행하지 않는다. 대신 신규 `runtime_tests.rs`가 담당한다.

---

## 4. 구현 상세

### 4.1 아키텍처 변경

```
[변경 전]
apply_lora_adapters ─┐
                     ├─> fuse_lora_weights_into ─> 인라인 쌍 구성 ─> warn+skip ─> add
apply_stage_lora_adapter ┘                                    ^
                                                              └── find_base_weight_name (실패 불가)
stage_runtime_adapters ──> 인라인 쌍 구성 ──> warn+skip ──> 항 스테이징

[변경 후]
apply_lora_adapters ─┐
                     ├─> fuse_lora_weights_into ─┐
apply_stage_lora_adapter ┘                       ├─> validate_adapter_tensors ─> Vec<FusablePair> | 단일 에러
                                                 │        ^
stage_runtime_adapters ──────────────────────────┘        └── find_base_weight_name -> Option<String>
```

### 4.2 주요 코드 변경

**파일: `src/lora/loader.rs`**

```rust
// 변경 전
let base_weight_name = find_base_weight_name(&base_name, base_weights)?;
let Some(base_weight) = base_weights.get(&base_weight_name) else {
    tracing::warn!("Base weight not found for LoRA layer {}: tried {}", base_name, base_weight_name);
    continue;
};

// 변경 후
let pairs = validate_adapter_tensors(base_weights, adapter_weights)?;
```

**변경 이유:** 해석과 존재 검사는 하나의 판단이며, 어떤 쓰기보다 앞서 어댑터 전체에 대해 내려진다. 그래야 실패가 모든 위반 항목을 지목하고 `base_weights`는 손대지 않은 상태로 남는다.

**파일: `src/lora/loader.rs`**

```rust
// 변경 후
if !violations.is_empty() {
    violations.sort();
    let count = violations.len();
    let noun = if count == 1 { "tensor" } else { "tensors" };
    anyhow::bail!(
        "{count} adapter {noun} cannot be applied to this model:\n  {}\n\
         Every tensor in a fusable adapter has to be one half of a <layer>.lora_a / \
         <layer>.lora_b pair whose base weight this checkpoint holds; skipping the rest \
         would serve weights that match neither the base model nor the fine-tune.",
        violations.join("\n  "),
    );
}
```

**변경 이유:** `WeightMap`은 `HashMap`이라 정렬하지 않은 보고는 실행마다 순서가 달라져 테스트도 diff도 어려워진다. 반환되는 `Vec<FusablePair>`도 같은 이유로 베이스 가중치 키 기준으로 정렬한다.

**파일: `src/lora/config.rs`**

```rust
// 변경 전
pub fn is_lora(&self) -> bool {
    matches!(self.fine_tune_type, FineTuneType::LoRA | FineTuneType::DoRA)
}

// 변경 후
pub fn is_fusable_lora(&self) -> bool {
    self.fine_tune_type == FineTuneType::LoRA
}
```

**변경 이유:** 옛 이름은 본문이 지키지 않는 것을 주장했고, 세 호출부 모두 이를 "이걸 적용해도 되나?" 게이트로 썼다. 호출부를 땜질하는 대신 이름을 바꿔 함정 자체를 없앴다.

### 4.3 데이터 모델 변경

없음. 온디스크 포맷, 설정 스키마, 와이어 포맷 모두 그대로다.

---

## 5. 학습 포인트

### 5.1 실패할 수 없는 해석 함수는 실패를 모든 호출부로 떠민다

**개념:**
조회 헬퍼가 부재 대신 최선의 추측을 반환하면, 반환 타입이 호출자에게 필요한 정보를 더는 담지 못한다. 그러면 모든 호출자가 "이게 실제로 해석된 건가?"를 각자 다시 판단해야 하고, 각자 독립적으로 틀릴 수 있다.

**이 PR에서의 적용:**
`find_base_weight_name`은 `Result<String>`을 반환하면서 아무것도 맞지 않을 때 `Ok(format!("{}.weight", lora_name))`으로 끝났다. 두 호출부 모두 그 뒤에 `contains_key` 검사와 `tracing::warn!`을 붙였고, 둘 다 계속 진행하는 쪽을 택했다. 시그니처를 `Option<String>`으로 바꾸자 부재를 성공으로 표현할 수 없게 되었고, 갈라져 있던 두 처리가 하나로 합쳐졌다.

**일반적인 사례:**
- "기본" 후보로 폴백하는 이름/경로 해석기
- 키가 설정되지 않았다고 보고하는 대신 기본값을 대입하는 설정 조회
- 설명해야 할 실패로부터 `Err` 갈래에 도달할 수 없는 모든 `Result`

**예시 코드:**
```rust
// 피해야 할 형태: Ok 값이 존재할 수도 없을 수도 있다.
fn resolve(name: &str, map: &Map) -> Result<String> {
    for c in candidates(name) { if map.contains_key(&c) { return Ok(c); } }
    Ok(format!("{name}.weight")) // "가장 그럴듯한 후보"
}

// 호출자가 한 번만 판단하게 강제하는 형태.
fn resolve(name: &str, map: &Map) -> Option<String> {
    candidates(name).into_iter().find(|c| map.contains_key(c))
}
```

### 5.2 입력에서 뽑은 개수는 출력에 대한 보고가 아니다

**개념:**
완료된 작업을 센 것이 아니라 입력을 잰 N으로 "N개 처리함"을 기록하면, 아무 문제가 없을 때만 정확하고 문제가 생겼을 때만 오해를 부르는 로그가 나온다.

**이 PR에서의 적용:**
`modified_count`는 어댑터 파일의 `.lora_a` 키를 셌다. 모든 쌍이 융합됐을 때만 실제와 일치했는데, 그때는 아무도 로그를 읽지 않는다. 건너뛴 경우마다 과장했는데, 그때는 누군가 읽는다.

**일반적인 사례:**
- 성공 개수 대신 배치 크기를 보고하는 배치 처리기
- 적용된 파일 수 대신 발견한 파일 수를 보고하는 마이그레이션 도구
- 결과를 만드는 루프보다 앞서 쓰인 모든 "N items" 로그

### 5.3 어댑터 융합과 양자화 체크포인트

**개념:**
4비트 MLX 체크포인트는 프로젝션의 `.weight`를 패킹된 `uint32`로 저장하고 `.scales` / `.biases` 평면을 따로 둔다. LoRA 델타는 `[out_features, in_features]` 형태의 조밀한 실수 행렬이다.

**이 PR에서의 적용:**
정직한 스모크 계획을 세우는 과정에서, 융합 `--adapter` 경로는 양자화 체크포인트로는 아예 검증할 수 없다는 사실이 드러났다. `models/mlx/qwen3-0.6b-4bit`는 `q_proj.weight`를 `[2048, 128]` `U32`로 저장하는데 델타는 `[2048, 1024]`이므로, 이 PR의 검증에 닿기도 전에 기존 shape 가드가 먼저 발동한다. 비융합 런타임 경로는 이를 올바르게 처리한다. `validate_pair_shapes`가 패킹된 폭이 아니라 scales 평면의 그룹 수로 `in_features`를 고정하기 때문이다. 이 PR이 바꾼 부분은 아니고, 다음에 어댑터 스모크를 계획하는 사람이 융합 경로에는 bf16 체크포인트를, `--lora`에는 양자화 체크포인트를 고르도록 기록해 둔다.

---

## 6. 추가 학습

### 핵심 용어

| 키워드 | 설명 | 이 PR에서의 의미 |
|--------|------|-----------------|
| `LoRA` | 저랭크 적응: 동결된 베이스 가중치에 `scale * B @ A`를 더함 | 이 코드가 적용하는 델타 |
| `DoRA` | 가중치 분해 LoRA: 저랭크 쌍에 출력 행별 크기 벡터를 더함 | 이 PR 전에는 LoRA로 통과, 이후에는 거부 |
| `fusion` | 모델 생성 전에 로딩 시점에 델타를 베이스 가중치에 더하는 것 | `--adapter`와 `--lora-fuse`가 타는 경로 |
| `runtime LoRA` | 쌍을 융합하지 않고 레이어가 forward마다 읽는 라이브 스케일 핸들 뒤에 두는 것 | #1439 이후 `--lora`의 기본 채널 |
| `base_layer` | 감싸인 동결 프로젝션에 대한 HuggingFace PEFT의 이름 | 베이스 가중치 후보 세 개 중 하나 |
| `LayerFilter` | 파이프라인 스테이지가 소유한 레이어 범위와 embedding/lm_head 플래그 | 스테이지 경로에서 적용 쌍 0개가 유효한 이유 |

### 관련 기술/프레임워크

- **mlx-lm**: 어댑터 명명과 스케일링의 레퍼런스 구현
  - https://github.com/ml-explore/mlx-lm
- **HuggingFace PEFT**: `base_layer` / `lora_A` / `lora_B` 명명 규약
  - https://github.com/huggingface/peft

### 관련 PR/이슈

- 이슈 #1328: 이 PR이 닫는 결함
- 이슈 #1439: 비융합 런타임 경로를 추가하면서 이 PR이 해소하는 `#1328 owns making both strict` 표식을 남김

---

## 7. 변경 요약

### 통계

| 항목 | 값 |
|------|-----|
| 변경 파일 | 11 |
| 추가 라인 | +941 |
| 삭제 라인 | -180 |
| 추가 테스트 | 17 |

### 카테고리별 변경

| 카테고리 | 개수 | 요약 |
|----------|------|------|
| 정합성 | 5 | 텐서 검증, 베이스 가중치 해석, 적용 쌍 개수, DoRA 거부, 런타임 경로 동등성 |
| 코드 품질 | 2 | 세 호출 경로가 검증기 공유, 테스트가 온디스크 어댑터 픽스처 공유 |
| 문서 | 1 | `docs/server-features.md`에 수용 규칙과 스테이지 예외 기록 |

### 관련 커밋

| 해시 | 유형 | 메시지 |
|------|------|--------|
| `fc6467a` | fix | fix(lora): refuse adapters whose tensors do not map onto the model |

---

## 8. 후속 조치

### 필수

- [ ] `models/mlx/qwen2.5-0.5b-bf16`에 대한 실제 체크포인트 스모크: `models/lora-dense-test`(72쌍 적용), `models/lora-runtime-test`(대응 없는 12쌍 보고), 텐서를 개명한 복사본, DoRA로 선언한 복사본. 정확한 명령과 기대값은 PR 본문에 있다.

### 모니터링 필요

- 기존에 정상 기동하던 배포에서 `adapter tensors cannot be applied`를 지목하는 시작 실패. 이런 실패는 회귀가 아니라 결함이 보고되는 것이지만, 해당 어댑터는 그동안 베이스 가중치를 서빙하고 있었으므로 운영자에게 알려야 한다.

### 향후 개선

- 어떤 스테이지도 쌍을 적용하지 않은 파이프라인 병렬 실행은 여전히 감지되지 않는다. 감지하려면 모든 스테이지 보고 후의 스테이지 간 집계가 필요하다.
- DoRA 융합(출력 행별로 `W' = m * (W + scale * B A) / ||W + scale * B A||`)은 미구현이다. 검증할 DoRA 체크포인트가 있어야 한다.
- 양자화 베이스에 대한 융합 어댑터 적용은 미지원이다(역양자화, 덧셈, 재양자화). 지금은 shape 가드에서 실패한다. 양자화 체크포인트에는 비융합 `--lora` 경로가 동작하는 답이다.
- HuggingFace PEFT의 `lora_A.weight` / `lora_B.weight` 표기는 이제 조용히 무시되는 대신 명확히 보고되지만, 여전히 읽지는 않는다.

---

## 부록

### A. 테스트 결과

| 명령 | 결과 |
|------|------|
| `cargo test --profile test-fast --features metal,accelerate --lib lora` | 61 passed, 0 failed |
| `cargo test --profile test-fast --features metal,accelerate --lib distributed::pipeline` | 323 passed, 0 failed |
| `cargo test --profile test-fast --features metal,accelerate --lib loading::` | 312 passed, 0 failed |
| `cargo clippy --profile test-fast --lib --tests --features metal,accelerate -- -D warnings` | clean |
| `cargo fmt --all -- --check` | clean |

### B. 성능 벤치마크

해당 없음. 검증은 어댑터 키 수로 제한된 로딩 시간에만 관여하고, 유효한 어댑터에 대한 MLX 호출 순서는 변하지 않는다.

### C. 참고 자료

- `src/lora/loader.rs`: `validate_adapter_tensors`, `find_base_weight_name`, `reject_unsupported_fine_tune_type`
- `src/lora/runtime.rs`: `stage_runtime_adapters`
- `docs/server-features.md`: "LoRA adapters"
